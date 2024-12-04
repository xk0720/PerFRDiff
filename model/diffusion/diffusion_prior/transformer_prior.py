import torch
import torch.nn as nn
from torch import einsum
import torch.nn.functional as F
from einops import rearrange, repeat
import math
from model.diffusion.diffusion_prior.rotary_embedding_torch import RotaryEmbedding
from enum import Enum
from einops.layers.torch import Rearrange
from model.diffusion.utils.util import prob_mask_like


def exists(val):
    return val is not None


def l2norm(t):
    return F.normalize(t, dim=-1)


class Activation(Enum):
    none = 'none'
    relu = 'relu'
    lrelu = 'lrelu'
    silu = 'silu'
    tanh = 'tanh'

    def get_act(self):
        if self == Activation.none:
            return nn.Identity()
        elif self == Activation.relu:
            return nn.ReLU()
        elif self == Activation.lrelu:
            return nn.LeakyReLU(negative_slope=0.2)
        elif self == Activation.silu:
            return nn.SiLU()
        elif self == Activation.tanh:
            return nn.Tanh()
        else:
            raise NotImplementedError()


class SwiGLU(nn.Module):
    """ used successfully in https://arxiv.org/abs/2204.0231 """

    def forward(self, x):
        x, gate = x.chunk(2, dim=-1)
        return x * F.silu(gate)


def timestep_embedding(timesteps, dim, max_period=10000, dtype=torch.float32):
    """
    Create sinusoidal timestep embeddings.
    :param timesteps: a 1-D Tensor of N indices, one per batch element.
                      These may be fractional.
    :param dim: the dimension of the output.
    :param max_period: controls the minimum frequency of the embeddings.
    :return: an [N x dim] Tensor of positional embeddings.
    """
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period) * torch.arange(start=0, end=half, dtype=dtype) / half
    ).to(device=timesteps.device)
    args = timesteps[:, None].type(dtype) * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding


class RelPosBias(nn.Module):
    def __init__(
            self,
            heads=8,
            num_buckets=32,
            max_distance=128,
    ):
        super().__init__()
        self.num_buckets = num_buckets
        self.max_distance = max_distance
        self.relative_attention_bias = nn.Embedding(num_buckets, heads)

    @staticmethod
    def _relative_position_bucket(
            relative_position,
            num_buckets=32,
            max_distance=128
    ):
        n = -relative_position
        n = torch.max(n, torch.zeros_like(n))

        max_exact = num_buckets // 2
        is_small = n < max_exact

        val_if_large = max_exact + (torch.log(n.float() / max_exact) / math.log(max_distance / max_exact) * (
                num_buckets - max_exact)).long()
        val_if_large = torch.min(val_if_large, torch.full_like(val_if_large, num_buckets - 1))
        return torch.where(is_small, n, val_if_large)

    def forward(self, i, j, *, device):
        q_pos = torch.arange(i, dtype=torch.long, device=device)
        k_pos = torch.arange(j, dtype=torch.long, device=device)
        rel_pos = rearrange(k_pos, 'j -> 1 j') - rearrange(q_pos, 'i -> i 1')
        rp_bucket = self._relative_position_bucket(rel_pos, num_buckets=self.num_buckets,
                                                   max_distance=self.max_distance)
        values = self.relative_attention_bias(rp_bucket)
        return rearrange(values, 'i j h -> h i j')


class LayerNorm(nn.Module):
    def __init__(self, dim, eps=1e-5, fp16_eps=1e-3, stable=False):
        super().__init__()
        self.eps = eps
        self.fp16_eps = fp16_eps
        self.stable = stable
        self.g = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        eps = self.eps if x.dtype == torch.float32 else self.fp16_eps

        if self.stable:
            x = x / x.amax(dim=-1, keepdim=True).detach()

        var = torch.var(x, dim=-1, unbiased=False, keepdim=True)
        mean = torch.mean(x, dim=-1, keepdim=True)
        return (x - mean) * (var + eps).rsqrt() * self.g


def FeedForward(
        dim,
        mult=4,
        dropout=0.,
        post_activation_norm=False
):
    """ post-activation norm https://arxiv.org/abs/2110.09456 """

    inner_dim = int(mult * dim)
    return nn.Sequential(
        LayerNorm(dim),
        nn.Linear(dim, inner_dim * 2, bias=False),
        SwiGLU(),
        LayerNorm(inner_dim) if post_activation_norm else nn.Identity(),
        nn.Dropout(dropout),
        nn.Linear(inner_dim, dim, bias=False)
    )


class Attention(nn.Module):
    def __init__(
            self,
            dim,
            *,
            dim_head=64,
            heads=8,
            dropout=0.,
            causal=False,
            rotary_emb=None,
            cosine_sim=True,
            cosine_sim_scale=16
    ):
        super().__init__()
        self.scale = cosine_sim_scale if cosine_sim else (dim_head ** -0.5)
        self.cosine_sim = cosine_sim

        self.heads = heads
        inner_dim = dim_head * heads

        self.causal = causal
        self.norm = LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)

        self.null_kv = nn.Parameter(torch.randn(2, dim_head))
        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_kv = nn.Linear(dim, dim_head * 2, bias=False)

        self.rotary_emb = rotary_emb

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim, bias=False),
            LayerNorm(dim)
        )

    def forward(self, x, mask=None, attn_bias=None):
        b, n, device = *x.shape[:2], x.device

        x = self.norm(x)
        q, k, v = (self.to_q(x), *self.to_kv(x).chunk(2, dim=-1))

        q = rearrange(q, 'b n (h d) -> b h n d', h=self.heads)
        q = q * self.scale

        # rotary embeddings
        if exists(self.rotary_emb):
            q, k = map(self.rotary_emb.rotate_queries_or_keys, (q, k))

        # add null key / value for classifier free guidance in prior net
        nk, nv = map(lambda t: repeat(t, 'd -> b 1 d', b=b), self.null_kv.unbind(dim=-2))
        k = torch.cat((nk, k), dim=-2)
        v = torch.cat((nv, v), dim=-2)

        # whether to use cosine sim
        if self.cosine_sim:
            q, k = map(l2norm, (q, k))

        q, k = map(lambda t: t * math.sqrt(self.scale), (q, k))

        # calculate query / key similarities
        sim = einsum('b h i d, b j d -> b h i j', q, k)

        # relative positional encoding (T5 style)
        if exists(attn_bias):
            sim = sim + attn_bias

        # masking
        max_neg_value = -torch.finfo(sim.dtype).max

        if exists(mask):
            mask = F.pad(mask, (1, 0), value=True)
            mask = rearrange(mask, 'b j -> b 1 1 j')
            sim = sim.masked_fill(~mask, max_neg_value)

        if self.causal:
            i, j = sim.shape[-2:]
            causal_mask = torch.ones((i, j), dtype=torch.bool, device=device).triu(j - i + 1)
            sim = sim.masked_fill(causal_mask, max_neg_value)

        # attention
        attn = sim.softmax(dim=-1, dtype=torch.float32)
        attn = attn.type(sim.dtype)
        attn = self.dropout(attn)

        # aggregate values
        out = einsum('b h i j, b j d -> b h i d', attn, v)

        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)


class CausalTransformer(nn.Module):
    def __init__(
            self,
            *,
            dim=512,  # latent_dim
            depth=4,
            dim_head=64,
            heads=8,
            ff_mult=4,
            norm_in=False,
            norm_out=True,
            attn_dropout=0.,
            ff_dropout=0.,
            final_proj=True,
            normformer=False,
            rotary_emb=True,
    ):
        super().__init__()
        self.init_norm = LayerNorm(dim) if norm_in else nn.Identity()  # from latest BLOOM model and Yandex's YaLM

        self.rel_pos_bias = RelPosBias(heads=heads)

        rotary_emb = RotaryEmbedding(dim=min(32, dim_head)) if rotary_emb else None

        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Attention(dim=dim, causal=True, dim_head=dim_head, heads=heads, dropout=attn_dropout,
                          rotary_emb=rotary_emb),
                FeedForward(dim=dim, mult=ff_mult, dropout=ff_dropout, post_activation_norm=normformer)
            ]))

        self.norm = LayerNorm(dim,
                              stable=True) if norm_out else nn.Identity()
        self.project_out = nn.Linear(dim, dim, bias=False) if final_proj else nn.Identity()

    def forward(self, x):
        n, device = x.shape[1], x.device

        x = self.init_norm(x)

        attn_bias = self.rel_pos_bias(n, n + 1, device=device)

        for attn, ff in self.layers:
            x = attn(x, attn_bias=attn_bias) + x
            x = ff(x) + x

        out = self.norm(x)
        return self.project_out(out)


class DiffusionPriorNetwork(nn.Module):
    def __init__(self,
                 audio_dim=78,
                 window_size=50,
                 _3dmm_dim=58,
                 speaker_emb_dim=512,
                 latent_dim=512,
                 depth=4,
                 num_time_layers=2,
                 num_time_embeds=1,
                 num_time_emb_channels=64,
                 time_last_act=False,
                 activation=Activation.silu,
                 use_learned_query=True,
                 s_audio_cond_drop_prob=0.0,
                 s_latentemb_cond_drop_prob=0.0,
                 s_3dmm_cond_drop_prob=0.0,
                 guidance_scale=1.0,
                 **kwargs):
        super().__init__()

        self.window_size = window_size
        self.latent_dim = latent_dim
        self.use_learned_query = use_learned_query
        self.num_time_emb_channels = num_time_emb_channels

        layers = []
        for i in range(num_time_layers):
            if i == 0:
                a = num_time_emb_channels
                b = latent_dim
            else:
                a = latent_dim
                b = latent_dim
            layers.append(nn.Linear(a, b))
            if i < num_time_layers - 1 or time_last_act:
                layers.append(activation.get_act())
        layers.append(Rearrange('b (n d) -> b n d', n=num_time_embeds))
        self.time_embed = nn.Sequential(*layers)

        self.to_audio_encodings = nn.Linear(audio_dim, latent_dim) if audio_dim != latent_dim else nn.Identity()
        self.to_speaker_latentemb = nn.Linear(speaker_emb_dim, latent_dim) \
            if speaker_emb_dim != latent_dim else nn.Identity()
        self.to_speaker_3dmmenc = nn.Linear(_3dmm_dim, latent_dim) if _3dmm_dim != latent_dim else nn.Identity()

        self.learned_query = nn.Parameter(torch.randn(latent_dim))

        self.guidance_scale = guidance_scale
        self.s_audio_cond_drop_prob = s_audio_cond_drop_prob
        self.s_latentemb_cond_drop_prob = s_latentemb_cond_drop_prob
        self.s_3dmm_cond_drop_prob = s_3dmm_cond_drop_prob
        self.null_s_audio_encodings = nn.Parameter(torch.randn(size=(1, self.window_size, self.latent_dim)))
        self.null_s_latent_embed = nn.Parameter(torch.randn(size=(1, 1, self.latent_dim)))
        self.null_s_3dmm_encodings = nn.Parameter(torch.randn(size=(1, self.window_size, self.latent_dim)))

        self.causal_transformer = CausalTransformer(dim=latent_dim, depth=depth, **kwargs)

    def forward_with_cond_scale(self, x_t, t, model_kwargs):
        self.s_audio_cond_drop_prob = 0.0
        self.s_latentemb_cond_drop_prob = 0.0
        self.s_3dmm_cond_drop_prob = 0.0
        logits = self.forward(x_t, t, model_kwargs)

        if self.guidance_scale <= 1.0:
            return logits

        self.s_audio_cond_drop_prob = 1.0
        self.s_latentemb_cond_drop_prob = 1.0
        self.s_3dmm_cond_drop_prob = 1.0
        null_logits = self.forward(x_t, t, model_kwargs)

        return null_logits + (logits - null_logits) * self.guidance_scale

    def forward(self, x_t, t, model_kwargs):
        assert x_t.shape[2] == self.latent_dim, \
            "x_t should have the same dimension as the latent dimension."

        t = timestep_embedding(t, self.num_time_emb_channels)
        time_emb = self.time_embed(t)
        bs, _, _ = time_emb.shape

        s_audio_encodings = model_kwargs.get("speaker_audio_encodings",
                                             torch.zeros(size=(bs, 0, self.latent_dim)).to(x_t.device))
        if s_audio_encodings.shape[1] > 0:
            s_audio_encodings = self.to_audio_encodings(s_audio_encodings)  # mapping

            audio_keep_mask = (
                prob_mask_like((s_audio_encodings.shape[0],),
                               (1 - self.s_audio_cond_drop_prob), device=x_t.device))
            audio_keep_mask = rearrange(audio_keep_mask, 'b -> b 1 1')
            s_audio_encodings = torch.where(
                audio_keep_mask,
                s_audio_encodings,
                self.null_s_audio_encodings.to(s_audio_encodings.device)
            )
        speaker_audio_encodings = s_audio_encodings

        s_latent_embed = model_kwargs.get("speaker_latent_emb",
                                          torch.zeros(size=(bs, 0, self.latent_dim)).to(x_t.device))
        if s_latent_embed.shape[1] > 0:
            s_latent_embed = self.to_speaker_latentemb(s_latent_embed)
            latentemb_keep_mask = (
                prob_mask_like((s_latent_embed.shape[0],),
                               (1 - self.s_latentemb_cond_drop_prob), device=s_latent_embed.device))
            latentemb_keep_mask = rearrange(latentemb_keep_mask, 'b -> b 1 1')
            s_latent_embed = torch.where(
                latentemb_keep_mask,
                s_latent_embed,
                self.null_s_latent_embed.to(s_latent_embed.device)
            )
        speaker_latent_emb = s_latent_embed

        s_3dmm_encodings = model_kwargs.get("speaker_3dmm_encodings",
                                            torch.zeros(size=(bs, 0, self.latent_dim)).to(x_t.device))
        if s_3dmm_encodings.shape[1] > 0:
            s_3dmm_encodings = self.to_speaker_3dmmenc(s_3dmm_encodings)
            _3dmm_keep_mask = (
                prob_mask_like((s_3dmm_encodings.shape[0],),
                               (1 - self.s_3dmm_cond_drop_prob), device=s_3dmm_encodings.device))
            _3dmm_keep_mask = rearrange(_3dmm_keep_mask, 'b -> b 1 1')
            s_3dmm_encodings = torch.where(
                _3dmm_keep_mask,
                s_3dmm_encodings,
                self.null_s_3dmm_encodings.to(s_3dmm_encodings.device)
            )
        speaker_3dmm_encodings = s_3dmm_encodings

        learned_queries = repeat(self.learned_query, 'd -> b 1 d', b=bs) \
            if self.use_learned_query else torch.zeros(size=(bs, 0, self.latent_dim)).to(x_t.device)

        tokens = torch.cat((
            speaker_audio_encodings,  # shape: (bs, window_size, dim)
            speaker_latent_emb,  # shape: (bs, 1, dim)
            speaker_3dmm_encodings,  # shape: (bs, window_size, dim)
            time_emb,  # shape: (bs, 1, dim)
            x_t,  # shape: (bs, 1, dim)
            learned_queries,  # shape: (bs, 1, dim)
        ), dim=-2)  # (bs, N', dim)

        tokens = self.causal_transformer(tokens)
        pred_image_embed = tokens[:, -1:, :]

        return pred_image_embed

    def get_model_name(self):
        return self.__class__.__name__
