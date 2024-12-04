from typing import Dict, List
import math
import torch
import torch.nn as nn
from torch import Tensor
from model.diffusion.operator.embeddings import (TimestepEmbedding, Timesteps)
from model.diffusion.operator.position_encoding import build_position_encoding
from model.diffusion.operator.cross_attention import (SkipTransformerEncoder,
                                                      TransformerDecoder,
                                                      TransformerDecoderLayer,
                                                      TransformerEncoder,
                                                      TransformerEncoderLayer)


def lengths_to_mask(lengths: List[int],
                    device: torch.device,
                    max_len: int = None) -> Tensor:
    lengths = torch.tensor(lengths, device=device)
    max_len = max_len if max_len else max(lengths)
    mask = torch.arange(max_len, device=device).expand(
        len(lengths), max_len) < lengths.unsqueeze(1)
    return mask


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


class TransformerDenoiser(nn.Module):
    def __init__(self,
                 encode_emotion: bool = False,
                 encode_3dmm: bool = False,
                 ablation_skip_connection: bool = True,
                 nfeats: int = 25,
                 latent_dim: int = 512,
                 ff_size: int = 1024,
                 num_layers: int = 7,
                 num_heads: int = 4,
                 dropout: float = 0.1,
                 normalize_before: bool = False,
                 activation: str = "gelu",
                 flip_sin_to_cos: bool = True,
                 return_intermediate_dec: bool = False,
                 position_embedding: str = "learned",
                 arch: str = "trans_enc",
                 freq_shift: int = 0,
                 time_encoded_dim: int = 64,
                 s_audio_dim: int = 78,
                 s_audio_scale: float = 0.04,
                 s_emotion_dim: int = 25,
                 l_embed_dim: int = 512,
                 s_embed_dim: int = 512,
                 personal_emb_dim: int = 512,
                 s_3dmm_dim: int = 58,
                 concat: str = "concat_first",
                 condition_concat: str = "token_concat",
                 guidance_scale: float = 7.5,
                 l_latent_embed_drop_prob: float = 0.2,
                 l_personal_embed_drop_prob: float = 0.2,
                 s_audio_enc_drop_prob: float = 0.2,
                 s_latent_embed_drop_prob: float = 0.2,
                 s_3dmm_enc_drop_prob: float = 0.2,
                 s_emotion_enc_drop_prob: float = 0.2,
                 past_l_emotion_drop_prob: float = 0.2,
                 **kwargs) -> None:

        super().__init__()

        self.encode_emotion = encode_emotion
        self.encode_3dmm = encode_3dmm
        self.s_audio_scale = s_audio_scale
        self.latent_dim = latent_dim
        self.ablation_skip_connection = ablation_skip_connection
        self.arch = arch
        self.concat = concat
        self.condition_concat = condition_concat
        self.guidance_scale = guidance_scale
        self.l_latent_embed_drop_prob = l_latent_embed_drop_prob
        self.l_personal_embed_drop_prob = l_personal_embed_drop_prob
        self.s_audio_enc_drop_prob = s_audio_enc_drop_prob
        self.s_latent_embed_drop_prob = s_latent_embed_drop_prob
        self.s_3dmm_enc_drop_prob = s_3dmm_enc_drop_prob
        self.s_emotion_enc_drop_prob = s_emotion_enc_drop_prob
        self.past_l_emotion_drop_prob = past_l_emotion_drop_prob

        self.to_emotion_embed = nn.Linear(nfeats, self.latent_dim) if nfeats != self.latent_dim else nn.Identity()
        self.to_emotion_feat = nn.Linear(self.latent_dim, nfeats) if self.latent_dim != nfeats else nn.Identity()

        self.time_proj = Timesteps(time_encoded_dim, flip_sin_to_cos, freq_shift)
        self.time_embedding = TimestepEmbedding(time_encoded_dim, self.latent_dim)

        self.speaker_latent_proj = nn.Sequential(nn.ReLU(), nn.Linear(s_embed_dim, self.latent_dim)) \
            if s_embed_dim != self.latent_dim else nn.Identity() 

        self.listener_latent_proj = nn.Sequential(nn.ReLU(), nn.Linear(l_embed_dim, self.latent_dim)) \
            if l_embed_dim != self.latent_dim else nn.Identity()

        self.speaker_audio_proj = nn.Linear(s_audio_dim, self.latent_dim) \
            if s_audio_dim != self.latent_dim else nn.Identity()

        if self.encode_3dmm:
            self.speaker_3dmm_proj = nn.Identity()
        else:
            assert s_3dmm_dim != self.latent_dim, "wrong dimension of raw 3dmm features."
            self.speaker_3dmm_proj = nn.Linear(s_3dmm_dim, self.latent_dim)

        if self.encode_emotion:
            self.speaker_emotion_proj = nn.Identity()
        else:
            assert s_emotion_dim != self.latent_dim, "wrong dimension of raw emotion features."
            self.speaker_emotion_proj = nn.Linear(s_emotion_dim, self.latent_dim)

        self.listener_personal_proj = nn.Linear(personal_emb_dim, self.latent_dim) \
            if personal_emb_dim != self.latent_dim else nn.Identity()

        self.listener_emotion_proj = nn.Linear(nfeats, self.latent_dim) \
            if nfeats != self.latent_dim else nn.Identity()

        self.query_pos = build_position_encoding(
            self.latent_dim, position_embedding=position_embedding)
        self.mem_pos = build_position_encoding(
            self.latent_dim, position_embedding=position_embedding)

        self.condition_proj = nn.Linear(self.latent_dim * 3, self.latent_dim) \
            if self.condition_concat == 'feat_concat' else nn.Identity()

        decoder_layer = TransformerDecoderLayer(
            self.latent_dim,
            num_heads,
            ff_size,
            dropout,
            activation,
            normalize_before,
        )

        if self.arch == "trans_enc":
            if self.ablation_skip_connection:
                encoder_layer = TransformerEncoderLayer(
                    self.latent_dim,
                    num_heads,
                    ff_size,
                    dropout,
                    activation,
                    normalize_before,
                )
                encoder_norm = nn.LayerNorm(self.latent_dim)
                self.encoder = SkipTransformerEncoder(encoder_layer,
                                                      num_layers, encoder_norm)
            else:
                encoder_layer = nn.TransformerEncoderLayer(
                    d_model=self.latent_dim,
                    nhead=num_heads,
                    dim_feedforward=ff_size,
                    dropout=dropout,
                    activation=activation)
                self.encoder = nn.TransformerEncoder(encoder_layer,
                                                     num_layers=num_layers)

        elif self.arch == "trans_dec":
            decoder_norm = nn.LayerNorm(self.latent_dim)
            self.decoder = TransformerDecoder(
                decoder_layer,
                num_layers,
                decoder_norm,
                return_intermediate=return_intermediate_dec,
            )
        else:
            raise ValueError(f"Not supported architechure{self.arch}!")

        if self.condition_concat == 'cascade':
            num_layer_ns2sa = 3
            num_layer_ns2sv = 3

            self.transformer_fusion_ns2sa = TransformerDecoder(
                decoder_layer, num_layer_ns2sa, nn.LayerNorm(self.latent_dim),
                return_intermediate=return_intermediate_dec,
            )

            self.transformer_fusion_ns2sv = TransformerDecoder(
                decoder_layer, num_layer_ns2sv, nn.LayerNorm(self.latent_dim),
                return_intermediate=return_intermediate_dec,
            )

            self.transformer_fusion_final = TransformerDecoder(
                decoder_layer, 1, nn.LayerNorm(self.latent_dim),
                return_intermediate=return_intermediate_dec,
            )

    def mask_cond(self, feature, mode='test', drop_prob=0.0):
        bs, _, _ = feature.shape

        if mode == 'test':
            uncond_feat, con_feat = feature.chunk(2)
            uncond_feat = torch.zeros_like(uncond_feat)
            feature = torch.cat((uncond_feat, con_feat), dim=0)

        else:
            if drop_prob > 0.0:
                mask = torch.bernoulli(
                    torch.ones(bs, device=feature.device) *
                    drop_prob).view(
                    bs, 1, 1)  # 1-> use null_cond, 0-> use real cond
                feature = feature * (1.0 - mask)

        return feature

    def get_model_kwargs(
            self,
            bs,
            mode,
            sample,
            model_kwargs,
    ):  # ALL CONDITIONS:

        listener_latent_embed = model_kwargs.get('listener_latent_embed')
        if listener_latent_embed is None or self.l_latent_embed_drop_prob >= 1.0:
            listener_latent_embed = torch.zeros(size=(bs, 0, self.latent_dim)).to(sample.device)
        else:
            listener_latent_embed = self.listener_latent_proj(listener_latent_embed)
            listener_latent_embed = self.mask_cond(listener_latent_embed, mode, self.l_latent_embed_drop_prob)
        listener_latent_embed = listener_latent_embed.permute(1, 0, 2).contiguous()

        listener_personal_embed = model_kwargs.get('listener_personal_embed')
        if listener_personal_embed is None or self.l_personal_embed_drop_prob >= 1.0:
            listener_personal_embed = torch.zeros(size=(bs, 0, self.latent_dim)).to(sample.device)
        else:
            listener_personal_embed = self.listener_personal_proj(listener_personal_embed)
            listener_personal_embed = self.mask_cond(listener_personal_embed, mode, self.l_personal_embed_drop_prob)
        listener_personal_embed = listener_personal_embed.permute(1, 0, 2).contiguous()

        speaker_audio_encodings = model_kwargs.get('speaker_audio_encodings')
        if speaker_audio_encodings is None or self.s_audio_enc_drop_prob >= 1.0:
            speaker_audio_encodings = torch.zeros(size=(bs, 0, self.latent_dim)).to(sample.device)
        else:
            speaker_audio_encodings = self.speaker_audio_proj(speaker_audio_encodings)
            speaker_audio_encodings = self.mask_cond(speaker_audio_encodings, mode, self.s_audio_enc_drop_prob)
            speaker_audio_encodings = self.s_audio_scale * speaker_audio_encodings
        speaker_audio_encodings = speaker_audio_encodings.permute(1, 0, 2).contiguous()

        speaker_latent_embed = model_kwargs.get('speaker_latent_embed')
        if speaker_latent_embed is None or self.s_latent_embed_drop_prob >= 1.0:
            speaker_latent_embed = torch.zeros(size=(bs, 0, self.latent_dim)).to(sample.device)
        else:
            speaker_latent_embed = self.speaker_latent_proj(speaker_latent_embed)
            speaker_latent_embed = self.mask_cond(speaker_latent_embed, mode, self.s_latent_embed_drop_prob)
        speaker_latent_embed = speaker_latent_embed.permute(1, 0, 2).contiguous()

        speaker_3dmm_encodings = model_kwargs.get("speaker_3dmm_encodings")
        if speaker_3dmm_encodings is None or self.s_3dmm_enc_drop_prob >= 1.0:
            speaker_3dmm_encodings = torch.zeros(size=(bs, 0, self.latent_dim)).to(sample.device)
        else:
            speaker_3dmm_encodings = self.speaker_3dmm_proj(speaker_3dmm_encodings)
            speaker_3dmm_encodings = self.mask_cond(speaker_3dmm_encodings, mode, self.s_3dmm_enc_drop_prob)
        speaker_3dmm_encodings = speaker_3dmm_encodings.permute(1, 0, 2).contiguous()

        speaker_emotion_encodings = model_kwargs.get("speaker_emotion_encodings")
        if speaker_emotion_encodings is None or self.s_emotion_enc_drop_prob >= 1.0:
            speaker_emotion_encodings = torch.zeros(size=(bs, 0, self.latent_dim)).to(sample.device)
        else:
            speaker_emotion_encodings = self.speaker_emotion_proj(speaker_emotion_encodings)
            speaker_emotion_encodings = self.mask_cond(speaker_emotion_encodings, mode, self.s_emotion_enc_drop_prob)
        speaker_emotion_encodings = speaker_emotion_encodings.permute(1, 0, 2).contiguous()

        past_listener_emotion = model_kwargs.get('past_listener_emotion')
        if past_listener_emotion is None or self.past_l_emotion_drop_prob >= 1.0:
            past_listener_emotion = torch.zeros(size=(bs, 0, self.latent_dim)).to(sample.device)
        else:
            past_listener_emotion = self.listener_emotion_proj(past_listener_emotion)
            past_listener_emotion = self.mask_cond(past_listener_emotion, mode, self.past_l_emotion_drop_prob)
        past_listener_emotion = past_listener_emotion.permute(1, 0, 2).contiguous()

        return (listener_latent_embed,
                listener_personal_embed,
                speaker_audio_encodings,
                speaker_latent_embed,
                speaker_3dmm_encodings,
                speaker_emotion_encodings,
                past_listener_emotion)

    def _forward(
            self,
            sample,
            time_embed,
            listener_latent_embed,
            listener_personal_embed,
            speaker_audio_encodings,
            speaker_latent_embed,
            speaker_3dmm_encodings,
            speaker_emotion_encodings,
            past_listener_emotion,
    ):

        sample = self.to_emotion_embed(sample)

        if self.condition_concat == 'token_concat':
            emb_latent = torch.cat((
                time_embed,
                speaker_audio_encodings,
                speaker_3dmm_encodings,
                speaker_emotion_encodings,
                speaker_latent_embed,
                listener_latent_embed,
                past_listener_emotion,
                listener_personal_embed,
            ), dim=0)

            embed_seq_len = emb_latent.shape[0]

            if self.arch == "trans_enc":
                if self.concat == "concat_first":
                    xseq = torch.cat((emb_latent, sample), dim=0)
                    xseq = self.query_pos(xseq)
                    tokens = self.encoder(xseq)
                    sample = tokens[embed_seq_len:]
                elif self.concat == "concat_last":
                    xseq = torch.cat((sample, emb_latent), dim=0)
                    xseq = self.query_pos(xseq)
                    tokens = self.encoder(xseq)
                    sample = tokens[:embed_seq_len]
                else:
                    raise NotImplementedError("{self.concat} is not supported.")

            elif self.arch == "trans_dec":
                # tgt    - [L~, bs, latent_dim]
                # memory - [token_num, bs, latent_dim]
                sample = self.query_pos(sample)
                emb_latent = self.mem_pos(emb_latent)
                sample = self.decoder(tgt=sample, memory=emb_latent).squeeze(0)

            else:
                raise NotImplementedError("{self.arch} is not supported.")

        elif self.condition_concat == 'feat_concat':
            assert speaker_emotion_encodings.shape[0] == speaker_3dmm_encodings.shape[0], \
                "we need to use temporal emotion encodings or raw AU features."

            emb_latent = torch.cat((
                speaker_audio_encodings,
                speaker_3dmm_encodings,
                speaker_emotion_encodings,
            ), dim=-1)

            emb_latent = self.condition_proj(emb_latent)
            emb_latent = torch.cat((time_embed, emb_latent), dim=0)
            embed_seq_len = emb_latent.shape[0]

            if self.arch == "trans_enc":
                if self.concat == "concat_first":
                    xseq = torch.cat((emb_latent, sample), dim=0)
                    xseq = self.query_pos(xseq)
                    tokens = self.encoder(xseq)
                    sample = tokens[embed_seq_len:]
                elif self.concat == "concat_last":
                    xseq = torch.cat((sample, emb_latent), dim=0)
                    xseq = self.query_pos(xseq)
                    tokens = self.encoder(xseq)
                    sample = tokens[:embed_seq_len]
                else:
                    raise NotImplementedError("{self.concat} is not supported.")

            elif self.arch == "trans_dec":
                # tgt    - [L~, bs, latent_dim]
                # memory - [token_num, bs, latent_dim]
                sample = self.query_pos(sample)
                emb_latent = self.mem_pos(emb_latent)
                sample = self.decoder(tgt=sample, memory=emb_latent).squeeze(0)

            else:
                raise NotImplementedError("{self.arch} is not supported.")

        elif self.condition_concat == 'cascade':
            sample = self.query_pos(sample)
            memory = torch.cat((
                time_embed,
                speaker_3dmm_encodings,
                speaker_latent_embed,
            ), dim=0)

            sample = self.transformer_fusion_ns2sv(tgt=sample, memory=memory).squeeze(0)
            memory = torch.cat((time_embed, speaker_audio_encodings), dim=0)
            sample = self.transformer_fusion_ns2sa(tgt=sample, memory=memory).squeeze(0)
            sample = self.transformer_fusion_final(tgt=sample, memory=sample).squeeze(0)

        sample = self.to_emotion_feat(sample)
        sample = sample.permute(1, 0, 2)

        return sample

    def forward_with_cond_scale(
            self,
            sample,  # noised x_t
            timesteps,
            model_kwargs,
    ):
        sample = torch.cat([sample] * 2, dim=0)
        bs, _, _ = sample.shape
        sample = sample.permute(1, 0, 2).contiguous()

        timesteps = torch.cat([timesteps] * 2, dim=0)
        time_emb = self.time_proj(timesteps)  # time_embedding
        time_emb = time_emb.to(dtype=sample.dtype)
        time_embed = self.time_embedding(time_emb).unsqueeze(0)

        if model_kwargs is None:
            model_kwargs = {}
        else:
            model_kwargs = model_kwargs.copy()
        # add null embeddings ...
        for k, v in model_kwargs.items():
            model_kwargs[k] = torch.cat(
                (torch.zeros_like(model_kwargs[k], dtype=model_kwargs[k].dtype), model_kwargs[k]),
                dim=0)

        (listener_latent_embed,
         listener_personal_embed,
         speaker_audio_encodings,
         speaker_latent_embed,
         speaker_3dmm_encodings,
         speaker_emotion_encodings,
         past_listener_emotion) = (
            self.get_model_kwargs(
                bs,
                'test',
                sample,
                model_kwargs,
            )
        )

        prediction = self._forward(
            sample,
            time_embed,
            listener_latent_embed,
            listener_personal_embed,
            speaker_audio_encodings,
            speaker_latent_embed,
            speaker_3dmm_encodings,
            speaker_emotion_encodings,
            past_listener_emotion,
        )

        pred_uncond, pred_cond = prediction.chunk(2)
        prediction = pred_uncond + self.guidance_scale * (pred_cond - pred_uncond)
        return prediction

    def forward(
            self,
            sample,  # noised x_t
            timesteps,
            model_kwargs,
            **kwargs
    ):
        bs, _, _ = sample.shape
        sample = sample.permute(1, 0, 2).contiguous()

        time_emb = self.time_proj(timesteps)  # time_embedding
        time_emb = time_emb.to(dtype=sample.dtype)
        time_embed = self.time_embedding(time_emb).unsqueeze(0)

        (listener_latent_embed,
         listener_personal_embed,
         speaker_audio_encodings,
         speaker_latent_embed,
         speaker_3dmm_encodings,
         speaker_emotion_encodings,
         past_listener_emotion) = (
            self.get_model_kwargs(
                bs,
                'train',
                sample,
                model_kwargs,
            )
        )

        output = self._forward(
            sample,
            time_embed,
            listener_latent_embed,
            listener_personal_embed,
            speaker_audio_encodings,
            speaker_latent_embed,
            speaker_3dmm_encodings,
            speaker_emotion_encodings,
            past_listener_emotion,
        )

        return output

    def get_model_name(self):
        return self.__class__.__name__
