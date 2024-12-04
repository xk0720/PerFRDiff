import torch
import torch.nn as nn
import torch.nn.functional as F


class PositionalEncoding(nn.Module):
    """
    compute sinusoid encoding.
    """

    def __init__(self, d_model, max_len, device):
        """
        constructor of sinusoid encoding class

        :param d_model: dimension of model
        :param max_len: max sequence length
        :param device: hardware device setting
        """
        super(PositionalEncoding, self).__init__()

        # same size with input matrix (for adding with input matrix)
        self.encoding = torch.zeros(max_len, d_model, device=device)
        self.encoding.requires_grad = False  # we don't need to compute gradient

        pos = torch.arange(0, max_len, device=device)
        pos = pos.float().unsqueeze(dim=1)
        # 1D => 2D unsqueeze to represent word's position

        _2i = torch.arange(0, d_model, step=2, device=device).float()
        # 'i' means index of d_model (e.g. embedding size = 50, 'i' = [0,50])
        # "step=2" means 'i' multiplied with two (same with 2 * i)

        self.encoding[:, 0::2] = torch.sin(pos / (10000 ** (_2i / d_model)))
        self.encoding[:, 1::2] = torch.cos(pos / (10000 ** (_2i / d_model)))
        # compute positional encoding to consider positional information of words

    def forward(self, x):
        # self.encoding
        encoding = self.encoding[:x.shape[1], :]
        return encoding.unsqueeze(0) # [1, seq_len, d_model]


class Transformer(nn.Module):
    def __init__(self, device, in_features, embed_dim, num_heads, num_layers, mlp_dim, seq_len, proj_dim,
                 proj_head="mlp", drop_prob=0.1, max_len=5000, pos_encoding="absolute", embed_layer="linear"):
        super(Transformer, self).__init__()

        self.num_heads = num_heads
        self.num_layers = num_layers
        self.mlp_dim = mlp_dim
        self.seq_len = seq_len
        self.proj_dim = proj_dim
        self.max_len = max_len
        self.embed_dim = embed_dim if embed_layer == "linear" else in_features
        self.embed_layer = nn.Linear(in_features, embed_dim) if embed_layer == "linear" else nn.Identity()

        self.pos_encoding = pos_encoding
        if pos_encoding == "learnable":
            self.pos_embed = nn.Parameter(torch.zeros(1, 1 + self.seq_len, self.embed_dim))
        elif pos_encoding == "absolute":
            self.pos_embed = PositionalEncoding(d_model=self.embed_dim, max_len=max_len, device=device)
        else:
            raise NotImplementedError('position encoding method not supported: {}'.format(pos_encoding))

        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.embed_dim))
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=self.embed_dim,
                nhead=self.num_heads,
                dim_feedforward=self.mlp_dim,
                batch_first=True),
            num_layers
        )  # encoder

        self.dropout = nn.Dropout(p=drop_prob) # dropout layer
        if proj_head == 'linear':
            self.proj_head = nn.Linear(embed_dim, proj_dim)
        elif proj_head == 'mlp':
            self.proj_head = nn.Sequential(
                nn.Linear(embed_dim, embed_dim),
                nn.ReLU(inplace=True),
                nn.Linear(embed_dim, proj_dim)
            )
        else:
            self.proj_head = nn.Identity()

    def forward(self, x):
        B = x.shape[0]
        x = self.embed_layer(x)
        x = torch.cat([self.cls_token.expand(B, -1, -1), x], dim=1)
        if self.pos_encoding == "absolute":
            x = x + self.pos_embed(x)
        elif self.pos_encoding == "learnable":
            x = x + self.pos_embed
        x = self.dropout(x)
        x = self.transformer(x)
        feat = x[:, 0, :]
        proj = F.normalize(self.proj_head(feat), dim=-1)
        return feat, proj # proj.shape: (bs, proj_dim)
