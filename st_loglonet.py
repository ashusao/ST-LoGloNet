import torch
import torch.nn as nn
import torch.nn.functional as F

from torchsummary import summary
from collections import OrderedDict


class ScaledDotProductAttention(nn.Module):

    def __init__(self, q_size):
        super(ScaledDotProductAttention, self).__init__()
        self.scale = q_size ** 0.5

    def forward(self, q, k, v):
        # q, k, v = (b_size, seq_len, feat_size)
        qk_t = q.bmm(k.transpose(-1, -2))
        score = F.softmax(qk_t / self.scale, dim=-1)
        v = score.bmm(v)
        return v, score


class AttentionHead(nn.Module):

    def __init__(self, dim_feat, dim_head):
        super(AttentionHead, self).__init__()
        self.linear_q = nn.Linear(dim_feat, dim_head)
        self.linear_k = nn.Linear(dim_feat, dim_head)
        self.linear_v = nn.Linear(dim_feat, dim_head)
        self.attention = ScaledDotProductAttention(dim_head)

    def forward(self, q, k, v):
        q = self.linear_q(q)
        k = self.linear_k(k)
        v = self.linear_v(v)
        v, score = self.attention(q, k, v)

        return v


class MultiHeadAttention(nn.Module):

    def __init__(self, n_head, dim_feat):
        super(MultiHeadAttention, self).__init__()

        self.dim_head = dim_feat // n_head
        self.dim_feat = dim_feat
        self.n_head = n_head

        self.heads = nn.ModuleList(
            [AttentionHead(dim_feat=dim_feat, dim_head=self.dim_head) for _ in range(self.n_head)]
        )
        self.linear = nn.Linear(self.n_head * self.dim_head, self.dim_feat)

    def forward(self, q, k, v):
        return self.linear(
            torch.cat([h(q, k, v) for h in self.heads], dim=-1)
        )


class EncoderLayer(nn.Module):

    def __init__(self, dim_feat=32, n_head=4, dim_ffn=64, dropout=0.1):
        super(EncoderLayer, self).__init__()
        self.dim_feat = dim_feat
        self.n_head = n_head
        self.dim_feed_fwd = dim_ffn

        self.dropout_1 = nn.Dropout(p=dropout)
        self.dropout_2 = nn.Dropout(p=dropout)

        self.multi_head_attention = MultiHeadAttention(n_head=n_head, dim_feat=dim_feat)
        self.layer_norm = nn.LayerNorm(dim_feat)
        self.ffn = self.point_wise_feed_forward(dim_in=dim_feat, dim_ffn=dim_ffn)

    def point_wise_feed_forward(self, dim_in=32, dim_ffn=64):
        return nn.Sequential(OrderedDict([
            ('fc1', nn.Linear(dim_in, dim_ffn)),
            ('elu1', nn.ELU()),
            ('fc2', nn.Linear(dim_ffn, dim_in))
        ]))

    def forward(self, X):
        # x = (batch, seq_len, dim_feat)
        tmp = X
        X = self.multi_head_attention(q=X, k=X, v=X) # self attention
        X = self.layer_norm(tmp + self.dropout_1(X))

        tmp = X
        X = self.ffn(X)
        X = self.layer_norm(tmp + self.dropout_2(X))

        return X

class Encoder(nn.Module):

    def __init__(self, dim_feat=32, n_head=4, dim_ffn=64, n_layer=1, dropout=0.1):
        super(Encoder, self).__init__()

        self.encoder_layers = nn.ModuleList([
            EncoderLayer(dim_feat=dim_feat, n_head=n_head, dim_ffn=dim_ffn, dropout=dropout)
            for _ in range(n_layer)
        ])

    def forward(self, X):

        for layer in self.encoder_layers:
            X = layer(X)

        return X


class PositionEmbedding(nn.Module):

    def __init__(self, seq_len, dim_embed=512, device=None):
        super(PositionEmbedding, self).__init__()
        self.seq_len = seq_len
        self.device = device
        self.position_embed = nn.Embedding(seq_len, dim_embed)

    def forward(self, x):  # (b, seq_len, dim_embed)
        pos = torch.arange(self.seq_len, device=self.device)
        pos = pos.unsqueeze(0).expand(x.size(0), -1)                # (b, seq_len)
        x = x + self.position_embed(pos)                            # (b, seq_len, dim_embed)

        return x

class ConvBnElu(nn.Module):

    def __init__(self, in_channels, out_channels, k_size):
        super(ConvBnElu, self).__init__()
        self.bn = nn.BatchNorm2d(out_channels)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=k_size,
                     stride=1, padding='same', bias=True)

    def forward(self, x):
        x = self.conv(x)
        x = F.elu(self.bn(x))
        return x


class RegionFunctionalityEncoder(nn.Module):

    def __init__(self, dim_in=1, dim_out=32):
        super(RegionFunctionalityEncoder, self).__init__()

        self.in_dim = dim_in
        self.out_dim = dim_out

        self.cnn_extractor = nn.Sequential(OrderedDict([
            ('conv1', ConvBnElu(in_channels=dim_in, out_channels=dim_out // 2, k_size=3)),
            ('conv2', ConvBnElu(in_channels=dim_out // 2, out_channels=dim_out, k_size=3)),
            ('conv1x1', ConvBnElu(in_channels=dim_out, out_channels=dim_out, k_size=1))
        ]))

    def forward(self, x):
        x = self.cnn_extractor(x)
        return x


class LocalSpatialEncoder(nn.Module):

    def __init__(self, dim_in=1, dim_out=32):
        super(LocalSpatialEncoder, self).__init__()

        self.in_dim = dim_in
        self.out_dim = dim_out

        self.cnn_extractor = nn.Sequential(OrderedDict([
            ('conv1', ConvBnElu(in_channels=dim_in, out_channels=dim_out // 4, k_size=3)),
            ('conv2', ConvBnElu(in_channels=dim_out // 4, out_channels=dim_out // 2, k_size=3)),
            ('conv3', ConvBnElu(in_channels=dim_out // 2, out_channels=dim_out, k_size=3)),
            ('conv1x1', ConvBnElu(in_channels=dim_out, out_channels=dim_out, k_size=1))
        ]))


    def forward(self, x):
        x = self.cnn_extractor(x)
        return x


class STLoGloNet(nn.Module):

    def __init__(self, len_conf=(4, 3, 2), n_c=1, n_poi=10, dim_feat=32, map_w=22, map_h=17,
                 dim_ts_feat=8, n_head_spatial=4, n_head_temporal=4,
                 n_layer_spatial=4, n_layer_temporal=4, dropout=0.1, device=None):
        super(STLoGloNet, self).__init__()

        self.len_conf = len_conf
        self.map_w = map_w
        self.map_h = map_h
        self.n_poi = n_poi
        self.n_layer_spatial = n_layer_spatial
        self.n_layer_temporal = n_layer_temporal
        self.ts_feat_dim = dim_ts_feat
        self.spatial_n_head = n_head_spatial
        self.temporal_n_head = n_head_temporal
        self.device = device

        self.n_c = n_c
        self.feat_dim = dim_feat

        self.dropout = nn.Dropout(p=dropout)

        self.local_spatial_encoder = LocalSpatialEncoder(dim_in=n_c, dim_out=dim_feat)
        self.poi_feature_extractor = RegionFunctionalityEncoder(dim_in=self.n_poi, dim_out=dim_feat)

        self.spatial_pos_embed = PositionEmbedding(seq_len=map_h*map_w, dim_embed=dim_feat, device=device)
        self.global_spatial_encoder = Encoder(dim_feat=dim_feat, n_head=n_head_spatial, dim_ffn=3 * dim_feat,
                                              n_layer=n_layer_spatial, dropout=dropout)

        self.temporal_pos_embed = PositionEmbedding(seq_len=sum(len_conf), dim_embed=dim_feat, device=device)
        self.temporal_encoder = Encoder(dim_feat=dim_feat, n_head=n_head_temporal, dim_ffn=3*dim_feat,
                                        n_layer=n_layer_temporal, dropout=dropout)

        self.ts_embedding = nn.Linear(in_features=dim_ts_feat, out_features=dim_feat)
        self.predict = nn.Linear(in_features=dim_feat, out_features=self.n_c * self.map_w * self.map_h)


    def forward(self, input_c, input_p, input_t, ts_c, ts_p, ts_t, input_poi):

        input_c = torch.stack([self.local_spatial_encoder(input_c[:, t, :, :, :]) for t in range(input_c.size(1))], dim=1)
        input_p = torch.stack([self.local_spatial_encoder(input_p[:, t, :, :, :]) for t in range(input_p.size(1))], dim=1)
        input_t = torch.stack([self.local_spatial_encoder(input_t[:, t, :, :, :]) for t in range(input_t.size(1))], dim=1)
        input_poi = self.poi_feature_extractor(input_poi)

        input_c = torch.permute(input_c, (0, 1, 3, 4, 2))
        input_p = torch.permute(input_p, (0, 1, 3, 4, 2))
        input_t = torch.permute(input_t, (0, 1, 3, 4, 2))
        input_poi = torch.permute(input_poi, (0, 2, 3, 1))

        input_c = torch.stack([input_c[:, t, :, :, :] + input_poi for t in range(input_c.size(1))],
                              dim=1)
        input_p = torch.stack([input_p[:, t, :, :, :] + input_poi for t in range(input_p.size(1))],
                              dim=1)
        input_t = torch.stack([input_t[:, t, :, :, :] + input_poi for t in range(input_t.size(1))],
                              dim=1)

        input_c = input_c.view(-1, input_c.size(1), self.map_w * self.map_h, self.feat_dim)  # (b, c, h*w, f)
        input_p = input_p.view(-1, input_p.size(1), self.map_w * self.map_h, self.feat_dim)  # (b, p, h*w, f)
        input_t = input_t.view(-1, input_t.size(1), self.map_w * self.map_h, self.feat_dim)  # (b, t, h*w, f)

        # Apply spatial position encoding in each timestamp
        input_c = torch.stack([self.spatial_pos_embed(input_c[:, t, :, :]) for t in range(input_c.size(1))], dim=1)
        input_p = torch.stack([self.spatial_pos_embed(input_p[:, t, :, :]) for t in range(input_p.size(1))], dim=1)
        input_t = torch.stack([self.spatial_pos_embed(input_t[:, t, :, :]) for t in range(input_t.size(1))], dim=1)

        # Regularize after positional encoding
        input_c = self.dropout(input_c)
        input_p = self.dropout(input_p)
        input_t = self.dropout(input_t)

        # spatial multi head attention
        input_c = torch.stack([self.global_spatial_encoder(input_c[:, t, :, :]) for t in range(input_c.size(1))], dim=1)
        input_p = torch.stack([self.global_spatial_encoder(input_p[:, t, :, :]) for t in range(input_p.size(1))], dim=1)
        input_t = torch.stack([self.global_spatial_encoder(input_t[:, t, :, :]) for t in range(input_t.size(1))], dim=1)

        ts_c = F.elu(self.ts_embedding(ts_c))
        ts_p = F.elu(self.ts_embedding(ts_p))
        ts_t = F.elu(self.ts_embedding(ts_t))

        # Compute mean feature vector of grid cells
        input_c = torch.mean(input_c, dim=-2)
        input_p = torch.mean(input_p, dim=-2)
        input_t = torch.mean(input_t, dim=-2)

        # concat the maps at each timestep to make a sequence
        x = torch.cat((input_c, input_p, input_t), dim=1)
        ts_embed = torch.cat((ts_c, ts_p, ts_t), dim=1)

        # Fuse input + postion embedding + time features
        x = self.temporal_pos_embed(x) + ts_embed
        x = self.temporal_encoder(x)

        out = F.tanh(self.predict(torch.mean(x, dim=1)))
        out = out.view(-1, self.n_c, self.map_h, self.map_w)

        return out

if __name__ == '__main__':

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = STLoGloNet(len_conf=(4, 3, 2), n_c=3, dim_feat=128, map_w=22, map_h=17,
                       dim_ts_feat=10, n_head_spatial=7, n_head_temporal=1,
                       n_layer_spatial=1, n_layer_temporal=1)
    #print(model)
    # model.to(device)

    summary(model, [(4, 3, 17, 22), (3, 3, 17, 22), (2, 3,  17, 22), (4, 10), (3, 10), (2, 10), (10, 17, 22)],
            device='cpu')