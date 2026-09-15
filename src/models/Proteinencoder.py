import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Optional
from typing import Tuple
import torch.nn.init as init
import torch.autograd

class Linear(nn.Module):
    def __init__(self, in_features: int, out_features: int, bias: bool = True) -> None:
        super(Linear, self).__init__()
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        init.xavier_uniform_(self.linear.weight)
        if bias:
            init.zeros_(self.linear.bias)

    def forward(self, x: Tensor) -> Tensor:
        return self.linear(x)

class GLU(nn.Module):
    def __init__(self, dim: int) -> None:
        super(GLU, self).__init__()
        self.dim = dim

    def forward(self, inputs: Tensor) -> Tensor:
        outputs, gate = inputs.chunk(2, dim=self.dim)
        return outputs * gate.sigmoid()


class Swish(nn.Module):
    def __init__(self):
        super(Swish, self).__init__()

    def forward(self, inputs: Tensor) -> Tensor:
        return inputs * inputs.sigmoid()


class Transpose(nn.Module):
    def __init__(self, shape: tuple):
        super(Transpose, self).__init__()
        self.shape = shape

    def forward(self, x: Tensor) -> Tensor:
        return x.transpose(*self.shape)


class RelativeMultiHeadAttention(nn.Module):
    def __init__(
            self,
            d_model: int = 512,
            num_heads: int = 16,
    ):
        super(RelativeMultiHeadAttention, self).__init__()
        assert d_model % num_heads == 0, "d_model % num_heads should be zero."
        self.d_model = d_model
        self.d_head = int(d_model / num_heads)
        self.num_heads = num_heads
        self.sqrt_dim = math.sqrt(d_model)
        self.query_proj = Linear(d_model, d_model)
        self.key_proj = Linear(d_model, d_model)
        self.value_proj = Linear(d_model, d_model)
        self.pos_proj = Linear(d_model, d_model, bias=False)
        self.u_bias = nn.Parameter(torch.Tensor(self.num_heads, self.d_head))
        self.v_bias = nn.Parameter(torch.Tensor(self.num_heads, self.d_head))
        torch.nn.init.xavier_uniform_(self.u_bias)
        torch.nn.init.xavier_uniform_(self.v_bias)

        self.out_proj = Linear(d_model, d_model)

    def forward(
            self,
            query: Tensor,
            key: Tensor,
            value: Tensor,
            pos_embedding: Tensor,
            mask: Tensor,
    ) -> Tensor:
        batch_size = value.size(0)
        query = self.query_proj(query).view(batch_size, -1, self.num_heads, self.d_head)
        key = self.key_proj(key).view(batch_size, -1, self.num_heads, self.d_head).permute(0, 2, 1, 3)
        value = self.value_proj(value).view(batch_size, -1, self.num_heads, self.d_head).permute(0, 2, 1, 3)
        pos_embedding = self.pos_proj(pos_embedding).view(batch_size, -1, self.num_heads, self.d_head)
        content_score = torch.matmul((query + self.u_bias).transpose(1, 2), key.transpose(2, 3))
        pos_score = torch.matmul((query + self.v_bias).transpose(1, 2), pos_embedding.permute(0, 2, 3, 1))

        pos_score = self._relative_shift(pos_score)

        score = (content_score + pos_score) / self.sqrt_dim

        if mask is not None:
            mask = mask.unsqueeze(1).unsqueeze(2)
            score = score.masked_fill(mask == 0, -1e9)
        attn = F.softmax(score, -1)
        context = torch.matmul(attn, value).transpose(1, 2)
        context = context.contiguous().view(batch_size, -1, self.d_model)
        return self.out_proj(context)

    def _relative_shift(self, pos_score: Tensor) -> Tensor:
        batch_size, num_heads, seq_length1, seq_length2 = pos_score.size()
        zeros = pos_score.new_zeros(batch_size, num_heads, seq_length1, 1)
        padded_pos_score = torch.cat([zeros, pos_score], dim=-1)
        padded_pos_score = padded_pos_score.view(batch_size, num_heads, seq_length2 + 1, seq_length1)
        pos_score = padded_pos_score[:, :, 1:].view_as(pos_score)

        return pos_score


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int = 512, max_len: int = 1000) -> None:
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model, requires_grad=False)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, length: int) -> Tensor:
        return self.pe[:, :length]


class FeedForwardModule(nn.Module):
    def __init__(
            self,
            encoder_dim: int = 512,
            expansion_factor: int = 4,
            dropout_p: float = 0.1,
    ) -> None:
        super(FeedForwardModule, self).__init__()
        self.sequential = nn.Sequential(
            nn.LayerNorm(encoder_dim),
            Linear(encoder_dim, encoder_dim * expansion_factor, bias=True),
            Swish(),
            nn.Dropout(p=dropout_p),
            Linear(encoder_dim * expansion_factor, encoder_dim, bias=True),
            nn.Dropout(p=dropout_p),
        )

    def forward(self, inputs: Tensor) -> Tensor:
        return self.sequential(inputs)

class DCNN(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super(DCNN, self).__init__()
        self.kernel_size = kernel_size
        self.conv_weights = nn.Parameter(torch.randn(out_channels, in_channels, kernel_size))
        self.conv_bias = nn.Parameter(torch.zeros(out_channels))

    def forward(self, x):
        x_unstacked = x.permute(0, 2, 1)  # [batch_size, in_channels, seq_length]
        output = F.conv1d(
            x_unstacked,
            self.conv_weights,
            bias=self.conv_bias,
            stride=1,
            padding=self.kernel_size // 2,
        )
        return output.permute(0, 2, 1)  # Return to shape: [batch_size, output_length, out_channels]
class ConvModule(nn.Module):
    def __init__(
            self,
            in_channels: int,
            kernel_size: int = 5,
            dropout_p: float = 0.1,
    ) -> None:
        super(ConvModule, self).__init__()
        self.dynamic_conv = DCNN(in_channels=in_channels, out_channels=in_channels,
                                             kernel_size=kernel_size)
        self.sequential = nn.Sequential(
            nn.LayerNorm(in_channels),
            Transpose(shape=(1, 2)),
            nn.Conv1d(in_channels=in_channels, out_channels=in_channels * 2,
                      kernel_size=1, stride=1, padding=0, bias=True, ),
            Swish(),
            nn.Conv1d(in_channels=in_channels * 2, out_channels=in_channels * 2, kernel_size=kernel_size,
                      stride=1, padding=(kernel_size - 1) // 2, bias=True,
                      ),
            GLU(dim=1),
            nn.BatchNorm1d(in_channels),
            nn.Conv1d(in_channels=in_channels, out_channels=in_channels,
                      kernel_size=1, stride=1, padding=0, bias=True, ),
            nn.Dropout(p=dropout_p),
        )

    def forward(self, inputs: Tensor) -> Tensor:
        x = self.dynamic_conv(inputs)  # 使用动态卷积层
        x = x.transpose(1, 2) # 转置为 [batch_size, output_length, in_channels * 2]
        x = F.layer_norm(x, x.shape[1:])  # 应用 LayerNorm，只归一化最后两维
        x = self.sequential(x.transpose(1, 2))  # 转置输入以适应 nn.Sequential
        #return self.sequential(inputs).transpose(1, 2)
        return x.transpose(1, 2)  # 返回到原始形状


class MultiHeadedSelfAttentionModule(nn.Module):
    def __init__(self, d_model: int, num_heads: int, dropout_p: float = 0.1, max_len: int = 1200):
        super(MultiHeadedSelfAttentionModule, self).__init__()
        self.positional_encoding = PositionalEncoding(d_model, max_len=max_len)
        self.layer_norm = nn.LayerNorm(d_model)
        self.attention = RelativeMultiHeadAttention(d_model, num_heads)
        self.dropout = nn.Dropout(p=dropout_p)

    def forward(self, inputs: Tensor, mask: Optional[Tensor] = None):
        batch_size, seq_length,hidden_dim= inputs.size()
        pos_embedding = self.positional_encoding(seq_length)
        pos_embedding = pos_embedding.repeat(batch_size, 1, 1)

        inputs = self.layer_norm(inputs)
        outputs = self.attention(inputs, inputs, inputs, pos_embedding=pos_embedding, mask=mask)

        return self.dropout(outputs)


class CNNTransBlock(nn.Module):
    def __init__(
            self,
            encoder_dim: int = 128,
            num_attention_heads: int = 4,
            feed_forward_expansion_factor: int = 4,
            feed_forward_dropout_p: float = 0.1,
            attention_dropout_p: float = 0.1,
            conv_dropout_p: float = 0.1,
            conv_kernel_size: int = 31,
            top_k: int = 5,
            k1: int = 3,
            max_len: int = 1200
    ):
        super(CNNTransBlock, self).__init__()

        self.MHSA_model = MultiHeadedSelfAttentionModule(
            d_model=encoder_dim,
            num_heads=num_attention_heads,
            dropout_p=attention_dropout_p,
            max_len=max_len
        )
        self.CNN_model = ConvModule(
            in_channels=encoder_dim,
            kernel_size=conv_kernel_size,
            dropout_p=conv_dropout_p,
        )
        """self.DCNN_model = DCNN(
            batch_size=64,  # 设为1以便后续处理，实际使用时需根据情况调整
            sentence_length=max_len,
            num_filters=[encoder_dim,encoder_dim],  # 设定过滤器数量
            embed_size=encoder_dim,
            top_k=top_k,
            k1=k1
        )"""
        self.FF_model = FeedForwardModule(
            encoder_dim=encoder_dim,
            expansion_factor=feed_forward_expansion_factor,
            dropout_p=feed_forward_dropout_p,
        )

    def forward(self, inputs: Tensor,mask: Tensor) -> Tuple[Tensor,Tensor]:
        MHSA_out = self.MHSA_model(inputs,mask)+inputs
        CNN_out = self.CNN_model(MHSA_out) + MHSA_out
        FFout = 0.5 * self.FF_model(CNN_out) + 0.5 * CNN_out
        return FFout


