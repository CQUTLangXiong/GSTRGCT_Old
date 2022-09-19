import torch
import torch.nn as nn
import torch.nn.functional as F
from GSTRGCT.model.auto_correlation import AutoCorrelation, AutoCorrelationLayer

class my_Layernorm(nn.Module):
    """
    Special designed layernorm for the seasonal part
    """
    def __init__(self, channels):
        super(my_Layernorm, self).__init__()
        self.layernorm = nn.LayerNorm(channels)

    def forward(self, x):
        x_hat = self.layernorm(x)
        return x_hat

class EncoderLayer(nn.Module):
    """
    Autoformer encoder layer with the progressive decomposition architecture
    """
    def __init__(self, attention, d_model, d_ff=None, dropout=0.1, activation="relu"):
        super(EncoderLayer, self).__init__()
        d_ff = d_ff or 4 * d_model
        self.attention = attention
        # self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1, bias=False)
        self.linear1 = nn.Linear(d_model, d_ff)
        # self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1, bias=False)
        self.linear2 =  nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu
        self.my_layer = my_Layernorm(d_model)


    def forward(self, x, attn_mask=None): # 3,12,512
        new_x, attn = self.attention(
            x, x, x,
            attn_mask=attn_mask
        )
        y = x + self.dropout(new_x) # 残差连接 #注意力机制计算的输出，为了提高泛化能力，随机以dropout的概率使得输出注意力为0       x, _ = self.decomp1(x)
        y1 = self.my_layer(y)
        y2 = self.dropout(self.activation(self.linear1(y1)))
        y2 = self.dropout(self.linear2(y2))
        out = self.my_layer(y2 + y1)
        return out, attn


class Encoder(nn.Module):
    """
    Autoformer encoder
    """
    def __init__(self, attn_layers, conv_layers=None, norm_layer=None):
        super(Encoder, self).__init__()
        self.attn_layers = nn.ModuleList(attn_layers)
        self.conv_layers = nn.ModuleList(conv_layers) if conv_layers is not None else None
        self.norm = norm_layer

    def forward(self, x, attn_mask=None):
        attns = []
        if self.conv_layers is not None:
            for attn_layer, conv_layer in zip(self.attn_layers, self.conv_layers):
                x, attn = attn_layer(x, attn_mask=attn_mask)
                x = conv_layer(x)
                attns.append(attn)
            x, attn = self.attn_layers[-1](x)
            attns.append(attn)
        else:
            for attn_layer in self.attn_layers:
                x, attn = attn_layer(x, attn_mask=attn_mask)
                attns.append(attn)

        if self.norm is not None:
            x = self.norm(x)

        return x, attns

        # self.projection = nn.Conv1d(in_channels=d_model, out_channels=c_out, kernel_size=3, stride=1, padding=1,
        #                             padding_mode='circular', bias=False)



if __name__ == '__main__':
    auto_correlation = AutoCorrelation(mask_flag=True, factor=1, scale=None, attention_dropout=0.1,output_attention=True)
    auto_correlation_layer = AutoCorrelationLayer(auto_correlation, d_model=64, n_heads=8, d_keys=None, d_values=None)
    pe_x = torch.rand(64,307,3, 12, 64)
    out, attn1 = auto_correlation_layer(queries=pe_x,keys=pe_x, values=pe_x, attn_mask=None)
    print(out.shape)
    encoder_layer = EncoderLayer(attention=auto_correlation_layer, d_model=64, d_ff=None, dropout=0.1, activation="relu")
    encoder_layer_out ,attn = encoder_layer(pe_x, attn_mask=None)
    print(encoder_layer_out.shape)
    attn_layers = [encoder_layer for l in range(2)]
    encoder = Encoder(attn_layers, conv_layers=None, norm_layer=my_Layernorm(64))
    encoder_out, attns = encoder(pe_x)
    print(encoder_out.shape)
    print(len(attns))


