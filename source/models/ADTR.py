import torch
import torch.nn as nn
import timm
import torch.nn.functional as F
import logging
timm_logger = logging.getLogger('timm')
timm_logger.setLevel(logging.ERROR)

class DyT(nn.Module):
    
    def __init__(self, C, init_a=0.5):
        super().__init__()
        
        self.a = nn.Parameter(torch.ones(1) * init_a)
        self.gamma = nn.Parameter(torch.ones(C))
        self.beta = nn.Parameter(torch.zeros(C))

    def forward(self, x):
        
        x = torch.tanh(self.a * x)
        return self.gamma * x + self.beta


class EfficientB5(nn.Module):
    def __init__(self):
        super().__init__()

        self.backbone = timm.create_model(
            'efficientnet_b5.sw_in12k',
            pretrained=True,
            features_only=True,
            out_indices=(0, 1, 2, 3, 4) 
        )
        
        for param in self.backbone.parameters():
            param.requires_grad = False

    def forward(self, x):
        self.backbone.eval()
        features = self.backbone(x)
        
        resized_features = [
            F.interpolate(feat, size=24, mode='bilinear', align_corners=False) 
            for feat in features
        ]
        
        return torch.cat(resized_features, dim=1)

class PatchEmbedding(nn.Module):
    def __init__(self, in_channels, d_model):
        super().__init__()
        self.projection = nn.Conv2d(in_channels, d_model, kernel_size=1, stride=1)

    def forward(self, x):
        x = self.projection(x) 
        x = x.flatten(2)  
        x = x.transpose(1, 2) 
        return x
    
class UnpatchEmbedding(nn.Module):
    def __init__(self, d_model, out_channels, patch_size):
        super().__init__()
        self.d_model = d_model
        self.out_channels = out_channels
        self.h = patch_size
        self.w = patch_size
        self.projection = nn.Conv2d(d_model, out_channels, kernel_size=1, stride=1)

    def forward(self, x):
        x = x.transpose(1, 2)
        x = x.view(x.shape[0], self.d_model, self.h, self.w)
        x = self.projection(x)
        return x
    
class PositionalEncoding(nn.Module):
    def __init__(self, patch_num, d_model):
        super().__init__()
        self.positional_embedding = nn.Parameter(torch.randn(1, patch_num, d_model))

    def forward(self, x):
        return x + self.positional_embedding
  
class AttentionHead(nn.Module):
  def __init__(self, d_model, head_size):
    super().__init__()
    self.head_size = head_size
    self.d_model = d_model
    self.query = nn.Linear(d_model, head_size)
    self.key = nn.Linear(d_model, head_size)
    self.value = nn.Linear(d_model, head_size)

  def forward(self, x):
    Q = self.query(x)
    K = self.key(x)
    V = self.value(x)
    attention = Q @ K.transpose(-2,-1)
    attention = attention / (self.head_size ** 0.5)
    attention = torch.softmax(attention, dim=-1)
    attention = attention @ V
    return attention
  
class MultiHeadAttention(nn.Module):
  def __init__(self, d_model, n_heads):
    super().__init__()
    self.head_size = d_model // n_heads
    self.d_model = d_model
    self.W_o = nn.Linear(d_model, d_model)
    self.heads = nn.ModuleList([AttentionHead(d_model, self.head_size) for _ in range(n_heads)])

  def forward(self, x):
    out = torch.cat([head(x) for head in self.heads], dim=-1)
    out = self.W_o(out)
    return out

class TransformerEncoderLayer(nn.Module):
  def __init__(self, d_model, n_heads, patch_num, r_mlp=4,dyt=True):
    super().__init__()
    self.d_model = d_model
    self.n_heads = n_heads
    self.patch_num = patch_num

    if dyt:
        self.norm1 = DyT(C=d_model)
        self.norm2 = DyT(C=d_model)
    else:
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
    
    self.mha = MultiHeadAttention(d_model, n_heads)
    self.mlp = nn.Sequential(
        nn.Linear(d_model, d_model*r_mlp),
        nn.GELU(),
        nn.Linear(d_model*r_mlp, d_model)
    )
    self.positional_encoding = PositionalEncoding(patch_num, d_model)

  def forward(self, x):
    x = self.positional_encoding(x)
    out = x + self.mha(self.norm1(x))
    out = out + self.mlp(self.norm2(out))
    return out

class AttentionHeadwithDecoder(nn.Module):
  def __init__(self, d_model, head_size):
    super().__init__()
    self.head_size = head_size
    self.query = nn.Linear(d_model, head_size)
    self.key = nn.Linear(d_model, head_size)
    self.value = nn.Linear(d_model, head_size)

  def forward(self, x, y):
    Q = self.query(x)
    K = self.key(y)
    V = self.value(y)
    attention = Q @ K.transpose(-2,-1)
    attention = attention / (self.head_size ** 0.5)
    attention = torch.softmax(attention, dim=-1)
    attention = attention @ V
    return attention
  
class MultiHeadCrossAttention(nn.Module):
  def __init__(self, d_model, n_heads):
    super().__init__()
    self.d_model = d_model
    self.n_heads = n_heads
    self.head_size = d_model // n_heads
    self.heads = nn.ModuleList([AttentionHeadwithDecoder(d_model, self.head_size) for _ in range(n_heads)])
    self.W_o = nn.Linear(d_model, d_model) 

  def forward(self, x, y):
    head_outputs = [head(x, y) for head in self.heads]
    out = torch.cat(head_outputs, dim=-1) 
    out = self.W_o(out)
    return out

class TransformerDecoderLayer(nn.Module):
  def __init__(self, d_model, n_heads, patch_num, r_mlp=4,dyt=True):
    super().__init__()
    self.patch_num = patch_num
    self.d_model = d_model
    self.n_heads = n_heads
    self.r_mlp = r_mlp

    if dyt:
        self.norm1 = DyT(C=d_model)
        self.norm2 = DyT(C=d_model)
        self.norm3 = DyT(C=d_model)
    else:
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
    
    self.self_attn = MultiHeadAttention(d_model, n_heads)
    self.cross_attn = MultiHeadCrossAttention(d_model, n_heads)
    
    self.positional_encoding1 = PositionalEncoding(patch_num, d_model)     
    
    self.ffn = nn.Sequential(
        nn.Linear(d_model, d_model * r_mlp),
        nn.GELU(),
        nn.Linear(d_model * r_mlp, d_model)
    )

  def forward(self, x, encoder_output):
    x = self.positional_encoding1(x)
    x = x + self.self_attn(self.norm1(x))
    x = x + self.cross_attn(self.norm2(x), encoder_output)
    x = x + self.ffn(self.norm3(x))
    
    return x
  
class Transformer_union(nn.Module):
    def __init__(self, d_model, patch_num, n_heads, n_layers,dyt=True):
        super().__init__()
        self.d_model = d_model
        self.patch_num = patch_num
        self.n_heads = n_heads
        self.n_layers = n_layers
        
        self.encoder_layers = nn.ModuleList([
            TransformerEncoderLayer(d_model, n_heads, patch_num,dyt=dyt) 
            for _ in range(n_layers)
        ])
        
        self.decoder_layers = nn.ModuleList([
            TransformerDecoderLayer(d_model, n_heads, patch_num,dyt=dyt) 
            for _ in range(n_layers)
        ])

    def forward(self, encoder_input, decoder_input):
        
        encoder_output = encoder_input
        for encoder_layer in self.encoder_layers:
            encoder_output = encoder_layer(encoder_output)
        
        
        decoder_output = decoder_input
        for decoder_layer in self.decoder_layers:
            decoder_output = decoder_layer(decoder_output, encoder_output)

        return decoder_output  
  
class ADTR(nn.Module):
    def __init__(self, d_model=1200, patch_num=24*24, n_heads=8, n_layers=4, img_size=24, n_channels=816, batch_size=4, use_dyt=True):
        super().__init__()
        self.d_model = d_model
        self.patch_num = patch_num
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.img_size = img_size
        self.n_channels = n_channels
        self.batch_size = batch_size
        
        self.learnable_tensor = nn.Parameter(torch.randn(1, patch_num, d_model))
        self.adtr = EfficientB5() 
        self.patch_embedding = PatchEmbedding(n_channels, d_model)
        self.tu = Transformer_union(d_model, patch_num, n_heads, n_layers,dyt=use_dyt) 
        self.unp = UnpatchEmbedding(d_model, n_channels, img_size)
      
    def forward(self, x):
        current_batch_size = x.shape[0]

        decoder_query = self.learnable_tensor.expand(current_batch_size, -1, -1)
        
        cnn_features = self.adtr(x)
        img = cnn_features 
        
        encoder_input = self.patch_embedding(cnn_features)
        
        transformer_output = self.tu(encoder_input, decoder_query)
        output = self.unp(transformer_output)

        return img, output
   
    def get_anomaly_map(self, original_features, reconstructed_features):
        feature_diff = original_features - reconstructed_features
        anomaly_map = torch.linalg.norm(feature_diff, ord=2, dim=1, keepdim=True)
        return anomaly_map
