import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class DyT(nn.Module):

    def __init__(self, C, init_a=0.5):
        super().__init__()

        self.a = nn.Parameter(torch.ones(1) * init_a)
        self.gamma = nn.Parameter(torch.ones(C))
        self.beta = nn.Parameter(torch.zeros(C))

    def forward(self, x):

        x = torch.tanh(self.a * x)
        return self.gamma * x + self.beta

class PatchEmbedding(nn.Module):
  def __init__(self, d_model, img_size, patch_size, n_channels):
    super().__init__()

    self.d_model = d_model # Dimensionality of Model
    self.img_size = img_size # Image Size
    self.patch_size = patch_size # Patch Size
    self.n_channels = n_channels # Number of Channels

    self.linear_project = nn.Conv2d(self.n_channels, self.d_model, kernel_size=self.patch_size, stride=self.patch_size)

  # B: Batch Size
  # C: Image Channels
  # H: Image Height
  # W: Image Width
  # P_col: Patch Column
  # P_row: Patch Row
  def forward(self, x):
    x = self.linear_project(x) # (B, C, H, W) -> (B, d_model, P_col, P_row)

    x = x.flatten(2) # (B, d_model, P_col, P_row) -> (B, d_model, P)

    x = x.transpose(1, 2) # (B, d_model, P) -> (B, P, d_model)

    return x

class PositionalEncoding(nn.Module):
  def __init__(self, d_model, max_seq_length):
    super().__init__()

    self.cls_token = nn.Parameter(torch.randn(1, 1, d_model)) # Classification Token

    # Creating positional encoding
    pe = torch.zeros(max_seq_length, d_model)

    for pos in range(max_seq_length):
      for i in range(d_model):
        if i % 2 == 0:
          pe[pos][i] = np.sin(pos/(10000 ** (i/d_model)))
        else:
          pe[pos][i] = np.cos(pos/(10000 ** ((i-1)/d_model)))

    self.register_buffer('pe', pe.unsqueeze(0))

  def forward(self, x):
    # Expand to have class token for every image in batch
    tokens_batch = self.cls_token.expand(x.size()[0], -1, -1)

    # Adding class tokens to the beginning of each embedding
    x = torch.cat((tokens_batch,x), dim=1)

    # Add positional encoding to embeddings
    x = x + self.pe

    return x

class AttentionHead(nn.Module):
  def __init__(self, d_model, head_size):
    super().__init__()
    self.head_size = head_size

    self.query = nn.Linear(d_model, head_size)
    self.key = nn.Linear(d_model, head_size)
    self.value = nn.Linear(d_model, head_size)

  def forward(self, x):
    # Obtaining Queries, Keys, and Values
    Q = self.query(x)
    K = self.key(x)
    V = self.value(x)

    # Dot Product of Queries and Keys
    attention = Q @ K.transpose(-2,-1)

    # Scaling
    attention = attention / (self.head_size ** 0.5)

    attention = torch.softmax(attention, dim=-1)

    attention = attention @ V

    return attention

class MultiHeadAttention(nn.Module):
  def __init__(self, d_model, n_heads):
    super().__init__()
    self.head_size = d_model // n_heads

    self.W_o = nn.Linear(d_model, d_model)

    self.heads = nn.ModuleList([AttentionHead(d_model, self.head_size) for _ in range(n_heads)])

  def forward(self, x):
    # Combine attention heads
    out = torch.cat([head(x) for head in self.heads], dim=-1)

    out = self.W_o(out)

    return out

class TransformerEncoder(nn.Module):
  def __init__(self, d_model, n_heads, r_mlp=4,use_dyT=False):
    super().__init__()
    self.d_model = d_model
    self.n_heads = n_heads

    if use_dyT:
    # Sub-Layer 1 Normalization
        self.ln1=DyT(C=d_model)
        self.ln2=DyT(C=d_model)
    else:
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)
    # Multi-Head Attention
    self.mha = MultiHeadAttention(d_model, n_heads)

    # Sub-Layer 2 Normalization


    # Multilayer Perception
    self.mlp = nn.Sequential(
        nn.Linear(d_model, d_model*r_mlp),
        nn.GELU(),
        nn.Linear(d_model*r_mlp, d_model)
    )

  def forward(self, x):
    # Residual Connection After Sub-Layer 1
    out = x + self.mha(self.ln1(x))

    # Residual Connection After Sub-Layer 2
    out = out + self.mlp(self.ln2(out))

    return out

class VisionTransformer(nn.Module):
  def __init__(self, d_model, img_size, patch_size, n_channels, n_heads, n_layers,use_DyT):
    super().__init__()

    assert img_size[0] % patch_size[0] == 0 and img_size[1] % patch_size[1] == 0, "img_size dimensions must be divisible by patch_size dimensions"
    assert d_model % n_heads == 0, "d_model must be divisible by n_heads"

    self.d_model = d_model # Dimensionality of model
    self.img_size = img_size # Image size
    self.patch_size = patch_size # Patch size
    self.n_channels = n_channels # Number of channels
    self.n_heads = n_heads # Number of attention heads

    self.n_patches = (self.img_size[0] * self.img_size[1]) // (self.patch_size[0] * self.patch_size[1])
    self.max_seq_length = self.n_patches + 1

    self.patch_embedding = PatchEmbedding(self.d_model, self.img_size, self.patch_size, self.n_channels)
    self.positional_encoding = PositionalEncoding( self.d_model, self.max_seq_length)
    self.transformer_encoder = nn.Sequential(*[TransformerEncoder( self.d_model, self.n_heads,use_dyT=use_DyT) for _ in range(n_layers)])



  def forward(self, images):
    x = self.patch_embedding(images)

    x = self.positional_encoding(x)

    x = self.transformer_encoder(x)

    return x

class Decoder(nn.Module):
    def __init__(self, in_channels=512, out_channels=3, base_channels=64,patch_size=16,img_size=512):
        super().__init__()
        self.patch_size=patch_size
        self.patch_dim = img_size // patch_size
        def up_block(in_ch, out_ch):
            return nn.Sequential(
                nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
                nn.Conv2d(in_ch, out_ch, kernel_size=3,stride=1,padding=1),
                nn.GroupNorm(8, out_ch),
                nn.GELU(),
                nn.Dropout2d(0.2)
            )

        # Upsample blocks (transpose conv)
        self.up1 = up_block(in_channels, base_channels * 8)  # 16 -> 32
        self.up2 = up_block(base_channels * 8, base_channels * 4)  # 32→64
        self.up3 = up_block(base_channels * 4, base_channels * 2) # 64→128
        self.up4 = up_block(base_channels * 2, base_channels)    # 128→256
        self.final = nn.Conv2d(base_channels, out_channels, kernel_size=3, padding=1)
        self.sigmoid= nn.Sigmoid()
        
    def forward(self, x):
        x = x[:, 1:, :] #remove CLS Token
        B, N, C = x.shape
        expected_patches = self.patch_dim * self.patch_dim
        if N != expected_patches:
            raise ValueError(f"Expected {expected_patches} tokens (got {N}). Check image/patch size.")

        x = x.permute(0, 2, 1).contiguous()  # [B, C, N]
        x = x.view(B, C, self.patch_dim, self.patch_dim) # [B, 768, 16, 16]

        x = self.up1(x)
        x = self.up2(x)
        x = self.up3(x)
        x = self.up4(x)
        x = self.sigmoid(self.final(x))
        return x

class ANOVit(nn.Module):
    def __init__(self, d_model, img_size, patch_size, n_channels, n_heads, n_layers,use_DyT=False):
        super().__init__()
        self.encoder =VisionTransformer(d_model, img_size, patch_size, n_channels, n_heads, n_layers,use_DyT)
        self.decoder =Decoder(in_channels=d_model,out_channels=n_channels,base_channels=64,patch_size=patch_size[0],img_size=img_size[0])

    def forward(self, x):
        features = self.encoder(x)             # [B, 197, 768]
        recon_image = self.decoder(features)   # [B, 3, H, W]
        return recon_image
      
    def get_anomaly_map(self, input_img,recon_img):
        feature_diff= (input_img - recon_img)
        combined_anomaly_map= torch.linalg.norm(feature_diff, ord=2, dim=1, keepdim=True)
        return combined_anomaly_map