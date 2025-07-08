"""
ADTR with FPN
This module implements the ADTR (Anomaly Detection Transformer) model with a Feature Pyramid Network (FPN) with efficientnet_b4 as the backbone.
"""


import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
import logging
import config
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

class AttentionHead(nn.Module):
    """
    A single head of the self-attention mechanism.
    """
    def __init__(self, d_model, head_size):
        super().__init__()
        self.head_size = head_size
        self.query = nn.Linear(d_model, head_size, bias=False)
        self.key = nn.Linear(d_model, head_size, bias=False)
        self.value = nn.Linear(d_model, head_size, bias=False)

    def forward(self, x):
        Q = self.query(x)
        K = self.key(x)
        V = self.value(x)
        attention = Q @ K.transpose(-2, -1) * (self.head_size**-0.5)
        attention = F.softmax(attention, dim=-1)
        return attention @ V

class MultiHeadAttention(nn.Module):
    """
    Multi-head self-attention mechanism.
    """
    def __init__(self, d_model, n_head):
        super().__init__()
        self.head_size = d_model // n_head
        self.heads = nn.ModuleList([AttentionHead(d_model, self.head_size) for _ in range(n_head)])
        self.W_o = nn.Linear(d_model, d_model)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        return self.W_o(out)

class AttentionHeadwithDecoder(nn.Module):
    """
    Attention head for cross-attention in the decoder.
    """
    def __init__(self, d_model, head_size):
        super().__init__()
        self.head_size = head_size
        self.query = nn.Linear(d_model, head_size, bias=False)
        self.key = nn.Linear(d_model, head_size, bias=False)
        self.value = nn.Linear(d_model, head_size, bias=False)

    def forward(self, x, y):
        Q = self.query(x)
        K = self.key(y)
        V = self.value(y)
        attention = Q @ K.transpose(-2, -1) * (self.head_size**-0.5)
        attention = F.softmax(attention, dim=-1)
        return attention @ V

class MultiHeadCrossAttention(nn.Module):
    """
    Multi-head cross-attention mechanism for the decoder.
    """
    def __init__(self, d_model, n_head):
        super().__init__()
        self.head_size = d_model // n_head
        self.heads = nn.ModuleList([AttentionHeadwithDecoder(d_model, self.head_size) for _ in range(n_head)])
        self.W_o = nn.Linear(d_model, d_model)

    def forward(self, x, y):
        out = torch.cat([h(x, y) for h in self.heads], dim=-1)
        return self.W_o(out)
    
class TransformerEncoderLayer(nn.Module):
    """
    A single layer of the Transformer encoder for ADTR.
    Positional encoding is handled outside this module.
    use_dyt: If True, use DyT for layer normalization.
    """
    def __init__(self, d_model, n_head, sequence_length, dim_feedforward=1024, use_dyt=False):
        super().__init__()
        #self.pos_embedding = nn.Parameter(torch.randn(1, sequence_length, d_model))
        self.ln1 = DyT(C=d_model) if use_dyt else nn.LayerNorm(d_model)
        self.mha = MultiHeadAttention(d_model, n_head)
        self.ln2 = DyT(C=d_model) if use_dyt else nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.GELU(),
            nn.Dropout(p=0.1),
            nn.Linear(dim_feedforward, d_model)
        )

    def forward(self, x):
        """
        Args:
            x: Input tensor of shape [batch_size, seq_len, d_model]
        """
        #x = x + self.pos_embedding
        x = x + self.mha(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x

class TransformerDecoderLayer(nn.Module):
    """
    A single layer of the Transformer decoder for ADTR.
    Positional encoding is handled outside this module.
    use_dyt: If True, use DyT for layer normalization.
    """
    def __init__(self, d_model, n_head, sequence_length, dim_feedforward=1024, use_dyt=False):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, n_head)
        #self.pos_embedding = nn.Parameter(torch.randn(1, sequence_length, d_model))
        self.ln1 = DyT(C=d_model) if use_dyt else nn.LayerNorm(d_model)
        self.cross_attn = MultiHeadCrossAttention(d_model, n_head)
        self.ln2 = DyT(C=d_model) if use_dyt else nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.GELU(),
            nn.Dropout(p=0.1),
            nn.Linear(dim_feedforward, d_model)
        )
        self.ln3 = DyT(C=d_model) if use_dyt else nn.LayerNorm(d_model)

    def forward(self, x, encoder_output):
        """
        Args:
            x: Input tensor from the previous decoder layer (i.e., query embedding).
            encoder_output: Output tensor from the encoder.
        """
        #x = x + self.pos_embedding
        # Self-attention over the queries. No mask needed for this architecture.
        x = x + self.self_attn(self.ln1(x))
        # Cross-attention with encoder output
        x = x + self.cross_attn(self.ln2(x), encoder_output)
        # Feed-forward network
        x = x + self.ffn(self.ln3(x))
        return x

class Transformer(nn.Module):
    """
    A complete Transformer model with an encoder-decoder architecture,
    adapted for the ADTR framework.
    """
    def __init__(self, d_model, n_head, n_encoder_layers, n_decoder_layers, sequence_length, dim_feedforward, use_dyt=False):
        super().__init__()
        self.pos_embedding_enc = nn.Parameter(torch.randn(1, sequence_length, d_model))
        self.encoder_layers = nn.ModuleList([
            TransformerEncoderLayer(d_model, n_head, sequence_length, dim_feedforward, use_dyt=use_dyt)
            for _ in range(n_encoder_layers)
        ])
        self.pos_embedding_dec = nn.Parameter(torch.randn(1, sequence_length, d_model))
        self.decoder_layers = nn.ModuleList([
            TransformerDecoderLayer(d_model, n_head, sequence_length, dim_feedforward, use_dyt=use_dyt)
            for _ in range(n_decoder_layers)
        ])

    def forward(self, src, tgt):
        """
        Args:
            src: The source sequence for the encoder (features + pos_embedding).
            tgt: The target sequence for the decoder (query_embedding).
        """
        # --- ENCODER PASS ---
        encoder_output = src + self.pos_embedding_enc
        for encoder_layer in self.encoder_layers:
            encoder_output = encoder_layer(encoder_output)

        # --- DECODER PASS ---
        decoder_output = tgt + self.pos_embedding_dec
        for decoder_layer in self.decoder_layers:
            decoder_output = decoder_layer(decoder_output, encoder_output)

        return decoder_output

class WeightedFeatureFusion(nn.Module):
    """
    Weighted Feature Fusion module for BiFPN.
    The sum is learned dinamically using learnable weights.
    """
    def __init__(self, num_inputs):
        super(WeightedFeatureFusion, self).__init__()
        # Create a learnable weight for each input feature map
        # nn.parameter tells PyTorch that this is a parameter that should be optimized during training.
        self.weights = nn.Parameter(torch.ones(num_inputs, dtype=torch.float32), requires_grad=True)
        self.relu = nn.ReLU()
        self.epsilon = 1e-4

    def forward(self, inputs):
        # Normalize weights to sum to 1 this is done to ensure that the fusion is a convex combination of the inputs
        # and prevent the feature map from becoming too large
        weights = self.relu(self.weights)
        weights = weights / (torch.sum(weights, dim=0) + self.epsilon)
        
        # Calculate the weighted sum of features
        fused_features = 0
        for i, x in enumerate(inputs):
            fused_features += weights[i] * x
        return fused_features

class BiFPN(nn.Module):
    def __init__(self, in_channels_list, out_channels):
        super(BiFPN, self).__init__()
        self.out_channels = out_channels

        # 1. Lateral connections (uniform 1x1 convolutions)
        self.lateral_convs = nn.ModuleList([
            nn.Conv2d(in_channels, out_channels, kernel_size=1) for in_channels in in_channels_list
        ])

        # 2. Refinement convolutions for each path
        self.refinement_convs = nn.ModuleList([
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1) for _ in in_channels_list
        ])

        # 3. Downsampling for the bottom-up path
        self.downsample_convs = nn.ModuleList([
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=2, padding=1) for _ in range(len(in_channels_list) - 1)
        ])

        # We need a fusion node for each point where features are combined.
        # Top-down fusion nodes (each takes 2 inputs)
        self.td_fusion_nodes = nn.ModuleList([
            WeightedFeatureFusion(num_inputs=2) for _ in range(len(in_channels_list) - 1)
        ])
        # Bottom-up fusion nodes (most take 3 inputs, last one takes 2)
        self.bu_fusion_nodes = nn.ModuleList([
            WeightedFeatureFusion(num_inputs=3) for _ in range(len(in_channels_list) - 1)
        ])
        
    def forward(self, features):
        # Initial 1x1 convolutions on backbone features
        laterals = [conv(feat) for conv, feat in zip(self.lateral_convs, features)]
        num_laterals = len(laterals)

        # --- Top-Down Pathway ---
        # Start with the coarsest lateral feature
        td_path = [laterals[-1]] 
        for i in range(num_laterals - 2, -1, -1):
            # Fuse the current lateral with the upsampled feature from the coarser level
            upsampled_feat = F.interpolate(td_path[-1], size=laterals[i].shape[2:], mode='bilinear', align_corners=False)
            fused_node = self.td_fusion_nodes[i]([laterals[i], upsampled_feat])
            td_path.append(fused_node)
        # Reverse the list to be in fine-to-coarse order [P3, P4, P5, ...]
        td_path.reverse()

        # --- Bottom-Up Pathway with Cross-Scale Connections ---
        output_features = [td_path[0]] # The finest level has no bottom-up input
        for i in range(num_laterals - 1):
            # Fuse:
            # 1. The original lateral feature (cross-scale connection)
            # 2. The feature from the top-down path
            # 3. The downsampled feature from the finer level in the bottom-up path
            downsampled_feat = self.downsample_convs[i](output_features[-1])
            fused_node = self.bu_fusion_nodes[i]([
                laterals[i + 1],          # Input 1: Original lateral
                td_path[i + 1],           # Input 2: Top-down path result
                downsampled_feat          # Input 3: Bottom-up path result
            ])
            output_features.append(fused_node)

        # Apply final 3x3 refinement convolutions to the final output
        output_features = [conv(feat) for conv, feat in zip(self.refinement_convs, output_features)]

        return output_features

class AdtrEmbedding(nn.Module):
    """Embedding stage of ADTR using a pre-trained backbone."""
    def __init__(self, out_channels=256):
        super(AdtrEmbedding, self).__init__()
        self.backbone = timm.create_model('efficientnet_b5.sw_in12k', pretrained=True, features_only=True, out_indices=(1,2,3,4))
        for param in self.backbone.parameters():
            param.requires_grad = False
            
        backbone_out_channels = self.backbone.feature_info.channels()
        self.fpn_out_channels = out_channels

        self.fpn = BiFPN(in_channels_list=backbone_out_channels, out_channels=self.fpn_out_channels)
        
            
    def forward(self, x):
        self.backbone.eval()
        features = self.backbone(x)
        features = self.fpn(features)
        target_size = features[2].shape[2:]
        resized_features = [F.interpolate(feat, size=target_size, mode='bilinear', align_corners=False) for feat in features]
        return torch.cat(resized_features, dim=1)

class AdtrReconstruction(nn.Module):
    """Reconstruction stage of ADTR using a Transformer."""
    def __init__(self, in_channels=256*5, transformer_dim=256, n_heads=8, n_encoder_layers=4, n_decoder_layers=4, dim_feedforward=1024, use_dyt=False, sequence_length=256):
        super(AdtrReconstruction, self).__init__()
        self.transformer_dim = transformer_dim
        self.input_proj = nn.Conv2d(in_channels, transformer_dim, kernel_size=1)
        self.transformer = Transformer(
            d_model=transformer_dim,
            n_head=n_heads,
            n_encoder_layers=n_encoder_layers,
            n_decoder_layers=n_decoder_layers,
            dim_feedforward=dim_feedforward,
            use_dyt=use_dyt,
            sequence_length=sequence_length
        )
        # Aux query
        self.query_embedding = nn.Parameter(torch.randn(1, sequence_length, transformer_dim))
        self.output_proj = nn.Conv2d(transformer_dim, in_channels, kernel_size=1)
        
    def forward(self, x):
        projected_x = self.input_proj(x)
        if self.training:
            noise = torch.randn_like(projected_x) * 0.1 
            projected_x = projected_x + noise
        N, C, H, W = projected_x.shape
        flattened_x = projected_x.flatten(2).permute(0, 2, 1)
        
        reconstructed_tokens = self.transformer(src=flattened_x, tgt=self.query_embedding.expand(N, -1, -1))
        reconstructed_feature_map = reconstructed_tokens.permute(0, 2, 1).reshape(N, C, H, W)
        return self.output_proj(reconstructed_feature_map)

class ADTR_FPN(nn.Module):
    """The complete ADTR model."""
    def __init__(self, in_channels=256*5, out_channels_fpn=256, transformer_dim=256):
        super(ADTR_FPN, self).__init__()
        use_dyt=config.USE_DYT
        img_size=config.IMG_SIZE[0]
        
        self.embedding = AdtrEmbedding(out_channels=out_channels_fpn)
        sequence_length = (img_size // 16) ** 2

        self.reconstruction = AdtrReconstruction(in_channels, transformer_dim, use_dyt=use_dyt, sequence_length=sequence_length, dim_feedforward=4*transformer_dim)

    def forward(self, x: torch.Tensor):
        original_features = self.embedding(x)
        reconstructed_features = self.reconstruction(original_features)
        return original_features, reconstructed_features

    def get_anomaly_map(self, original_features, reconstructed_features):
        feature_diff = original_features - reconstructed_features
        anomaly_map = torch.linalg.norm(feature_diff, ord=2, dim=1, keepdim=True)
        return anomaly_map