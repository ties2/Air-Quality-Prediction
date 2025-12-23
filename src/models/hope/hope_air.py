"""
HOPE-Air: HOPE Architecture Adapted for Air Quality Prediction

Implements a self-modifying recurrent architecture with Continuum Memory System
for continual air pollution forecasting. Based on Google's Nested Learning paradigm.

Key components:
1. Temporal Encoder: Processes historical pollution time series
2. CMS: Multi-frequency memory for different temporal patterns
3. Self-Modifying Titans: Surprise-gated memory updates
4. Prediction Head: Multi-target pollution forecasting

Reference: "Nested Learning: The Illusion of Deep Learning Architectures"
           Behrouz et al., NeurIPS 2025
"""

import math
from typing import Optional, Dict, Any, Tuple, List
from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F

from .cms import ContinuumMemorySystem, CMSConfig


@dataclass
class HOPEAirConfig:
    """Configuration for HOPE-Air model."""
    # Input/Output
    n_parameters: int = 6  # PM2.5, PM10, NO2, O3, CO, SO2
    history_window: int = 168  # 7 days hourly
    forecast_horizon: int = 24  # 24 hours ahead
    
    # Architecture
    hidden_dim: int = 256
    n_layers: int = 6
    n_heads: int = 8
    dropout: float = 0.1
    
    # CMS configuration
    cms_config: CMSConfig = field(default_factory=lambda: CMSConfig(
        n_levels=3,
        frequencies=[0.1, 0.5, 0.9],
        memory_dim=128,
        hidden_dim=256,
        memory_depth=2
    ))
    
    # Titans / Surprise-gated learning
    use_titans: bool = True
    surprise_threshold: float = 0.01
    memory_size: int = 1024
    n_memory_steps: int = 2
    
    # Auxiliary features
    use_temporal_encoding: bool = True
    use_spatial_encoding: bool = False
    n_spatial_features: int = 2  # lat, lon


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for time series."""
    
    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [seq_len, batch, dim] or [batch, seq_len, dim]"""
        if x.dim() == 3 and x.shape[0] != x.shape[1]:
            # Assume [batch, seq, dim]
            x = x + self.pe[:x.size(1)].transpose(0, 1)
        else:
            x = x + self.pe[:x.size(0)]
        return self.dropout(x)


class TemporalEncoder(nn.Module):
    """
    Temporal encoder for air quality time series.
    
    Uses a combination of:
    - Learnable parameter embeddings
    - Hour-of-day and day-of-week cyclical encodings
    - Optional spatial embeddings
    """
    
    def __init__(
        self,
        n_parameters: int,
        hidden_dim: int,
        use_temporal: bool = True,
        use_spatial: bool = False,
        n_spatial_features: int = 2,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.n_parameters = n_parameters
        self.hidden_dim = hidden_dim
        self.use_temporal = use_temporal
        self.use_spatial = use_spatial
        
        # Input projection: parameter values to hidden dim
        self.value_projection = nn.Linear(n_parameters, hidden_dim)
        
        # Parameter-specific embeddings
        self.param_embedding = nn.Embedding(n_parameters, hidden_dim)
        
        # Temporal encodings
        if use_temporal:
            # Hour of day (24 categories)
            self.hour_embedding = nn.Embedding(24, hidden_dim // 4)
            # Day of week (7 categories)
            self.dow_embedding = nn.Embedding(7, hidden_dim // 4)
            # Month (12 categories)
            self.month_embedding = nn.Embedding(12, hidden_dim // 4)
            # Is weekend
            self.weekend_embedding = nn.Embedding(2, hidden_dim // 4)
            
            # Combine temporal features
            self.temporal_projection = nn.Linear(hidden_dim, hidden_dim)
        
        # Spatial encoding
        if use_spatial:
            self.spatial_projection = nn.Linear(n_spatial_features, hidden_dim)
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(hidden_dim, dropout=dropout)
        
        # Layer norm
        self.layer_norm = nn.LayerNorm(hidden_dim)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
    
    def forward(
        self,
        x: torch.Tensor,
        temporal_features: Optional[torch.Tensor] = None,
        spatial_features: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Encode input time series.
        
        Args:
            x: Pollution values [batch, seq_len, n_parameters]
            temporal_features: [batch, seq_len, 4] - hour, dow, month, is_weekend
            spatial_features: [batch, n_spatial_features] - lat, lon
        
        Returns:
            Encoded representation [batch, seq_len, hidden_dim]
        """
        batch_size, seq_len, _ = x.shape
        
        # Project values
        h = self.value_projection(x)  # [B, L, hidden_dim]
        
        # Add temporal encoding
        if self.use_temporal and temporal_features is not None:
            hour = temporal_features[..., 0].long()  # [B, L]
            dow = temporal_features[..., 1].long()
            month = temporal_features[..., 2].long()
            weekend = temporal_features[..., 3].long()
            
            hour_emb = self.hour_embedding(hour % 24)
            dow_emb = self.dow_embedding(dow % 7)
            month_emb = self.month_embedding(month % 12)
            weekend_emb = self.weekend_embedding(weekend.clamp(0, 1))
            
            temporal = torch.cat([hour_emb, dow_emb, month_emb, weekend_emb], dim=-1)
            temporal = self.temporal_projection(temporal)
            h = h + temporal
        
        # Add spatial encoding
        if self.use_spatial and spatial_features is not None:
            spatial = self.spatial_projection(spatial_features)  # [B, hidden_dim]
            spatial = spatial.unsqueeze(1).expand(-1, seq_len, -1)
            h = h + spatial
        
        # Add positional encoding
        h = self.pos_encoder(h)
        
        # Layer norm and dropout
        h = self.layer_norm(h)
        h = self.dropout(h)
        
        return h


class SurpriseGatedMemory(nn.Module):
    """
    Surprise-gated memory module from Titans architecture.
    
    Memory updates are gated by "surprise" - how unexpected the input is.
    This helps the model:
    - Focus learning on novel/anomalous patterns (e.g., pollution spikes)
    - Avoid redundant updates for predictable data
    - Maintain stability during normal operation
    """
    
    def __init__(
        self,
        hidden_dim: int,
        memory_size: int = 1024,
        surprise_threshold: float = 0.01,
        n_memory_steps: int = 2
    ):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.memory_size = memory_size
        self.surprise_threshold = surprise_threshold
        self.n_memory_steps = n_memory_steps
        
        # Memory bank
        self.register_buffer(
            "memory_bank",
            torch.zeros(memory_size, hidden_dim)
        )
        self.register_buffer("memory_usage", torch.zeros(memory_size))
        self.register_buffer("write_ptr", torch.tensor(0))
        
        # Query/Key/Value projections for memory access
        self.query_proj = nn.Linear(hidden_dim, hidden_dim)
        self.key_proj = nn.Linear(hidden_dim, hidden_dim)
        self.value_proj = nn.Linear(hidden_dim, hidden_dim)
        
        # Surprise predictor
        self.surprise_predictor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim)
        )
        
        # Output projection
        self.output_proj = nn.Linear(hidden_dim, hidden_dim)
        
        # Layer norm
        self.layer_norm = nn.LayerNorm(hidden_dim)
    
    def compute_surprise(
        self,
        x: torch.Tensor,
        predicted: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute surprise as prediction error magnitude.
        
        Surprise = ||x - predicted|| / ||x||
        """
        error = x - predicted
        error_norm = torch.norm(error, dim=-1, keepdim=True)
        input_norm = torch.norm(x, dim=-1, keepdim=True) + 1e-8
        surprise = error_norm / input_norm
        return surprise
    
    def read_memory(self, query: torch.Tensor) -> torch.Tensor:
        """Read from memory using attention."""
        # Query projection
        q = self.query_proj(query)  # [B, L, D]
        
        # Key/Value from memory bank
        k = self.key_proj(self.memory_bank)  # [M, D]
        v = self.value_proj(self.memory_bank)
        
        # Attention
        scores = torch.einsum("bld,md->blm", q, k) / math.sqrt(self.hidden_dim)
        
        # Weight by usage (prioritize frequently used memories)
        usage_weights = F.softmax(self.memory_usage, dim=0)
        scores = scores * usage_weights.unsqueeze(0).unsqueeze(0)
        
        attn = F.softmax(scores, dim=-1)
        read_out = torch.einsum("blm,md->bld", attn, v)
        
        return read_out
    
    def write_memory(
        self,
        content: torch.Tensor,
        surprise: torch.Tensor
    ):
        """Write to memory, gated by surprise."""
        batch_size, seq_len, _ = content.shape
        
        # Only write if training
        if not self.training:
            return
        
        # Average over batch and sequence
        content_avg = content.mean(dim=(0, 1))  # [D]
        surprise_avg = surprise.mean()
        
        # Only write if sufficiently surprising
        if surprise_avg > self.surprise_threshold:
            # Write to memory at current position
            ptr = self.write_ptr.item()
            self.memory_bank[ptr] = content_avg
            self.memory_usage[ptr] += 1
            
            # Advance pointer
            self.write_ptr = (self.write_ptr + 1) % self.memory_size
    
    def forward(
        self,
        x: torch.Tensor,
        n_steps: Optional[int] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass with surprise-gated memory.
        
        Args:
            x: Input [batch, seq_len, hidden_dim]
            n_steps: Number of memory update steps
        
        Returns:
            output: Enhanced representation
            surprise: Surprise values for each position
        """
        n_steps = n_steps or self.n_memory_steps
        
        # Predict what the input "should" be
        predicted = self.surprise_predictor(x)
        
        # Compute surprise
        surprise = self.compute_surprise(x, predicted)
        
        # Read from memory
        memory_read = self.read_memory(x)
        
        # Iterative memory update steps
        h = x
        for _ in range(n_steps):
            # Combine with memory
            h = h + memory_read
            h = self.layer_norm(h)
            
            # Update memory with surprising content
            self.write_memory(h, surprise)
        
        # Output projection
        output = self.output_proj(h)
        
        return output, surprise
    
    def reset_memory(self):
        """Reset memory bank."""
        self.memory_bank.zero_()
        self.memory_usage.zero_()
        self.write_ptr.zero_()


class HOPEAirBlock(nn.Module):
    """
    Single HOPE-Air transformer block with CMS integration.
    
    Combines:
    1. Self-attention for sequence modeling
    2. CMS for multi-frequency memory
    3. Feed-forward network
    """
    
    def __init__(
        self,
        hidden_dim: int,
        n_heads: int,
        cms_config: CMSConfig,
        dropout: float = 0.1
    ):
        super().__init__()
        
        # Multi-head self-attention
        self.self_attn = nn.MultiheadAttention(
            hidden_dim,
            n_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # CMS
        self.cms = ContinuumMemorySystem(hidden_dim, cms_config)
        
        # Feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.Dropout(dropout)
        )
        
        # Layer norms
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.norm3 = nn.LayerNorm(hidden_dim)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
    
    def forward(
        self,
        x: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Forward pass through HOPE-Air block.
        
        Args:
            x: Input [batch, seq_len, hidden_dim]
            attn_mask: Attention mask for causal masking
        
        Returns:
            output: Processed representation
            info: Dictionary with CMS info
        """
        # Self-attention with residual
        residual = x
        x = self.norm1(x)
        attn_out, _ = self.self_attn(x, x, x, attn_mask=attn_mask)
        x = residual + self.dropout(attn_out)
        
        # CMS with residual
        residual = x
        x = self.norm2(x)
        cms_out, cms_info = self.cms(x)
        x = residual + self.dropout(cms_out)
        
        # FFN with residual
        residual = x
        x = self.norm3(x)
        ffn_out = self.ffn(x)
        x = residual + ffn_out
        
        return x, cms_info


class HOPEAir(nn.Module):
    """
    Full HOPE-Air model for continual air quality prediction.
    
    Architecture:
    1. Temporal Encoder: Embeds input time series with temporal features
    2. Stack of HOPE-Air Blocks: Self-attention + CMS
    3. Surprise-Gated Memory: Prioritizes learning from anomalies
    4. Prediction Head: Multi-target pollution forecasting
    
    Example:
        >>> config = HOPEAirConfig(n_parameters=6, forecast_horizon=24)
        >>> model = HOPEAir(config)
        >>> 
        >>> # Input: 7 days of hourly pollution data
        >>> x = torch.randn(32, 168, 6)  # [batch, history, params]
        >>> temporal = torch.randint(0, 24, (32, 168, 4))  # temporal features
        >>> 
        >>> # Predict next 24 hours
        >>> predictions, info = model(x, temporal_features=temporal)
        >>> print(predictions.shape)  # [32, 24, 6]
    """
    
    def __init__(self, config: HOPEAirConfig):
        super().__init__()
        
        self.config = config
        
        # Temporal encoder
        self.encoder = TemporalEncoder(
            n_parameters=config.n_parameters,
            hidden_dim=config.hidden_dim,
            use_temporal=config.use_temporal_encoding,
            use_spatial=config.use_spatial_encoding,
            n_spatial_features=config.n_spatial_features,
            dropout=config.dropout
        )
        
        # Stack of HOPE-Air blocks
        self.blocks = nn.ModuleList([
            HOPEAirBlock(
                hidden_dim=config.hidden_dim,
                n_heads=config.n_heads,
                cms_config=config.cms_config,
                dropout=config.dropout
            )
            for _ in range(config.n_layers)
        ])
        
        # Surprise-gated memory (optional)
        if config.use_titans:
            self.titans = SurpriseGatedMemory(
                hidden_dim=config.hidden_dim,
                memory_size=config.memory_size,
                surprise_threshold=config.surprise_threshold,
                n_memory_steps=config.n_memory_steps
            )
        else:
            self.titans = None
        
        # Prediction head
        self.prediction_head = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.GELU(),
            nn.LayerNorm(config.hidden_dim),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim, config.forecast_horizon * config.n_parameters)
        )
        
        # Final layer norm
        self.final_norm = nn.LayerNorm(config.hidden_dim)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize model weights."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, std=0.02)
    
    def forward(
        self,
        x: torch.Tensor,
        temporal_features: Optional[torch.Tensor] = None,
        spatial_features: Optional[torch.Tensor] = None,
        return_all_info: bool = False
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Forward pass for air quality prediction.
        
        Args:
            x: Pollution values [batch, history_window, n_parameters]
            temporal_features: [batch, history_window, 4]
            spatial_features: [batch, n_spatial_features]
            return_all_info: Whether to return detailed layer info
        
        Returns:
            predictions: [batch, forecast_horizon, n_parameters]
            info: Dictionary with model diagnostics
        """
        batch_size = x.shape[0]
        
        # Encode input
        h = self.encoder(x, temporal_features, spatial_features)
        
        # Create causal attention mask
        seq_len = h.shape[1]
        causal_mask = torch.triu(
            torch.ones(seq_len, seq_len, device=h.device),
            diagonal=1
        ).bool()
        
        # Process through HOPE-Air blocks
        all_cms_info = []
        for block in self.blocks:
            h, cms_info = block(h, attn_mask=causal_mask)
            if return_all_info:
                all_cms_info.append(cms_info)
        
        # Surprise-gated memory
        surprise = None
        if self.titans is not None:
            h, surprise = self.titans(h)
        
        # Final normalization
        h = self.final_norm(h)
        
        # Use last position for prediction
        last_hidden = h[:, -1, :]  # [batch, hidden_dim]
        
        # Predict
        pred_flat = self.prediction_head(last_hidden)
        predictions = pred_flat.view(
            batch_size,
            self.config.forecast_horizon,
            self.config.n_parameters
        )
        
        # Build info dict
        info = {
            "last_hidden": last_hidden,
            "surprise": surprise,
        }
        
        if return_all_info:
            info["cms_info"] = all_cms_info
        
        return predictions, info
    
    def reset_memory(self):
        """Reset all memory states (CMS and Titans)."""
        for block in self.blocks:
            block.cms.reset_all_memory()
        
        if self.titans is not None:
            self.titans.reset_memory()
    
    def get_memory_summary(self) -> Dict[str, Any]:
        """Get summary of all memory states."""
        summary = {}
        
        for i, block in enumerate(self.blocks):
            block_summary = block.cms.get_memory_summary()
            for key, value in block_summary.items():
                summary[f"block_{i}_{key}"] = value
        
        if self.titans is not None:
            summary["titans_usage"] = self.titans.memory_usage
        
        return summary
    
    def count_parameters(self) -> int:
        """Count total trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def create_hope_air_small() -> HOPEAir:
    """Create small HOPE-Air model (~10M parameters)."""
    config = HOPEAirConfig(
        hidden_dim=128,
        n_layers=4,
        n_heads=4,
        cms_config=CMSConfig(
            n_levels=2,
            frequencies=[0.2, 0.8],
            memory_dim=64,
            hidden_dim=128
        )
    )
    return HOPEAir(config)


def create_hope_air_base() -> HOPEAir:
    """Create base HOPE-Air model (~35M parameters)."""
    config = HOPEAirConfig(
        hidden_dim=256,
        n_layers=6,
        n_heads=8,
        cms_config=CMSConfig(
            n_levels=3,
            frequencies=[0.1, 0.5, 0.9],
            memory_dim=128,
            hidden_dim=256
        )
    )
    return HOPEAir(config)


def create_hope_air_large() -> HOPEAir:
    """Create large HOPE-Air model (~120M parameters)."""
    config = HOPEAirConfig(
        hidden_dim=512,
        n_layers=12,
        n_heads=16,
        cms_config=CMSConfig(
            n_levels=5,
            frequencies=[0.05, 0.2, 0.5, 0.8, 0.95],
            memory_dim=256,
            hidden_dim=512
        )
    )
    return HOPEAir(config)


if __name__ == "__main__":
    # Example usage
    print("Creating HOPE-Air Base model...")
    model = create_hope_air_base()
    
    print(f"Total parameters: {model.count_parameters():,}")
    
    # Simulate input
    batch_size = 16
    history_window = 168  # 7 days
    n_params = 6
    
    x = torch.randn(batch_size, history_window, n_params)
    temporal = torch.stack([
        torch.randint(0, 24, (batch_size, history_window)),  # hour
        torch.randint(0, 7, (batch_size, history_window)),   # day of week
        torch.randint(0, 12, (batch_size, history_window)),  # month
        torch.randint(0, 2, (batch_size, history_window)),   # weekend
    ], dim=-1)
    
    # Forward pass
    print("Running forward pass...")
    predictions, info = model(x, temporal_features=temporal)
    
    print(f"Input shape: {x.shape}")
    print(f"Predictions shape: {predictions.shape}")
    print(f"Last hidden shape: {info['last_hidden'].shape}")
    if info['surprise'] is not None:
        print(f"Surprise shape: {info['surprise'].shape}")
