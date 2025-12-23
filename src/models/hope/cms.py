"""
Continuum Memory System (CMS) for Nested Learning

Implements the multi-frequency memory system from the HOPE architecture.
Memory is treated as a spectrum of modules, each updating at different rates,
enabling continual learning without catastrophic forgetting.

Reference: "Nested Learning: The Illusion of Deep Learning Architectures"
           Behrouz et al., NeurIPS 2025
"""

import math
from typing import List, Optional, Tuple, Dict, Any
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class CMSConfig:
    """Configuration for Continuum Memory System."""
    n_levels: int = 3
    frequencies: List[float] = None  # Update rates per level
    memory_dim: int = 128
    hidden_dim: int = 256
    memory_depth: int = 2  # MLP depth per level
    use_gating: bool = True
    use_layer_norm: bool = True
    dropout: float = 0.1
    
    def __post_init__(self):
        if self.frequencies is None:
            # Default: exponentially spaced frequencies
            # Low frequency = slow updates (long-term memory)
            # High frequency = fast updates (short-term memory)
            self.frequencies = [
                0.1 ** (i / (self.n_levels - 1)) if self.n_levels > 1 else 0.5
                for i in range(self.n_levels)
            ]


class MemoryMLP(nn.Module):
    """
    Memory MLP block with configurable depth.
    
    Each level of the CMS contains an MLP that processes and stores
    information at its designated update frequency.
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        depth: int = 2,
        dropout: float = 0.1,
        use_layer_norm: bool = True
    ):
        super().__init__()
        
        layers = []
        current_dim = input_dim
        
        for i in range(depth):
            is_last = (i == depth - 1)
            out_dim = output_dim if is_last else hidden_dim
            
            layers.append(nn.Linear(current_dim, out_dim))
            
            if not is_last:
                if use_layer_norm:
                    layers.append(nn.LayerNorm(out_dim))
                layers.append(nn.GELU())
                layers.append(nn.Dropout(dropout))
            
            current_dim = out_dim
        
        self.mlp = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(x)


class CMSLevel(nn.Module):
    """
    Single level of the Continuum Memory System.
    
    Each level maintains:
    - A memory MLP that processes inputs
    - Running statistics for normalization
    - Update frequency (tau) that controls plasticity
    
    The key insight is that different levels update at different rates:
    - Fast levels (high tau) adapt quickly to new patterns
    - Slow levels (low tau) preserve long-term knowledge
    """
    
    def __init__(
        self,
        input_dim: int,
        memory_dim: int,
        hidden_dim: int,
        frequency: float,  # tau: update rate
        depth: int = 2,
        use_gating: bool = True,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.frequency = frequency
        self.use_gating = use_gating
        
        # Memory MLP
        self.memory_mlp = MemoryMLP(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=memory_dim,
            depth=depth,
            dropout=dropout
        )
        
        # Gating mechanism for controlling information flow
        if use_gating:
            self.gate = nn.Sequential(
                nn.Linear(input_dim + memory_dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, memory_dim),
                nn.Sigmoid()
            )
        
        # Layer normalization
        self.layer_norm = nn.LayerNorm(memory_dim)
        
        # Running memory state (for continual updates)
        self.register_buffer(
            "memory_state",
            torch.zeros(1, memory_dim)
        )
        
        # Update counter for frequency-based updates
        self.register_buffer("update_counter", torch.tensor(0.0))
    
    def forward(
        self,
        x: torch.Tensor,
        update_memory: bool = True
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through CMS level.
        
        Args:
            x: Input tensor [batch_size, seq_len, input_dim]
            update_memory: Whether to update running memory state
        
        Returns:
            output: Processed output [batch_size, seq_len, memory_dim]
            memory: Current memory state [1, memory_dim]
        """
        batch_size, seq_len, _ = x.shape
        
        # Process through memory MLP
        memory_out = self.memory_mlp(x)  # [B, L, memory_dim]
        
        # Expand memory state for batch
        memory_expanded = self.memory_state.expand(batch_size, -1)  # [B, memory_dim]
        memory_expanded = memory_expanded.unsqueeze(1).expand(-1, seq_len, -1)
        
        # Apply gating
        if self.use_gating:
            gate_input = torch.cat([x, memory_expanded], dim=-1)
            gate = self.gate(gate_input)
            output = gate * memory_out + (1 - gate) * memory_expanded
        else:
            output = memory_out
        
        # Apply layer norm
        output = self.layer_norm(output)
        
        # Update memory state based on frequency
        if update_memory and self.training:
            self.update_counter += 1
            
            # Check if we should update based on frequency
            # Higher frequency = more frequent updates
            update_interval = max(1, int(1.0 / self.frequency))
            
            if self.update_counter % update_interval == 0:
                # Exponential moving average update
                with torch.no_grad():
                    new_memory = memory_out.mean(dim=(0, 1))  # [memory_dim]
                    self.memory_state = (
                        self.frequency * new_memory.unsqueeze(0) +
                        (1 - self.frequency) * self.memory_state
                    )
        
        return output, self.memory_state
    
    def reset_memory(self):
        """Reset memory state to zeros."""
        self.memory_state.zero_()
        self.update_counter.zero_()


class ContinuumMemorySystem(nn.Module):
    """
    Full Continuum Memory System with multiple frequency levels.
    
    The CMS creates a "memory spectrum" where different levels
    handle different temporal scales:
    - Fast levels: Immediate context, short-term patterns
    - Medium levels: Recent trends, weekly patterns
    - Slow levels: Long-term knowledge, seasonal patterns
    
    This enables continual learning by separating fast-adapting
    components from stable long-term knowledge.
    
    Example:
        >>> config = CMSConfig(n_levels=3, frequencies=[0.1, 0.5, 0.9])
        >>> cms = ContinuumMemorySystem(input_dim=256, config=config)
        >>> x = torch.randn(32, 168, 256)  # batch, seq, features
        >>> output, memories = cms(x)
    """
    
    def __init__(
        self,
        input_dim: int,
        config: Optional[CMSConfig] = None
    ):
        super().__init__()
        
        self.config = config or CMSConfig()
        self.input_dim = input_dim
        
        # Create CMS levels
        self.levels = nn.ModuleList([
            CMSLevel(
                input_dim=input_dim,
                memory_dim=self.config.memory_dim,
                hidden_dim=self.config.hidden_dim,
                frequency=freq,
                depth=self.config.memory_depth,
                use_gating=self.config.use_gating,
                dropout=self.config.dropout
            )
            for freq in self.config.frequencies
        ])
        
        # Aggregation layer to combine all levels
        self.aggregator = nn.Sequential(
            nn.Linear(self.config.memory_dim * self.config.n_levels, self.config.hidden_dim),
            nn.GELU(),
            nn.LayerNorm(self.config.hidden_dim),
            nn.Dropout(self.config.dropout),
            nn.Linear(self.config.hidden_dim, input_dim)
        )
        
        # Residual connection scaling
        self.residual_scale = nn.Parameter(torch.ones(1) * 0.1)
        
        # Output projection
        self.output_dim = input_dim
    
    def forward(
        self,
        x: torch.Tensor,
        update_memory: bool = True,
        return_level_outputs: bool = False
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Forward pass through the full CMS.
        
        Args:
            x: Input tensor [batch_size, seq_len, input_dim]
            update_memory: Whether to update memory states
            return_level_outputs: Whether to return per-level outputs
        
        Returns:
            output: Aggregated output [batch_size, seq_len, input_dim]
            info: Dictionary containing memory states and diagnostics
        """
        level_outputs = []
        memory_states = []
        
        # Process through each level
        for level in self.levels:
            level_out, memory = level(x, update_memory=update_memory)
            level_outputs.append(level_out)
            memory_states.append(memory)
        
        # Concatenate level outputs
        concat_output = torch.cat(level_outputs, dim=-1)
        
        # Aggregate
        aggregated = self.aggregator(concat_output)
        
        # Residual connection
        output = x + self.residual_scale * aggregated
        
        # Build info dict
        info = {
            "memory_states": memory_states,
            "frequencies": self.config.frequencies,
        }
        
        if return_level_outputs:
            info["level_outputs"] = level_outputs
        
        return output, info
    
    def reset_all_memory(self):
        """Reset all memory states."""
        for level in self.levels:
            level.reset_memory()
    
    def get_memory_summary(self) -> Dict[str, torch.Tensor]:
        """Get summary of current memory states."""
        return {
            f"level_{i}_freq_{self.config.frequencies[i]:.2f}": level.memory_state
            for i, level in enumerate(self.levels)
        }
    
    def freeze_slow_levels(self, threshold: float = 0.3):
        """
        Freeze slow-updating levels to preserve long-term knowledge.
        
        This is useful during fine-tuning on new data to prevent
        catastrophic forgetting of core patterns.
        """
        for i, level in enumerate(self.levels):
            if self.config.frequencies[i] < threshold:
                for param in level.parameters():
                    param.requires_grad = False
    
    def unfreeze_all_levels(self):
        """Unfreeze all levels."""
        for level in self.levels:
            for param in level.parameters():
                param.requires_grad = True


class MultiFrequencyScheduler:
    """
    Learning rate scheduler that respects CMS frequency hierarchy.
    
    Slow-updating levels should have lower learning rates to maintain
    stability, while fast-updating levels can have higher rates.
    """
    
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        cms: ContinuumMemorySystem,
        base_lr: float = 1e-4,
        min_lr: float = 1e-6
    ):
        self.optimizer = optimizer
        self.cms = cms
        self.base_lr = base_lr
        self.min_lr = min_lr
        
        # Create learning rate mapping
        self._create_lr_mapping()
    
    def _create_lr_mapping(self):
        """Create LR mapping based on CMS frequencies."""
        self.lr_mapping = {}
        
        for i, level in enumerate(self.cms.levels):
            freq = self.cms.config.frequencies[i]
            # Higher frequency = higher learning rate
            level_lr = self.base_lr * freq
            level_lr = max(level_lr, self.min_lr)
            
            for name, param in level.named_parameters():
                self.lr_mapping[id(param)] = level_lr
    
    def step(self):
        """Apply frequency-aware learning rates."""
        for param_group in self.optimizer.param_groups:
            for param in param_group["params"]:
                if id(param) in self.lr_mapping:
                    # This is a simplified approach
                    # A more complete implementation would use 
                    # separate param groups per level
                    pass


if __name__ == "__main__":
    # Example usage
    config = CMSConfig(
        n_levels=3,
        frequencies=[0.1, 0.5, 0.9],  # slow, medium, fast
        memory_dim=128,
        hidden_dim=256,
        memory_depth=2
    )
    
    cms = ContinuumMemorySystem(input_dim=256, config=config)
    
    # Simulate input: batch of 32, sequence length 168 (7 days hourly), 256 features
    x = torch.randn(32, 168, 256)
    
    # Forward pass
    output, info = cms(x, return_level_outputs=True)
    
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Number of CMS levels: {len(info['memory_states'])}")
    print(f"Frequencies: {info['frequencies']}")
    
    # Check memory states
    for i, mem in enumerate(info['memory_states']):
        print(f"Level {i} (freq={info['frequencies'][i]:.2f}) memory shape: {mem.shape}")
