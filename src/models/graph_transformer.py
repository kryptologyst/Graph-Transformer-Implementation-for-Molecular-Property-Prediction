"""Enhanced Graph Transformer models for molecular property prediction."""

from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GraphTransformer, global_mean_pool, global_max_pool, global_add_pool
from torch_geometric.data import Batch, Data

from .positional_encoding import (
    LaplacianPositionalEncoding,
    RandomWalkPositionalEncoding,
    LearnablePositionalEncoding,
    SinusoidalPositionalEncoding,
)


class GraphTransformerNet(nn.Module):
    """Enhanced Graph Transformer for molecular property prediction.
    
    This model combines Graph Transformer with positional encodings and
    multiple pooling strategies for better molecular property prediction.
    
    Args:
        in_channels: Number of input node features.
        hidden_channels: Number of hidden channels.
        num_layers: Number of transformer layers.
        num_heads: Number of attention heads.
        dropout: Dropout probability.
        use_positional_encoding: Whether to use positional encoding.
        positional_encoding_type: Type of positional encoding.
        positional_encoding_dim: Dimension of positional encoding.
        use_edge_attr: Whether to use edge attributes.
        edge_attr_dim: Dimension of edge attributes.
        pooling: Type of global pooling.
        use_residual: Whether to use residual connections.
    """
    
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int = 64,
        num_layers: int = 3,
        num_heads: int = 4,
        dropout: float = 0.1,
        use_positional_encoding: bool = True,
        positional_encoding_type: str = "laplacian",
        positional_encoding_dim: int = 16,
        use_edge_attr: bool = True,
        edge_attr_dim: int = 3,
        pooling: str = "mean",
        use_residual: bool = True,
    ):
        super().__init__()
        
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.use_positional_encoding = use_positional_encoding
        self.use_edge_attr = use_edge_attr
        self.pooling = pooling
        self.use_residual = use_residual
        
        # Input projection
        self.input_proj = nn.Linear(in_channels, hidden_channels)
        
        # Positional encoding
        if use_positional_encoding:
            if positional_encoding_type == "laplacian":
                self.pos_enc = LaplacianPositionalEncoding(positional_encoding_dim)
            elif positional_encoding_type == "random_walk":
                self.pos_enc = RandomWalkPositionalEncoding(positional_encoding_dim)
            elif positional_encoding_type == "learnable":
                self.pos_enc = LearnablePositionalEncoding(positional_encoding_dim)
            elif positional_encoding_type == "sinusoidal":
                self.pos_enc = SinusoidalPositionalEncoding(positional_encoding_dim)
            else:
                raise ValueError(f"Unknown positional encoding type: {positional_encoding_type}")
            
            self.pos_proj = nn.Linear(positional_encoding_dim, hidden_channels)
        
        # Graph Transformer
        self.transformer = GraphTransformer(
            in_channels=hidden_channels,
            hidden_channels=hidden_channels,
            num_layers=num_layers,
            heads=num_heads,
            dropout=dropout,
            edge_dim=edge_attr_dim if use_edge_attr else None,
        )
        
        # Output layers
        self.lin1 = nn.Linear(hidden_channels, hidden_channels)
        self.lin2 = nn.Linear(hidden_channels, hidden_channels // 2)
        self.lin3 = nn.Linear(hidden_channels // 2, 1)
        
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(hidden_channels)
        
    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: Optional[torch.Tensor] = None,
        batch: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass.
        
        Args:
            x: Node features of shape [num_nodes, in_channels].
            edge_index: Edge indices of shape [2, num_edges].
            edge_attr: Edge attributes of shape [num_edges, edge_attr_dim].
            batch: Batch assignment of shape [num_nodes].
            
        Returns:
            torch.Tensor: Graph-level predictions of shape [batch_size].
        """
        # Input projection
        x = self.input_proj(x)
        
        # Add positional encoding
        if self.use_positional_encoding:
            pos_enc = self.pos_enc(x, edge_index, batch)
            pos_features = self.pos_proj(pos_enc)
            x = x + pos_features
        
        # Apply transformer
        x_transformed = self.transformer(x, edge_index, edge_attr)
        
        # Residual connection
        if self.use_residual:
            x = x + x_transformed
        else:
            x = x_transformed
        
        # Layer normalization
        x = self.norm(x)
        
        # Global pooling
        if batch is None:
            batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)
        
        if self.pooling == "mean":
            x = global_mean_pool(x, batch)
        elif self.pooling == "max":
            x = global_max_pool(x, batch)
        elif self.pooling == "add":
            x = global_add_pool(x, batch)
        elif self.pooling == "attention":
            x = self.attention_pooling(x, batch)
        else:
            raise ValueError(f"Unknown pooling type: {self.pooling}")
        
        # Output layers
        x = F.relu(self.lin1(x))
        x = self.dropout(x)
        x = F.relu(self.lin2(x))
        x = self.dropout(x)
        x = self.lin3(x)
        
        return x.squeeze(-1)
    
    def attention_pooling(self, x: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
        """Attention-based global pooling.
        
        Args:
            x: Node features of shape [num_nodes, hidden_channels].
            batch: Batch assignment of shape [num_nodes].
            
        Returns:
            torch.Tensor: Graph-level features of shape [batch_size, hidden_channels].
        """
        # Compute attention weights
        attention_weights = torch.softmax(
            torch.sum(x * x.mean(dim=0, keepdim=True), dim=1, keepdim=True), dim=0
        )
        
        # Apply attention pooling
        pooled_features = []
        for i in range(batch.max().item() + 1):
            mask = batch == i
            if mask.any():
                graph_x = x[mask]
                graph_weights = attention_weights[mask]
                pooled = torch.sum(graph_x * graph_weights, dim=0)
                pooled_features.append(pooled)
            else:
                pooled_features.append(torch.zeros_like(x[0]))
        
        return torch.stack(pooled_features)


class MultiScaleGraphTransformer(nn.Module):
    """Multi-scale Graph Transformer with different aggregation strategies.
    
    This model combines multiple Graph Transformers with different scales
    and aggregation strategies for enhanced molecular property prediction.
    
    Args:
        in_channels: Number of input node features.
        hidden_channels: Number of hidden channels.
        num_layers: Number of transformer layers.
        num_heads: Number of attention heads.
        dropout: Dropout probability.
        num_scales: Number of different scales.
        use_edge_attr: Whether to use edge attributes.
        edge_attr_dim: Dimension of edge attributes.
    """
    
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int = 64,
        num_layers: int = 3,
        num_heads: int = 4,
        dropout: float = 0.1,
        num_scales: int = 3,
        use_edge_attr: bool = True,
        edge_attr_dim: int = 3,
    ):
        super().__init__()
        
        self.num_scales = num_scales
        
        # Multiple Graph Transformers with different configurations
        self.transformers = nn.ModuleList([
            GraphTransformerNet(
                in_channels=in_channels,
                hidden_channels=hidden_channels,
                num_layers=num_layers,
                num_heads=num_heads,
                dropout=dropout,
                use_positional_encoding=True,
                positional_encoding_type=["laplacian", "random_walk", "sinusoidal"][i % 3],
                positional_encoding_dim=16,
                use_edge_attr=use_edge_attr,
                edge_attr_dim=edge_attr_dim,
                pooling=["mean", "max", "attention"][i % 3],
            )
            for i in range(num_scales)
        ])
        
        # Fusion layer
        self.fusion = nn.Sequential(
            nn.Linear(hidden_channels * num_scales, hidden_channels),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels, hidden_channels // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels // 2, 1),
        )
        
    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: Optional[torch.Tensor] = None,
        batch: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass.
        
        Args:
            x: Node features of shape [num_nodes, in_channels].
            edge_index: Edge indices of shape [2, num_edges].
            edge_attr: Edge attributes of shape [num_edges, edge_attr_dim].
            batch: Batch assignment of shape [num_nodes].
            
        Returns:
            torch.Tensor: Graph-level predictions of shape [batch_size].
        """
        # Get predictions from each transformer
        predictions = []
        for transformer in self.transformers:
            pred = transformer(x, edge_index, edge_attr, batch)
            predictions.append(pred)
        
        # Concatenate predictions
        combined = torch.stack(predictions, dim=1)  # [batch_size, num_scales]
        
        # Apply fusion layer
        output = self.fusion(combined.view(combined.size(0), -1))
        
        return output.squeeze(-1)


class GraphTransformerWithAttention(nn.Module):
    """Graph Transformer with attention visualization capabilities.
    
    This model extends the basic Graph Transformer with attention
    weight extraction for interpretability.
    
    Args:
        in_channels: Number of input node features.
        hidden_channels: Number of hidden channels.
        num_layers: Number of transformer layers.
        num_heads: Number of attention heads.
        dropout: Dropout probability.
        use_edge_attr: Whether to use edge attributes.
        edge_attr_dim: Dimension of edge attributes.
    """
    
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int = 64,
        num_layers: int = 3,
        num_heads: int = 4,
        dropout: float = 0.1,
        use_edge_attr: bool = True,
        edge_attr_dim: int = 3,
    ):
        super().__init__()
        
        self.hidden_channels = hidden_channels
        self.num_layers = num_layers
        self.num_heads = num_heads
        
        # Input projection
        self.input_proj = nn.Linear(in_channels, hidden_channels)
        
        # Graph Transformer
        self.transformer = GraphTransformer(
            in_channels=hidden_channels,
            hidden_channels=hidden_channels,
            num_layers=num_layers,
            heads=num_heads,
            dropout=dropout,
            edge_dim=edge_attr_dim if use_edge_attr else None,
        )
        
        # Output layers
        self.lin1 = nn.Linear(hidden_channels, hidden_channels)
        self.lin2 = nn.Linear(hidden_channels, hidden_channels // 2)
        self.lin3 = nn.Linear(hidden_channels // 2, 1)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: Optional[torch.Tensor] = None,
        batch: Optional[torch.Tensor] = None,
        return_attention: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Forward pass.
        
        Args:
            x: Node features of shape [num_nodes, in_channels].
            edge_index: Edge indices of shape [2, num_edges].
            edge_attr: Edge attributes of shape [num_edges, edge_attr_dim].
            batch: Batch assignment of shape [num_nodes].
            return_attention: Whether to return attention weights.
            
        Returns:
            torch.Tensor or Tuple[torch.Tensor, torch.Tensor]: 
                Graph-level predictions and optionally attention weights.
        """
        # Input projection
        x = self.input_proj(x)
        
        # Apply transformer
        x = self.transformer(x, edge_index, edge_attr)
        
        # Global pooling
        if batch is None:
            batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)
        
        x = global_mean_pool(x, batch)
        
        # Output layers
        x = F.relu(self.lin1(x))
        x = self.dropout(x)
        x = F.relu(self.lin2(x))
        x = self.dropout(x)
        x = self.lin3(x)
        
        if return_attention:
            # Extract attention weights (simplified version)
            attention_weights = self._extract_attention_weights(x, edge_index, batch)
            return x.squeeze(-1), attention_weights
        
        return x.squeeze(-1)
    
    def _extract_attention_weights(
        self, x: torch.Tensor, edge_index: torch.Tensor, batch: torch.Tensor
    ) -> torch.Tensor:
        """Extract attention weights for visualization.
        
        Args:
            x: Node features.
            edge_index: Edge indices.
            batch: Batch assignment.
            
        Returns:
            torch.Tensor: Attention weights.
        """
        # Simplified attention weight extraction
        # In practice, you would need to modify the GraphTransformer
        # to return attention weights
        num_edges = edge_index.size(1)
        attention_weights = torch.ones(num_edges, device=x.device) / num_edges
        return attention_weights
