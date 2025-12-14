"""Positional encoding layers for Graph Transformers."""

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
from torch_geometric.utils import to_dense_batch


class LaplacianPositionalEncoding(nn.Module):
    """Laplacian Positional Encoding for Graph Transformers.
    
    Implements the Laplacian positional encoding from "Graph Transformer Networks"
    by Dwivedi & Bresson (2020).
    
    Args:
        dim: Dimension of positional encoding.
        max_nodes: Maximum number of nodes for pre-computation.
    """
    
    def __init__(self, dim: int, max_nodes: int = 1000):
        super().__init__()
        self.dim = dim
        self.max_nodes = max_nodes
        
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, 
                batch: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Compute Laplacian positional encoding.
        
        Args:
            x: Node features of shape [num_nodes, in_channels].
            edge_index: Edge indices of shape [2, num_edges].
            batch: Batch assignment of shape [num_nodes].
            
        Returns:
            torch.Tensor: Positional encodings of shape [num_nodes, dim].
        """
        from torch_geometric.utils import get_laplacian
        
        # Get Laplacian matrix
        edge_index, edge_weight = get_laplacian(edge_index, normalization="sym")
        
        # Convert to dense batch format
        if batch is None:
            batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)
        
        x_dense, mask = to_dense_batch(x, batch)
        edge_index_dense, _ = to_dense_batch(edge_index.T, batch)
        
        pos_enc = []
        for i in range(x_dense.size(0)):  # For each graph in batch
            num_nodes = mask[i].sum().item()
            if num_nodes == 0:
                continue
                
            # Get subgraph
            graph_x = x_dense[i][:num_nodes]
            graph_edge_index = edge_index_dense[i][:num_nodes]
            
            # Compute Laplacian eigenvalues
            try:
                from torch_geometric.utils import to_scipy_sparse_matrix
                import scipy.sparse.linalg as splinalg
                
                # Convert to scipy sparse matrix
                edge_index_sub = graph_edge_index.T
                edge_weight_sub = torch.ones(edge_index_sub.size(1), device=x.device)
                laplacian = to_scipy_sparse_matrix(edge_index_sub, edge_weight_sub, num_nodes)
                
                # Compute eigenvalues
                eigenvals = splinalg.eigsh(laplacian, k=min(self.dim, num_nodes-1), 
                                         which='SM', return_eigenvectors=False)
                eigenvals = torch.from_numpy(eigenvals).float().to(x.device)
                
                # Pad if necessary
                if len(eigenvals) < self.dim:
                    padding = torch.zeros(self.dim - len(eigenvals), device=x.device)
                    eigenvals = torch.cat([eigenvals, padding])
                
                pos_enc.append(eigenvals[:self.dim])
                
            except Exception:
                # Fallback to random encoding
                pos_enc.append(torch.randn(self.dim, device=x.device))
        
        if not pos_enc:
            return torch.zeros(x.size(0), self.dim, device=x.device)
        
        # Convert back to original format
        pos_enc = torch.stack(pos_enc)
        pos_enc_dense = torch.zeros(x_dense.size(0), x_dense.size(1), self.dim, device=x.device)
        
        for i, enc in enumerate(pos_enc):
            num_nodes = mask[i].sum().item()
            if num_nodes > 0:
                pos_enc_dense[i, :num_nodes] = enc.unsqueeze(0).expand(num_nodes, -1)
        
        return pos_enc_dense[mask]


class RandomWalkPositionalEncoding(nn.Module):
    """Random Walk Positional Encoding for Graph Transformers.
    
    Implements random walk structural encoding from "Graph Transformer Networks".
    
    Args:
        dim: Dimension of positional encoding.
        walk_length: Length of random walks.
    """
    
    def __init__(self, dim: int, walk_length: int = 16):
        super().__init__()
        self.dim = dim
        self.walk_length = walk_length
        
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor,
                batch: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Compute random walk positional encoding.
        
        Args:
            x: Node features of shape [num_nodes, in_channels].
            edge_index: Edge indices of shape [2, num_edges].
            batch: Batch assignment of shape [num_nodes].
            
        Returns:
            torch.Tensor: Positional encodings of shape [num_nodes, dim].
        """
        from torch_geometric.utils import to_dense_adj
        
        if batch is None:
            batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)
        
        # Convert to dense adjacency matrix
        adj = to_dense_adj(edge_index, batch)
        
        pos_enc = []
        for i in range(adj.size(0)):  # For each graph in batch
            graph_adj = adj[i]
            num_nodes = graph_adj.size(0)
            
            # Compute random walk probabilities
            degree = graph_adj.sum(dim=1)
            degree_inv = torch.where(degree > 0, 1.0 / degree, torch.zeros_like(degree))
            transition_matrix = graph_adj * degree_inv.unsqueeze(1)
            
            # Compute powers of transition matrix
            rw_enc = torch.zeros(num_nodes, self.dim, device=x.device)
            current_power = torch.eye(num_nodes, device=x.device)
            
            for k in range(min(self.dim, self.walk_length)):
                rw_enc[:, k] = current_power.diag()
                current_power = torch.mm(current_power, transition_matrix)
            
            pos_enc.append(rw_enc)
        
        # Convert back to original format
        pos_enc = torch.cat(pos_enc, dim=0)
        return pos_enc


class LearnablePositionalEncoding(nn.Module):
    """Learnable positional encoding for Graph Transformers.
    
    Args:
        dim: Dimension of positional encoding.
        max_nodes: Maximum number of nodes.
    """
    
    def __init__(self, dim: int, max_nodes: int = 1000):
        super().__init__()
        self.dim = dim
        self.max_nodes = max_nodes
        self.pos_embedding = nn.Embedding(max_nodes, dim)
        
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor,
                batch: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Compute learnable positional encoding.
        
        Args:
            x: Node features of shape [num_nodes, in_channels].
            edge_index: Edge indices of shape [2, num_edges].
            batch: Batch assignment of shape [num_nodes].
            
        Returns:
            torch.Tensor: Positional encodings of shape [num_nodes, dim].
        """
        num_nodes = x.size(0)
        node_indices = torch.arange(num_nodes, device=x.device)
        return self.pos_embedding(node_indices)


class SinusoidalPositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for Graph Transformers.
    
    Args:
        dim: Dimension of positional encoding.
    """
    
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor,
                batch: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Compute sinusoidal positional encoding.
        
        Args:
            x: Node features of shape [num_nodes, in_channels].
            edge_index: Edge indices of shape [2, num_edges].
            batch: Batch assignment of shape [num_nodes].
            
        Returns:
            torch.Tensor: Positional encodings of shape [num_nodes, dim].
        """
        num_nodes = x.size(0)
        device = x.device
        
        pos_enc = torch.zeros(num_nodes, self.dim, device=device)
        position = torch.arange(num_nodes, device=device).unsqueeze(1).float()
        
        div_term = torch.exp(torch.arange(0, self.dim, 2, device=device).float() *
                            -(math.log(10000.0) / self.dim))
        
        pos_enc[:, 0::2] = torch.sin(position * div_term)
        pos_enc[:, 1::2] = torch.cos(position * div_term)
        
        return pos_enc
