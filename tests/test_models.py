"""Unit tests for Graph Transformer implementation."""

import pytest
import torch
import torch.nn as nn
from torch_geometric.data import Data

from src.models.graph_transformer import (
    GraphTransformerNet,
    MultiScaleGraphTransformer,
    GraphTransformerWithAttention,
)
from src.layers.positional_encoding import (
    LaplacianPositionalEncoding,
    RandomWalkPositionalEncoding,
    LearnablePositionalEncoding,
    SinusoidalPositionalEncoding,
)
from src.data.dataset import create_synthetic_dataset
from src.eval.metrics import GraphTransformerEvaluator
from src.utils.core import get_device, set_seed


class TestGraphTransformerNet:
    """Test GraphTransformerNet model."""
    
    def test_initialization(self):
        """Test model initialization."""
        model = GraphTransformerNet(
            in_channels=9,
            hidden_channels=64,
            num_layers=3,
            num_heads=4,
            dropout=0.1,
        )
        
        assert isinstance(model, nn.Module)
        assert model.in_channels == 9
        assert model.hidden_channels == 64
    
    def test_forward_pass(self):
        """Test forward pass."""
        model = GraphTransformerNet(
            in_channels=9,
            hidden_channels=64,
            num_layers=3,
            num_heads=4,
            dropout=0.1,
        )
        
        # Create test data
        x = torch.randn(10, 9)
        edge_index = torch.randint(0, 10, (2, 20))
        edge_attr = torch.randn(20, 3)
        batch = torch.zeros(10, dtype=torch.long)
        
        # Forward pass
        output = model(x, edge_index, edge_attr, batch)
        
        assert output.shape == (1,)  # Single graph prediction
        assert torch.isfinite(output).all()
    
    def test_different_pooling(self):
        """Test different pooling strategies."""
        poolings = ["mean", "max", "add", "attention"]
        
        for pooling in poolings:
            model = GraphTransformerNet(
                in_channels=9,
                hidden_channels=64,
                pooling=pooling,
            )
            
            x = torch.randn(10, 9)
            edge_index = torch.randint(0, 10, (2, 20))
            edge_attr = torch.randn(20, 3)
            batch = torch.zeros(10, dtype=torch.long)
            
            output = model(x, edge_index, edge_attr, batch)
            assert output.shape == (1,)
            assert torch.isfinite(output).all()


class TestPositionalEncodings:
    """Test positional encoding implementations."""
    
    def test_laplacian_encoding(self):
        """Test Laplacian positional encoding."""
        pos_enc = LaplacianPositionalEncoding(dim=16)
        
        x = torch.randn(10, 9)
        edge_index = torch.randint(0, 10, (2, 20))
        batch = torch.zeros(10, dtype=torch.long)
        
        encoding = pos_enc(x, edge_index, batch)
        assert encoding.shape == (10, 16)
        assert torch.isfinite(encoding).all()
    
    def test_random_walk_encoding(self):
        """Test Random Walk positional encoding."""
        pos_enc = RandomWalkPositionalEncoding(dim=16)
        
        x = torch.randn(10, 9)
        edge_index = torch.randint(0, 10, (2, 20))
        batch = torch.zeros(10, dtype=torch.long)
        
        encoding = pos_enc(x, edge_index, batch)
        assert encoding.shape == (10, 16)
        assert torch.isfinite(encoding).all()
    
    def test_learnable_encoding(self):
        """Test Learnable positional encoding."""
        pos_enc = LearnablePositionalEncoding(dim=16)
        
        x = torch.randn(10, 9)
        edge_index = torch.randint(0, 10, (2, 20))
        batch = torch.zeros(10, dtype=torch.long)
        
        encoding = pos_enc(x, edge_index, batch)
        assert encoding.shape == (10, 16)
        assert torch.isfinite(encoding).all()
    
    def test_sinusoidal_encoding(self):
        """Test Sinusoidal positional encoding."""
        pos_enc = SinusoidalPositionalEncoding(dim=16)
        
        x = torch.randn(10, 9)
        edge_index = torch.randint(0, 10, (2, 20))
        batch = torch.zeros(10, dtype=torch.long)
        
        encoding = pos_enc(x, edge_index, batch)
        assert encoding.shape == (10, 16)
        assert torch.isfinite(encoding).all()


class TestDataPipeline:
    """Test data pipeline functionality."""
    
    def test_synthetic_dataset(self):
        """Test synthetic dataset creation."""
        dataset = create_synthetic_dataset(num_graphs=10)
        
        assert len(dataset) == 10
        
        for data in dataset:
            assert isinstance(data, Data)
            assert data.x.shape[1] == 9  # Node features
            assert data.edge_attr.shape[1] == 3  # Edge features
            assert data.y.shape == (1,)  # Target


class TestEvaluation:
    """Test evaluation metrics."""
    
    def test_evaluator_initialization(self):
        """Test evaluator initialization."""
        device = get_device()
        evaluator = GraphTransformerEvaluator(device)
        
        assert evaluator.device == device
        assert "mae" in evaluator.metrics
    
    def test_metrics_computation(self):
        """Test metrics computation."""
        device = get_device()
        evaluator = GraphTransformerEvaluator(device)
        
        # Create dummy model
        model = GraphTransformerNet(
            in_channels=9,
            hidden_channels=64,
        )
        model.to(device)
        
        # Create dummy data
        dataset = create_synthetic_dataset(num_graphs=5)
        
        # Create dummy dataloader
        from torch_geometric.loader import DataLoader
        dataloader = DataLoader(dataset, batch_size=2)
        
        # Evaluate
        results = evaluator.evaluate(model, dataloader)
        
        assert "mae" in results
        assert "rmse" in results
        assert "r2" in results
        assert "pearson" in results
        
        for metric, value in results.items():
            assert isinstance(value, float)
            assert torch.isfinite(torch.tensor(value))


class TestUtilities:
    """Test utility functions."""
    
    def test_device_detection(self):
        """Test device detection."""
        device = get_device()
        assert isinstance(device, torch.device)
    
    def test_seed_setting(self):
        """Test seed setting."""
        set_seed(42)
        # This is hard to test directly, but we can ensure it doesn't raise errors
        assert True


if __name__ == "__main__":
    pytest.main([__file__])
