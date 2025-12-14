#!/usr/bin/env python3
"""
Project 418: Graph Transformers Implementation - Simple Example

This is a simplified example demonstrating the basic Graph Transformer implementation.
For the full production-ready implementation with advanced features, see the src/ directory.

Graph Transformers combine the power of self-attention with the inductive biases 
of graph structures. Unlike traditional GNNs that rely purely on local neighborhoods, 
transformers can capture long-range dependencies in graphs using attention over node pairs.

This example implements a basic Graph Transformer using PyTorch Geometric on the ZINC dataset.
"""

import torch
import torch.nn.functional as F
from torch_geometric.datasets import ZINC
from torch_geometric.nn import GraphTransformer, global_mean_pool
from torch_geometric.loader import DataLoader


class SimpleGraphTransformer(torch.nn.Module):
    """Simple Graph Transformer for molecular property prediction."""
    
    def __init__(self, in_channels: int, hidden_channels: int):
        super().__init__()
        self.transformer = GraphTransformer(
            in_channels=in_channels, 
            hidden_channels=hidden_channels,
            num_layers=3, 
            heads=4, 
            dropout=0.1
        )
        self.lin1 = torch.nn.Linear(hidden_channels, hidden_channels)
        self.lin2 = torch.nn.Linear(hidden_channels, 1)
 
    def forward(self, x, edge_index, edge_attr, batch):
        x = self.transformer(x, edge_index, edge_attr)
        x = global_mean_pool(x, batch)
        x = F.relu(self.lin1(x))
        return self.lin2(x).squeeze()


def main():
    """Main function demonstrating basic Graph Transformer training."""
    print("Graph Transformer Implementation - Simple Example")
    print("=" * 50)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load ZINC molecular graph dataset
    print("Loading ZINC dataset...")
    train_dataset = ZINC(root='./data/ZINC', split='train')
    val_dataset = ZINC(root='./data/ZINC', split='val')
    test_dataset = ZINC(root='./data/ZINC', split='test')
    
    # Create data loaders
    train_loader = DataLoader(train_dataset[:1000], batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset[:200], batch_size=32)
    test_loader = DataLoader(test_dataset[:200], batch_size=32)
    
    print(f"Training samples: {len(train_dataset[:1000])}")
    print(f"Validation samples: {len(val_dataset[:200])}")
    print(f"Test samples: {len(test_dataset[:200])}")
    
    # Create model
    model = SimpleGraphTransformer(
        in_channels=train_dataset.num_node_features, 
        hidden_channels=64
    ).to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    loss_fn = torch.nn.L1Loss()  # MAE
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Training function
    def train():
        model.train()
        total_loss = 0
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            pred = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
            loss = loss_fn(pred, batch.y.view(-1))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        return total_loss / len(train_loader)
    
    # Validation function
    def evaluate(loader):
        model.eval()
        total_loss = 0
        with torch.no_grad():
            for batch in loader:
                batch = batch.to(device)
                pred = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
                loss = loss_fn(pred, batch.y.view(-1))
                total_loss += loss.item()
        return total_loss / len(loader)
    
    # Training loop
    print("\nStarting training...")
    print("Epoch | Train MAE | Val MAE")
    print("-" * 30)
    
    for epoch in range(1, 21):
        train_loss = train()
        val_loss = evaluate(val_loader)
        print(f"{epoch:5d} | {train_loss:9.4f} | {val_loss:8.4f}")
    
    # Final evaluation
    print("\nFinal evaluation on test set...")
    test_loss = evaluate(test_loader)
    print(f"Test MAE: {test_loss:.4f}")
    
    print("\nTraining completed!")
    print("\nFor advanced features, see the full implementation in src/ directory:")
    print("- Multiple Graph Transformer variants")
    print("- Advanced positional encodings")
    print("- Comprehensive evaluation metrics")
    print("- Interactive Streamlit demo")
    print("- Production-ready training pipeline")


if __name__ == "__main__":
    main()