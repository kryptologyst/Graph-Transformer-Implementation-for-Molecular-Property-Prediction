#!/usr/bin/env python3
"""Main training script for Graph Transformer implementation."""

import argparse
import os
import sys
from typing import Dict

import torch
import yaml
from omegaconf import DictConfig, OmegaConf

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from src.data.dataset import GraphDataModule
from src.models.graph_transformer import (
    GraphTransformerNet,
    MultiScaleGraphTransformer,
    GraphTransformerWithAttention,
)
from src.train.trainer import train_model
from src.utils.core import get_device, set_seed, create_directories


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train Graph Transformer on ZINC dataset")
    
    parser.add_argument(
        "--config",
        type=str,
        default="configs/default.yaml",
        help="Path to configuration file",
    )
    
    parser.add_argument(
        "--model",
        type=str,
        default="graph_transformer",
        choices=["graph_transformer", "multi_scale", "attention"],
        help="Model type to train",
    )
    
    parser.add_argument(
        "--data_root",
        type=str,
        default="./data",
        help="Root directory for data",
    )
    
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Batch size",
    )
    
    parser.add_argument(
        "--num_epochs",
        type=int,
        default=100,
        help="Number of training epochs",
    )
    
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=0.001,
        help="Learning rate",
    )
    
    parser.add_argument(
        "--hidden_channels",
        type=int,
        default=64,
        help="Number of hidden channels",
    )
    
    parser.add_argument(
        "--num_layers",
        type=int,
        default=3,
        help="Number of transformer layers",
    )
    
    parser.add_argument(
        "--num_heads",
        type=int,
        default=4,
        help="Number of attention heads",
    )
    
    parser.add_argument(
        "--dropout",
        type=float,
        default=0.1,
        help="Dropout probability",
    )
    
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )
    
    parser.add_argument(
        "--use_wandb",
        action="store_true",
        help="Use Weights & Biases logging",
    )
    
    parser.add_argument(
        "--use_tensorboard",
        action="store_true",
        default=True,
        help="Use TensorBoard logging",
    )
    
    parser.add_argument(
        "--eval_only",
        action="store_true",
        help="Only evaluate, don't train",
    )
    
    parser.add_argument(
        "--checkpoint",
        type=str,
        help="Path to checkpoint for evaluation",
    )
    
    return parser.parse_args()


def load_config(config_path: str, args) -> DictConfig:
    """Load configuration from file and override with command line arguments."""
    if os.path.exists(config_path):
        config = OmegaConf.load(config_path)
    else:
        # Create default config
        config = OmegaConf.create({
            "model": {
                "name": "graph_transformer",
                "in_channels": 9,
                "hidden_channels": 64,
                "num_layers": 3,
                "num_heads": 4,
                "dropout": 0.1,
                "use_positional_encoding": True,
                "positional_encoding_type": "laplacian",
                "positional_encoding_dim": 16,
                "use_edge_attr": True,
                "edge_attr_dim": 3,
                "pooling": "mean",
                "use_residual": True,
            },
            "training": {
                "batch_size": 32,
                "learning_rate": 0.001,
                "weight_decay": 1e-4,
                "num_epochs": 100,
                "patience": 20,
                "min_delta": 1e-4,
                "gradient_clip_norm": 1.0,
                "use_amp": False,
            },
            "data": {
                "dataset": "ZINC",
                "data_root": "./data",
                "train_size": 10000,
                "val_size": 1000,
                "test_size": 1000,
                "num_workers": 4,
                "pin_memory": True,
                "shuffle": True,
            },
            "evaluation": {
                "metrics": ["mae", "rmse", "r2"],
                "save_predictions": True,
                "save_embeddings": False,
            },
            "logging": {
                "log_dir": "./logs",
                "use_wandb": False,
                "wandb_project": "graph-transformer-zinc",
                "use_tensorboard": True,
                "log_interval": 10,
                "save_interval": 50,
            },
            "paths": {
                "checkpoints_dir": "./checkpoints",
                "assets_dir": "./assets",
                "results_dir": "./results",
            },
            "seed": 42,
            "deterministic": True,
        })
    
    # Override with command line arguments
    if args.data_root:
        config.data.data_root = args.data_root
    if args.batch_size:
        config.training.batch_size = args.batch_size
    if args.num_epochs:
        config.training.num_epochs = args.num_epochs
    if args.learning_rate:
        config.training.learning_rate = args.learning_rate
    if args.hidden_channels:
        config.model.hidden_channels = args.hidden_channels
    if args.num_layers:
        config.model.num_layers = args.num_layers
    if args.num_heads:
        config.model.num_heads = args.num_heads
    if args.dropout:
        config.model.dropout = args.dropout
    if args.seed:
        config.seed = args.seed
    if args.use_wandb:
        config.logging.use_wandb = True
    if args.use_tensorboard:
        config.logging.use_tensorboard = True
    
    return config


def create_model(config: DictConfig, model_type: str) -> torch.nn.Module:
    """Create model based on configuration and type.
    
    Args:
        config: Configuration.
        model_type: Type of model to create.
        
    Returns:
        torch.nn.Module: Created model.
    """
    model_config = config.model
    
    if model_type == "graph_transformer":
        model = GraphTransformerNet(
            in_channels=model_config.in_channels,
            hidden_channels=model_config.hidden_channels,
            num_layers=model_config.num_layers,
            num_heads=model_config.num_heads,
            dropout=model_config.dropout,
            use_positional_encoding=model_config.use_positional_encoding,
            positional_encoding_type=model_config.positional_encoding_type,
            positional_encoding_dim=model_config.positional_encoding_dim,
            use_edge_attr=model_config.use_edge_attr,
            edge_attr_dim=model_config.edge_attr_dim,
            pooling=model_config.pooling,
            use_residual=model_config.use_residual,
        )
    elif model_type == "multi_scale":
        model = MultiScaleGraphTransformer(
            in_channels=model_config.in_channels,
            hidden_channels=model_config.hidden_channels,
            num_layers=model_config.num_layers,
            num_heads=model_config.num_heads,
            dropout=model_config.dropout,
            num_scales=3,
            use_edge_attr=model_config.use_edge_attr,
            edge_attr_dim=model_config.edge_attr_dim,
        )
    elif model_type == "attention":
        model = GraphTransformerWithAttention(
            in_channels=model_config.in_channels,
            hidden_channels=model_config.hidden_channels,
            num_layers=model_config.num_layers,
            num_heads=model_config.num_heads,
            dropout=model_config.dropout,
            use_edge_attr=model_config.use_edge_attr,
            edge_attr_dim=model_config.edge_attr_dim,
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    return model


def main():
    """Main training function."""
    args = parse_args()
    
    # Load configuration
    config = load_config(args.config, args)
    
    # Set random seed
    set_seed(config.seed)
    
    # Create directories
    create_directories([
        config.paths.checkpoints_dir,
        config.paths.assets_dir,
        config.paths.results_dir,
        config.logging.log_dir,
    ])
    
    # Get device
    device = get_device()
    print(f"Using device: {device}")
    
    # Create data module
    data_module = GraphDataModule(
        data_root=config.data.data_root,
        batch_size=config.training.batch_size,
        num_workers=config.data.num_workers,
        pin_memory=config.data.pin_memory,
        train_size=config.data.train_size,
        val_size=config.data.val_size,
        test_size=config.data.test_size,
    )
    
    # Setup data
    data_module.setup()
    
    # Print dataset info
    dataset_info = data_module.get_dataset_info()
    print("Dataset Information:")
    for key, value in dataset_info.items():
        print(f"  {key}: {value}")
    
    # Create model
    model = create_model(config, args.model)
    print(f"Created {args.model} model with {sum(p.numel() for p in model.parameters()):,} parameters")
    
    # Train or evaluate
    if args.eval_only:
        if not args.checkpoint:
            raise ValueError("Checkpoint path required for evaluation only mode")
        
        # Load checkpoint
        checkpoint = torch.load(args.checkpoint, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        
        # Evaluate
        from src.train.trainer import GraphTransformerTrainer
        trainer = GraphTransformerTrainer(
            model=model,
            data_module=data_module,
            config=config,
            device=device,
            use_wandb=False,
            use_tensorboard=False,
        )
        results = trainer.evaluate()
        
    else:
        # Train model
        results = train_model(
            model=model,
            data_module=data_module,
            config=config,
            device=device,
            use_wandb=config.logging.use_wandb,
            use_tensorboard=config.logging.use_tensorboard,
        )
    
    # Save results
    results_path = os.path.join(config.paths.results_dir, f"{args.model}_results.yaml")
    with open(results_path, "w") as f:
        yaml.dump(results, f)
    
    print("Training completed successfully!")
    print(f"Results saved to: {results_path}")


if __name__ == "__main__":
    main()
