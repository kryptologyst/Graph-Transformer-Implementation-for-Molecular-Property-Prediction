#!/usr/bin/env python3
"""Evaluation script for Graph Transformer models."""

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
from src.eval.metrics import GraphTransformerEvaluator, ModelComparison
from src.utils.core import get_device, set_seed


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Evaluate Graph Transformer models")
    
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to model checkpoint",
    )
    
    parser.add_argument(
        "--config",
        type=str,
        default="configs/default.yaml",
        help="Path to configuration file",
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
        help="Batch size for evaluation",
    )
    
    parser.add_argument(
        "--output",
        type=str,
        default="./results/evaluation_results.yaml",
        help="Output file for results",
    )
    
    return parser.parse_args()


def load_model_from_checkpoint(checkpoint_path: str, device: torch.device) -> torch.nn.Module:
    """Load model from checkpoint.
    
    Args:
        checkpoint_path: Path to checkpoint file.
        device: Device to load model on.
        
    Returns:
        torch.nn.Module: Loaded model.
    """
    checkpoint = torch.load(checkpoint_path, map_location=device)
    config = checkpoint["config"]
    
    # Determine model type from config or checkpoint
    model_type = config.get("model", {}).get("name", "graph_transformer")
    
    if model_type == "graph_transformer":
        model = GraphTransformerNet(
            in_channels=config["model"]["in_channels"],
            hidden_channels=config["model"]["hidden_channels"],
            num_layers=config["model"]["num_layers"],
            num_heads=config["model"]["num_heads"],
            dropout=config["model"]["dropout"],
            use_positional_encoding=config["model"]["use_positional_encoding"],
            positional_encoding_type=config["model"]["positional_encoding_type"],
            positional_encoding_dim=config["model"]["positional_encoding_dim"],
            use_edge_attr=config["model"]["use_edge_attr"],
            edge_attr_dim=config["model"]["edge_attr_dim"],
            pooling=config["model"]["pooling"],
            use_residual=config["model"]["use_residual"],
        )
    elif model_type == "multi_scale":
        model = MultiScaleGraphTransformer(
            in_channels=config["model"]["in_channels"],
            hidden_channels=config["model"]["hidden_channels"],
            num_layers=config["model"]["num_layers"],
            num_heads=config["model"]["num_heads"],
            dropout=config["model"]["dropout"],
            num_scales=3,
            use_edge_attr=config["model"]["use_edge_attr"],
            edge_attr_dim=config["model"]["edge_attr_dim"],
        )
    elif model_type == "attention":
        model = GraphTransformerWithAttention(
            in_channels=config["model"]["in_channels"],
            hidden_channels=config["model"]["hidden_channels"],
            num_layers=config["model"]["num_layers"],
            num_heads=config["model"]["num_heads"],
            dropout=config["model"]["dropout"],
            use_edge_attr=config["model"]["use_edge_attr"],
            edge_attr_dim=config["model"]["edge_attr_dim"],
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    
    return model


def main():
    """Main evaluation function."""
    args = parse_args()
    
    # Load configuration
    if os.path.exists(args.config):
        config = OmegaConf.load(args.config)
    else:
        config = OmegaConf.create({
            "data": {
                "data_root": args.data_root,
                "batch_size": args.batch_size,
            },
            "evaluation": {
                "metrics": ["mae", "rmse", "r2", "pearson"],
            },
        })
    
    # Set random seed
    set_seed(42)
    
    # Get device
    device = get_device()
    print(f"Using device: {device}")
    
    # Create data module
    data_module = GraphDataModule(
        data_root=config.data.data_root,
        batch_size=config.data.batch_size,
    )
    
    # Setup data
    data_module.setup()
    
    # Print dataset info
    dataset_info = data_module.get_dataset_info()
    print("Dataset Information:")
    for key, value in dataset_info.items():
        print(f"  {key}: {value}")
    
    # Load model
    print(f"Loading model from {args.checkpoint}")
    model = load_model_from_checkpoint(args.checkpoint, device)
    
    # Print model info
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {num_params:,}")
    
    # Create evaluator
    evaluator = GraphTransformerEvaluator(device, config.evaluation.metrics)
    
    # Evaluate on test set
    print("Evaluating on test set...")
    test_results = evaluator.evaluate(model, data_module.test_dataloader())
    
    # Evaluate on validation set
    print("Evaluating on validation set...")
    val_results = evaluator.evaluate(model, data_module.val_dataloader())
    
    # Print results
    print("\nTest Results:")
    for metric, value in test_results.items():
        print(f"  {metric.upper()}: {value:.4f}")
    
    print("\nValidation Results:")
    for metric, value in val_results.items():
        print(f"  {metric.upper()}: {value:.4f}")
    
    # Combine results
    results = {
        "checkpoint": args.checkpoint,
        "model_parameters": num_params,
        "dataset_info": dataset_info,
        "test_results": test_results,
        "validation_results": val_results,
    }
    
    # Save results
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as f:
        yaml.dump(results, f)
    
    print(f"\nResults saved to: {args.output}")


if __name__ == "__main__":
    main()
