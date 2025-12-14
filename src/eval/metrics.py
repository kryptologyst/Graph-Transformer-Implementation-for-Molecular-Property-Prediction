"""Evaluation metrics and utilities for Graph Transformer models."""

from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from torchmetrics import MeanAbsoluteError, MeanSquaredError, R2Score
from torchmetrics.regression import PearsonCorrCoef


class GraphTransformerEvaluator:
    """Evaluator for Graph Transformer models on molecular property prediction.
    
    Args:
        device: Device to run evaluation on.
        metrics: List of metrics to compute.
    """
    
    def __init__(
        self,
        device: torch.device,
        metrics: Optional[List[str]] = None,
    ):
        self.device = device
        self.metrics = metrics or ["mae", "rmse", "r2", "pearson"]
        
        # Initialize torchmetrics
        self.torch_metrics = {
            "mae": MeanAbsoluteError(),
            "mse": MeanSquaredError(),
            "r2": R2Score(),
            "pearson": PearsonCorrCoef(),
        }
        
        # Move metrics to device
        for metric in self.torch_metrics.values():
            metric.to(device)
    
    def evaluate(
        self,
        model: nn.Module,
        dataloader: torch.utils.data.DataLoader,
        return_predictions: bool = False,
    ) -> Union[Dict[str, float], Tuple[Dict[str, float], List[float], List[float]]]:
        """Evaluate model on a dataset.
        
        Args:
            model: Model to evaluate.
            dataloader: Data loader for evaluation.
            return_predictions: Whether to return predictions and targets.
            
        Returns:
            Dict[str, float] or Tuple[Dict[str, float], List[float], List[float]]:
                Evaluation metrics and optionally predictions and targets.
        """
        model.eval()
        
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            for batch in dataloader:
                batch = batch.to(self.device)
                
                # Get predictions
                predictions = model(
                    batch.x,
                    batch.edge_index,
                    batch.edge_attr,
                    batch.batch,
                )
                
                all_predictions.extend(predictions.cpu().numpy())
                all_targets.extend(batch.y.cpu().numpy())
        
        # Convert to tensors
        predictions_tensor = torch.tensor(all_predictions, device=self.device)
        targets_tensor = torch.tensor(all_targets, device=self.device)
        
        # Compute metrics
        results = {}
        
        if "mae" in self.metrics:
            results["mae"] = self.torch_metrics["mae"](predictions_tensor, targets_tensor).item()
        
        if "rmse" in self.metrics:
            mse = self.torch_metrics["mse"](predictions_tensor, targets_tensor).item()
            results["rmse"] = np.sqrt(mse)
        
        if "r2" in self.metrics:
            results["r2"] = self.torch_metrics["r2"](predictions_tensor, targets_tensor).item()
        
        if "pearson" in self.metrics:
            results["pearson"] = self.torch_metrics["pearson"](predictions_tensor, targets_tensor).item()
        
        # Compute additional metrics using sklearn
        if "mae_sklearn" in self.metrics:
            results["mae_sklearn"] = mean_absolute_error(all_targets, all_predictions)
        
        if "rmse_sklearn" in self.metrics:
            results["rmse_sklearn"] = np.sqrt(mean_squared_error(all_targets, all_predictions))
        
        if "r2_sklearn" in self.metrics:
            results["r2_sklearn"] = r2_score(all_targets, all_predictions)
        
        if return_predictions:
            return results, all_predictions, all_targets
        
        return results
    
    def reset_metrics(self):
        """Reset all metrics."""
        for metric in self.torch_metrics.values():
            metric.reset()


class ModelComparison:
    """Compare multiple models on the same dataset.
    
    Args:
        evaluator: Evaluator instance.
        models: Dictionary of model names to models.
    """
    
    def __init__(
        self,
        evaluator: GraphTransformerEvaluator,
        models: Dict[str, nn.Module],
    ):
        self.evaluator = evaluator
        self.models = models
    
    def compare(
        self,
        dataloader: torch.utils.data.DataLoader,
        return_predictions: bool = False,
    ) -> Union[Dict[str, Dict[str, float]], Tuple[Dict[str, Dict[str, float]], Dict[str, List[float]]]]:
        """Compare all models on the dataset.
        
        Args:
            dataloader: Data loader for evaluation.
            return_predictions: Whether to return predictions.
            
        Returns:
            Dict[str, Dict[str, float]] or Tuple[Dict[str, Dict[str, float]], Dict[str, List[float]]]:
                Comparison results and optionally predictions.
        """
        results = {}
        all_predictions = {}
        
        for name, model in self.models.items():
            print(f"Evaluating {name}...")
            
            if return_predictions:
                model_results, predictions, targets = self.evaluator.evaluate(
                    model, dataloader, return_predictions=True
                )
                all_predictions[name] = predictions
            else:
                model_results = self.evaluator.evaluate(model, dataloader)
            
            results[name] = model_results
        
        if return_predictions:
            return results, all_predictions
        
        return results
    
    def create_leaderboard(
        self,
        results: Dict[str, Dict[str, float]],
        metric: str = "mae",
        ascending: bool = True,
    ) -> List[Tuple[str, float]]:
        """Create a leaderboard sorted by a specific metric.
        
        Args:
            results: Results from comparison.
            metric: Metric to sort by.
            ascending: Whether to sort in ascending order.
            
        Returns:
            List[Tuple[str, float]]: Sorted leaderboard.
        """
        leaderboard = []
        
        for name, metrics in results.items():
            if metric in metrics:
                leaderboard.append((name, metrics[metric]))
        
        leaderboard.sort(key=lambda x: x[1], reverse=not ascending)
        
        return leaderboard


def compute_error_analysis(
    predictions: List[float],
    targets: List[float],
    bins: int = 10,
) -> Dict[str, Union[List[float], List[int], List[float]]]:
    """Compute error analysis across different target value ranges.
    
    Args:
        predictions: Model predictions.
        targets: Ground truth targets.
        bins: Number of bins for analysis.
        
    Returns:
        Dict[str, Union[List[float], List[int], List[float]]]: Error analysis results.
    """
    predictions = np.array(predictions)
    targets = np.array(targets)
    
    # Compute errors
    errors = predictions - targets
    abs_errors = np.abs(errors)
    
    # Bin targets
    target_min, target_max = np.min(targets), np.max(targets)
    bin_edges = np.linspace(target_min, target_max, bins + 1)
    bin_indices = np.digitize(targets, bin_edges) - 1
    bin_indices = np.clip(bin_indices, 0, bins - 1)
    
    # Compute statistics per bin
    bin_centers = []
    bin_counts = []
    bin_mae = []
    bin_rmse = []
    bin_r2 = []
    
    for i in range(bins):
        mask = bin_indices == i
        if np.sum(mask) > 0:
            bin_centers.append((bin_edges[i] + bin_edges[i + 1]) / 2)
            bin_counts.append(np.sum(mask))
            bin_mae.append(np.mean(abs_errors[mask]))
            bin_rmse.append(np.sqrt(np.mean(errors[mask] ** 2)))
            
            # R2 for this bin
            if np.var(targets[mask]) > 0:
                bin_r2.append(1 - np.sum(errors[mask] ** 2) / np.sum((targets[mask] - np.mean(targets[mask])) ** 2))
            else:
                bin_r2.append(0.0)
    
    return {
        "bin_centers": bin_centers,
        "bin_counts": bin_counts,
        "bin_mae": bin_mae,
        "bin_rmse": bin_rmse,
        "bin_r2": bin_r2,
    }


def compute_residual_analysis(
    predictions: List[float],
    targets: List[float],
) -> Dict[str, float]:
    """Compute residual analysis for model evaluation.
    
    Args:
        predictions: Model predictions.
        targets: Ground truth targets.
        
    Returns:
        Dict[str, float]: Residual analysis results.
    """
    predictions = np.array(predictions)
    targets = np.array(targets)
    
    residuals = predictions - targets
    
    # Compute statistics
    mean_residual = np.mean(residuals)
    std_residual = np.std(residuals)
    
    # Normality test (simplified)
    from scipy import stats
    shapiro_stat, shapiro_p = stats.shapiro(residuals)
    
    # Heteroscedasticity test (simplified)
    # Correlation between residuals and targets
    hetero_corr = np.corrcoef(np.abs(residuals), targets)[0, 1]
    
    return {
        "mean_residual": mean_residual,
        "std_residual": std_residual,
        "shapiro_stat": shapiro_stat,
        "shapiro_p": shapiro_p,
        "hetero_corr": hetero_corr,
    }
