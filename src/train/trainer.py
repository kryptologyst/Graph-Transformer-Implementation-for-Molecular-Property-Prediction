"""Training utilities for Graph Transformer models."""

import os
import time
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from ..utils.core import EarlyStopping, get_device, set_seed
from ..data.dataset import GraphDataModule
from ..eval.metrics import GraphTransformerEvaluator


class GraphTransformerTrainer:
    """Trainer for Graph Transformer models.
    
    Args:
        model: Model to train.
        data_module: Data module for training.
        config: Training configuration.
        device: Device to train on.
        use_wandb: Whether to use Weights & Biases logging.
        use_tensorboard: Whether to use TensorBoard logging.
    """
    
    def __init__(
        self,
        model: nn.Module,
        data_module: GraphDataModule,
        config: Dict,
        device: Optional[torch.device] = None,
        use_wandb: bool = False,
        use_tensorboard: bool = True,
    ):
        self.model = model
        self.data_module = data_module
        self.config = config
        self.device = device or get_device()
        self.use_wandb = use_wandb
        self.use_tensorboard = use_tensorboard
        
        # Move model to device
        self.model.to(self.device)
        
        # Initialize optimizer
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=config["training"]["learning_rate"],
            weight_decay=config["training"]["weight_decay"],
        )
        
        # Initialize loss function
        self.criterion = nn.L1Loss()  # MAE loss
        
        # Initialize evaluator
        self.evaluator = GraphTransformerEvaluator(self.device)
        
        # Initialize early stopping
        self.early_stopping = EarlyStopping(
            patience=config["training"]["patience"],
            min_delta=config["training"]["min_delta"],
        )
        
        # Initialize logging
        if self.use_tensorboard:
            self.writer = SummaryWriter(config["logging"]["log_dir"])
        
        if self.use_wandb:
            import wandb
            wandb.init(
                project=config["logging"]["wandb_project"],
                config=config,
            )
        
        # Training state
        self.current_epoch = 0
        self.best_val_loss = float("inf")
        self.train_losses = []
        self.val_losses = []
        
    def train_epoch(self) -> float:
        """Train for one epoch.
        
        Returns:
            float: Average training loss.
        """
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        train_loader = self.data_module.train_dataloader()
        
        with tqdm(train_loader, desc=f"Epoch {self.current_epoch}") as pbar:
            for batch in train_loader:
                batch = batch.to(self.device)
                
                # Forward pass
                self.optimizer.zero_grad()
                predictions = self.model(
                    batch.x,
                    batch.edge_index,
                    batch.edge_attr,
                    batch.batch,
                )
                
                loss = self.criterion(predictions, batch.y.view(-1))
                
                # Backward pass
                loss.backward()
                
                # Gradient clipping
                if self.config["training"]["gradient_clip_norm"] > 0:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.config["training"]["gradient_clip_norm"],
                    )
                
                self.optimizer.step()
                
                total_loss += loss.item()
                num_batches += 1
                
                # Update progress bar
                pbar.set_postfix({"loss": f"{loss.item():.4f}"})
        
        avg_loss = total_loss / num_batches
        return avg_loss
    
    def validate_epoch(self) -> float:
        """Validate for one epoch.
        
        Returns:
            float: Average validation loss.
        """
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        
        val_loader = self.data_module.val_dataloader()
        
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(self.device)
                
                predictions = self.model(
                    batch.x,
                    batch.edge_index,
                    batch.edge_attr,
                    batch.batch,
                )
                
                loss = self.criterion(predictions, batch.y.view(-1))
                total_loss += loss.item()
                num_batches += 1
        
        avg_loss = total_loss / num_batches
        return avg_loss
    
    def train(self) -> Dict[str, float]:
        """Train the model.
        
        Returns:
            Dict[str, float]: Training results.
        """
        print("Starting training...")
        print(f"Device: {self.device}")
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        
        start_time = time.time()
        
        for epoch in range(self.config["training"]["num_epochs"]):
            self.current_epoch = epoch
            
            # Train
            train_loss = self.train_epoch()
            self.train_losses.append(train_loss)
            
            # Validate
            val_loss = self.validate_epoch()
            self.val_losses.append(val_loss)
            
            # Log metrics
            if self.use_tensorboard:
                self.writer.add_scalar("Loss/Train", train_loss, epoch)
                self.writer.add_scalar("Loss/Validation", val_loss, epoch)
            
            if self.use_wandb:
                import wandb
                wandb.log({
                    "epoch": epoch,
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                })
            
            # Print progress
            print(f"Epoch {epoch:03d}: Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
            
            # Check for best model
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.save_checkpoint(is_best=True)
            
            # Early stopping
            if self.early_stopping(val_loss, self.model):
                print(f"Early stopping at epoch {epoch}")
                break
            
            # Save checkpoint
            if epoch % self.config["logging"]["save_interval"] == 0:
                self.save_checkpoint(is_best=False)
        
        training_time = time.time() - start_time
        print(f"Training completed in {training_time:.2f} seconds")
        
        # Final evaluation
        final_results = self.evaluate()
        
        return {
            "best_val_loss": self.best_val_loss,
            "final_train_loss": self.train_losses[-1],
            "final_val_loss": self.val_losses[-1],
            "training_time": training_time,
            "num_epochs": self.current_epoch + 1,
            **final_results,
        }
    
    def evaluate(self) -> Dict[str, float]:
        """Evaluate the model on test set.
        
        Returns:
            Dict[str, float]: Evaluation results.
        """
        print("Evaluating model...")
        
        # Evaluate on test set
        test_results = self.evaluator.evaluate(
            self.model,
            self.data_module.test_dataloader(),
        )
        
        # Evaluate on validation set
        val_results = self.evaluator.evaluate(
            self.model,
            self.data_module.val_dataloader(),
        )
        
        # Print results
        print("Test Results:")
        for metric, value in test_results.items():
            print(f"  {metric.upper()}: {value:.4f}")
        
        print("Validation Results:")
        for metric, value in val_results.items():
            print(f"  {metric.upper()}: {value:.4f}")
        
        return {
            "test_" + k: v for k, v in test_results.items()
        }
    
    def save_checkpoint(self, is_best: bool = False):
        """Save model checkpoint.
        
        Args:
            is_best: Whether this is the best model.
        """
        checkpoint = {
            "epoch": self.current_epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "best_val_loss": self.best_val_loss,
            "train_losses": self.train_losses,
            "val_losses": self.val_losses,
            "config": self.config,
        }
        
        # Save regular checkpoint
        checkpoint_path = os.path.join(
            self.config["paths"]["checkpoints_dir"],
            f"checkpoint_epoch_{self.current_epoch}.pt",
        )
        torch.save(checkpoint, checkpoint_path)
        
        # Save best model
        if is_best:
            best_path = os.path.join(
                self.config["paths"]["checkpoints_dir"],
                "best_model.pt",
            )
            torch.save(checkpoint, best_path)
            print(f"Best model saved at epoch {self.current_epoch}")
    
    def load_checkpoint(self, checkpoint_path: str):
        """Load model checkpoint.
        
        Args:
            checkpoint_path: Path to checkpoint file.
        """
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.current_epoch = checkpoint["epoch"]
        self.best_val_loss = checkpoint["best_val_loss"]
        self.train_losses = checkpoint["train_losses"]
        self.val_losses = checkpoint["val_losses"]
        
        print(f"Checkpoint loaded from epoch {self.current_epoch}")
    
    def close(self):
        """Close logging and cleanup."""
        if self.use_tensorboard:
            self.writer.close()
        
        if self.use_wandb:
            import wandb
            wandb.finish()


def train_model(
    model: nn.Module,
    data_module: GraphDataModule,
    config: Dict,
    device: Optional[torch.device] = None,
    use_wandb: bool = False,
    use_tensorboard: bool = True,
) -> Dict[str, float]:
    """Train a Graph Transformer model.
    
    Args:
        model: Model to train.
        data_module: Data module for training.
        config: Training configuration.
        device: Device to train on.
        use_wandb: Whether to use Weights & Biases logging.
        use_tensorboard: Whether to use TensorBoard logging.
        
    Returns:
        Dict[str, float]: Training results.
    """
    trainer = GraphTransformerTrainer(
        model=model,
        data_module=data_module,
        config=config,
        device=device,
        use_wandb=use_wandb,
        use_tensorboard=use_tensorboard,
    )
    
    try:
        results = trainer.train()
        return results
    finally:
        trainer.close()
