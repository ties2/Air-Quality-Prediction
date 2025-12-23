"""
Continual Learning Trainer for HOPE-Air

Implements training infrastructure for continual air quality prediction,
including:
- Sequential task training
- Experience replay
- Forgetting metrics (BWT, FWT)
- Elastic Weight Consolidation (EWC)

The goal is to train on streaming data without catastrophic forgetting.
"""

import os
import copy
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple, Callable
from dataclasses import dataclass, field
from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Subset
import numpy as np
from tqdm import tqdm
from loguru import logger

try:
    import mlflow
    HAS_MLFLOW = True
except ImportError:
    HAS_MLFLOW = False


@dataclass
class ContinualTrainerConfig:
    """Configuration for continual learning trainer."""
    # Basic training
    learning_rate: float = 1e-4
    weight_decay: float = 1e-5
    batch_size: int = 32
    epochs_per_task: int = 20
    gradient_clip: float = 1.0
    
    # Continual learning
    n_tasks: int = 12
    task_duration: str = "7d"
    
    # Experience replay
    use_replay: bool = True
    replay_buffer_size: int = 1000
    replay_ratio: float = 0.3
    replay_strategy: str = "reservoir"
    
    # Regularization
    use_ewc: bool = True
    ewc_lambda: float = 5000.0
    n_fisher_samples: int = 200
    
    # Validation
    val_split: float = 0.15
    eval_on_all_tasks: bool = True
    
    # Checkpointing
    checkpoint_dir: str = "models/"
    save_every_n_tasks: int = 1
    
    # Logging
    log_interval: int = 100
    use_mlflow: bool = True
    
    # Device
    device: str = "cuda"


class ReplayBuffer:
    """Experience replay buffer for continual learning."""
    
    def __init__(self, capacity: int, strategy: str = "reservoir"):
        self.capacity = capacity
        self.strategy = strategy
        self.buffer: List[Dict[str, torch.Tensor]] = []
        self.priorities: List[float] = []
        self.n_seen = 0
    
    def add(self, sample: Dict[str, torch.Tensor], priority: float = 1.0):
        """Add sample to buffer."""
        self.n_seen += 1
        
        if self.strategy == "reservoir":
            if len(self.buffer) < self.capacity:
                self.buffer.append(sample)
                self.priorities.append(priority)
            else:
                idx = np.random.randint(0, self.n_seen)
                if idx < self.capacity:
                    self.buffer[idx] = sample
                    self.priorities[idx] = priority
        
        elif self.strategy == "fifo":
            if len(self.buffer) < self.capacity:
                self.buffer.append(sample)
                self.priorities.append(priority)
            else:
                self.buffer.pop(0)
                self.priorities.pop(0)
                self.buffer.append(sample)
                self.priorities.append(priority)
    
    def sample(self, n: int) -> List[Dict[str, torch.Tensor]]:
        """Sample n items from buffer."""
        if len(self.buffer) == 0:
            return []
        n = min(n, len(self.buffer))
        indices = np.random.choice(len(self.buffer), n, replace=False)
        return [self.buffer[i] for i in indices]
    
    def __len__(self) -> int:
        return len(self.buffer)


class EWC:
    """Elastic Weight Consolidation for preventing catastrophic forgetting."""
    
    def __init__(self, model: nn.Module, lambda_ewc: float = 5000.0, n_samples: int = 200):
        self.model = model
        self.lambda_ewc = lambda_ewc
        self.n_samples = n_samples
        self.saved_params: Dict[int, Dict[str, torch.Tensor]] = {}
        self.fisher: Dict[int, Dict[str, torch.Tensor]] = {}
        self.task_count = 0
    
    def compute_fisher(self, dataloader: DataLoader, criterion: Callable):
        """Compute Fisher information matrix for current task."""
        fisher = {}
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                fisher[name] = torch.zeros_like(param)
        
        self.model.eval()
        n_samples = 0
        
        for batch in dataloader:
            if n_samples >= self.n_samples:
                break
            
            x = batch["x"].to(next(self.model.parameters()).device)
            y = batch["y"].to(next(self.model.parameters()).device)
            temporal = batch.get("temporal")
            if temporal is not None:
                temporal = temporal.to(x.device)
            
            self.model.zero_grad()
            pred, _ = self.model(x, temporal_features=temporal)
            loss = criterion(pred, y)
            loss.backward()
            
            for name, param in self.model.named_parameters():
                if param.requires_grad and param.grad is not None:
                    fisher[name] += param.grad.pow(2)
            
            n_samples += x.size(0)
        
        # Normalize
        for name in fisher:
            fisher[name] /= n_samples
        
        # Store
        self.fisher[self.task_count] = fisher
        self.saved_params[self.task_count] = {
            name: param.clone().detach()
            for name, param in self.model.named_parameters()
            if param.requires_grad
        }
        self.task_count += 1
    
    def penalty(self) -> torch.Tensor:
        """Compute EWC penalty."""
        if self.task_count == 0:
            return torch.tensor(0.0)
        
        penalty = 0.0
        for task_id in range(self.task_count):
            for name, param in self.model.named_parameters():
                if name in self.fisher[task_id]:
                    penalty += (
                        self.fisher[task_id][name] *
                        (param - self.saved_params[task_id][name]).pow(2)
                    ).sum()
        
        return self.lambda_ewc * penalty


class ContinualMetrics:
    """Metrics for evaluating continual learning performance."""
    
    def __init__(self, n_tasks: int):
        self.n_tasks = n_tasks
        # Performance matrix: R[i,j] = performance on task j after training on task i
        self.performance_matrix = np.zeros((n_tasks, n_tasks))
        self.current_task = 0
    
    def update(self, task_trained: int, task_evaluated: int, metric: float):
        """Update performance matrix."""
        self.performance_matrix[task_trained, task_evaluated] = metric
    
    def backward_transfer(self) -> float:
        """
        Backward Transfer (BWT): How much learning new tasks affects old tasks.
        Negative BWT indicates forgetting.
        
        BWT = (1/T-1) * sum_{i=1}^{T-1} (R[T,i] - R[i,i])
        """
        if self.current_task < 2:
            return 0.0
        
        T = self.current_task
        bwt = 0.0
        for i in range(T - 1):
            bwt += self.performance_matrix[T-1, i] - self.performance_matrix[i, i]
        return bwt / (T - 1)
    
    def forward_transfer(self) -> float:
        """
        Forward Transfer (FWT): How much learning previous tasks helps new tasks.
        
        FWT = (1/T-1) * sum_{i=2}^{T} (R[i-1,i] - baseline[i])
        """
        if self.current_task < 2:
            return 0.0
        
        T = self.current_task
        fwt = 0.0
        for i in range(1, T):
            # Compare with random baseline (assumed 0)
            fwt += self.performance_matrix[i-1, i]
        return fwt / (T - 1)
    
    def average_accuracy(self) -> float:
        """Average accuracy across all tasks after final training."""
        if self.current_task == 0:
            return 0.0
        T = self.current_task
        return np.mean(self.performance_matrix[T-1, :T])
    
    def forgetting_measure(self) -> float:
        """
        Forgetting Measure: Maximum performance drop on any task.
        
        FM = (1/T-1) * sum_{i=1}^{T-1} max_{j in 1..T-1} (R[j,i] - R[T,i])
        """
        if self.current_task < 2:
            return 0.0
        
        T = self.current_task
        fm = 0.0
        for i in range(T - 1):
            max_perf = max(self.performance_matrix[j, i] for j in range(i, T))
            fm += max_perf - self.performance_matrix[T-1, i]
        return fm / (T - 1)


class ContinualTrainer:
    """
    Trainer for continual learning with HOPE-Air.
    
    Handles:
    - Sequential task training
    - Experience replay
    - EWC regularization
    - Comprehensive forgetting metrics
    """
    
    def __init__(
        self,
        model: nn.Module,
        config: ContinualTrainerConfig
    ):
        self.model = model
        self.config = config
        self.device = torch.device(config.device if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        
        # Optimizer
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay
        )
        
        # Loss function
        self.criterion = nn.MSELoss()
        
        # Replay buffer
        if config.use_replay:
            self.replay_buffer = ReplayBuffer(
                config.replay_buffer_size,
                config.replay_strategy
            )
        else:
            self.replay_buffer = None
        
        # EWC
        if config.use_ewc:
            self.ewc = EWC(
                model,
                lambda_ewc=config.ewc_lambda,
                n_samples=config.n_fisher_samples
            )
        else:
            self.ewc = None
        
        # Metrics
        self.metrics = ContinualMetrics(config.n_tasks)
        
        # Training state
        self.current_task = 0
        self.global_step = 0
        self.task_dataloaders: List[DataLoader] = []
        
        # Checkpointing
        self.checkpoint_dir = Path(config.checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    def train_task(
        self,
        task_id: int,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None
    ) -> Dict[str, float]:
        """Train on a single task."""
        logger.info(f"Training on Task {task_id}")
        self.current_task = task_id
        self.task_dataloaders.append(train_loader)
        
        best_val_loss = float("inf")
        task_metrics = defaultdict(list)
        
        for epoch in range(self.config.epochs_per_task):
            # Training
            train_loss = self._train_epoch(train_loader, task_id)
            task_metrics["train_loss"].append(train_loss)
            
            # Validation
            if val_loader is not None:
                val_loss = self._validate(val_loader)
                task_metrics["val_loss"].append(val_loss)
                
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    self._save_checkpoint(f"task_{task_id}_best.pt")
            
            logger.info(
                f"Task {task_id} Epoch {epoch+1}/{self.config.epochs_per_task} - "
                f"Train Loss: {train_loss:.4f}" +
                (f" Val Loss: {val_loss:.4f}" if val_loader else "")
            )
        
        # Update EWC Fisher information
        if self.ewc is not None:
            logger.info("Computing Fisher information for EWC...")
            self.ewc.compute_fisher(train_loader, self.criterion)
        
        # Evaluate on all previous tasks
        if self.config.eval_on_all_tasks:
            self._evaluate_all_tasks()
        
        # Update metrics
        self.metrics.current_task = task_id + 1
        
        # Save checkpoint
        if (task_id + 1) % self.config.save_every_n_tasks == 0:
            self._save_checkpoint(f"task_{task_id}_final.pt")
        
        return {
            "final_train_loss": task_metrics["train_loss"][-1],
            "best_val_loss": best_val_loss,
            "bwt": self.metrics.backward_transfer(),
            "fwt": self.metrics.forward_transfer(),
            "avg_acc": self.metrics.average_accuracy(),
        }
    
    def _train_epoch(self, dataloader: DataLoader, task_id: int) -> float:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        n_batches = 0
        
        pbar = tqdm(dataloader, desc=f"Task {task_id}")
        for batch in pbar:
            loss = self._train_step(batch, task_id)
            total_loss += loss
            n_batches += 1
            
            pbar.set_postfix({"loss": f"{loss:.4f}"})
            self.global_step += 1
        
        return total_loss / n_batches
    
    def _train_step(self, batch: Dict[str, torch.Tensor], task_id: int) -> float:
        """Single training step."""
        # Move to device
        x = batch["x"].to(self.device)
        y = batch["y"].to(self.device)
        temporal = batch.get("temporal")
        if temporal is not None:
            temporal = temporal.to(self.device)
        
        # Mix with replay buffer
        if self.replay_buffer is not None and len(self.replay_buffer) > 0:
            n_replay = int(self.config.batch_size * self.config.replay_ratio)
            replay_samples = self.replay_buffer.sample(n_replay)
            
            if replay_samples:
                replay_x = torch.stack([s["x"] for s in replay_samples]).to(self.device)
                replay_y = torch.stack([s["y"] for s in replay_samples]).to(self.device)
                
                x = torch.cat([x, replay_x], dim=0)
                y = torch.cat([y, replay_y], dim=0)
                
                if temporal is not None:
                    replay_temporal = torch.stack([s["temporal"] for s in replay_samples]).to(self.device)
                    temporal = torch.cat([temporal, replay_temporal], dim=0)
        
        # Forward pass
        self.optimizer.zero_grad()
        pred, _ = self.model(x, temporal_features=temporal)
        
        # Loss
        loss = self.criterion(pred, y)
        
        # Add EWC penalty
        if self.ewc is not None and task_id > 0:
            ewc_loss = self.ewc.penalty()
            loss = loss + ewc_loss
        
        # Backward pass
        loss.backward()
        
        # Gradient clipping
        if self.config.gradient_clip > 0:
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                self.config.gradient_clip
            )
        
        self.optimizer.step()
        
        # Add to replay buffer
        if self.replay_buffer is not None:
            for i in range(min(10, x.size(0))):  # Add subset to buffer
                sample = {
                    "x": x[i].cpu(),
                    "y": y[i].cpu(),
                }
                if temporal is not None:
                    sample["temporal"] = temporal[i].cpu()
                self.replay_buffer.add(sample, priority=loss.item())
        
        return loss.item()
    
    def _validate(self, dataloader: DataLoader) -> float:
        """Validate on a dataloader."""
        self.model.eval()
        total_loss = 0.0
        n_batches = 0
        
        with torch.no_grad():
            for batch in dataloader:
                x = batch["x"].to(self.device)
                y = batch["y"].to(self.device)
                temporal = batch.get("temporal")
                if temporal is not None:
                    temporal = temporal.to(self.device)
                
                pred, _ = self.model(x, temporal_features=temporal)
                loss = self.criterion(pred, y)
                
                total_loss += loss.item()
                n_batches += 1
        
        return total_loss / n_batches
    
    def _evaluate_all_tasks(self):
        """Evaluate on all previous tasks to measure forgetting."""
        self.model.eval()
        
        for task_id, loader in enumerate(self.task_dataloaders):
            loss = self._validate(loader)
            # Convert loss to accuracy-like metric (lower loss = higher performance)
            # Using negative loss as "performance" so higher is better
            performance = -loss
            self.metrics.update(self.current_task, task_id, performance)
            
            logger.info(f"  Task {task_id} loss: {loss:.4f}")
    
    def _save_checkpoint(self, filename: str):
        """Save model checkpoint."""
        path = self.checkpoint_dir / filename
        torch.save({
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "current_task": self.current_task,
            "global_step": self.global_step,
            "config": self.config,
            "metrics": {
                "performance_matrix": self.metrics.performance_matrix,
                "bwt": self.metrics.backward_transfer(),
                "fwt": self.metrics.forward_transfer(),
                "avg_acc": self.metrics.average_accuracy(),
            }
        }, path)
        logger.info(f"Saved checkpoint: {path}")
    
    def load_checkpoint(self, path: str):
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.current_task = checkpoint["current_task"]
        self.global_step = checkpoint["global_step"]
        logger.info(f"Loaded checkpoint: {path}")
    
    def get_continual_metrics(self) -> Dict[str, float]:
        """Get all continual learning metrics."""
        return {
            "backward_transfer": self.metrics.backward_transfer(),
            "forward_transfer": self.metrics.forward_transfer(),
            "average_accuracy": self.metrics.average_accuracy(),
            "forgetting_measure": self.metrics.forgetting_measure(),
            "n_tasks_trained": self.current_task + 1,
        }


class AirQualityDataset(Dataset):
    """Dataset for air quality time series prediction."""
    
    def __init__(
        self,
        data: np.ndarray,  # [n_samples, history + forecast, n_params]
        history_window: int = 168,
        forecast_horizon: int = 24,
        temporal_data: Optional[np.ndarray] = None
    ):
        self.data = torch.FloatTensor(data)
        self.history_window = history_window
        self.forecast_horizon = forecast_horizon
        self.temporal_data = torch.LongTensor(temporal_data) if temporal_data is not None else None
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sequence = self.data[idx]
        x = sequence[:self.history_window]
        y = sequence[self.history_window:self.history_window + self.forecast_horizon]
        
        sample = {"x": x, "y": y}
        
        if self.temporal_data is not None:
            sample["temporal"] = self.temporal_data[idx, :self.history_window]
        
        return sample


def create_tasks_from_data(
    data: np.ndarray,
    n_tasks: int,
    history_window: int = 168,
    forecast_horizon: int = 24,
    temporal_data: Optional[np.ndarray] = None,
    batch_size: int = 32
) -> List[Tuple[DataLoader, DataLoader]]:
    """
    Split data into sequential tasks for continual learning.
    
    Args:
        data: Full dataset [n_samples, seq_len, n_params]
        n_tasks: Number of tasks to create
        history_window: Input sequence length
        forecast_horizon: Output sequence length
        temporal_data: Temporal features
        batch_size: Batch size for dataloaders
    
    Returns:
        List of (train_loader, val_loader) tuples, one per task
    """
    n_samples = len(data)
    samples_per_task = n_samples // n_tasks
    
    tasks = []
    
    for task_id in range(n_tasks):
        start_idx = task_id * samples_per_task
        end_idx = start_idx + samples_per_task if task_id < n_tasks - 1 else n_samples
        
        task_data = data[start_idx:end_idx]
        task_temporal = temporal_data[start_idx:end_idx] if temporal_data is not None else None
        
        dataset = AirQualityDataset(
            task_data,
            history_window,
            forecast_horizon,
            task_temporal
        )
        
        # Train/val split
        n_val = int(len(dataset) * 0.15)
        n_train = len(dataset) - n_val
        
        train_dataset = Subset(dataset, range(n_train))
        val_dataset = Subset(dataset, range(n_train, len(dataset)))
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        
        tasks.append((train_loader, val_loader))
    
    return tasks


if __name__ == "__main__":
    # Example usage
    from src.models.hope.hope_air import create_hope_air_small
    
    # Create model
    model = create_hope_air_small()
    
    # Create trainer
    config = ContinualTrainerConfig(
        n_tasks=4,
        epochs_per_task=5,
        use_replay=True,
        use_ewc=True
    )
    trainer = ContinualTrainer(model, config)
    
    # Simulate data
    n_samples = 1000
    history = 168
    forecast = 24
    n_params = 6
    
    data = np.random.randn(n_samples, history + forecast, n_params).astype(np.float32)
    temporal = np.random.randint(0, 24, (n_samples, history, 4))
    
    # Create tasks
    tasks = create_tasks_from_data(data, n_tasks=4, temporal_data=temporal)
    
    # Train sequentially
    for task_id, (train_loader, val_loader) in enumerate(tasks):
        metrics = trainer.train_task(task_id, train_loader, val_loader)
        print(f"Task {task_id} metrics: {metrics}")
    
    # Final continual metrics
    print("\nFinal Continual Learning Metrics:")
    print(trainer.get_continual_metrics())
