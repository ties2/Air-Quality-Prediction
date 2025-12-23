#!/usr/bin/env python3
"""
Training script for HOPE-Air continual air quality prediction.

Usage:
    # Train with default config
    python scripts/train.py

    # Train with specific config
    python scripts/train.py --config config/experiments/hope_full_cms.yaml

    # Continual learning mode
    python scripts/train.py --continual --n-tasks 12

    # Specific city
    python scripts/train.py --city Delhi --days 90
"""

import os
import sys
import argparse
from pathlib import Path
from datetime import datetime
import asyncio

import torch
import numpy as np
import pandas as pd
from loguru import logger
import yaml

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.hope.hope_air import HOPEAir, HOPEAirConfig, create_hope_air_base
from src.models.hope.cms import CMSConfig
from src.training.continual_trainer import (
    ContinualTrainer,
    ContinualTrainerConfig,
    create_tasks_from_data,
    AirQualityDataset
)
from src.data.openaq_client import OpenAQDataFetcher

try:
    import mlflow
    HAS_MLFLOW = True
except ImportError:
    HAS_MLFLOW = False


def parse_args():
    parser = argparse.ArgumentParser(description="Train HOPE-Air model")
    
    # Config
    parser.add_argument("--config", type=str, default=None,
                        help="Path to config YAML file")
    
    # Data
    parser.add_argument("--city", type=str, default="Delhi",
                        help="City to fetch data for")
    parser.add_argument("--days", type=int, default=30,
                        help="Days of historical data to fetch")
    parser.add_argument("--data-path", type=str, default=None,
                        help="Path to preprocessed data (skip API fetch)")
    
    # Model
    parser.add_argument("--model-size", type=str, default="base",
                        choices=["small", "base", "large"],
                        help="Model size preset")
    parser.add_argument("--hidden-dim", type=int, default=None,
                        help="Override hidden dimension")
    parser.add_argument("--n-layers", type=int, default=None,
                        help="Override number of layers")
    
    # Training
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--device", type=str, default="cuda")
    
    # Continual learning
    parser.add_argument("--continual", action="store_true",
                        help="Use continual learning mode")
    parser.add_argument("--n-tasks", type=int, default=12,
                        help="Number of sequential tasks")
    parser.add_argument("--use-ewc", action="store_true", default=True,
                        help="Use Elastic Weight Consolidation")
    parser.add_argument("--use-replay", action="store_true", default=True,
                        help="Use experience replay")
    
    # Logging
    parser.add_argument("--experiment-name", type=str, default="hope_air_training")
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default="models/")
    
    return parser.parse_args()


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def prepare_data(
    city: str,
    days: int,
    history_window: int = 168,
    forecast_horizon: int = 24
) -> tuple:
    """
    Fetch and prepare data for training.
    
    Returns:
        data: numpy array [n_samples, history + forecast, n_params]
        temporal: numpy array [n_samples, history, 4]
        param_names: list of parameter names
    """
    logger.info(f"Fetching data for {city} ({days} days)...")
    
    # Fetch from OpenAQ
    fetcher = OpenAQDataFetcher()
    df = asyncio.run(fetcher.fetch_city_data(
        city=city,
        parameters=["pm25", "pm10", "no2", "o3", "co", "so2"],
        days=days
    ))
    
    if df.empty:
        raise ValueError(f"No data found for {city}")
    
    logger.info(f"Fetched {len(df)} records")
    
    # Get parameter columns (exclude metadata)
    param_cols = [c for c in df.columns if c not in 
                  ["datetime", "location_id", "location_name", "latitude", "longitude"]]
    
    # Group by location and create sequences
    sequences = []
    temporal_features = []
    
    for loc_id, loc_df in df.groupby("location_id"):
        loc_df = loc_df.sort_values("datetime")
        
        # Resample to hourly and fill gaps
        loc_df = loc_df.set_index("datetime")
        loc_df = loc_df[param_cols].resample("H").mean()
        loc_df = loc_df.interpolate(method="linear", limit=6)
        loc_df = loc_df.dropna()
        
        if len(loc_df) < history_window + forecast_horizon:
            continue
        
        # Create sliding windows
        values = loc_df.values
        for i in range(len(values) - history_window - forecast_horizon + 1):
            seq = values[i:i + history_window + forecast_horizon]
            sequences.append(seq)
            
            # Temporal features: hour, day_of_week, month, is_weekend
            timestamps = loc_df.index[i:i + history_window]
            temporal = np.stack([
                timestamps.hour,
                timestamps.dayofweek,
                timestamps.month,
                (timestamps.dayofweek >= 5).astype(int)
            ], axis=-1)
            temporal_features.append(temporal)
    
    if not sequences:
        raise ValueError(f"Not enough continuous data for {city}")
    
    data = np.array(sequences, dtype=np.float32)
    temporal = np.array(temporal_features, dtype=np.int64)
    
    logger.info(f"Created {len(data)} training sequences")
    logger.info(f"Data shape: {data.shape}")
    
    return data, temporal, param_cols


def create_model(args, n_parameters: int = 6) -> HOPEAir:
    """Create HOPE-Air model based on arguments."""
    
    # Base config based on size
    if args.model_size == "small":
        hidden_dim = 128
        n_layers = 4
        n_heads = 4
        cms_levels = 2
        cms_freqs = [0.2, 0.8]
    elif args.model_size == "large":
        hidden_dim = 512
        n_layers = 12
        n_heads = 16
        cms_levels = 5
        cms_freqs = [0.05, 0.2, 0.5, 0.8, 0.95]
    else:  # base
        hidden_dim = 256
        n_layers = 6
        n_heads = 8
        cms_levels = 3
        cms_freqs = [0.1, 0.5, 0.9]
    
    # Override with args if provided
    if args.hidden_dim:
        hidden_dim = args.hidden_dim
    if args.n_layers:
        n_layers = args.n_layers
    
    config = HOPEAirConfig(
        n_parameters=n_parameters,
        history_window=168,
        forecast_horizon=24,
        hidden_dim=hidden_dim,
        n_layers=n_layers,
        n_heads=n_heads,
        cms_config=CMSConfig(
            n_levels=cms_levels,
            frequencies=cms_freqs,
            memory_dim=hidden_dim // 2,
            hidden_dim=hidden_dim
        )
    )
    
    return HOPEAir(config)


def train_standard(
    model: HOPEAir,
    train_loader,
    val_loader,
    args
) -> dict:
    """Standard (non-continual) training."""
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=1e-5
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.epochs,
        eta_min=1e-6
    )
    criterion = torch.nn.MSELoss()
    
    best_val_loss = float("inf")
    history = {"train_loss": [], "val_loss": []}
    
    for epoch in range(args.epochs):
        # Training
        model.train()
        train_loss = 0.0
        n_batches = 0
        
        for batch in train_loader:
            x = batch["x"].to(device)
            y = batch["y"].to(device)
            temporal = batch.get("temporal")
            if temporal is not None:
                temporal = temporal.to(device)
            
            optimizer.zero_grad()
            pred, _ = model(x, temporal_features=temporal)
            loss = criterion(pred, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            train_loss += loss.item()
            n_batches += 1
        
        train_loss /= n_batches
        history["train_loss"].append(train_loss)
        
        # Validation
        model.eval()
        val_loss = 0.0
        n_batches = 0
        
        with torch.no_grad():
            for batch in val_loader:
                x = batch["x"].to(device)
                y = batch["y"].to(device)
                temporal = batch.get("temporal")
                if temporal is not None:
                    temporal = temporal.to(device)
                
                pred, _ = model(x, temporal_features=temporal)
                loss = criterion(pred, y)
                val_loss += loss.item()
                n_batches += 1
        
        val_loss /= n_batches
        history["val_loss"].append(val_loss)
        
        scheduler.step()
        
        # Logging
        logger.info(
            f"Epoch {epoch+1}/{args.epochs} - "
            f"Train Loss: {train_loss:.4f} - Val Loss: {val_loss:.4f}"
        )
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(
                model.state_dict(),
                Path(args.output_dir) / "hope_air_best.pt"
            )
    
    return {
        "best_val_loss": best_val_loss,
        "final_train_loss": history["train_loss"][-1],
        "history": history
    }


def train_continual(
    model: HOPEAir,
    data: np.ndarray,
    temporal: np.ndarray,
    args
) -> dict:
    """Continual learning training."""
    
    # Create trainer config
    trainer_config = ContinualTrainerConfig(
        learning_rate=args.lr,
        batch_size=args.batch_size,
        epochs_per_task=args.epochs // args.n_tasks,
        n_tasks=args.n_tasks,
        use_replay=args.use_replay,
        use_ewc=args.use_ewc,
        checkpoint_dir=args.output_dir,
        device=args.device
    )
    
    # Create trainer
    trainer = ContinualTrainer(model, trainer_config)
    
    # Create tasks from data
    tasks = create_tasks_from_data(
        data,
        n_tasks=args.n_tasks,
        history_window=168,
        forecast_horizon=24,
        temporal_data=temporal,
        batch_size=args.batch_size
    )
    
    # Train on each task sequentially
    all_metrics = []
    for task_id, (train_loader, val_loader) in enumerate(tasks):
        logger.info(f"\n{'='*50}")
        logger.info(f"Starting Task {task_id + 1}/{args.n_tasks}")
        logger.info(f"{'='*50}")
        
        task_metrics = trainer.train_task(task_id, train_loader, val_loader)
        all_metrics.append(task_metrics)
        
        # Log continual metrics
        continual_metrics = trainer.get_continual_metrics()
        logger.info(f"Continual Metrics after Task {task_id + 1}:")
        logger.info(f"  Backward Transfer (BWT): {continual_metrics['backward_transfer']:.4f}")
        logger.info(f"  Forward Transfer (FWT): {continual_metrics['forward_transfer']:.4f}")
        logger.info(f"  Average Accuracy: {continual_metrics['average_accuracy']:.4f}")
        logger.info(f"  Forgetting Measure: {continual_metrics['forgetting_measure']:.4f}")
    
    return {
        "task_metrics": all_metrics,
        "final_continual_metrics": trainer.get_continual_metrics()
    }


def main():
    args = parse_args()
    
    # Setup logging
    logger.add(
        Path(args.output_dir) / "training.log",
        rotation="10 MB"
    )
    
    # Load config if provided
    if args.config:
        config = load_config(args.config)
        # Override args with config values
        for key, value in config.get("training", {}).items():
            if hasattr(args, key):
                setattr(args, key, value)
    
    # Create output directory
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    
    # Prepare data
    if args.data_path:
        # Load preprocessed data
        data = np.load(args.data_path)
        temporal = np.load(args.data_path.replace(".npy", "_temporal.npy"))
        param_cols = ["pm25", "pm10", "no2", "o3", "co", "so2"]
    else:
        # Fetch from OpenAQ
        data, temporal, param_cols = prepare_data(
            args.city,
            args.days
        )
        
        # Save preprocessed data
        np.save(Path(args.output_dir) / "data.npy", data)
        np.save(Path(args.output_dir) / "data_temporal.npy", temporal)
    
    # Create model
    model = create_model(args, n_parameters=len(param_cols))
    logger.info(f"Model parameters: {model.count_parameters():,}")
    
    # Setup MLflow
    if HAS_MLFLOW and args.experiment_name:
        mlflow.set_experiment(args.experiment_name)
        run_name = args.run_name or f"{args.city}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        mlflow.start_run(run_name=run_name)
        mlflow.log_params({
            "city": args.city,
            "days": args.days,
            "model_size": args.model_size,
            "batch_size": args.batch_size,
            "learning_rate": args.lr,
            "continual": args.continual,
            "n_tasks": args.n_tasks if args.continual else 1,
        })
    
    # Train
    if args.continual:
        results = train_continual(model, data, temporal, args)
    else:
        # Create dataloaders for standard training
        from torch.utils.data import DataLoader, random_split
        
        dataset = AirQualityDataset(data, temporal_data=temporal)
        train_size = int(0.85 * len(dataset))
        val_size = len(dataset) - train_size
        train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
        
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
        
        results = train_standard(model, train_loader, val_loader, args)
    
    # Log final results
    logger.info("\n" + "="*50)
    logger.info("Training Complete!")
    logger.info("="*50)
    
    if args.continual:
        final_metrics = results["final_continual_metrics"]
        logger.info(f"Final Backward Transfer: {final_metrics['backward_transfer']:.4f}")
        logger.info(f"Final Forward Transfer: {final_metrics['forward_transfer']:.4f}")
        logger.info(f"Final Average Accuracy: {final_metrics['average_accuracy']:.4f}")
        
        if HAS_MLFLOW:
            mlflow.log_metrics(final_metrics)
    else:
        logger.info(f"Best Validation Loss: {results['best_val_loss']:.4f}")
        
        if HAS_MLFLOW:
            mlflow.log_metric("best_val_loss", results['best_val_loss'])
    
    if HAS_MLFLOW:
        mlflow.end_run()
    
    logger.info(f"Model saved to {args.output_dir}")


if __name__ == "__main__":
    main()
