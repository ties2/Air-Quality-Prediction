# Nested Learning for Continual Air Quality Prediction

> Applying Google's Nested Learning paradigm (NeurIPS 2025) to urban air pollution forecasting using OpenAQ streaming data

## Overview

This project implements a **Continual Air Quality Prediction System** using **Nested Learning** principles and the **HOPE (Hierarchically Optimizing Processing Ensembles)** architecture. The system continuously adapts to evolving pollution patterns without catastrophic forgetting—addressing a critical limitation of traditional deep learning models in dynamic, streaming environments.

### Why Nested Learning for Air Pollution?

Urban air quality generates massive streaming data that changes rapidly:
- **Rush-hour spikes** in traffic-related pollutants (NO₂, CO)
- **Seasonal patterns** (winter inversions, summer ozone)
- **Expanding sensor networks** adding new data sources
- **Non-stationary distributions** from policy changes or new emission sources

Traditional ML models struggle with these dynamics—they either forget past patterns when retrained or fail to adapt to new ones. Nested Learning's **multi-frequency update mechanism** and **Continuum Memory System (CMS)** directly address this by:

1. **Fast-updating modules** for immediate changes (hourly/daily pollution spikes)
2. **Slow-updating modules** for long-term patterns (seasonal trends, urban growth impacts)
3. **Self-modifying architecture** that learns how to learn from streaming data

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    HOPE-Air Architecture                        │
├─────────────────────────────────────────────────────────────────┤
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐             │
│  │   Fast CMS  │  │  Medium CMS │  │   Slow CMS  │             │
│  │   (hourly)  │  │   (daily)   │  │  (seasonal) │             │
│  │  τ = 0.1    │  │   τ = 0.5   │  │   τ = 0.9   │             │
│  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘             │
│         │                │                │                     │
│         └────────────────┼────────────────┘                     │
│                          ▼                                      │
│              ┌───────────────────┐                              │
│              │  Self-Modifying   │                              │
│              │  Titans Module    │                              │
│              │  (Surprise-Gated) │                              │
│              └─────────┬─────────┘                              │
│                        │                                        │
│                        ▼                                        │
│              ┌───────────────────┐                              │
│              │   Prediction Head │                              │
│              │   (Multi-target)  │                              │
│              └───────────────────┘                              │
│                        │                                        │
│            ┌───────────┴───────────┐                            │
│            ▼           ▼           ▼                            │
│         [PM2.5]     [NO₂]       [O₃]                           │
└─────────────────────────────────────────────────────────────────┘
```

## Key Features

- **Continual Learning**: Adapts to new data without forgetting (measured via backward transfer)
- **Multi-Scale Memory**: CMS with configurable update frequencies for different temporal patterns
- **Surprise-Gated Learning**: Prioritizes learning from anomalous events (pollution spikes)
- **Real-Time Streaming**: Direct integration with OpenAQ API v3
- **Multi-Pollutant Forecasting**: Simultaneous prediction of PM2.5, PM10, NO₂, O₃, CO, SO₂
- **Experiment Tracking**: Full MLflow integration for reproducibility

## Project Structure

```
nested_air_pollution/
├── README.md               # This file
├── LICENSE                 # MIT License
├── .gitignore              
├── requirements.txt        # Python dependencies
├── setup.py                # Package installation
├── pyproject.toml          # Modern Python packaging
├── Dockerfile              
├── docker-compose.yml      
│
├── config/                 # Configuration files
│   ├── config.yaml         # Main hyperparameters
│   ├── cities/             # City-specific configs
│   │   ├── delhi.yaml
│   │   ├── beijing.yaml
│   │   └── los_angeles.yaml
│   └── experiments/        # Experiment presets
│       ├── baseline_lstm.yaml
│       ├── hope_small.yaml
│       └── hope_full_cms.yaml
│
├── data/                   # Data storage (git-ignored)
│   ├── raw/                # Raw API responses
│   ├── processed/          # Cleaned time series
│   ├── tasks/              # Prepared continual learning tasks
│   └── cache/              # API response cache
│
├── docs/                   # Documentation
│   ├── architecture.md     # Detailed architecture docs
│   ├── api.md              # Internal API documentation
│   ├── experiments.md      # Experiment reproduction guide
│   └── nested_learning_primer.md  # NL concepts explained
│
├── notebooks/              # Jupyter notebooks
│   ├── 01_openaq_exploration.ipynb
│   ├── 02_data_preprocessing.ipynb
│   ├── 03_baseline_models.ipynb
│   ├── 04_hope_training.ipynb
│   ├── 05_continual_evaluation.ipynb
│   └── 06_visualization.ipynb
│
├── src/                    # Source code
│   ├── __init__.py
│   │
│   ├── data/               # Data pipeline
│   │   ├── __init__.py
│   │   ├── openaq_client.py      # OpenAQ API v3 wrapper
│   │   ├── stream_handler.py     # Real-time data streaming
│   │   ├── preprocessor.py       # Cleaning, normalization
│   │   ├── feature_engineer.py   # Temporal features, lags
│   │   └── task_generator.py     # Continual learning task creation
│   │
│   ├── models/             # Model implementations
│   │   ├── __init__.py
│   │   ├── base.py               # Base model interface
│   │   ├── baselines/
│   │   │   ├── lstm.py           # LSTM baseline
│   │   │   ├── transformer.py    # Standard Transformer
│   │   │   └── gru.py            # GRU baseline
│   │   ├── hope/
│   │   │   ├── __init__.py
│   │   │   ├── cms.py            # Continuum Memory System
│   │   │   ├── titans.py         # Self-modifying Titans module
│   │   │   ├── hope_air.py       # HOPE adapted for air quality
│   │   │   └── deep_optimizer.py # Deep Momentum GD
│   │   └── attention/
│   │       ├── linear_attention.py
│   │       └── associative_memory.py
│   │
│   ├── training/           # Training infrastructure
│   │   ├── __init__.py
│   │   ├── trainer.py            # Main training loop
│   │   ├── continual_trainer.py  # Continual learning trainer
│   │   ├── schedulers.py         # LR schedulers for CMS
│   │   └── callbacks.py          # Training callbacks
│   │
│   ├── evaluation/         # Evaluation metrics
│   │   ├── __init__.py
│   │   ├── metrics.py            # MAE, RMSE, MAPE, etc.
│   │   ├── forgetting.py         # Backward/forward transfer
│   │   └── visualization.py      # Plotting utilities
│   │
│   ├── deployment/         # Deployment scripts
│   │   ├── __init__.py
│   │   ├── app.py                # FastAPI service
│   │   ├── predict.py            # Batch inference
│   │   └── alert_system.py       # Pollution alert logic
│   │
│   ├── utils/              # Utilities
│   │   ├── __init__.py
│   │   ├── logging.py            # Logging setup
│   │   ├── config.py             # Config loading
│   │   └── helpers.py            # General utilities
│   │
│   └── pipeline.py         # Main orchestration
│
├── scripts/                # CLI scripts
│   ├── fetch_data.py       # Download OpenAQ data
│   ├── train.py            # Training entrypoint
│   ├── evaluate.py         # Evaluation entrypoint
│   └── serve.py            # Start API server
│
├── tests/                  # Unit tests
│   ├── test_data.py
│   ├── test_models.py
│   ├── test_cms.py
│   └── test_continual.py
│
├── models/                 # Saved models (git-ignored)
│   └── .gitkeep
│
├── reports/                # Output reports
│   ├── figures/
│   └── metrics/
│
├── mlflow/                 # MLflow tracking (git-ignored)
│   └── .gitkeep
│
└── workflows/              # CI/CD
    └── .github/workflows/
        └── ci.yaml
```

## Quick Start

### 1. Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/nested_air_pollution.git
cd nested_air_pollution

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or: venv\Scripts\activate  # Windows

# Install dependencies
pip install -e .
```

### 2. Get OpenAQ API Key

1. Sign up at https://openaq.org
2. Get your API key from the dashboard
3. Set it as an environment variable:

```bash
export OPENAQ_API_KEY="your-api-key-here"
```

### 3. Fetch Initial Data

```bash
# Download historical data for Delhi (default: last 30 days)
python scripts/fetch_data.py --city Delhi --days 30

# Or specify multiple cities
python scripts/fetch_data.py --cities Delhi,Beijing,LosAngeles --days 90
```

### 4. Train a Model

```bash
# Train HOPE model with default config
python scripts/train.py --config config/experiments/hope_small.yaml

# Train with continual learning (sequential tasks)
python scripts/train.py --config config/experiments/hope_full_cms.yaml --continual

# Compare with baseline
python scripts/train.py --config config/experiments/baseline_lstm.yaml
```

### 5. Evaluate

```bash
# Evaluate on held-out test set
python scripts/evaluate.py --checkpoint models/hope_best.pt

# Measure forgetting across tasks
python scripts/evaluate.py --checkpoint models/hope_best.pt --measure-forgetting
```

## Configuration

### Main Config (`config/config.yaml`)

```yaml
# Data settings
data:
  api_key: ${OPENAQ_API_KEY}  # From environment
  cities: ["Delhi", "Beijing", "Los Angeles"]
  parameters: ["pm25", "pm10", "no2", "o3", "co", "so2"]
  history_window: 168  # 7 days of hourly data
  forecast_horizon: 24  # Predict 24 hours ahead
  cache_enabled: true

# Model settings
model:
  name: "hope_air"
  hidden_dim: 256
  n_layers: 6
  n_heads: 8
  
  # CMS configuration
  cms:
    n_levels: 3
    frequencies: [0.1, 0.5, 0.9]  # Fast, medium, slow
    memory_dim: 128
  
  # Surprise-gated learning
  surprise_threshold: 0.01
  
# Training settings
training:
  batch_size: 32
  learning_rate: 1e-4
  epochs: 100
  early_stopping_patience: 10
  
  # Continual learning
  continual:
    enabled: true
    task_duration: "7d"  # New task every 7 days
    replay_buffer_size: 1000
    
# Logging
logging:
  mlflow_tracking_uri: "mlflow/"
  log_interval: 100
```

## Experiments

### Experiment 1: Continual Prediction with HOPE

Adapt HOPE to forecast PM2.5 using sequential data streams. Train on historical data, then simulate continual updates.

```bash
python scripts/train.py \
  --config config/experiments/hope_full_cms.yaml \
  --city Delhi \
  --continual \
  --tasks 12  # 12 weekly tasks
```

### Experiment 2: Multi-Frequency Memory Ablation

Compare CMS configurations to understand optimal frequency settings.

```bash
# Single frequency (no CMS)
python scripts/train.py --config config/experiments/ablation_single_freq.yaml

# Three frequencies (default)
python scripts/train.py --config config/experiments/hope_full_cms.yaml

# Five frequencies
python scripts/train.py --config config/experiments/ablation_five_freq.yaml
```

### Experiment 3: Cross-City Transfer

Train on one city, adapt to another to measure transfer learning.

```bash
python scripts/train.py \
  --config config/experiments/transfer.yaml \
  --source-city Delhi \
  --target-city Beijing
```

## Evaluation Metrics

| Metric | Description |
|--------|-------------|
| MAE | Mean Absolute Error on test set |
| RMSE | Root Mean Squared Error |
| MAPE | Mean Absolute Percentage Error |
| BWT | Backward Transfer (forgetting measure) |
| FWT | Forward Transfer (generalization) |
| ACC | Average Continual Accuracy |

## References

- **Nested Learning Paper**: Behrouz, A., et al. "Nested Learning: The Illusion of Deep Learning Architectures." NeurIPS 2025.
- **HOPE Architecture**: Google Research Blog, November 2025.
- **OpenAQ Documentation**: https://docs.openaq.org

## License

MIT License - See [LICENSE](LICENSE) for details.

## Contributing

Contributions welcome! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## Citation

If you use this code in your research, please cite:

```bibtex
@software{nested_air_pollution,
  title={Nested Learning for Continual Air Quality Prediction},
  author={Your Name},
  year={2025},
  url={https://github.com/yourusername/nested_air_pollution}
}
```
