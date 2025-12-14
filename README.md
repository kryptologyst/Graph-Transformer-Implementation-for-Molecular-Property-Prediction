# Graph Transformer Implementation for Molecular Property Prediction

A production-ready implementation of Graph Transformers for molecular property prediction on the ZINC dataset. This project showcases advanced graph neural network architectures with self-attention mechanisms, positional encodings, and comprehensive evaluation frameworks.

## Features

- **Multiple Graph Transformer Variants**: Basic Graph Transformer, Multi-scale Graph Transformer, and Attention-visualization Graph Transformer
- **Advanced Positional Encodings**: Laplacian, Random Walk, Learnable, and Sinusoidal positional encodings
- **Comprehensive Evaluation**: MAE, RMSE, R², Pearson correlation, and error analysis
- **Interactive Demo**: Streamlit-based web application for molecular property prediction
- **Production Ready**: Type hints, comprehensive documentation, configuration management, and reproducible results
- **Modern Stack**: PyTorch 2.x, PyTorch Geometric, OmegaConf, TensorBoard, and Weights & Biases support

## Quick Start

### Installation

1. Clone the repository:
```bash
git clone https://github.com/kryptologyst/Graph-Transformer-Implementation-for-Molecular-Property-Prediction.git
cd Graph-Transformer-Implementation-for-Molecular-Property-Prediction
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up pre-commit hooks (optional):
```bash
pre-commit install
```

### Training

Train a Graph Transformer model on the ZINC dataset:

```bash
python scripts/train.py --model graph_transformer --num_epochs 50
```

Available model types:
- `graph_transformer`: Basic Graph Transformer with positional encoding
- `multi_scale`: Multi-scale Graph Transformer with different aggregation strategies
- `attention`: Graph Transformer with attention visualization capabilities

### Evaluation

Evaluate a trained model:

```bash
python scripts/train.py --eval_only --checkpoint checkpoints/best_model.pt
```

### Interactive Demo

Launch the Streamlit demo:

```bash
streamlit run demo/streamlit_demo.py
```

## Project Structure

```
0418_Graph_transformers_implementation/
├── src/                          # Source code
│   ├── models/                   # Model implementations
│   │   └── graph_transformer.py  # Graph Transformer variants
│   ├── layers/                   # Custom layers
│   │   └── positional_encoding.py # Positional encoding implementations
│   ├── data/                     # Data handling
│   │   └── dataset.py            # Dataset classes and utilities
│   ├── train/                    # Training utilities
│   │   └── trainer.py           # Training loop and utilities
│   ├── eval/                     # Evaluation
│   │   └── metrics.py            # Evaluation metrics and utilities
│   └── utils/                    # Utilities
│       └── core.py               # Core utilities
├── configs/                      # Configuration files
│   └── default.yaml              # Default configuration
├── scripts/                      # Training and evaluation scripts
│   └── train.py                  # Main training script
├── demo/                         # Interactive demos
│   └── streamlit_demo.py         # Streamlit demo
├── tests/                        # Unit tests
├── data/                         # Data directory
├── checkpoints/                  # Model checkpoints
├── logs/                         # Training logs
├── assets/                       # Generated assets
├── results/                      # Evaluation results
├── requirements.txt              # Python dependencies
├── .gitignore                    # Git ignore file
└── README.md                     # This file
```

## Model Architectures

### Graph Transformer

The basic Graph Transformer implementation includes:

- **Self-attention mechanism**: Captures long-range dependencies in molecular graphs
- **Positional encodings**: Multiple encoding strategies for graph structure
- **Residual connections**: Improves gradient flow and training stability
- **Global pooling**: Aggregates node-level features to graph-level predictions

### Multi-scale Graph Transformer

Extends the basic model with:

- **Multiple scales**: Different positional encodings and pooling strategies
- **Feature fusion**: Combines predictions from multiple scales
- **Enhanced representation**: Better capture of molecular patterns

### Graph Transformer with Attention

Provides interpretability through:

- **Attention visualization**: Extract and visualize attention weights
- **Interpretable predictions**: Understand model decision-making
- **Attention analysis**: Analyze which parts of the molecule are important

## Positional Encodings

### Laplacian Positional Encoding

Uses the Laplacian matrix eigenvalues to encode graph structure:

```python
from src.layers.positional_encoding import LaplacianPositionalEncoding

pos_enc = LaplacianPositionalEncoding(dim=16)
encoding = pos_enc(x, edge_index, batch)
```

### Random Walk Positional Encoding

Encodes graph structure through random walk probabilities:

```python
from src.layers.positional_encoding import RandomWalkPositionalEncoding

pos_enc = RandomWalkPositionalEncoding(dim=16, walk_length=16)
encoding = pos_enc(x, edge_index, batch)
```

### Learnable Positional Encoding

Learns positional encodings during training:

```python
from src.layers.positional_encoding import LearnablePositionalEncoding

pos_enc = LearnablePositionalEncoding(dim=16, max_nodes=1000)
encoding = pos_enc(x, edge_index, batch)
```

### Sinusoidal Positional Encoding

Uses sinusoidal functions for positional encoding:

```python
from src.layers.positional_encoding import SinusoidalPositionalEncoding

pos_enc = SinusoidalPositionalEncoding(dim=16)
encoding = pos_enc(x, edge_index, batch)
```

## Configuration

The project uses YAML configuration files for easy experimentation:

```yaml
# Model configuration
model:
  name: "graph_transformer"
  in_channels: 9
  hidden_channels: 64
  num_layers: 3
  num_heads: 4
  dropout: 0.1
  use_positional_encoding: true
  positional_encoding_type: "laplacian"
  positional_encoding_dim: 16
  use_edge_attr: true
  edge_attr_dim: 3
  pooling: "mean"
  use_residual: true

# Training configuration
training:
  batch_size: 32
  learning_rate: 0.001
  weight_decay: 1e-4
  num_epochs: 100
  patience: 20
  min_delta: 1e-4
  gradient_clip_norm: 1.0
  use_amp: false
```

## Evaluation Metrics

The implementation includes comprehensive evaluation metrics:

- **MAE (Mean Absolute Error)**: Primary regression metric
- **RMSE (Root Mean Square Error)**: Penalizes larger errors more
- **R² (Coefficient of Determination)**: Measures explained variance
- **Pearson Correlation**: Measures linear relationship strength
- **Error Analysis**: Analyzes errors across different target value ranges
- **Residual Analysis**: Statistical analysis of prediction residuals

## Data Pipeline

### ZINC Dataset

The implementation uses the ZINC molecular dataset with:

- **Node features**: 9-dimensional atom features
- **Edge features**: 3-dimensional bond features
- **Target**: Molecular property (regression task)
- **Splits**: Train/validation/test splits
- **Synthetic data**: Optional synthetic data generation for augmentation

### Data Loading

```python
from src.data.dataset import GraphDataModule

data_module = GraphDataModule(
    data_root="./data",
    batch_size=32,
    train_size=10000,
    val_size=1000,
    test_size=1000,
)

data_module.setup()
train_loader = data_module.train_dataloader()
```

## Training

### Basic Training

```python
from src.train.trainer import train_model

results = train_model(
    model=model,
    data_module=data_module,
    config=config,
    device=device,
    use_wandb=True,
    use_tensorboard=True,
)
```

### Advanced Features

- **Early stopping**: Prevents overfitting
- **Gradient clipping**: Stabilizes training
- **Mixed precision**: Optional AMP support
- **Checkpointing**: Saves best and regular checkpoints
- **Logging**: TensorBoard and Weights & Biases integration

## Interactive Demo

The Streamlit demo provides:

- **Molecule generation**: Create random molecular graphs
- **Visualization**: Interactive graph visualization with Plotly
- **Property prediction**: Real-time molecular property prediction
- **Model comparison**: Compare different model variants
- **Batch prediction**: Predict properties for multiple molecules
- **Statistics**: Molecule and prediction statistics

### Running the Demo

```bash
streamlit run demo/streamlit_demo.py
```

## Reproducibility

The implementation ensures reproducibility through:

- **Deterministic seeding**: Random seeds for all libraries
- **Device fallback**: CUDA → MPS → CPU fallback chain
- **Configuration management**: YAML-based configuration
- **Version control**: Git with proper .gitignore
- **Dependency management**: requirements.txt with pinned versions

## Performance Optimization

### Model Optimization

- **Efficient attention**: Optimized self-attention implementation
- **Memory management**: Efficient batch processing
- **GPU utilization**: Optimized for CUDA and MPS devices

### Training Optimization

- **Data loading**: Multi-process data loading with pin_memory
- **Gradient accumulation**: Support for large effective batch sizes
- **Mixed precision**: Optional AMP for faster training

## Testing

Run the test suite:

```bash
pytest tests/
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Run the test suite
6. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- PyTorch Geometric team for the excellent graph neural network library
- ZINC dataset creators for providing molecular data
- Transformer architecture inventors for the revolutionary attention mechanism
- Graph neural network community for continuous research and development

## Troubleshooting

### Common Issues

1. **CUDA out of memory**: Reduce batch size or use gradient accumulation
2. **Import errors**: Ensure all dependencies are installed correctly
3. **Data loading issues**: Check data paths and permissions
4. **Model loading errors**: Verify checkpoint compatibility

### Getting Help

- Check the issues section for common problems
- Create a new issue with detailed error information
- Include system information and error traces

## Future Work

- [ ] Support for more molecular datasets
- [ ] Additional positional encoding strategies
- [ ] Graph generation capabilities
- [ ] Multi-task learning support
- [ ] Distributed training support
- [ ] Model compression and quantization
- [ ] Web-based model serving
- [ ] Integration with molecular databases
- [ ] Real-time molecular property prediction API
- [ ] Advanced visualization tools
# Graph-Transformer-Implementation-for-Molecular-Property-Prediction
