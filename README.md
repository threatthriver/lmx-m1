# Long Context Transformer

A PyMLX implementation of a transformer model with sliding window attention mechanism for handling extended context lengths efficiently.

## Features

- **Sliding Window Attention**: Implements an efficient attention mechanism that processes text in windows, enabling handling of longer sequences while maintaining computational efficiency.
- **Extended Context Length**: Supports sequences up to 32K tokens.
- **MLX Framework**: Built using Apple's MLX framework for optimal performance on Apple Silicon.
- **Modular Architecture**: Clean, modular implementation of transformer components for easy customization.

## Model Architecture

The model consists of several key components:

1. **SlidingWindowAttention**: Processes attention in fixed-size windows to reduce memory usage and computation time.
2. **TransformerBlock**: Standard transformer block with attention and feed-forward layers.
3. **LongContextTransformer**: Main model class combining embeddings, transformer blocks, and output projection.

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### Basic Example

```python
from model import ModelConfig, LongContextTransformer
import mlx.core as mx

# Initialize model configuration
config = ModelConfig(
    vocab_size=32000,
    hidden_size=4096,
    num_attention_heads=32,
    num_hidden_layers=32
)

# Create model instance
model = LongContextTransformer(config)

# Prepare input
input_ids = mx.random.randint(0, config.vocab_size, (1, 1024))  # Batch size 1, sequence length 1024

# Generate output
logits = model(input_ids)
```

### Configuration

The model can be customized through the `ModelConfig` class:

```python
config = ModelConfig(
    vocab_size=32000,          # Vocabulary size
    hidden_size=4096,          # Hidden layer dimension
    num_attention_heads=32,    # Number of attention heads
    num_hidden_layers=32,      # Number of transformer layers
    intermediate_size=11008,   # FFN intermediate layer size
    max_position_embeddings=32768,  # Maximum sequence length
    attention_window=1024      # Sliding window size
)
```

## Testing

Run the test suite:

```bash
python test_model.py
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- MLX framework by Apple
- Transformer architecture based on the "Attention Is All You Need" paper
- Sliding window attention mechanism inspired by Longformer