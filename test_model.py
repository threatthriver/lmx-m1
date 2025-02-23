import mlx.core as mx
from model import ModelConfig, LongContextTransformer

def test_model():
    # Initialize model configuration
    config = ModelConfig(
        vocab_size=32000,
        hidden_size=512,  # Smaller for testing
        num_attention_heads=8,
        num_hidden_layers=4,
        intermediate_size=2048,
        max_position_embeddings=2048,
        attention_window=256
    )
    
    # Create model instance
    model = LongContextTransformer(config)
    
    # Create sample input data
    batch_size = 2
    seq_length = 512
    input_ids = mx.random.randint(0, config.vocab_size, (batch_size, seq_length))
    
    # Optional inputs
    position_ids = mx.tile(mx.arange(seq_length)[None, :], (batch_size, 1))
    attention_mask = mx.ones((batch_size, 1, seq_length, seq_length))
    
    # Forward pass
    logits = model(input_ids, position_ids, attention_mask)
    
    # Check output shape
    expected_shape = (batch_size, seq_length, config.vocab_size)
    actual_shape = logits.shape
    
    print(f"Test Results:")
    print(f"Input shape: {input_ids.shape}")
    print(f"Expected output shape: {expected_shape}")
    print(f"Actual output shape: {actual_shape}")
    print(f"Shape test: {'Passed' if actual_shape == expected_shape else 'Failed'}")
    print(f"Output contains NaN: {'Yes' if mx.isnan(logits).any() else 'No'}")

if __name__ == '__main__':
    test_model()