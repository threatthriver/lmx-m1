import mlx.core as mx
import mlx.nn as nn
from dataclasses import dataclass
from typing import Optional, Tuple
from einops import rearrange

@dataclass
class ModelConfig:
    vocab_size: int = 32000
    hidden_size: int = 4096
    num_attention_heads: int = 32
    num_hidden_layers: int = 32
    intermediate_size: int = 11008
    max_position_embeddings: int = 32768  # Extended context length
    attention_window: int = 1024  # Sliding window size
    layer_norm_eps: float = 1e-5
    initializer_range: float = 0.02
    use_cache: bool = True

class SlidingWindowAttention(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.num_attention_heads = config.num_attention_heads
        self.hidden_size = config.hidden_size
        self.head_dim = config.hidden_size // config.num_attention_heads
        self.window_size = config.attention_window
        
        self.q_proj = nn.Linear(config.hidden_size, config.hidden_size)
        self.k_proj = nn.Linear(config.hidden_size, config.hidden_size)
        self.v_proj = nn.Linear(config.hidden_size, config.hidden_size)
        self.o_proj = nn.Linear(config.hidden_size, config.hidden_size)
        
    def __call__(self, x: mx.array, attention_mask: Optional[mx.array] = None) -> mx.array:
        batch_size, seq_length, _ = x.shape
        
        # Project queries, keys, and values
        q = rearrange(self.q_proj(x), 'b n (h d) -> b h n d', h=self.num_attention_heads)
        k = rearrange(self.k_proj(x), 'b n (h d) -> b h n d', h=self.num_attention_heads)
        v = rearrange(self.v_proj(x), 'b n (h d) -> b h n d', h=self.num_attention_heads)
        
        # Compute attention scores with sliding window
        scores = []
        for i in range(0, seq_length, self.window_size):
            end_idx = min(i + self.window_size, seq_length)
            window_q = q[:, :, i:end_idx, :]
            window_k = k[:, :, max(0, i - self.window_size):end_idx + self.window_size, :]
            window_scores = mx.matmul(window_q, window_k.transpose(0, 1, 3, 2))
            window_scores = window_scores / mx.sqrt(self.head_dim)
            
            if attention_mask is not None:
                window_mask = attention_mask[:, :, i:end_idx, max(0, i - self.window_size):end_idx + self.window_size]
                window_scores = window_scores + window_mask
            
            scores.append(window_scores)
        
        attention = mx.softmax(mx.concatenate(scores, axis=2), axis=-1)
        
        # Apply attention to values
        context = mx.matmul(attention, v)
        context = rearrange(context, 'b h n d -> b n (h d)')
        
        return self.o_proj(context)

class TransformerBlock(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.attention = SlidingWindowAttention(config)
        self.mlp = nn.Sequential(
            nn.Linear(config.hidden_size, config.intermediate_size),
            nn.GELU(),
            nn.Linear(config.intermediate_size, config.hidden_size)
        )
        self.ln1 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.ln2 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        
    def __call__(self, x: mx.array, attention_mask: Optional[mx.array] = None) -> mx.array:
        # Self-attention with residual connection
        h = x + self.attention(self.ln1(x), attention_mask)
        # FFN with residual connection
        out = h + self.mlp(self.ln2(h))
        return out

class LongContextTransformer(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        
        self.embeddings = nn.Embedding(config.vocab_size, config.hidden_size)
        self.position_embeddings = nn.Embedding(
            config.max_position_embeddings, config.hidden_size
        )
        
        self.layers = [TransformerBlock(config) for _ in range(config.num_hidden_layers)]
        self.ln_f = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        
        # Output projection
        self.output = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        
    def __call__(
        self,
        input_ids: mx.array,
        position_ids: Optional[mx.array] = None,
        attention_mask: Optional[mx.array] = None
    ) -> mx.array:
        # Get input embeddings
        hidden_states = self.embeddings(input_ids)
        
        # Add positional embeddings
        if position_ids is None:
            position_ids = mx.arange(input_ids.shape[1])[None, :]
        hidden_states = hidden_states + self.position_embeddings(position_ids)
        
        # Apply transformer layers
        for layer in self.layers:
            hidden_states = layer(hidden_states, attention_mask)
        
        # Apply final layer norm
        hidden_states = self.ln_f(hidden_states)
        
        # Project to vocabulary
        logits = self.output(hidden_states)
        
        return logits