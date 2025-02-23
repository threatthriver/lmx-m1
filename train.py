import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from model import ModelConfig, LongContextTransformer
from typing import Optional, Dict, Any
import json
from tqdm import tqdm
import numpy as np

def load_jsonl(file_path: str) -> list:
    with open(file_path, 'r') as f:
        return [json.loads(line) for line in f]

class Trainer:
    def __init__(
        self,
        model: LongContextTransformer,
        optimizer: optim.Optimizer,
        max_steps: int = 1000,
        batch_size: int = 1,  # Reduced batch size for memory efficiency
        grad_accum_steps: int = 4,  # Gradient accumulation steps
        grad_clip: float = 1.0,
        save_every: int = 100,
        memory_limit_gb: float = 1.0  # Memory limit in GB
    ):
        self.model = model
        self.optimizer = optimizer
        self.max_steps = max_steps
        self.batch_size = batch_size
        self.grad_accum_steps = grad_accum_steps
        self.grad_clip = grad_clip
        self.save_every = save_every
        self.memory_limit_bytes = int(memory_limit_gb * 1024 * 1024 * 1024)  # Convert GB to bytes

    def compute_loss(self, batch: Dict[str, mx.array]) -> mx.array:
        logits = self.model(
            batch['input_ids'],
            position_ids=batch.get('position_ids'),
            attention_mask=batch.get('attention_mask')
        )
        
        # Shift logits and labels for next token prediction
        shift_logits = logits[:, :-1, :]
        shift_labels = batch['input_ids'][:, 1:]
        
        # Compute cross entropy loss
        loss = nn.losses.cross_entropy(
            shift_logits.reshape(-1, shift_logits.shape[-1]),
            shift_labels.reshape(-1),
            reduction='mean'
        )
        return loss

    def train_step(self, batch: Dict[str, mx.array], is_accumulating: bool = False):
        # Forward and backward pass
        loss, grads = nn.value_and_grad(self.model, self.compute_loss)(batch)
        
        # Scale loss and gradients for gradient accumulation
        loss = loss / self.grad_accum_steps
        grads = {k: v / self.grad_accum_steps for k, v in grads.items()}
        
        # Accumulate gradients
        if not hasattr(self, '_accumulated_grads'):
            self._accumulated_grads = grads
        else:
            self._accumulated_grads = {k: self._accumulated_grads[k] + grads[k] 
                                      for k in grads.keys()}
        
        # Only update weights after accumulating enough gradients
        if not is_accumulating:
            # Clip gradients
            if self.grad_clip > 0.0:
                self._accumulated_grads = optim.clip_grad_norm(self._accumulated_grads, self.grad_clip)
            
            # Update parameters
            self.optimizer.update(self.model, self._accumulated_grads)
            self._accumulated_grads = None
        
        return loss

    def train(self, train_data: str, valid_data: Optional[str] = None):
        # Load training data
        train_examples = load_jsonl(train_data)
        if valid_data:
            valid_examples = load_jsonl(valid_data)
        
        # Training loop
        step = 0
        train_losses = []
        accum_step = 0
        
        with tqdm(total=self.max_steps) as pbar:
            while step < self.max_steps:
                # Sample batch
                batch_indices = np.random.choice(
                    len(train_examples),
                    size=self.batch_size
                )
                batch = {
                    'input_ids': mx.array([
                        train_examples[i]['input_ids'] for i in batch_indices
                    ])
                }
                
                # Check if we're still accumulating gradients
                is_accumulating = (accum_step + 1) % self.grad_accum_steps != 0
                
                # Training step with gradient accumulation
                loss = self.train_step(batch, is_accumulating)
                train_losses.append(float(loss) * self.grad_accum_steps)  # Scale loss back
                
                # Update progress
                accum_step += 1
                if not is_accumulating:
                    step += 1
                    pbar.update(1)
                    pbar.set_description(f'Loss: {np.mean(train_losses[-self.grad_accum_steps:]):.4f}')
                
                # Save checkpoint
                if step > 0 and step % self.save_every == 0:
                    mx.savez(
                        f'checkpoint_{step}.npz',
                        **self.model.parameters()
                    )
                    
                    # Validate if validation data is provided
                    if valid_data:
                        valid_losses = []
                        for i in range(0, len(valid_examples), self.batch_size):
                            # Prepare validation batch
                            batch_end = min(i + self.batch_size, len(valid_examples))
                            valid_batch = {
                                'input_ids': mx.array([
                                    valid_examples[j]['input_ids'] 
                                    for j in range(i, batch_end)
                                ])
                            }
                            
                            # Compute validation loss
                            with mx.stop_gradient():
                                valid_loss = self.compute_loss(valid_batch)
                            valid_losses.append(float(valid_loss))
                        
                        # Calculate metrics
                        avg_valid_loss = np.mean(valid_losses)
                        perplexity = np.exp(avg_valid_loss)
                        print(f'\nValidation Loss: {avg_valid_loss:.4f}')
                        print(f'Perplexity: {perplexity:.4f}\n')

def main():
    # Initialize model configuration with reduced size
    config = ModelConfig(
        vocab_size=32000,
        hidden_size=1024,  # Reduced from 4096
        num_attention_heads=16,  # Reduced from 32
        num_hidden_layers=12,  # Reduced from 32
        intermediate_size=2048  # Reduced from 11008
    )
    
    # Create model and optimizer
    model = LongContextTransformer(config)
    optimizer = optim.Adam(learning_rate=1e-4)
    
    # Initialize optimizer state with model parameters
    mx.eval(model.parameters())
    
    # Initialize trainer with memory-efficient settings
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        max_steps=10000,
        batch_size=1,  # Reduced batch size
        grad_accum_steps=8,  # Accumulate gradients over 8 steps
        grad_clip=1.0,
        save_every=500,
        memory_limit_gb=1.0
    )
    
    # Start training
    trainer.train(
        train_data='train.jsonl',
        valid_data='valid.jsonl'
    )

if __name__ == '__main__':
    main()