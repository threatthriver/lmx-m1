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
        batch_size: int = 4,
        grad_clip: float = 1.0,
        save_every: int = 100
    ):
        self.model = model
        self.optimizer = optimizer
        self.max_steps = max_steps
        self.batch_size = batch_size
        self.grad_clip = grad_clip
        self.save_every = save_every

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
            shift_labels.reshape(-1)
        )
        return loss

    def train_step(self, batch: Dict[str, mx.array]):
        loss, grads = nn.value_and_grad(self.model, self.compute_loss)(batch)
        
        # Clip gradients
        if self.grad_clip > 0.0:
            grads = optim.clip_by_norm(grads, self.grad_clip)
        
        # Update parameters
        self.optimizer.update(self.model, grads)
        return loss

    def train(self, train_data: str, valid_data: Optional[str] = None):
        # Load training data
        train_examples = load_jsonl(train_data)
        if valid_data:
            valid_examples = load_jsonl(valid_data)
        
        # Training loop
        step = 0
        train_losses = []
        
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
                
                # Training step
                loss = self.train_step(batch)
                train_losses.append(float(loss))
                
                # Update progress
                step += 1
                pbar.update(1)
                pbar.set_description(f'Loss: {loss:.4f}')
                
                # Save checkpoint
                if step % self.save_every == 0:
                    mx.savez(
                        f'checkpoint_{step}.npz',
                        **self.model.parameters()
                    )
                    
                    # Validate if validation data is provided
                    if valid_data:
                        self.model.eval()
                        valid_loss = self.evaluate(valid_examples)
                        print(f'\nValidation Loss: {valid_loss:.4f}')
                        self.model.train()

    def evaluate(self, eval_examples: list) -> float:
        total_loss = 0.0
        num_batches = 0
        
        for i in range(0, len(eval_examples), self.batch_size):
            batch = {
                'input_ids': mx.array([
                    ex['input_ids'] for ex in \
                    eval_examples[i:i + self.batch_size]
                ])
            }
            loss = self.compute_loss(batch)
            total_loss += float(loss)
            num_batches += 1
        
        return total_loss / num_batches

def main():
    # Initialize model and training
    config = ModelConfig()
    model = LongContextTransformer(config)
    
    # Initialize optimizer
    optimizer = optim.Adam(learning_rate=1e-4)
    
    # Create trainer
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        max_steps=10000,
        batch_size=4,
        grad_clip=1.0,
        save_every=100
    )
    
    # Start training
    trainer.train(
        train_data='train.jsonl',
        valid_data='valid.jsonl'
    )

if __name__ == '__main__':
    main()