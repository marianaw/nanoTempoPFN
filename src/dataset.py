import jax
import jax.numpy as jnp
import numpy as np


# Dataset: generate long sequences and chunk them with sliding window
class ChunkedDataset:
    """Generate long sequences and yield chunks via sliding window."""
    
    def __init__(self, gen_fn, n_seqs=100, horizon=200, chunk_size=32, stride=1):
        self.gen_fn = gen_fn
        self.chunk_size = chunk_size
        self.stride = stride
        self.refresh(n_seqs, horizon)
    
    def refresh(self, n_seqs=100, horizon=200):
        """Generate new sequences and extract chunks."""
        seqs, _, _ = self.gen_fn(n_samples=n_seqs, horizon=horizon)
        self.all_chunks_x = []
        self.all_chunks_y = []
        self.all_chunks_mask = []
        
        for i in range(len(seqs)):
            seq_x = seqs[i, :-1, :1]      # (L, 1) - values
            seq_y = seqs[i, 1:, :1]      # (L, 1) - values
            seq_mask = seqs[i, :-1, -1]   # (L,) - validity mask
            valid_len = int(seq_mask.sum())
            
            # Extract chunks using sliding window (only from valid region)
            for start in range(0, max(1, valid_len - self.chunk_size + 1), self.stride):
                end = start + self.chunk_size
                if end <= valid_len:
                    self.all_chunks_x.append(seq_x[start:end])
                    self.all_chunks_y.append(seq_y[start:end])
                    self.all_chunks_mask.append(seq_mask[start:end])
        
        self.all_chunks_x = np.array(self.all_chunks_x)      # (N_chunks, chunk_size, 1)
        self.all_chunks_y = np.array(self.all_chunks_y)      # (N_chunks, chunk_size, 1)
        self.all_chunks_mask = np.array(self.all_chunks_mask) # (N_chunks, chunk_size)
        print(f"Created {len(self.all_chunks_x)} chunks from {n_seqs} sequences")
    
    def sample_batch(self, batch_size, rng_key=None):
        """Sample a random batch of chunks."""
        if rng_key is None:
            idx = np.random.choice(len(self.all_chunks_x), size=batch_size, replace=True)
        else:
            idx = jax.random.choice(rng_key, len(self.all_chunks_x), shape=(batch_size,), replace=True)
        return self.all_chunks_x[idx], self.all_chunks_y[idx], self.all_chunks_mask[idx]
