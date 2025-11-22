import sys
import os
import jax
import jax.numpy as jnp
import optax
import equinox as eqx
import numpy as np
from pathlib import Path

# --- PATH SETUP ---
current_path = Path(__file__).resolve().parent.parent
sys.path.append(str(current_path))

# 1. Import the Correct Model Class (Physics Version)
from balnlp.modeling.fractal_net import BalochiPhysicsTransformer

# 2. Import your Config
from balnlp.modeling.config import model_config, train_config

def get_batch(data, batch_size, seq_len):
    """
    Randomly selects a chunk of text for training.
    """
    # Ensure we don't go out of bounds
    max_idx = len(data) - seq_len - 1
    ix = np.random.randint(0, max_idx, batch_size)

    # Stack arrays
    x = np.stack([data[i: i + seq_len] for i in ix])
    y = np.stack([data[i + 1: i + seq_len + 1] for i in ix])
    return jnp.array(x), jnp.array(y)

def main():
    # --- PATHS ---
    DATA_PATH = current_path / "data" / "balochi_training_data.npy"
    MODEL_SAVE = current_path / "models" / "balochi_physics.eqx"

    if not DATA_PATH.exists():
        print(f"❌ Data not found at {DATA_PATH}")
        print("   Run 'scripts/tokenize_data.py' first.")
        return

    # Load Data
    raw_data = np.load(DATA_PATH, mmap_mode='r')
    print(f"🚀 Starting Physics Training on {len(raw_data):,} tokens...")
    print(f"⚙️  Config: Vocab={model_config.vocab_size}, Dims={model_config.embed_dim}, Depth={model_config.fractal_iterations}")

    # --- 1. SETUP LEARNING RATE SCHEDULER ---
    # We define this BEFORE the optimizer so we can use it.
    total_steps = train_config.total_steps

    scheduler = optax.warmup_cosine_decay_schedule(
        init_value=0.0,           # Start at 0 (Prevents Fractal Explosion)
        peak_value=train_config.learning_rate, # Go up to 3e-4
        warmup_steps=100,         # Warmup for 100 steps
        decay_steps=total_steps,  # Fade out over time
        end_value=1e-6            # End near zero
    )

    # --- 2. INITIALIZE MODEL ---
    key = jax.random.PRNGKey(train_config.seed)

    model = BalochiPhysicsTransformer(
        vocab_size=model_config.vocab_size,
        dim=model_config.embed_dim,
        key=key
    )

    # --- 3. INITIALIZE OPTIMIZER (With Weight Decay) ---
    # weight_decay=1e-2 fights Overfitting
    optimizer = optax.adamw(learning_rate=scheduler, weight_decay=1e-2)
    opt_state = optimizer.init(eqx.filter(model, eqx.is_array))

    # --- LOSS FUNCTION ---
    @eqx.filter_value_and_grad
    def compute_loss(model, x, y):
        # Run the Physics Model
        # Output shape: [Batch, Seq_Len, Vocab]
        logits = jax.vmap(model)(x)

        # Calculate Error
        loss = optax.softmax_cross_entropy_with_integer_labels(
            logits.reshape(-1, model_config.vocab_size),
            y.reshape(-1)
        )
        return jnp.mean(loss)

    # --- OPTIMIZATION STEP (Compiled on GPU/CPU) ---
    @eqx.filter_jit
    def make_step(model, opt_state, x, y):
        loss, grads = compute_loss(model, x, y)
        updates, opt_state = optimizer.update(grads, opt_state, model)
        model = eqx.apply_updates(model, updates)
        return loss, model, opt_state

    # --- TRAINING LOOP ---
    print(">>> Entering Quantum-Fractal Simulation Loop...")

    for step in range(total_steps):
        # Get Batch
        xb, yb = get_batch(raw_data, train_config.batch_size, train_config.seq_len)

        # Train
        loss, model, opt_state = make_step(model, opt_state, xb, yb)

        if step % train_config.log_interval == 0:
            # Get current Learning Rate for logging
            current_lr = scheduler(step)
            print(f"Step {step} | Energy Loss: {loss.item():.4f} | LR: {current_lr:.6f}")

    # --- SAVE ---
    # Ensure folder exists
    os.makedirs(MODEL_SAVE.parent, exist_ok=True)
    eqx.tree_serialise_leaves(str(MODEL_SAVE), model)
    print(f"✅ Physics Model Saved to: {MODEL_SAVE}")

if __name__ == "__main__":
    main()