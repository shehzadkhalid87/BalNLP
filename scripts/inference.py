import sys
import os
import jax
import jax.numpy as jnp
import equinox as eqx
import numpy as np
from pathlib import Path

# Setup Path
current_path = Path(__file__).resolve().parent.parent
sys.path.append(str(current_path))

from balnlp.modeling.fractal_net import BalochiTransformer
from balnlp.bal_tokenizer.sentencepiece_tokenizer import BalSentencePieceTokenizer
from balnlp.modeling.config import model_config


class BalochiGenerator:
    def __init__(self):
        print(">>> Loading Real-Time Engine...")

        # 1. Paths
        self.model_path = current_path / "models" / "balochi_physics.eqx"
        self.tokenizer_path = current_path / "models" / "tokenizer" / "balochi_bpe.model"

        # 2. Load Tokenizer
        self.tokenizer = BalSentencePieceTokenizer(str(self.tokenizer_path))

        # 3. Load Model Structure
        # We need a dummy key just to initialize the shape
        key = jax.random.PRNGKey(0)
        self.model = BalochiPhysicsTransformer(
            model_config.vocab_size,
            model_config.embed_dim,
            key
        )

        # 4. Load Trained Weights
        # This puts your trained "brain" into the structure
        self.model = eqx.tree_deserialise_leaves(str(self.model_path), self.model)
        print("✅ Model Loaded Successfully.")

    def generate(self, start_text, max_new_tokens=20, temperature=0.7):
        """
        Real-Time Generation Loop
        """
        # Convert Text -> Numbers
        input_ids = self.tokenizer.encode(start_text)

        # JAX requires fixed shapes, but for inference we loop dynamically
        # This is a simple autoregressive loop
        for _ in range(max_new_tokens):
            # Prepare input for JAX (Add batch dimension)
            # We take the last 64 tokens to fit context window
            ctx = input_ids[-64:]
            # Pad if too short (JAX is strict about shapes)
            if len(ctx) < 64:
                ctx = [0] * (64 - len(ctx)) + ctx

            x_input = jnp.array([ctx])  # Batch size 1

            # Run Model
            logits = jax.vmap(self.model)(x_input)

            # Get prediction for the LAST token in sequence
            last_token_logits = logits[0, -1, :]

            # SAMPLING STRATEGY (Not just Argmax)
            # Apply Temperature
            scaled_logits = last_token_logits / temperature

            # Softmax to get probabilities
            probs = jax.nn.softmax(scaled_logits)

            # Convert to Numpy for random choice
            probs_np = np.array(probs)

            # Sample next token based on probability
            next_token = np.random.choice(len(probs_np), p=probs_np)

            # Append to sequence
            input_ids.append(int(next_token))

            # Stop if EOS (End of Sentence) generated
            if next_token == self.tokenizer.eos_id:
                break

        # Decode Numbers -> Text
        return self.tokenizer.decode(input_ids)


if __name__ == "__main__":
    # TEST RUN
    engine = BalochiGenerator()

    print("\n--- PROTOTYPE TEST ---")
    prompt = "بلوچی زبان"
    print(f"Input: {prompt}")

    output = engine.generate(prompt, max_new_tokens=15, temperature=0.8)
    print(f"Output: {output}")