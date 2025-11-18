import os
import tempfile
from typing import List, Optional

try:
    import sentencepiece as spm
except ImportError:
    spm = None


class BalSentencePieceTokenizer:
    """
    SentencePiece tokenizer wrapper for Balochi text.
    """

    def __init__(self, model_prefix: str = "balochi_sp"):
        self.model_prefix = model_prefix
        self.sp_model = None

    def train(
        self, texts: List[str], vocab_size: int = 10000, save_dir: Optional[str] = None
    ):
        """Train SentencePiece model."""
        if spm is None:
            raise ImportError(
                "sentencepiece is required. Install with: pip install sentencepiece"
            )

        # Save texts to temporary file
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            for text in texts:
                f.write(text + "\n")
            temp_file = f.name

        try:
            # Train model
            spm.SentencePieceTrainer.train(
                input=temp_file,
                model_prefix=self.model_prefix,
                vocab_size=vocab_size,
                character_coverage=0.9995,
                model_type="bpe",
                pad_id=0,
                unk_id=1,
                bos_id=2,
                eos_id=3,
                pad_piece="<pad>",
                unk_piece="<unk>",
                bos_piece="<s>",
                eos_piece="</s>",
            )

            # Load model
            self.sp_model = spm.SentencePieceProcessor()
            self.sp_model.load(f"{self.model_prefix}.model")

            # Save to directory if specified
            if save_dir:
                self.save(save_dir)

        finally:
            os.unlink(temp_file)

    def encode(self, text: str) -> List[int]:
        """Encode text to token IDs."""
        if self.sp_model is None:
            raise ValueError("Tokenizer not trained. Call train() first.")
        return self.sp_model.encode_as_ids(text)

    def decode(self, token_ids: List[int]) -> str:
        """Decode token IDs to text."""
        if self.sp_model is None:
            raise ValueError("Tokenizer not trained. Call train() first.")
        return self.sp_model.decode_ids(token_ids)

    def save(self, save_dir: str):
        """Save tokenizer files."""
        os.makedirs(save_dir, exist_ok=True)

        # Copy model files to save directory
        import shutil

        for ext in [".model", ".vocab"]:
            src = f"{self.model_prefix}{ext}"
            dst = os.path.join(save_dir, f"sp_model{ext}")
            if os.path.exists(src):
                shutil.copy2(src, dst)

    def load(self, save_dir: str):
        """Load tokenizer from files."""
        if spm is None:
            raise ImportError(
                "sentencepiece is required. Install with: pip install sentencepiece"
            )

        model_path = os.path.join(save_dir, "sp_model.model")
        self.sp_model = spm.SentencePieceProcessor()
        self.sp_model.load(model_path)
