"""FinBERT encoder for financial news articles (768-dim vectors)."""

from __future__ import annotations

import logging
from typing import Any

LOGGER = logging.getLogger(__name__)

try:
    import torch
except ModuleNotFoundError:  # pragma: no cover
    torch = None  # type: ignore[assignment]

try:
    from transformers import AutoTokenizer, AutoModel
except ModuleNotFoundError:  # pragma: no cover
    AutoTokenizer = None  # type: ignore[assignment]
    AutoModel = None  # type: ignore[assignment]

try:
    from tqdm import tqdm
except ModuleNotFoundError:  # pragma: no cover
    def tqdm(iterable, **kwargs):  # type: ignore[misc]
        """Fallback if tqdm not available."""
        return iterable


class FinBertEncoder:
    """Convert news text into 768-dimensional FinBERT embeddings (finance-optimized)."""

    def __init__(
        self,
        model_name: str = "ProsusAI/finbert",
        device: str = "cuda",
        batch_size: int = 32,
        embedding_dim: int = 768,
    ) -> None:
        self.model_name = model_name
        self.embedding_dim = embedding_dim
        self.device = self._resolve_device(device)
        self.batch_size = max(1, int(batch_size))

        if AutoTokenizer is None or AutoModel is None:
            raise RuntimeError(
                "transformers library is not installed. Install it to use FinBertEncoder."
            )

        try:
            self.tokenizer: Any = AutoTokenizer.from_pretrained(model_name)
            self.model: Any = AutoModel.from_pretrained(model_name)
            self.model = self.model.to(self.device)
            self.model.eval()
            LOGGER.info("Loaded FinBERT model=%s on device=%s", model_name, self.device)
        except Exception as exc:
            LOGGER.error("Failed to load FinBERT model=%s: %s", model_name, exc)
            raise

    def _zero_vector(self) -> list[float]:
        """Return a zero vector of the expected dimension."""
        return [0.0] * self.embedding_dim

    @staticmethod
    def _resolve_device(device: str) -> str:
        """Fallback to CPU if requested device cannot be used."""
        normalized = (device or "cpu").strip().lower()
        if normalized == "cuda":
            if torch is not None and hasattr(torch, "cuda") and torch.cuda.is_available():
                return "cuda"
            LOGGER.warning("CUDA requested but unavailable. Falling back to CPU.")
            return "cpu"
        return "cpu"

    def encode(self, text: str) -> list[float]:
        """Encode a single text into a 768-dim FinBERT vector.
        
        Returns:
            A list of floats representing the embedding.
        """
        if not text or not text.strip():
            LOGGER.warning("encode() received empty text; returning zero vector.")
            return self._zero_vector()

        try:
            # Tokenize and truncate to 512 tokens (FinBERT max)
            inputs = self.tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=512,
            ).to(self.device)

            # Forward pass
            with torch.no_grad():
                outputs = self.model(**inputs)

            # Extract [CLS] token embedding (first token of last hidden state)
            embedding = outputs.last_hidden_state[:, 0, :].detach().cpu()

            if torch is not None and isinstance(embedding, torch.Tensor):
                vector = embedding.squeeze(0).tolist()
            else:
                vector = list(embedding.squeeze(0))

            if len(vector) != self.embedding_dim:
                LOGGER.warning(
                    "Embedding dim mismatch (expected=%d got=%d); normalizing length.",
                    self.embedding_dim,
                    len(vector),
                )
                return self._normalize_dim(vector)

            return vector
        except Exception as exc:
            LOGGER.error("Embedding failed: %s", exc)
            return self._zero_vector()

    def encode_batch(
        self,
        texts: list[str],
        batch_size: int | None = None,
    ) -> list[list[float]]:
        """Encode a batch of texts into 768-dim vectors.
        
        Args:
            texts: List of text strings to encode
            batch_size: Override the default batch size for this call
            
        Returns:
            A list of embeddings (each is a list of floats), same length as input.
        """
        if not texts:
            return []

        effective_batch_size = max(1, int(batch_size or self.batch_size))
        vectors: list[list[float]] = [self._zero_vector() for _ in texts]

        # Filter empty texts but track their indices
        non_empty: list[tuple[int, str]] = [
            (idx, value.strip())
            for idx, value in enumerate(texts)
            if value and value.strip()
        ]

        if not non_empty:
            LOGGER.warning("encode_batch() received only empty texts; returning zero vectors.")
            return vectors

        indices = [idx for idx, _ in non_empty]
        payload = [value for _, value in non_empty]

        try:
            num_batches = (len(payload) + effective_batch_size - 1) // effective_batch_size
            
            for batch_idx in tqdm(range(num_batches), desc="Encoding batches"):
                start_idx = batch_idx * effective_batch_size
                end_idx = min(start_idx + effective_batch_size, len(payload))
                batch_texts = payload[start_idx:end_idx]

                # Tokenize batch (with padding)
                inputs = self.tokenizer(
                    batch_texts,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=512,
                ).to(self.device)

                # Forward pass
                with torch.no_grad():
                    outputs = self.model(**inputs)

                # Extract [CLS] token embeddings (first token of each sequence)
                embeddings = outputs.last_hidden_state[:, 0, :].detach().cpu()

                if torch is not None and isinstance(embeddings, torch.Tensor):
                    matrix = embeddings.tolist()
                else:
                    matrix = embeddings.tolist() if hasattr(embeddings, "tolist") else list(embeddings)

                # Map back to original indices
                for orig_idx, (idx, row) in enumerate(
                    zip(range(start_idx, end_idx), matrix, strict=False)
                ):
                    vector = list(row)
                    orig_position = indices[idx]
                    vectors[orig_position] = self._normalize_dim(vector)

            LOGGER.info("Embedded %d/%d texts.", len(non_empty), len(texts))
            return vectors
        except Exception as exc:
            LOGGER.error("Batch embedding failed: %s", exc)
            return vectors

    def _normalize_dim(self, vector: list[float]) -> list[float]:
        """Pad/truncate vectors to expected dimension."""
        if len(vector) == self.embedding_dim:
            return vector
        if len(vector) > self.embedding_dim:
            return vector[: self.embedding_dim]
        return vector + [0.0] * (self.embedding_dim - len(vector))
