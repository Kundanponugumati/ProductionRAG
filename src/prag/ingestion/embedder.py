"""Local embeddings with sentence-transformers (no API cost)."""

from sentence_transformers import SentenceTransformer


class Embedder:
    def __init__(
        self,
        model: str,
        device: str | None = None,
        batch_size: int = 32,
        query_instruction: str = "",
    ) -> None:
        self.model = SentenceTransformer(model, device=device)
        self.batch_size = batch_size
        self.query_instruction = query_instruction

    @property
    def dim(self) -> int:
        return self.model.get_embedding_dimension()  # pyright: ignore[reportReturnType]

    def embed_documents(self, texts: list[str], show_progress: bool = False) -> list[list[float]]:
        # Normalised vectors → cosine similarity == dot product.
        vectors = self.model.encode(
            texts,
            batch_size=self.batch_size,
            normalize_embeddings=True,
            show_progress_bar=show_progress,
        )
        return vectors.tolist()

    def embed_query(self, text: str) -> list[float]:
        return self.embed_queries([text])[0]

    def embed_queries(self, texts: list[str]) -> list[list[float]]:
        prefixed = [self.query_instruction + t for t in texts]
        vectors = self.model.encode(prefixed, batch_size=self.batch_size, normalize_embeddings=True)
        return vectors.tolist()
