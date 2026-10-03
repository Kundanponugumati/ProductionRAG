"""Qdrant vector store: one point per chunk, full chunk metadata in the payload."""

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

from prag.models import Chunk, SearchResult


class VectorStore:
    def __init__(self, url: str, collection: str) -> None:
        self.client = QdrantClient(url=url)
        self.collection = collection

    def ensure_collection(self, dim: int) -> None:
        if not self.client.collection_exists(self.collection):
            self.client.create_collection(
                self.collection,
                vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
            )
            return
        # Fail loudly if the collection was built with a different embedding model.
        existing = self.client.get_collection(self.collection).config.params.vectors.size
        if existing != dim:
            raise ValueError(
                f"Collection '{self.collection}' has dim {existing}, embedder has {dim}. "
                "Use a new collection name or delete the old one."
            )

    def upsert(self, chunks: list[Chunk], vectors: list[list[float]]) -> None:
        points = [
            PointStruct(id=chunk.id, vector=vector, payload=chunk.model_dump())
            for chunk, vector in zip(chunks, vectors, strict=True)
        ]
        self.client.upsert(self.collection, points=points, wait=True)

    def search(self, query_vector: list[float], k: int = 5) -> list[SearchResult]:
        response = self.client.query_points(
            self.collection, query=query_vector, limit=k, with_payload=True
        )
        return [
            SearchResult(chunk=Chunk.model_validate(point.payload), score=point.score)
            for point in response.points
        ]

    def count(self) -> int:
        if not self.client.collection_exists(self.collection):
            return 0
        return self.client.count(self.collection, exact=True).count

    def delete_collection(self) -> None:
        self.client.delete_collection(self.collection)
