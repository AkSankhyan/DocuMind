import os
from typing import List, Dict, Any, Optional
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams, PointStruct
from fastembed import TextEmbedding
from flashrank import Ranker, RerankRequest

from .schema import DocumentChunk
from . import config


class VectorStoreManager:
    """
    Production-ready Vector Store & Reranking Manager.
    - Persistent local storage via Qdrant
    - Local CPU-optimized dense embeddings via FastEmbed (BGE-small-en)
    - Cross-encoder reranking via FlashRank (TinyBERT)
    """

    def __init__(self):
        # Initialize Qdrant with persistent local on-disk storage
        self.client = QdrantClient(path=config.QDRANT_DIR)
        
        # Initialize FastEmbed embedding model (Lazy loaded or on init)
        print(f"Loading embedding model: {config.EMBEDDING_MODEL}...")
        self.embedding_model = TextEmbedding(model_name=config.EMBEDDING_MODEL)
        
        # Embedding vector size for BAAI/bge-small-en-v1.5 is 384
        self.vector_size = 384
        
        # Initialize FlashRank reranker
        print(f"Loading reranker model: {config.RERANK_MODEL}...")
        self.reranker = Ranker(model_name=config.RERANK_MODEL)

    def create_collection_if_not_exists(self, collection_id: str):
        """Ensures a Qdrant collection exists for the given document/session."""
        # Sanitize collection name (letters, numbers, underscores, hyphens)
        safe_name = collection_id.replace(".", "_").replace(" ", "_")
        existing_collections = [col.name for col in self.client.get_collections().collections]
        
        if safe_name not in existing_collections:
            self.client.create_collection(
                collection_name=safe_name,
                vectors_config=VectorParams(size=self.vector_size, distance=Distance.COSINE)
            )
        return safe_name

    def index_chunks(self, collection_id: str, chunks: List[DocumentChunk]) -> int:
        """
        Embeds and upserts document chunks into the Qdrant collection.
        """
        if not chunks:
            return 0

        safe_name = self.create_collection_if_not_exists(collection_id)
        
        # Extract text contents for embedding
        texts = [chunk.content for chunk in chunks]
        
        # Generate embeddings using FastEmbed
        embeddings_generator = self.embedding_model.embed(texts)
        embeddings = list(embeddings_generator)

        # Build Qdrant points
        points = []
        for idx, (chunk, vector) in enumerate(zip(chunks, embeddings)):
            points.append(PointStruct(
                id=idx,
                vector=vector.tolist(),
                payload={
                    "chunk_id": chunk.chunk_id,
                    "content": chunk.content,
                    "metadata": chunk.metadata
                }
            ))

        # Upsert in batches of 100
        batch_size = 100
        for i in range(0, len(points), batch_size):
            self.client.upsert(
                collection_name=safe_name,
                points=points[i:i + batch_size]
            )

        print(f"Successfully indexed {len(points)} chunks into collection '{safe_name}'")
        return len(points)

    def search(self, collection_id: str, query: str, top_k: int = 10) -> List[DocumentChunk]:
        """
        Performs vector similarity search against the collection.
        """
        safe_name = collection_id.replace(".", "_").replace(" ", "_")
        existing_collections = [col.name for col in self.client.get_collections().collections]
        
        if safe_name not in existing_collections:
            print(f"Collection '{safe_name}' does not exist in vector store.")
            return []

        # Embed query
        query_vector = list(self.embedding_model.embed([query]))[0].tolist()

        # Search Qdrant
        search_results = self.client.query_points(
            collection_name=safe_name,
            query=query_vector,
            limit=top_k
        ).points

        chunks: List[DocumentChunk] = []
        for point in search_results:
            payload = point.payload or {}
            chunks.append(DocumentChunk(
                chunk_id=payload.get("chunk_id", str(point.id)),
                content=payload.get("content", ""),
                metadata=payload.get("metadata", {}),
                score=point.score
            ))

        return chunks

    def rerank(self, query: str, chunks: List[DocumentChunk], top_k: int = 4) -> List[DocumentChunk]:
        """
        Re-ranks retrieved chunks using FlashRank cross-encoder.
        Significantly improves retrieval precision and eliminates hallucinations.
        """
        if not chunks:
            return []

        # Prepare passages for FlashRank
        passages = [
            {"id": chunk.chunk_id, "text": chunk.content, "metadata": chunk.metadata}
            for chunk in chunks
        ]

        rerank_req = RerankRequest(query=query, passages=passages)
        results = self.reranker.rerank(rerank_req)

        # Map back to DocumentChunks
        chunk_lookup = {chunk.chunk_id: chunk for chunk in chunks}
        reranked_chunks: List[DocumentChunk] = []

        for item in results[:top_k]:
            cid = item["id"]
            if cid in chunk_lookup:
                c = chunk_lookup[cid]
                reranked_chunks.append(DocumentChunk(
                    chunk_id=c.chunk_id,
                    content=c.content,
                    metadata=c.metadata,
                    score=float(item.get("score", 0.0))
                ))

        return reranked_chunks

    def list_collections(self) -> List[str]:
        """Returns all existing collections."""
        return [col.name for col in self.client.get_collections().collections]

    def delete_collection(self, collection_id: str):
        """Deletes a collection and its vector index."""
        safe_name = collection_id.replace(".", "_").replace(" ", "_")
        self.client.delete_collection(safe_name)
