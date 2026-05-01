"""
File that serves or the injection stage
"""

import uuid
import os
from enum import StrEnum
from itertools import zip_longest
from pathlib import Path
from typing import Any, Protocol

from fastembed.common.types import NumpyArray
import numpy as np
from fastembed import LateInteractionTextEmbedding, TextEmbedding
from tqdm import tqdm
from numpy.typing import NDArray
from qdrant_client import QdrantClient
from qdrant_client.models import (
    CollectionStatus,
    Distance,
    HnswConfigDiff,
    MultiVectorComparator,
    MultiVectorConfig,
    PointStruct,
    VectorParams,
)

from app.models import Collections, CollectionVectorType, EmbeddingNames

_NAMESPACE = uuid.UUID("00000000-0000-0000-0000-000000000001")
_ROOT_PATH = Path(
    __file__
).parent.parent  # IMPORTANT!: change once file location is changed


class CollectionConfigTypes(StrEnum):
    """
    Available config options
    """

    DEFAULT = "default"
    LATE_INTERACTION = "late-interaction"


class DataEntity(Protocol):
    """
    Protocol of data that will be injected
    """

    def to_searchable_text(self) -> str:
        """
        Transform to a single texttual representation of structured data
        """
        ...

    def to_payload(self) -> dict[str, str | float | int]:
        """
        Convert to the dictionary representation of the data model
        """
        ...


class BatchSequentialEmbedding:
    def __init__(self, dense_model: TextEmbedding, colbert_model: LateInteractionTextEmbedding | None) -> None:
        self.dense_model = dense_model
        self.colbert_model = colbert_model
        self._parallel = os.cpu_count()


    def _embed_in_batches(self, model: TextEmbedding | LateInteractionTextEmbedding, texts: list[str], batch_size: int, desc: str = "Embedding"):
        """Embed texts in batches with progress bar. Returns list of arrays."""
        results: list[NumpyArray] = []
        total = len(texts)
        for i in tqdm(range(0, total, batch_size), desc=desc):
            batch = texts[i : i + batch_size]
            batch_embeddings = list(model.embed(batch, parallel=self._parallel))
            results.extend(batch_embeddings)

        return results

    def embed(self, texts: list[str], batch_size: int):
        """
        Run dense then ColBERT sequentially. 
        Safer on RAM, use this if parallel version causes memory pressure.
        """

        dense_results = self._embed_in_batches(self.dense_model, texts, desc="Dense embedding", batch_size=batch_size)
        colbert_results = self._embed_in_batches(self.colbert_model, texts, desc="ColBERT embedding", batch_size=batch_size) if self.colbert_model else []
        return dense_results, colbert_results


class QdrantInjectCollection:
    """
    Class for handling qdrant collection and injections
    """

    def __init__(
        self,
        client_url: str,
        name: Collections,
        *,
        distance: Distance = Distance.COSINE,
        collection_type: CollectionConfigTypes = CollectionConfigTypes.LATE_INTERACTION,
        dense_model_name: EmbeddingNames | None = None,
        interaction_model_name: EmbeddingNames | None = None,
    ) -> None:
        self._name = name
        self._client = QdrantClient(location=client_url)
        self._distance = distance
        self._collection_type = collection_type
        self._created = self._client.collection_exists(name)
        self.dense_model: None | TextEmbedding = None
        self.interaction_model: None | LateInteractionTextEmbedding = None
        self._interaction_model_name = (
            interaction_model_name
            if interaction_model_name
            else EmbeddingNames.LATE_COLBERT_128
        )
        self._dense_model_name = (
            dense_model_name if dense_model_name else EmbeddingNames.DENSE_BAAI_384
        )

    @property
    def client(self) -> QdrantClient:
        return self._client

    @property
    def name(self) -> Collections:
        return self._name

    @property
    def _vector_config(self):
        """
        Vector configuration for the collection
        """
        if self._collection_type == CollectionConfigTypes.DEFAULT:
            return VectorParams(
                size=self._client.get_embedding_size(self._dense_model_name),
                distance=self._distance,
            )
        return {
            str(CollectionVectorType.MAIN_VECTOR): VectorParams(
                size=self._client.get_embedding_size(self._dense_model_name),
                distance=self._distance,
            ),
            str(CollectionVectorType.LATE_VECTOR): VectorParams(
                size=self._client.get_embedding_size(self._interaction_model_name),
                distance=self._distance,
                multivector_config=MultiVectorConfig(
                    comparator=MultiVectorComparator.MAX_SIM
                ),
                hnsw_config=HnswConfigDiff(m=0),
            ),
        }

    def _doc_id(self, doc: DataEntity) -> str:
        """
        Create document based id
        """
        return str(uuid.uuid5(_NAMESPACE, doc.to_searchable_text()))

    def check_status(self) -> CollectionStatus:
        """
        Check if the collection is ready
        """
        if not self._created:
            return CollectionStatus.GREY
        return self._client.get_collection(self._name).status

    def _create(self):
        self.initialize_embedding_classes()
        if self._created:
            return False
        collection = self._client.create_collection(
            self._name,
            self._vector_config
        )

        self._created = True
        return collection

    def initialize_embedding_classes(self):
        if self.dense_model is None:
            self.dense_model = TextEmbedding(model_name=self._dense_model_name)
        if self._collection_type != CollectionConfigTypes.DEFAULT and self.interaction_model is None:
            self.interaction_model = LateInteractionTextEmbedding(
                model_name=self._interaction_model_name
            )

    def _embed_and_save(self, name: str, data: list[DataEntity], batch_size = 256):
        if not self.dense_model:
            raise RuntimeError("Should initialize dense model first")
        embedding_strategy = BatchSequentialEmbedding(
            self.dense_model,
            self.interaction_model
        )
        dense_path = Path(f"{_ROOT_PATH}/embeddings/dense_embeddings_{name}.npy")
        has_late_interaction = (
            self._collection_type == CollectionConfigTypes.LATE_INTERACTION
        )
        interaction_path: None | Path = None
        interaction: NDArray | None = None
        dense: NDArray | None = None
        if has_late_interaction:
            interaction_path = Path(f"{_ROOT_PATH}/embeddings/interaction_embeddings_{name}.npy")

        cache_complete = dense_path.exists() and (
            interaction_path is None or interaction_path.exists()
        )
        if cache_complete:
            dense = np.load(dense_path)
            if interaction_path is not None:
                interaction = np.load(interaction_path, allow_pickle=True)
        else:
            dense_embedding, interaction_embedding = embedding_strategy.embed(
                texts=[element.to_searchable_text() for element in data],
                batch_size=batch_size,
            )
            dense = np.array(dense_embedding)
            dense_path.parent.mkdir(parents=True, exist_ok=True)
            np.save(dense_path, dense)
            if interaction_path is not None:
                interaction = np.array(interaction_embedding, dtype=object)
                np.save(interaction_path, interaction)

        return dense, interaction

    def _batch_upsert(self, points: list[PointStruct], batch_size: int = 100):
        def batch(iterable, size):
            for i in range(0, len(iterable), size):
                yield iterable[i:i + size]
        for batch_points in batch(points, batch_size):
            self._client.upsert(
                collection_name=self._name,
                points=batch_points
            )

    def upsert_new_elements(self, name: str, data: list[DataEntity]):
        """
        Add new embedded elements to the collection
        """
        self._create()
        dense_embedding, interaction_embedding = self._embed_and_save(name, data)
        if dense_embedding is None:
            raise ValueError("Dense array embedding cannot be None")
        if (
            interaction_embedding is None
            and self._collection_type == CollectionConfigTypes.LATE_INTERACTION
        ):
            raise ValueError("Interaction array embedding cannot be None")

        def create_vectors(
            dense_el: Any, interaction_el: Any
        ) -> list[float] | dict[str, Any]:
            if interaction_el is None:
                return dense_el.tolist()
            return {
                str(CollectionVectorType.MAIN_VECTOR): dense_el.tolist(),
                str(CollectionVectorType.LATE_VECTOR): interaction_el.tolist(),
            }

        if len(data) != len(dense_embedding):
            raise ValueError("The embeddings should be the same size as data")
        points = [
            PointStruct(
                id=self._doc_id(doc),
                vector=create_vectors(dense_el, interaction_el),
                payload=doc.to_payload(),
            )
            for doc, dense_el, interaction_el in zip_longest(
                data, dense_embedding, interaction_embedding or [], fillvalue=None
            )
            if doc is not None
        ]

        self._batch_upsert(points)
