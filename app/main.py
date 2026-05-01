"""
Top-level orchestrator: wires Qdrant ingestion and the search registry.
"""

from qdrant_client import QdrantClient

from app.dataset import VoyageDataset, WikiCityDataset
from app.inject_collection import QdrantInjectCollection
from app.models import Collections, HierarchyLevel, SearchStrategies
from app.search import SearchImplementor, SearchParams


class RerankingSearch:
    def __init__(
        self,
        location: str,
        collection_name: Collections = Collections.CITIES_POI,
    ) -> None:
        self._collection_name = collection_name
        self._client = QdrantClient(location=location)
        self._collection_handler = QdrantInjectCollection(self._client, collection_name)
        self._collection_handler.initialize_embedding_classes()
        if not self._collection_handler.dense_model:
            raise RuntimeError("Cannot access dense model embedding")
        self._search_handler = SearchImplementor(
            self._client,
            collection_name,
            self._collection_handler.dense_model,
            self._collection_handler.interaction_model,
        )

    def ingest(self, force: bool = False) -> None:
        if not force and self._client.count(self._collection_name).count > 0:
            return

        wiki_cities = WikiCityDataset().get_dataset()
        self._collection_handler.upsert_new_elements(HierarchyLevel.CITY, wiki_cities, 16)

        voyage_pois = VoyageDataset()
        voyage_pois.cities = [c.city for c in wiki_cities]
        wiki_pois = voyage_pois.get_dataset()
        self._collection_handler.upsert_new_elements(HierarchyLevel.POI, wiki_pois, 16)

    def search(
        self,
        text: str,
        strategy: SearchStrategies | None = None,
        params: SearchParams | None = None,
    ):
        return self._search_handler.search(text, strategy, params)
