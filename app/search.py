"""
Contains Search related classes
"""

from abc import ABC, abstractmethod
from typing import Any, Generic, TypeVar, override

from fastembed import LateInteractionTextEmbedding, TextEmbedding
from fastembed.rerank.cross_encoder import TextCrossEncoder
from pydantic import BaseModel
from qdrant_client import QdrantClient
from qdrant_client.conversions.common_types import ScoredPoint
from qdrant_client.http.models import PointGroup
from qdrant_client.models import FieldCondition, Filter, MatchValue, Prefetch

from app.models import CollectionVectorType, HierarchyLevel, SearchStrategies, WikiCity, WikiPOI


def _format_payload(payload: dict | None) -> str:
    """Dispatch a Qdrant payload to its model and format a one-line summary."""
    if payload is None:
        return "<empty payload>"
    level = payload.get("level")
    if level == HierarchyLevel.CITY:
        city = WikiCity.to_model(payload)
        return f"{city.city}: {city.abstract}"
    if level == HierarchyLevel.POI:
        poi = WikiPOI.to_model(payload)
        return f"{poi.title} in {poi.article} ({poi.type}): {poi.description}"
    return f"<unknown level={level}>"


class SearchParams(BaseModel):
    pass


class BasicSearchParams(SearchParams):
    limit: int = 30


class CitySearchParams(SearchParams):
    city: str
    limit: int = 10
    prefetch_limit: int = 50


class BottomUpDiscoveryParams(SearchParams):
    prefetch_limit: int = 100
    group_size: int = 5
    limit: int = 10


class TopDownDiscoveryParams(SearchParams):
    top_cities: int = 3
    pois_per_city: int = 5
    prefetch_limit: int = 30
    limit: int = 40

class SearchEncoderParams(SearchParams):
    prefetch_limit: int = 100
    final_limit: int = 10


ReturnTypeT = TypeVar("ReturnTypeT")
ParamsT = TypeVar("ParamsT", bound=SearchParams)


class SearchStrategy(ABC, Generic[ReturnTypeT, ParamsT]):
    """
    Common interface for classes that use different search capabilities & tools
    """

    def __init__(self, client: QdrantClient, collection_name: str) -> None:
        self._client = client
        self._collection_name = collection_name

    @abstractmethod
    def search(self, text: str, params: ParamsT) -> list[ReturnTypeT]:
        """
        Differentiator search functionality
        """

    @abstractmethod
    def get_name(self) -> SearchStrategies:
        """
        Name of the strategy
        """
    @staticmethod
    @abstractmethod
    def to_readable_str(results: list[ReturnTypeT]) -> str:
        """
        Get readable string
        """

class SimpleSearch(SearchStrategy[ScoredPoint, BasicSearchParams]):
    def __init__(
        self, client: QdrantClient, collection_name: str, dense_model: TextEmbedding
    ) -> None:
        super().__init__(client, collection_name)
        self._dense_model = dense_model

    @override
    def search(self, text: str, params: BasicSearchParams) -> list[ScoredPoint]:
        dense_query = list(self._dense_model.embed([text]))

        result = self._client.query_points(
            self._collection_name,
            query=dense_query[0],
            using=str(CollectionVectorType.MAIN_VECTOR),
            with_vectors=True,
            with_payload=True,
            limit=params.limit,
        )
        return result.points

    @override
    def get_name(self) -> SearchStrategies:
        return SearchStrategies.SIMPLE_SEARCH

    @staticmethod
    def to_readable_str(results: list[ScoredPoint]) -> str:
        return "\n".join(
            f"[score={p.score:.4f}] {_format_payload(p.payload)}" for p in results
        )


class ExplorationSearch(SearchStrategy[ScoredPoint, CitySearchParams]):
    def __init__(
        self,
        client: QdrantClient,
        collection_name: str,
        dense_model: TextEmbedding,
        interaction_model: LateInteractionTextEmbedding,
    ) -> None:
        super().__init__(client, collection_name)
        self._dense_model = dense_model
        self._interaction_model = interaction_model

    @override
    def search(self, text: str, params: CitySearchParams) -> list[ScoredPoint]:
        dense_query = list(self._dense_model.embed([text]))[0].tolist()
        colbert_query = list(self._interaction_model.embed([text]))[0].tolist()

        result = self._client.query_points(
            self._collection_name,
            prefetch=Prefetch(
                query=dense_query,
                using=str(CollectionVectorType.MAIN_VECTOR),
                limit=params.prefetch_limit,
                filter=Filter(
                    must=[
                        FieldCondition(
                            key="article",
                            match=MatchValue(value=params.city),
                        ),
                        FieldCondition(
                            key="level",
                            match=MatchValue(value=HierarchyLevel.POI),
                        ),
                    ]
                ),
            ),
            query=colbert_query,
            using=str(CollectionVectorType.LATE_VECTOR),
            limit=params.limit,
        )
        return result.points

    @override
    def get_name(self) -> SearchStrategies:
        return SearchStrategies.EXPLORATION_SEARCH

    @staticmethod
    def to_readable_str(results: list[ScoredPoint]) -> str:
        return "\n".join(
            f"[score={p.score:.4f}] {_format_payload(p.payload)}" for p in results
        )

class BottomUpDiscoverySearch(SearchStrategy[PointGroup, BottomUpDiscoveryParams]):
    def __init__(
        self,
        client: QdrantClient,
        collection_name: str,
        dense_model: TextEmbedding,
        interaction_model: LateInteractionTextEmbedding,
    ) -> None:
        super().__init__(client, collection_name)
        self._dense_model = dense_model
        self._interaction_model = interaction_model

    @override
    def search(self, text: str, params: BottomUpDiscoveryParams) -> list[PointGroup]:
        dense_query = list(self._dense_model.embed([text]))[0].tolist()
        colbert_query = list(self._interaction_model.embed([text]))[0].tolist()

        result = self._client.query_points_groups(
            collection_name=self._collection_name,
            prefetch=Prefetch(
                query=dense_query,
                using=str(CollectionVectorType.MAIN_VECTOR),
                limit=params.prefetch_limit,
                filter=Filter(
                    must=[FieldCondition(key="level", match=MatchValue(value=HierarchyLevel.POI))]
                ),
            ),
            query=colbert_query,
            using=str(CollectionVectorType.LATE_VECTOR),
            group_by="article",
            group_size=params.group_size,
            limit=params.limit,
        )
        return result.groups

    @override
    def get_name(self) -> SearchStrategies:
        return SearchStrategies.BOTTOM_UP_DISCOVERY

    @staticmethod
    def to_readable_str(results: list[PointGroup]) -> str:
        lines: list[str] = []
        for group in results:
            lines.append(f"== {group.id} ==")
            for hit in group.hits:
                lines.append(f"  [score={hit.score:.4f}] {_format_payload(hit.payload)}")
        return "\n".join(lines)


class TopDownDiscoverySearch(SearchStrategy[dict[str, Any], TopDownDiscoveryParams]):
    def __init__(
        self,
        client: QdrantClient,
        collection_name: str,
        dense_model: TextEmbedding,
        interaction_model: LateInteractionTextEmbedding,
    ) -> None:
        super().__init__(client, collection_name)
        self._dense_model = dense_model
        self._interaction_model = interaction_model

    @override
    def search(self, text: str, params: TopDownDiscoveryParams) -> list[dict[str, Any]]:
        dense_query = list(self._dense_model.embed([text]))[0].tolist()
        colbert_query = list(self._interaction_model.embed([text]))[0].tolist()

        cities_result = self._client.query_points(
            collection_name=self._collection_name,
            prefetch=Prefetch(
                query=dense_query,
                using=str(CollectionVectorType.MAIN_VECTOR),
                limit=params.prefetch_limit,
                filter=Filter(
                    must=[FieldCondition(key="level", match=MatchValue(value=HierarchyLevel.CITY))]
                ),
            ),
            query=colbert_query,
            using=str(CollectionVectorType.LATE_VECTOR),
            limit=params.top_cities,
        )

        results: dict[str, list[ScoredPoint]] = {}
        for city_point in cities_result.points:
            if city_point.payload is None:
                continue
            city_name = city_point.payload["city"]
            pois = self._client.query_points(
                collection_name=self._collection_name,
                prefetch=Prefetch(
                    query=dense_query,
                    using=str(CollectionVectorType.MAIN_VECTOR),
                    limit=params.limit,
                    filter=Filter(
                        must=[
                            FieldCondition(
                                key="article", match=MatchValue(value=city_name)
                            ),
                            FieldCondition(key="level", match=MatchValue(value=HierarchyLevel.POI)),
                        ]
                    ),
                ),
                query=colbert_query,
                using=str(CollectionVectorType.LATE_VECTOR),
                limit=params.pois_per_city,
            )
            results[city_name] = pois.points

        return [{"key": k, "value": v} for k, v in results.items()]

    @override
    def get_name(self) -> SearchStrategies:
        return SearchStrategies.TOP_DOWN_DISCOVERY

    @staticmethod
    def to_readable_str(results: list[dict[str, Any]]) -> str:
        lines: list[str] = []
        for entry in results:
            city_name = entry["key"]
            pois: list[ScoredPoint] = entry["value"]
            lines.append(f"== {city_name} ==")
            for poi in pois:
                lines.append(f"  [score={poi.score:.4f}] {_format_payload(poi.payload)}")
        return "\n".join(lines)


class CrossEncodingSearch(SearchStrategy[tuple[ScoredPoint, float], SearchEncoderParams]):
    def __init__(
        self,
        client: QdrantClient,
        collection_name: str,
        dense_model: TextEmbedding,
        encoder_model: str = 'Xenova/ms-marco-MiniLM-L-6-v2'
    ) -> None:
        super().__init__(client, collection_name)
        self._dense_model = dense_model
        self._cross_encoder = TextCrossEncoder(encoder_model)

    @override
    def search(self, text: str, params: SearchEncoderParams) -> list[tuple[ScoredPoint, float]]:
        dense_query = list(self._dense_model.embed([text]))[0].tolist()

        candidates = self._client.query_points(
            collection_name=self._collection_name,
            query=dense_query,
            using=str(CollectionVectorType.MAIN_VECTOR),
            limit=params.prefetch_limit,
            query_filter=Filter(must=[
                FieldCondition(key="level", match=MatchValue(value=HierarchyLevel.POI)),
            ]),
        )

        scored_candidates = [
            (p, p.payload["text"]) for p in candidates.points
            if p.payload and "text" in p.payload
        ]
        if not scored_candidates:
            return []

        candidate_texts = [t for _, t in scored_candidates]
        scores = list(self._cross_encoder.rerank(text, candidate_texts))

        reranked = sorted(
            zip([p for p, _ in scored_candidates], scores),
            key=lambda x: x[1],
            reverse=True,
        )[:params.final_limit]
        return reranked

    @override
    def get_name(self) -> SearchStrategies:
        return SearchStrategies.SIMPLE_ENCODING_SEARCH

    @staticmethod
    def to_readable_str(results: list[tuple[ScoredPoint, float]]) -> str:
        return "\n".join(
            f"[rerank={score:.4f} qdrant={p.score:.4f}] {_format_payload(p.payload)}"
            for p, score in results
        )


_DEFAULT_PARAMS: dict[SearchStrategies, type[SearchParams]] = {
    SearchStrategies.SIMPLE_SEARCH: BasicSearchParams,
    SearchStrategies.EXPLORATION_SEARCH: CitySearchParams,
    SearchStrategies.BOTTOM_UP_DISCOVERY: BottomUpDiscoveryParams,
    SearchStrategies.TOP_DOWN_DISCOVERY: TopDownDiscoveryParams,
    SearchStrategies.SIMPLE_ENCODING_SEARCH: SearchEncoderParams,
}


SEARCH_STRATEGIE_CLASSES: dict[SearchStrategies, type[SearchStrategy]] = {
    SearchStrategies.SIMPLE_SEARCH: SimpleSearch,
    SearchStrategies.EXPLORATION_SEARCH: ExplorationSearch,
    SearchStrategies.BOTTOM_UP_DISCOVERY: BottomUpDiscoverySearch,
    SearchStrategies.TOP_DOWN_DISCOVERY: TopDownDiscoverySearch,
    SearchStrategies.SIMPLE_ENCODING_SEARCH: CrossEncodingSearch,
}

class SearchImplementor:
    """
    Registry/dispatcher for available search strategies.
    """

    def __init__(
        self,
        client: QdrantClient,
        collection_name: str,
        dense_model: TextEmbedding,
        interaction_model: LateInteractionTextEmbedding | None,
    ) -> None:
        self._default_strategy = SearchStrategies.TOP_DOWN_DISCOVERY

        search_repository: list[SearchStrategy[Any, Any]] = [
            SimpleSearch(client, collection_name, dense_model),
            CrossEncodingSearch(client, collection_name, dense_model),
        ]

        if interaction_model is not None:
            search_repository.extend([
                ExplorationSearch(client, collection_name, dense_model, interaction_model),
                BottomUpDiscoverySearch(client, collection_name, dense_model, interaction_model),
                TopDownDiscoverySearch(client, collection_name, dense_model, interaction_model),
            ])

        self._search_repository: dict[SearchStrategies, SearchStrategy[Any, Any]] = {
            strategy.get_name(): strategy for strategy in search_repository
        }

    def does_strategy_exist(self, strategy: SearchStrategies) -> bool:
        return strategy in self._search_repository

    def search(
        self,
        text: str,
        strategy: SearchStrategies | None = None,
        params: SearchParams | None = None,
    ):
        strategy = strategy if strategy is not None else self._default_strategy

        if strategy not in self._search_repository:
            raise ValueError(
                f"Strategy '{strategy}' not available. "
                f"Available: {list(self._search_repository.keys())}"
            )

        if params is None:
            params = _DEFAULT_PARAMS[strategy]()

        return self._search_repository[strategy].search(text, params)