"""
File that specfies the actual models used by the application
"""
from enum import StrEnum
from typing import Generic, Optional, TypedDict, TypeVar

from pydantic import BaseModel


class Collections(StrEnum):
    """
    Qdrant collection names 
    """
    CITIES_POI = "cities_poi"

class DatasetSrc(StrEnum):
    """
    Sources of the datasets
    """
    WIKI_VOYAGE_EU = "ashmib/wikivoyage-eu-city-embeddings"
    LOCAL_VOYAGE_LISTINGS = './datasets/wikivoyage-listings-en.csv'


class EmbeddingNames(StrEnum):
    """
    Embeddings models that are used by the application (part of fastembed)
    """
    DENSE_BAAI_384 = 'BAAI/bge-small-en'
    DENSE_JINAAI_512 = 'jinaai/jina-embeddings-v2-small-en'
    LATE_COLBERT_128 = 'colbert-ir/colbertv2.0'

class CollectionVectorType(StrEnum):
    """
    Stored Vector names
    """
    MAIN_VECTOR = 'dense'
    LATE_VECTOR = 'late_interaction'


class HierarchyLevel(StrEnum):
    """
    Travel data level. Tells what kind of travel object is stored
    """
    CITY = 'city'
    POI = 'poi'

class SearchStrategies(StrEnum):
    """
    Search strategies that might be used in the application
    """
    SIMPLE_SEARCH  = 'simple_search'
    EXPLORATION_SEARCH = 'exploration_search'
    BOTTOM_UP_DISCOVERY = 'bottom-up-discovery'
    TOP_DOWN_DISCOVERY = 'top-down-discovery'
    SIMPLE_ENCODING_SEARCH = 'encoding-search'

class WikiCity(BaseModel):
    """
    Object that is received from wikivoayage dataset
    """
    city: str
    country: str
    lat: float
    lng: float
    population: float
    abstract: str

    def to_searchable_text(self) -> str:
        """
        transform model for embedding
        """
        return f"{self.city}, {self.country}. {self.abstract}"

    def to_payload(self) -> dict:
        """
        transform model to object form 
        """
        return {
            "city": self.city,
            "country": self.country,
            "latitude": self.lat,
            "longitude": self.lng,
            "population": self.population,
            "level": HierarchyLevel.CITY,
        }

class WikiPOI(BaseModel):
    """
    Object that is received from voyage poi dataset
    """
    # Core fields (almost no NaN)
    article: str
    type: str
    title: str
    description: str
    text: str # concatenation of various fields

    # Optional metadata
    alt: Optional[str] = None
    wikidata: Optional[str] = None
    wikipedia: Optional[str] = None

    # Location / contact
    address: Optional[str] = None
    directions: Optional[str] = None
    phone: Optional[str] = None
    tollFree: Optional[str] = None
    email: Optional[str] = None
    fax: Optional[str] = None
    url: Optional[str] = None

    # Operational info
    hours: Optional[str] = None
    checkIn: Optional[str] = None
    checkOut: Optional[str] = None
    price: Optional[str] = None

    # Media
    image: Optional[str] = None

    # Geo
    latitude: Optional[float] = None
    longitude: Optional[float] = None

    # Rare / mostly missing → still optional
    wifi: Optional[str] = None
    accessibility: Optional[str] = None

    # Metadata
    lastEdit: Optional[str] = None  # could be datetime if parsed

    def to_searchable_text(self) -> str:
        """
        transform model for embedding
        """
        return self.text

    def to_payload(self) -> dict:
        """
        transform model to object form
        """
        return {
            **self.model_dump(),
            "level": HierarchyLevel.POI,
        }


DataT = TypeVar("DataT")

class MetaData(TypedDict):
    """
    Metadate of the objects
    """
    source: str

class DataWithMeta(BaseModel, Generic[DataT]):
    """
    Data enriches with metadata
    """
    data: DataT
    metadata: MetaData
