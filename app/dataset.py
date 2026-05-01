from abc import ABC, abstractmethod
from typing import Generic, TypeVar, cast

import numpy as np
import pandas as pd

from app.models import DatasetSrc, WikiCity, WikiPOI
from datasets import load_dataset

T = TypeVar("T")


class BaseDataset(ABC, Generic[T]):
    """
    Common interface for datasets that produce DataEntity-compatible records.
    """

    @abstractmethod
    def get_dataset(self) -> list[T]:
        ...


class WikiCityDataset(BaseDataset[WikiCity]):
    def __init__(self):
        wiki_voyage_eu = load_dataset(DatasetSrc.WIKI_VOYAGE_EU)
        wiki_voyage_eu = wiki_voyage_eu['train']

        self._dataset = [WikiCity(**cast(dict, wiki_document)) for wiki_document in wiki_voyage_eu]

    def get_dataset(self) -> list[WikiCity]:
        return self._dataset


class VoyageDataset(BaseDataset[WikiPOI]):
    def __init__(self, cities_to_take: list[str], text_max_size: int = 256) -> None:
        self.voyage_df = pd.read_csv(DatasetSrc.LOCAL_VOYAGE_LISTINGS, encoding="utf-8")
        self._text_max_size = text_max_size
        self._cities_to_take = cities_to_take
    

    def _preprocess(self):
        keep_cols = ["article", "type", "title", "description",
            "price", "latitude", "longitude", "address", "url", "hours"]


        voyage_df = self.voyage_df.dropna(subset=["description"], inplace=False)

        voyage_df= voyage_df[keep_cols]

        if not isinstance(voyage_df, pd.DataFrame):
            raise RuntimeError(f"Should be of the type dataframe instead of {voyage_df.__class__}")

        else:
            voyage_df = cast(pd.DataFrame, voyage_df)
        self.voyage_df = voyage_df

    def _filter_cities(self):
        eu_df = self.voyage_df[self.voyage_df['article'].isin(self._cities_to_take)]
        if isinstance(eu_df, pd.DataFrame):
            eu_df = eu_df.dropna(subset=['description'])
            self.voyage_df = eu_df


    def _add_text_field(self):
        self.voyage_df["text"] = (
            self.voyage_df["title"].fillna("")
            + " — " + self.voyage_df["type"].fillna("")
            + " in " + self.voyage_df["article"].fillna("")
            + ". " + self.voyage_df["description"]
        )

        self.voyage_df['text'] = self.voyage_df['text'].str[:self._text_max_size]
        assert self.voyage_df['text'].isna().any() == np.False_

    def get_dataset(self) -> list[WikiPOI]:
        self._preprocess()
        self._filter_cities()
        self._add_text_field()
        return [WikiPOI(**record) for record in self.voyage_df.to_dict(orient='records')]
