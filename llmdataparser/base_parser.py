from abc import ABC, abstractmethod
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Generic, TypeVar

import datasets

# Define the generic type variable
T = TypeVar("T", bound="ParseEntry")


@dataclass(frozen=True)
class ParseEntry:
    """A simple base class for entries, customizable by each dataset parser."""


class DatasetParser(ABC, Generic[T]):
    """
    Abstract base class defining the interface for all dataset parsers.
    """

    def __init__(self):
        self._parsed_data: list[T] = []

    @abstractmethod
    def load(self, **kwargs: Any) -> None:
        pass

    @abstractmethod
    def parse(self, split_names: str | list[str] | None = None, **kwargs: Any) -> None:
        """
        Parse the loaded dataset into self._parsed_data.
        """

    @property
    def get_parsed_data(self) -> list[T]:
        if not hasattr(self, "_parsed_data") or not self._parsed_data:
            raise ValueError("Parsed data has not been initialized.")
        return self._parsed_data

    @abstractmethod
    def process_entry(self, row: dict[str, Any]) -> T:
        pass


# Base class for Hugging Face datasets
class HuggingFaceDatasetParser(DatasetParser[T]):
    """
    Base class for parsers that use datasets from Hugging Face.
    """

    _data_source: str  # Class variable for the dataset name

    def __init__(self):
        self.raw_data = None
        self.task_names = []
        super().__init__()

    def get_task_names(self) -> list[str]:
        return self.task_names

    @staticmethod
    @lru_cache(maxsize=3)
    def load_dataset_cached(
        data_source: str, config_name: str = "default", **kwargs: Any
    ):
        """
        Cached static method to load a dataset from Hugging Face.
        """
        return datasets.load_dataset(data_source, config_name, **kwargs)

    def load(
        self,
        data_source: str | None = None,
        config_name: str = "all",
        trust_remote_code: bool = True,
        split: str | None = None,
        **kwargs: Any,
    ) -> None:
        """
        Load the dataset using the Hugging Face datasets library.
        """
        # Use class-level data_source if not provided
        data_source = data_source or self._data_source
        if not data_source:
            raise ValueError("The 'data_source' class variable must be defined.")

        # Call the cached static method
        self.raw_data = self.load_dataset_cached(
            data_source,
            config_name=config_name,
            trust_remote_code=trust_remote_code,
            split=split,
            **kwargs,
        )
        self.task_names = list(self.raw_data.keys())
        print(
            f"Loaded dataset with {len(self.task_names)} tasks: {', '.join(self.task_names)}."
        )
        # Additional common initialization can be added here
