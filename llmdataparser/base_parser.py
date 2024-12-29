from abc import ABC, abstractmethod
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, ClassVar, Generic, TypeVar

import datasets

# Define the generic type variable
T = TypeVar("T", bound="ParseEntry")


@dataclass(frozen=True, kw_only=True, slots=True)
class ParseEntry:
    """A simple base class for entries, customizable by each dataset parser."""

    prompt: str
    answer: str
    raw_question: str
    raw_answer: str


@dataclass(frozen=True, kw_only=True, slots=True)
class DatasetDescription:
    """Standardized description of a dataset."""

    name: str
    purpose: str
    source: str
    language: str
    format: str
    characteristics: str
    citation: str | None = None
    additional_info: dict[str, Any] | None = None

    @classmethod
    def create(
        cls,
        name: str,
        purpose: str,
        source: str,
        language: str,
        format: str,
        characteristics: str,
        citation: str | None = None,
        additional_info: dict[str, Any] | None = None,
    ) -> "DatasetDescription":
        return cls(
            name=name,
            purpose=purpose,
            source=source,
            language=language,
            format=format,
            characteristics=characteristics,
            citation=citation,
            additional_info=additional_info,
        )


@dataclass(frozen=True, kw_only=True, slots=True)
class EvaluationMetric:
    """Description of an evaluation metric for a dataset."""

    name: str
    type: str
    description: str
    implementation: str
    primary: bool

    @classmethod
    def create(
        cls, name: str, type: str, description: str, implementation: str, primary: bool
    ) -> "EvaluationMetric":
        return cls(
            name=name,
            type=type,
            description=description,
            implementation=implementation,
            primary=primary,
        )


class DatasetParser(Generic[T], ABC):
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
    def process_entry(
        self, row: dict[str, Any], task_name: str | None = None, **kwargs: Any
    ) -> T:
        """
        Process a single entry from the dataset.

        Args:
            row: A dictionary representing a single entry from the dataset.
            task_name: Optional task name for the entry.
            **kwargs: Additional keyword arguments.

        Returns:
            T: The processed entry, typically an instance of a subclass of ParseEntry.
        """

    @abstractmethod
    def get_dataset_description(self) -> DatasetDescription:
        """Returns a standardized description of the dataset."""

    @abstractmethod
    def get_evaluation_metrics(self) -> list[EvaluationMetric]:
        """Returns the recommended evaluation metrics for the dataset."""


@dataclass(frozen=True, kw_only=True, slots=True)
class HuggingFaceParseEntry(ParseEntry):
    """ParseEntry with an additional task_name field."""

    task_name: str


class HuggingFaceDatasetParser(DatasetParser[T]):
    """
    Base class for parsers that use datasets from Hugging Face.
    """

    # _data_source is the name of the dataset, e.g. "lighteval/MATH"
    _data_source: ClassVar[str]
    # _task_names is the list of tasks in the dataset, e.g. ["algebra", "geometry", "statistics"]
    _task_names: ClassVar[list[str]]
    # _default_task is the default task to use if no task is specified, e.g. "algebra"
    _default_task: ClassVar[str]
    # _default_system_prompt is the default system prompt to use if no system prompt is specified
    _default_system_prompt: ClassVar[str]
    # _hidden_task_names is the list of task names that are hidden in the dataset, e.g. ["math", "physics", "chemistry"]
    _hidden_task_names: ClassVar[list[str]] = []

    def __init__(self, system_prompt: str | None = None, **kwargs):
        """
        Initialize a HuggingFaceDatasetParser.

        Args:
            system_prompt: Optional custom system prompt to use instead of the default.
                         If not provided, will use the class's _default_system_prompt.
            **kwargs: Additional keyword arguments passed to the parent class.
        """
        super().__init__()
        # raw_data is the dataset loaded from HuggingFace
        self.raw_data: dict[str, Any] | None = None
        # split_names is the list of splits in the dataset, e.g. ["train", "test", "validation"]
        self.split_names: list[str] = []
        # _current_task is the task currently being processed, e.g. "algebra"
        self._current_task: str = ""
        # _system_prompt is the system prompt currently being used
        self._system_prompt: str = system_prompt or self._default_system_prompt

    def _get_current_task(self, data_entry: dict[str, Any] | None = None) -> str:
        """
        Get the currently loaded task name.

        Args:
            data_entry: Optional dictionary containing entry data that might include task information

        Returns:
            str: The task name from either the data entry (if available) or the currently set task
        """
        # If data_entry is provided and contains task information, use it
        if data_entry is not None and hasattr(self, "_get_task_from_entry"):
            try:
                return self._get_task_from_entry(data_entry)
            except (KeyError, AttributeError):
                pass

        # Otherwise return the task set during load()
        return self._current_task or self._default_task

    @property
    def task_names(self) -> list[str]:
        """Get all available task names."""
        return self._task_names

    @property
    def total_tasks(self) -> int:
        """Get total number of available tasks."""
        return len(self._task_names)

    @property
    def get_huggingface_link(self) -> str:
        return "https://huggingface.co/datasets/" + self._data_source

    @staticmethod
    @lru_cache(maxsize=3)
    def load_dataset_cached(
        data_source: str, task_name: str = "default", **kwargs: Any
    ):
        """
        Cached static method to load a dataset from Hugging Face.
        """
        return datasets.load_dataset(data_source, task_name, **kwargs)

    def parse(
        self,
        split_names: str | list[str] | None = None,
        force: bool = False,
        **kwargs: Any,
    ) -> None:
        """
        Parse the MATH dataset splits into structured entries.

        Args:
            split_names: Dataset splits to parse. Can be:
                - None: Parse all available splits
                - str: Parse a single split (e.g., "train")
                - list[str]: Parse multiple splits (e.g., ["train", "test"])
            force: If True, overwrites existing parsed data without confirmation.
                If False and parsed data exists, prompts for confirmation.
            **kwargs: Additional keyword arguments passed to process_entry

        Raises:
            ValueError: If no data is loaded or if a specified split name doesn't exist
        """
        if self.raw_data is None:
            raise ValueError("No data loaded. Please load the dataset first.")

        if self._parsed_data and not force:
            response = input(
                f"Found {len(self._parsed_data)} existing parsed entries. "
                "Do you want to overwrite them? [y/N]: "
            ).lower()
            if response not in ("y", "yes"):
                print("Parsing cancelled. Existing data preserved.")
                return

        self._parsed_data.clear()

        # Dataset with splits
        if split_names is None:
            split_names = self.split_names
        elif isinstance(split_names, str):
            split_names = [split_names]

        for split_name in split_names:
            if split_name not in self.split_names:
                raise ValueError(f"Split '{split_name}' not found in the dataset.")

            dataset_split = self.raw_data[split_name]
            total_entries = len(dataset_split)
            print(f"Processing {split_name} split with {total_entries} entries...")

            for index, entry in enumerate(dataset_split, start=1):
                try:
                    task_name = self._get_current_task(data_entry=entry)
                    parsed_entry = self.process_entry(entry, task_name, **kwargs)
                    self._parsed_data.append(parsed_entry)

                    # Print progress every 100 entries
                    if index % 100 == 0:
                        print(
                            f"Processed {index}/{total_entries} entries from '{split_name}'"
                        )

                except Exception as e:
                    print(f"Error processing entry {index} in {split_name}: {str(e)}")
                    continue

            print(f"Completed parsing {index} entries from '{split_name}'")

        print(f"Total parsed entries: {len(self._parsed_data)}")

    def load(
        self,
        task_name: str | None = None,
        trust_remote_code: bool = True,
        split: str | None = None,
        **kwargs: Any,
    ) -> None:
        """
        Load the dataset using the Hugging Face datasets library.
        """
        # Set the task name
        self._current_task = task_name or self._default_task

        # Call the cached static method
        raw_data = self.load_dataset_cached(
            self._data_source,
            task_name=self._current_task,
            trust_remote_code=trust_remote_code,
            split=split,
            **kwargs,
        )

        # Handle split-specific loading
        if split:
            self.raw_data = {split: raw_data}
            self.split_names = [split]
        else:
            self.raw_data = raw_data
            self.split_names = list(raw_data.keys())

        print(
            f"Loaded dataset with {len(self.split_names)} groups: {', '.join(self.split_names)}."
        )

    def __repr__(self) -> str:
        status = "loaded" if self.raw_data is not None else "not loaded"
        parsed_count = len(self._parsed_data) if self._parsed_data else 0
        return (
            f"{self.__class__.__name__}("
            f"data_source='{self._data_source}', "
            f"task='{self._current_task}', "
            f"status='{status}', "
            f"parsed_entries={parsed_count}"
            ")"
        )

    def __str__(self) -> str:
        return self.__repr__()
