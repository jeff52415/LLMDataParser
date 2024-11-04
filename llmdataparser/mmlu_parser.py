from dataclasses import dataclass
from typing import Any

from llmdataparser.base_parser import HuggingFaceDatasetParser, ParseEntry
from llmdataparser.prompts import MMLU_SYSTEM_PROMPT


@dataclass(frozen=True)
class MMLUParseEntry(ParseEntry):
    """
    Custom entry class for MMLU, with fields specific to this dataset parser.
    """

    prompt: str
    answer_letter: str

    @classmethod
    def create(cls, prompt: str, answer_letter: str) -> "MMLUParseEntry":
        if answer_letter not in {"A", "B", "C", "D"}:
            raise ValueError(
                f"Invalid answer_letter '{answer_letter}'; must be one of 'A', 'B', 'C', 'D'."
            )
        return cls(prompt=prompt, answer_letter=answer_letter)


class MMLUDatasetParser(HuggingFaceDatasetParser[MMLUParseEntry]):
    _data_source = "cais/mmlu"

    def __init__(self, system_prompt: str = MMLU_SYSTEM_PROMPT):
        super().__init__()  # Properly initialize the base class
        self.parsed_data: list[MMLUParseEntry] = []
        self.task_names: list[str] = []
        self.subject_list: set[str] = set()
        self.system_prompt: str = system_prompt
        super().__init__()

    def parse(self, split_names: str | list[str] | None = None, **kwargs: Any) -> None:
        self.parsed_data.clear()
        if self.raw_data is None:
            raise ValueError("No data loaded. Please load the dataset first.")

        if split_names is None:
            split_names = self.task_names
        elif isinstance(split_names, str):
            split_names = [split_names]

        for split_name in split_names:
            if split_name not in self.task_names:
                raise ValueError(f"Task '{split_name}' not found in the dataset.")

            dataset_split = self.raw_data[split_name]
            for index, entry in enumerate(dataset_split, start=1):
                data_entry = self.process_entry(entry, **kwargs)
                self._parsed_data.append(data_entry)
                self.subject_list.add(entry.get("subject", "Unknown"))
            print(f"Parsed {index} data points from task '{split_name}'.")

        print(
            f"Number of subjects: {len(self.subject_list)}. "
            "For more details, please check the `self.subject_list` attribute."
        )

    def process_entry(self, row: dict[str, Any], **kwargs) -> MMLUParseEntry:
        """
        Generate a prompt and expected answer from the given row.

        Args:
            row (dict[str, Any]): A data point to be formatted.

        Returns:
            MMLUParseEntry: The formatted entry object.
        """
        choices = "\n".join(
            f"{chr(65 + i)}. {choice}" for i, choice in enumerate(row["choices"])
        )
        prompt = (
            f"{self.system_prompt}\nQuestion: {row['question']}\n{choices}\nAnswer:"
        )
        answer_letter = chr(65 + row["answer"])  # Convert index to 'A', 'B', 'C', 'D'

        return MMLUParseEntry.create(prompt, answer_letter)
