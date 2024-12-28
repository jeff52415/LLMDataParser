from dataclasses import dataclass
from typing import Any, ClassVar

from llmdataparser.base_parser import HuggingFaceDatasetParser, HuggingFaceParseEntry
from llmdataparser.prompts import MBPP_SYSTEM_PROMPT


@dataclass(frozen=True, kw_only=True, slots=True)
class MBPPParseEntry(HuggingFaceParseEntry):
    """Custom entry class for MBPP, with fields specific to this dataset parser."""

    task_id: int
    test_list: list[str]
    test_setup_code: str
    challenge_test_list: list[str]
    source_file: str

    @classmethod
    def create(
        cls,
        prompt: str,
        answer: str,
        raw_question: str,
        task_id: int,
        test_list: list[str],
        test_setup_code: str,
        challenge_test_list: list[str],
        task_name: str,
        source_file: str,
    ) -> "MBPPParseEntry":
        if not isinstance(task_id, int):
            raise ValueError("Task ID must be an integer")

        return cls(
            prompt=prompt,
            answer=answer,
            raw_question=raw_question,
            raw_answer=answer,  # In MBPP, the code solution is the raw answer
            task_id=task_id,
            test_list=test_list,
            test_setup_code=test_setup_code,
            challenge_test_list=challenge_test_list,
            task_name=task_name,
            source_file=source_file,
        )


class MBPPDatasetParser(HuggingFaceDatasetParser[MBPPParseEntry]):
    """Parser for the MBPP (Mostly Basic Python Programming) dataset."""

    _data_source: ClassVar[str] = "google-research-datasets/mbpp"
    _default_task: ClassVar[str] = "full"  # Can be 'full' or 'sanitized'
    _task_names: ClassVar[list[str]] = ["full", "sanitized"]
    _default_system_prompt: ClassVar[str] = MBPP_SYSTEM_PROMPT

    def process_entry(
        self, row: dict[str, Any], task_name: str | None = None, **kwargs: Any
    ) -> MBPPParseEntry:
        """Process a single MBPP entry."""
        raw_question = row.get("text", row.get("prompt"))
        answer = row["code"]
        task_id = row["task_id"]
        test_list = row["test_list"]
        test_setup_code = row.get("test_setup_code", "")
        challenge_test_list = row.get("challenge_test_list", [])

        # Combine system prompt with the task description
        prompt = f"{self._system_prompt}\n\nTask: {raw_question}"

        # Use task_name if provided, otherwise use default
        task = task_name or self._get_current_task(row)
        source_file = row.get("source_file", "")

        return MBPPParseEntry.create(
            prompt=prompt,
            answer=answer,
            raw_question=raw_question,
            task_id=task_id,
            test_list=test_list,
            test_setup_code=test_setup_code,
            challenge_test_list=challenge_test_list,
            task_name=task,
            source_file=source_file,
        )


if __name__ == "__main__":
    # Example usage
    parser = MBPPDatasetParser()

    # Load the dataset
    parser.load()

    # Parse all splits
    parser.parse()

    # Get parsed data
    parsed_data = parser.get_parsed_data

    # Print example entry
    if parsed_data:
        example = parsed_data[0]
        print("\nExample parsed entry:")
        print(f"Task ID: {example.task_id}")
        print(f"Task: {example.raw_question}")
        print(f"Solution:\n{example.answer}")
        print(f"Test Cases:\n{example.test_list}")
