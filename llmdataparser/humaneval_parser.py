from dataclasses import dataclass
from typing import Any, ClassVar

from llmdataparser.base_parser import HuggingFaceDatasetParser, HuggingFaceParseEntry
from llmdataparser.prompts import HUMANEVAL_SYSTEM_PROMPT


@dataclass(frozen=True, kw_only=True, slots=True)
class HumanEvalParseEntry(HuggingFaceParseEntry):
    """Custom entry class for HumanEval, with fields specific to this dataset parser."""

    task_id: str
    task_name: str
    entry_point: str
    test: str

    @classmethod
    def create(
        cls,
        prompt: str,
        answer: str,
        raw_question: str,
        task_id: str,
        entry_point: str,
        test: str,
        task_name: str,
    ) -> "HumanEvalParseEntry":
        if not task_id:
            raise ValueError("Task ID cannot be empty")
        if not entry_point:
            raise ValueError("Entry point cannot be empty")
        return cls(
            prompt=prompt,
            answer=answer,
            raw_question=raw_question,
            raw_answer=answer,  # In HumanEval, the canonical solution is the raw answer
            task_id=task_id,
            entry_point=entry_point,
            test=test,
            task_name=task_name,
        )


class HumanEvalDatasetParser(HuggingFaceDatasetParser[HumanEvalParseEntry]):
    """Parser for the HumanEval dataset."""

    _data_source: ClassVar[str] = "openai/openai_humaneval"
    _default_task: ClassVar[str] = "openai_humaneval"
    _task_names: ClassVar[list[str]] = ["openai_humaneval"]
    _default_system_prompt: ClassVar[str] = HUMANEVAL_SYSTEM_PROMPT

    def process_entry(
        self, row: dict[str, Any], task_name: str | None = None, **kwargs: Any
    ) -> HumanEvalParseEntry:
        """Process a single HumanEval entry."""
        raw_question = row["prompt"]
        answer = row["canonical_solution"]
        task_id = row["task_id"]
        entry_point = row["entry_point"]
        test = row["test"]

        # Combine system prompt with the function signature and docstring
        prompt = f"{self._system_prompt}\n\n{raw_question}"

        # Use task_name if provided, otherwise use default
        task = task_name or self._get_current_task(row)

        return HumanEvalParseEntry.create(
            prompt=prompt,
            answer=answer,
            raw_question=raw_question,
            task_id=task_id,
            entry_point=entry_point,
            test=test,
            task_name=task,  # Guarantee non-None
        )


class HumanEvalDatasetPlusParser(HumanEvalDatasetParser):
    """Parser for the HumanEval dataset."""

    _data_source: ClassVar[str] = "evalplus/humanevalplus"
    _default_task: ClassVar[str] = "default"
    _task_names: ClassVar[list[str]] = ["default"]
    _default_system_prompt: ClassVar[str] = HUMANEVAL_SYSTEM_PROMPT

    def process_entry(
        self, row: dict[str, Any], task_name: str | None = None, **kwargs: Any
    ) -> HumanEvalParseEntry:
        """Process a single HumanEval entry."""
        raw_question = row["prompt"]
        answer = row["canonical_solution"]
        task_id = row["task_id"]
        entry_point = row["entry_point"]
        test = row["test"]

        # Combine system prompt with the function signature and docstring
        prompt = f"{self._system_prompt}\n\n{raw_question}"

        # Use task_name if provided, otherwise use default
        task = task_name or self._get_current_task(row)

        return HumanEvalParseEntry.create(
            prompt=prompt,
            answer=answer,
            raw_question=raw_question,
            task_id=task_id,
            entry_point=entry_point,
            test=test,
            task_name=task,  # task is guaranteed to be str from _get_current_task
        )


if __name__ == "__main__":
    # Example usage
    parser = HumanEvalDatasetParser()

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
        print(f"Entry Point: {example.entry_point}")
        print(f"Prompt:\n{example.prompt}")
        print(f"Solution:\n{example.answer}")

    parser = HumanEvalDatasetPlusParser()
    parser.load()
    parser.parse()
    parsed_data = parser.get_parsed_data
    if parsed_data:
        example = parsed_data[0]
        print("\nExample parsed entry:")
        print(f"Task: {example.task_name}")
        print(f"Question: {example.raw_question}")
        print(f"Correct Answer: {example.answer}")
