from dataclasses import dataclass
from typing import Any, ClassVar, List

from llmdataparser.base_parser import HuggingFaceDatasetParser, HuggingFaceParseEntry
from llmdataparser.prompts import IFEVAL_SYSTEM_PROMPT  # You'll need to create this


@dataclass(frozen=True, kw_only=True, slots=True)
class IFEvalParseEntry(HuggingFaceParseEntry):
    """Custom entry class for IFEval, with fields specific to this dataset parser."""

    key: int
    instruction_id_list: List[str]
    kwargs: dict[str, Any]

    @classmethod
    def create(
        cls,
        prompt: str,
        answer: str,
        raw_question: str,
        raw_answer: str,
        key: int,
        instruction_id_list: List[str],
        kwargs: dict[str, Any],
        task_name: str,
    ) -> "IFEvalParseEntry":
        return cls(
            prompt=prompt,
            answer=answer,
            raw_question=raw_question,
            raw_answer=raw_answer,
            key=key,
            instruction_id_list=instruction_id_list,
            kwargs=kwargs,
            task_name=task_name,
        )


class IFEvalDatasetParser(HuggingFaceDatasetParser[IFEvalParseEntry]):
    """Parser for the IFEval dataset."""

    _data_source: ClassVar[str] = "google/IFEval"
    _default_task: ClassVar[str] = "default"
    _task_names: ClassVar[list[str]] = ["default"]
    _default_system_prompt: ClassVar[str] = IFEVAL_SYSTEM_PROMPT

    def process_entry(
        self, row: dict[str, Any], task_name: str | None = None, **kwargs: Any
    ) -> IFEvalParseEntry:
        """Process a single IFEval entry."""
        # Extract fields from the row
        key = row["key"]
        raw_question = row["prompt"]  # The prompt is the raw question in this case
        instruction_id_list = row["instruction_id_list"]
        kwargs_data = row["kwargs"]

        # For IFEval, we don't have explicit answers in the dataset
        # We'll use empty strings as placeholders
        answer = ""
        raw_answer = ""

        # Combine system prompt with the instruction prompt
        prompt = f"{self._system_prompt}\n\n{raw_question}"

        # Use task_name if provided, otherwise use default
        task = task_name or self._get_current_task(row)

        return IFEvalParseEntry.create(
            prompt=prompt,
            answer=answer,
            raw_question=raw_question,
            raw_answer=raw_answer,
            key=key,
            instruction_id_list=instruction_id_list,
            kwargs=kwargs_data,
            task_name=task,
        )


if __name__ == "__main__":
    # Example usage
    parser = IFEvalDatasetParser()
    parser.load()
    parser.parse()

    parsed_data = parser.get_parsed_data
    if parsed_data:
        example = parsed_data[0]
        print("\nExample parsed entry:")
        print(f"Key: {example.key}")
        print(f"Prompt: {example.prompt}")
        print(f"Instruction IDs: {example.instruction_id_list}")
        print(f"kwargs: {example.kwargs}")
