from dataclasses import dataclass
from typing import Any, ClassVar

from llmdataparser.base_parser import HuggingFaceDatasetParser, HuggingFaceParseEntry


@dataclass(frozen=True, kw_only=True, slots=True)
class MATHParseEntry(HuggingFaceParseEntry):
    """Custom entry class for MATH dataset, with fields specific to this dataset parser."""

    level: str
    task_name: str
    solution: str

    @classmethod
    def create(
        cls,
        prompt: str,
        answer: str,
        raw_question: str,
        raw_answer: str,
        level: str,
        task_name: str,
        solution: str,
    ) -> "MATHParseEntry":
        return cls(
            prompt=prompt,
            answer=answer,
            raw_question=raw_question,
            raw_answer=raw_answer,
            level=level,
            task_name=task_name,
            solution=solution,
        )


class MATHDatasetParser(HuggingFaceDatasetParser[MATHParseEntry]):
    """Parser for the MATH dataset."""

    _data_source: ClassVar[str] = "lighteval/MATH"
    _task_names: ClassVar[list[str]] = [
        "algebra",
        "geometry",
        "calculus",
        "prealgebra",
        "intermediate_algebra",
        "number_theory",
        "precalculus",
        "all",
    ]
    _default_task: ClassVar[str] = "all"
    _default_system_prompt: ClassVar[
        str
    ] = "Solve the following mathematics problem step by step:"
    _valid_levels: ClassVar[set[str]] = {
        f"Level {i}" for i in range(1, 6)
    }  # Levels 1-5 are valid

    def _get_task_from_entry(self, data_entry: dict[str, Any]) -> str:
        """Get the task name from the data entry or fall back to current task."""
        entry_type = data_entry.get("type")
        if entry_type and (entry_type in self._task_names):
            return entry_type
        return self._current_task or self._default_task

    def process_entry(
        self, row: dict[str, Any], task_name: str | None = None, **kwargs: Any
    ) -> MATHParseEntry:
        """Process a single MATH dataset entry."""
        task = task_name or self._get_current_task(row)

        # Validate and normalize level
        level = row.get("level")
        if level not in self._valid_levels:
            level = "Unknown"

        return MATHParseEntry.create(
            prompt=f"{self._system_prompt}\n{row['problem']}",
            answer=row["solution"],
            raw_question=row["problem"],
            raw_answer=row["solution"],
            level=level,
            task_name=task,
            solution=row["solution"],
        )


if __name__ == "__main__":
    # Example usage of MATH parser
    parser = MATHDatasetParser()

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
        print(f"Task: {example.task_name}")
        print(f"Level: {example.level}")
        print(f"Question: {example.raw_question}")
        print(f"Solution: {example.solution}")
