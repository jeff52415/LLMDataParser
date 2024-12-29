from dataclasses import dataclass
from typing import Any, ClassVar

from llmdataparser.base_parser import (
    DatasetDescription,
    EvaluationMetric,
    HuggingFaceDatasetParser,
    HuggingFaceParseEntry,
)


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
    _default_system_prompt: ClassVar[str] = (
        "Solve the following mathematics problem step by step:"
    )
    _valid_levels: ClassVar[set[str]] = {
        f"Level {i}" for i in range(1, 6)
    }  # Levels 1-5 are valid

    def _get_task_from_entry(self, data_entry: dict[str, Any]) -> str:
        """Get the task name from the data entry or fall back to current task."""
        entry_type: str = data_entry.get("type", "")
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

    def get_dataset_description(self) -> DatasetDescription:
        """Returns description of the MATH dataset."""
        return DatasetDescription.create(
            name="MATH",
            purpose="Measure mathematical problem-solving capabilities in machine learning models",
            source="Hendrycks et al., UC Berkeley (NeurIPS 2021)",
            language="English",
            format="Competition mathematics problems with step-by-step solutions",
            characteristics=(
                "Collection of 12,500 challenging competition mathematics problems designed to "
                "evaluate mathematical reasoning. Problems include step-by-step solutions that "
                "can be used to teach models to generate answer derivations and explanations. "
                "Problems are categorized by subject area and difficulty level (1-5)."
            ),
            citation="""@article{hendrycksmath2021,
    title={Measuring Mathematical Problem Solving With the MATH Dataset},
    author={Dan Hendrycks and Collin Burns and Saurav Kadavath and Akul Arora and Steven Basart and Eric Tang and Dawn Song and Jacob Steinhardt},
    journal={NeurIPS},
    year={2021}
    }""",
            additional_info={
                "difficulty_levels": "1-5",
                "topics": [
                    "algebra",
                    "geometry",
                    "calculus",
                    "prealgebra",
                    "intermediate_algebra",
                    "number_theory",
                    "precalculus",
                ],
                "size": "12,500 problems",
                "evaluation_note": "Exact match equivalence calculated using sympy library",
                "homepage": "https://github.com/hendrycks/math",
            },
        )

    def get_evaluation_metrics(self) -> list[EvaluationMetric]:
        """Returns recommended evaluation metrics for MATH dataset."""
        return [
            EvaluationMetric.create(
                name="symbolic_equivalence",
                type="exact_match",
                description="Verifies answer correctness using symbolic mathematics (e.g., sympy) to check mathematical equivalence.",
                implementation="sympy_equivalence_checker",
                primary=True,
            ),
            EvaluationMetric.create(
                name="solution_presence",
                type="text",
                description="Ensures that a complete step-by-step solution is provided, demonstrating how the answer is derived.",
                implementation="solution_completeness_checker",
                primary=True,
            ),
            EvaluationMetric.create(
                name="reasoning_validity",
                type="text",
                description="Evaluates the logical correctness and mathematical reasoning in the solution's derivation steps.",
                implementation="reasoning_validator",
                primary=True,
            ),
            EvaluationMetric.create(
                name="mathematical_notation",
                type="text",
                description="Checks for the correct use of mathematical notation and symbolic representation to ensure clarity.",
                implementation="notation_validator",
                primary=False,
            ),
            EvaluationMetric.create(
                name="solution_clarity",
                type="text",
                description="Assesses the clarity, readability, and coherence of the solution steps to enhance interpretability.",
                implementation="clarity_scorer",
                primary=False,
            ),
        ]


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
