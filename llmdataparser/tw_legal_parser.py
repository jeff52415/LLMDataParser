from dataclasses import dataclass
from typing import Any, Final

from llmdataparser.base_parser import (
    DatasetDescription,
    EvaluationMetric,
    HuggingFaceDatasetParser,
    HuggingFaceParseEntry,
)

TW_LEGAL_VALID_ANSWERS: Final[set[str]] = {"A", "B", "C", "D"}
TW_LEGAL_VALID_ANSWER_STR: Final[str] = ", ".join(sorted(TW_LEGAL_VALID_ANSWERS))


@dataclass(frozen=True, kw_only=True, slots=True)
class TWLegalParseEntry(HuggingFaceParseEntry):
    """Custom entry class for Taiwan Legal Benchmark, with fields specific to this dataset parser."""

    raw_choices: list[str]

    @classmethod
    def create(
        cls,
        question: str,
        answer: str,
        raw_question: str,
        raw_choices: list[str],
        raw_answer: str,
        task_name: str,
    ) -> "TWLegalParseEntry":
        if answer not in TW_LEGAL_VALID_ANSWERS:
            raise ValueError(
                f"Invalid answer_letter '{answer}'; must be one of {TW_LEGAL_VALID_ANSWER_STR}"
            )
        return cls(
            question=question,
            answer=answer,
            raw_question=raw_question,
            raw_answer=raw_answer,
            raw_choices=raw_choices,
            task_name=task_name,
        )


class TWLegalDatasetParser(HuggingFaceDatasetParser[TWLegalParseEntry]):
    """Parser for the Taiwan Legal Benchmark dataset."""

    _data_source = "lianghsun/tw-legal-benchmark-v1"
    _default_task = "default"
    _task_names = ["default"]

    def process_entry(
        self, row: dict[str, Any], task_name: str | None = None, **kwargs: Any
    ) -> TWLegalParseEntry:
        """Process a single Taiwan Legal Benchmark entry."""
        # Extract choices in order
        task = task_name or self._get_current_task(row)
        raw_choices = [row["A"], row["B"], row["C"], row["D"]]
        choices = "\n".join(
            f"{chr(65 + i)}. {choice}" for i, choice in enumerate(raw_choices)
        )
        raw_question = row["question"]
        raw_answer = row["answer"]

        question = f"Question: {raw_question}\n{choices}\nAnswer:"

        return TWLegalParseEntry.create(
            question=question,
            answer=raw_answer,
            raw_question=raw_question,
            raw_choices=raw_choices,
            raw_answer=raw_answer,
            task_name=task,
        )

    def get_dataset_description(self) -> DatasetDescription:
        """Returns description of the Taiwan Legal Benchmark dataset."""
        return DatasetDescription.create(
            name="Taiwan Legal Benchmark",
            language="Traditional Chinese",
            purpose="Evaluate models on Taiwan-specific legal knowledge and understanding",
            source="Taiwan Bar Examination questions",
            category=["Taiwan", "General Knowledge and Reasoning", "Legal"],
            format="Multiple choice questions (A/B/C/D)",
            characteristics=(
                "Contains questions from Taiwan's bar examination, testing understanding "
                "of Taiwan's legal system, terminology, and concepts"
            ),
            citation="""
                url={https://huggingface.co/datasets/lianghsun/tw-legal-benchmark-v1}
            """,
        )

    def get_evaluation_metrics(self) -> list[EvaluationMetric]:
        """Returns recommended evaluation metrics for Taiwan Legal Benchmark."""
        return [
            EvaluationMetric.create(
                name="accuracy",
                type="classification",
                description="Overall percentage of correctly answered legal questions",
                implementation="datasets.load_metric('accuracy')",
                primary=True,
            ),
        ]


if __name__ == "__main__":
    # Example usage
    parser = TWLegalDatasetParser()
    parser.load()
    parser.parse()

    # Get parsed data with correct type
    parsed_data = parser.get_parsed_data

    # Print example entry
    if parsed_data:
        example = parsed_data[0]
        print("\nExample parsed entry:")
        print(f"Question: {example.question}")
        print("Choices:")
        for i, choice in enumerate(example.raw_choices):
            print(f"{chr(65 + i)}. {choice}")
        print(f"Correct Answer: {example.answer}")
        print(f"Task Name: {example.task_name}")
