from dataclasses import dataclass
from typing import Any, ClassVar

from llmdataparser.base_parser import (
    DatasetDescription,
    EvaluationMetric,
    HuggingFaceDatasetParser,
    HuggingFaceParseEntry,
)
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

    def get_dataset_description(self) -> DatasetDescription:
        """Returns a description of the MBPP dataset."""
        return DatasetDescription.create(
            name="Mostly Basic Python Problems (MBPP)",
            purpose="A benchmark for evaluating code generation capabilities using entry-level Python programming problems",
            source="https://github.com/google-research/google-research/tree/master/mbpp",
            language="English and Python",
            category=["Programming"],
            format="Task descriptions in English with corresponding Python solutions and automated test cases",
            characteristics=(
                "Contains approximately 1,000 crowd-sourced Python programming problems "
                "designed for entry-level programmers. Problems cover programming fundamentals "
                "and standard library functionality. Each problem includes a task description, "
                "code solution, and 3 automated test cases. A subset of the data has been "
                "hand-verified by the authors."
            ),
            citation=(
                "@article{austin2021program,\n"
                "  title={Program Synthesis with Large Language Models},\n"
                "  author={Austin, Jacob and Odena, Augustus and Nye, Maxwell and Bosma, Maarten and Michalewski, Henryk and Dohan, David and Jiang, Ellen and Cai, Carrie and Terry, Michael and Le, Quoc and others},\n"
                "  journal={arXiv preprint arXiv:2108.07732},\n"
                "  year={2021}\n"
                "}"
            ),
            additional_info={
                "size": "~1,000 programming problems",
                "splits": "Available in full or sanitized versions",
                "test_coverage": "Each problem includes 3 automated test cases",
                "verification": "Subset of data has been hand-verified by authors",
            },
        )

    def get_evaluation_metrics(self) -> list[EvaluationMetric]:
        """Returns the recommended evaluation metrics for MBPP dataset."""
        return [
            EvaluationMetric.create(
                name="pass@k",
                type="code_evaluation",
                description="Percentage of problems where at least one solution in k generations passes all test cases",
                implementation="custom_pass_at_k",
                primary=True,
            ),
            EvaluationMetric.create(
                name="test_case_success_rate",
                type="code_evaluation",
                description="Percentage of test cases passed across all problems",
                implementation="custom_test_success_rate",
                primary=False,
            ),
            EvaluationMetric.create(
                name="syntax_validity",
                type="code_evaluation",
                description="Verifies that generated code is syntactically valid Python",
                implementation="custom_syntax_check",
                primary=False,
            ),
            EvaluationMetric.create(
                name="code_similarity",
                type="similarity",
                description="Similarity between generated code and reference solution",
                implementation="evaluate.load('code_eval')",
                primary=False,
            ),
        ]


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
