from dataclasses import dataclass
from typing import Any, ClassVar

from llmdataparser.base_parser import (
    DatasetDescription,
    EvaluationMetric,
    HuggingFaceDatasetParser,
    HuggingFaceParseEntry,
)
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

    def get_dataset_description(self) -> DatasetDescription:
        """Returns description of the HumanEval dataset."""
        return DatasetDescription.create(
            name="HumanEval",
            purpose="Evaluate code generation capabilities through Python programming tasks",
            source="OpenAI",
            language="Python",
            format="Function signatures with docstrings and unit tests",
            characteristics=(
                "Collection of 164 hand-written Python programming problems. Each problem "
                "includes a function signature, docstring, example test cases, and hidden unit "
                "tests. Problems test basic programming, algorithms, and data structure skills"
            ),
            citation="""@article{chen2021codex,
title={Evaluating Large Language Models Trained on Code},
author={Mark Chen and Jerry Tworek and Heewoo Jun and Qiming Yuan and Henrique Ponde de Oliveira Pinto and Jared Kaplan and Harri Edwards and Yuri Burda and Nicholas Joseph and Greg Brockman and Alex Ray and Raul Puri and Gretchen Krueger and Michael Petrov and Heidy Khlaaf and Girish Sastry and Pamela Mishkin and Brooke Chan and Scott Gray and Nick Ryder and Mikhail Pavlov and Alethea Power and Lukasz Kaiser and Mohammad Bavarian and Clemens Winter and Philippe Tillet and Felipe Petroski Such and Dave Cummings and Matthias Plappert and Fotios Chantzis and Elizabeth Barnes and Ariel Herbert-Voss and William Hebgen Guss and Alex Nichol and Alex Paino and Nikolas Tezak and Jie Tang and Igor Babuschkin and Suchir Balaji and Shantanu Jain and William Saunders and Christopher Hesse and Andrew N. Carr and Jan Leike and Josh Achiam and Vedant Misra and Evan Morikawa and Alec Radford and Matthew Knight and Miles Brundage and Mira Murati and Katie Mayer and Peter Welinder and Bob McGrew and Dario Amodei and Sam McCandlish and Ilya Sutskever and Wojciech Zaremba},
year={2021},
eprint={2107.03374},
archivePrefix={arXiv},
primaryClass={cs.LG}
}""",
        )

    def get_evaluation_metrics(self) -> list[EvaluationMetric]:
        """Returns recommended evaluation metrics for HumanEval."""
        return [
            EvaluationMetric.create(
                name="pass@k",
                type="code",
                description="Probability that correct solution appears at least once in k samples",
                implementation="custom_pass_at_k",
                primary=True,
            ),
            EvaluationMetric.create(
                name="test_success_rate",
                type="code",
                description="Percentage of test cases passed by the generated solution",
                implementation="custom_test_executor",
                primary=False,
            ),
            EvaluationMetric.create(
                name="type_correctness",
                type="code",
                description="Verification of type hints and type safety in generated code",
                implementation="custom_type_checker",
                primary=False,
            ),
            EvaluationMetric.create(
                name="code_style",
                type="code",
                description="Compliance with Python best practices and PEP 8 guidelines",
                implementation="custom_style_checker",
                primary=False,
            ),
            EvaluationMetric.create(
                name="runtime_efficiency",
                type="code",
                description="Analysis of time and space complexity of the solution",
                implementation="custom_complexity_analyzer",
                primary=False,
            ),
        ]


class HumanEvalDatasetPlusParser(HumanEvalDatasetParser):
    """Parser for the enhanced HumanEval Plus dataset with 80x more comprehensive test coverage."""

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

    def get_dataset_description(self) -> DatasetDescription:
        """Returns description of the HumanEval Plus dataset."""
        return DatasetDescription.create(
            name="HumanEval Plus",
            purpose="Enhanced evaluation of code generation with 80x more test coverage",
            source="EvalPlus",
            language="Python",
            format="Function signatures with docstrings and comprehensive test suites",
            characteristics=(
                "Significantly enhanced version of HumanEval with 80x more test cases. "
                "Includes extensive edge cases, boundary conditions, stress tests, and "
                "error handling scenarios to rigorously evaluate code correctness and robustness. "
                "Each problem has been augmented with comprehensive testing to catch subtle bugs "
                "and ensure production-quality code generation."
            ),
            citation="""@inproceedings{evalplus,
title = {Is Your Code Generated by Chat{GPT} Really Correct? Rigorous Evaluation of Large Language Models for Code Generation},
author = {Liu, Jiawei and Xia, Chunqiu Steven and Wang, Yuyao and Zhang, Lingming},
booktitle = {Thirty-seventh Conference on Neural Information Processing Systems},
year = {2023},
url = {https://openreview.net/forum?id=1qvx610Cu7},
}""",
        )

    def get_evaluation_metrics(self) -> list[EvaluationMetric]:
        """Returns recommended evaluation metrics for HumanEval Plus."""
        return [
            EvaluationMetric.create(
                name="pass@k",
                type="code",
                description="Probability that correct solution appears at least once in k samples",
                implementation="custom_pass_at_k",
                primary=True,
            ),
            EvaluationMetric.create(
                name="test_coverage",
                type="code",
                description="Percentage of edge cases and stress tests passed by the solution",
                implementation="custom_coverage_checker",
                primary=False,
            ),
            EvaluationMetric.create(
                name="error_handling",
                type="code",
                description="Assessment of solution's robustness in handling invalid inputs and edge cases",
                implementation="custom_error_handler",
                primary=False,
            ),
            EvaluationMetric.create(
                name="performance_stress",
                type="code",
                description="Evaluation of solution performance under high load and stress conditions",
                implementation="custom_stress_tester",
                primary=False,
            ),
            EvaluationMetric.create(
                name="code_quality",
                type="code",
                description="Analysis of code readability, maintainability and adherence to Python best practices",
                implementation="custom_quality_checker",
                primary=False,
            ),
        ]


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
