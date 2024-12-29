from dataclasses import dataclass
from typing import Any, ClassVar

from llmdataparser.base_parser import (
    DatasetDescription,
    EvaluationMetric,
    HuggingFaceDatasetParser,
    HuggingFaceParseEntry,
)
from llmdataparser.prompts import GSM8K_SYSTEM_PROMPT


@dataclass(frozen=True, kw_only=True, slots=True)
class GSM8KParseEntry(HuggingFaceParseEntry):
    """Custom entry class for GSM8K, with fields specific to this dataset parser."""

    solution: str
    numerical_answer: int | float
    task_name: str

    @classmethod
    def create(
        cls,
        prompt: str,
        answer: str,
        raw_question: str,
        raw_answer: str,
        solution: str,
        numerical_answer: int | float,
        task_name: str,
    ) -> "GSM8KParseEntry":
        return cls(
            prompt=prompt,
            answer=answer,
            raw_question=raw_question,
            raw_answer=raw_answer,
            solution=solution,
            numerical_answer=numerical_answer,
            task_name=task_name,
        )


class GSM8KDatasetParser(HuggingFaceDatasetParser[GSM8KParseEntry]):
    """Parser for the GSM8K dataset."""

    _data_source: ClassVar[str] = "openai/gsm8k"
    _task_names: ClassVar[list[str]] = ["main", "socratic"]
    _default_task: ClassVar[str] = "main"
    _default_system_prompt: ClassVar[str] = GSM8K_SYSTEM_PROMPT

    def process_entry(
        self, row: dict[str, Any], task_name: str | None = None, **kwargs: Any
    ) -> GSM8KParseEntry:
        """Process a single GSM8K entry."""
        task = task_name or self._get_current_task(row)
        raw_question = row["question"]
        raw_answer = row["answer"]

        # Extract numerical answer (always after '####' in GSM8K)
        numerical_str = raw_answer.split("####")[-1].strip().replace(",", "")
        # Convert string to number
        try:
            numerical_answer = float(numerical_str)
            if numerical_answer.is_integer():
                numerical_answer = int(numerical_answer)
        except ValueError:
            raise ValueError(f"Could not convert '{numerical_str}' to number")

        # Extract solution (everything before '####')
        solution = raw_answer.split("####")[0].strip()

        prompt = f"{self._system_prompt}\n{raw_question}"

        return GSM8KParseEntry.create(
            prompt=prompt,
            answer=str(numerical_answer),
            raw_question=raw_question,
            raw_answer=raw_answer,
            solution=solution,
            numerical_answer=numerical_answer,  # Now guaranteed to be int or float
            task_name=task,  # Guarantee non-None
        )

    def get_dataset_description(self) -> DatasetDescription:
        """Returns description of the GSM8K dataset."""
        return DatasetDescription.create(
            name="Grade School Math 8K (GSM8K)",
            purpose="Evaluate mathematical reasoning capabilities through word problems",
            source="OpenAI",
            language="English",
            format="Word problems with step-by-step solutions and numerical answers",
            characteristics=(
                "Collection of 8.5K grade school math word problems that require "
                "multi-step reasoning. Problems gradually increase in difficulty "
                "and cover basic arithmetic, word problems, and elementary algebra"
            ),
            citation="""@article{cobbe2021gsm8k,
title={Training Verifiers to Solve Math Word Problems},
author={Cobbe, Karl and Kosaraju, Vineet and Bavarian, Mohammad and Chen, Mark and Jun, Heewoo and Kaiser, Lukasz and Plappert, Matthias and Tworek, Jerry and Hilton, Jacob and Nakano, Reiichiro and Hesse, Christopher and Schulman, John},
journal={arXiv preprint arXiv:2110.14168},
year={2021}
}""",
        )

    def get_evaluation_metrics(self) -> list[EvaluationMetric]:
        """Returns recommended evaluation metrics for GSM8K."""
        return [
            EvaluationMetric.create(
                name="exact_match",
                type="string",
                description="Exact match comparison between predicted and correct numerical answers",
                implementation="custom_exact_match",
                primary=True,
            ),
            EvaluationMetric.create(
                name="solution_validity",
                type="text",
                description="Assessment of whether the solution steps are mathematically valid and complete",
                implementation="custom_solution_validator",
                primary=True,
            ),
            EvaluationMetric.create(
                name="step_accuracy",
                type="numerical",
                description="Accuracy of intermediate calculation steps (e.g., <<48/2=24>>)",
                implementation="custom_step_accuracy",
                primary=True,
            ),
            EvaluationMetric.create(
                name="step_count",
                type="numerical",
                description="Analysis of the number of reasoning steps in solutions",
                implementation="custom_step_counter",
                primary=False,
            ),
        ]


if __name__ == "__main__":
    from pprint import pprint

    parser = GSM8KDatasetParser()
    parser.load()
    parser.parse()

    parsed_data = parser.get_parsed_data
    pprint(parsed_data[0].prompt)
    pprint(parsed_data[0].answer)
    pprint(parsed_data[0].raw_question)
    pprint(parsed_data[0].raw_answer)
    pprint(parsed_data[0].solution)
    pprint(parsed_data[0].numerical_answer)
