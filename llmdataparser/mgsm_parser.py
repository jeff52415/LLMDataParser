from dataclasses import dataclass
from typing import Any, ClassVar

from llmdataparser.base_parser import (
    DatasetDescription,
    EvaluationMetric,
    HuggingFaceDatasetParser,
    HuggingFaceParseEntry,
)
from llmdataparser.prompts import MGSM_SYSTEM_PROMPT


@dataclass(frozen=True, kw_only=True, slots=True)
class MGSMParseEntry(HuggingFaceParseEntry):
    """Custom entry class for MGSM, with fields specific to this dataset parser."""

    numerical_answer: int | float
    equation_solution: str | None
    language: str

    @classmethod
    def create(
        cls,
        prompt: str,
        answer: str,
        raw_question: str,
        raw_answer: str,
        numerical_answer: int | float,
        equation_solution: str | None,
        task_name: str,
        language: str,
    ) -> "MGSMParseEntry":
        return cls(
            prompt=prompt,
            answer=answer,
            raw_question=raw_question,
            raw_answer=raw_answer,
            numerical_answer=numerical_answer,
            equation_solution=equation_solution,
            task_name=task_name,
            language=language,
        )


class MGSMDatasetParser(HuggingFaceDatasetParser[MGSMParseEntry]):
    """Parser for the MGSM (Multilingual Grade School Math) dataset."""

    _data_source: ClassVar[str] = "juletxara/mgsm"
    _default_task: ClassVar[str] = "en"
    _task_names: ClassVar[list[str]] = [
        "bn",
        "de",
        "en",
        "es",
        "fr",
        "ja",
        "ru",
        "sw",
        "te",
        "th",
        "zh",
    ]
    _default_system_prompt: ClassVar[str] = MGSM_SYSTEM_PROMPT

    def process_entry(
        self, row: dict[str, Any], task_name: str | None = None, **kwargs: Any
    ) -> MGSMParseEntry:
        """
        Process a single MGSM entry.

        Args:
            row: Dictionary containing the MGSM entry fields
            task_name: Language code for the current task

        Returns:
            MGSMParseEntry: Processed entry with prompt, answer, and metadata
        """
        task = task_name or self._get_current_task(row)
        raw_question = row["question"]
        raw_answer = row["answer"] if row["answer"] else ""
        numerical_answer = row["answer_number"]
        equation_solution = row["equation_solution"]

        # Construct the prompt with the system prompt and question
        prompt = f"{self._system_prompt}\n{raw_question}"

        # Use numerical answer as string for the answer field if no detailed answer is provided
        answer = raw_answer if raw_answer else str(numerical_answer)

        return MGSMParseEntry.create(
            prompt=prompt,
            answer=answer,
            raw_question=raw_question,
            raw_answer=raw_answer,
            numerical_answer=numerical_answer,
            equation_solution=equation_solution,
            task_name=task,
            language=task,
        )

    def get_dataset_description(self) -> DatasetDescription:
        """Returns a description of the Multilingual Grade School Math dataset."""
        return DatasetDescription.create(
            name="Multilingual Grade School Math (MGSM)",
            purpose="Evaluate multilingual chain-of-thought reasoning capabilities in mathematical problem solving",
            source="https://huggingface.co/datasets/juletxara/mgsm",
            language="Multilingual (11 languages)",
            format="Word problems with numerical answers and solution steps",
            category=["Math", "MultiLingual"],
            characteristics=(
                "Human-translated version of 250 GSM8K problems into 10 additional languages. "
                "Each problem includes the original question from GSM8K, its translations, "
                "numerical answer, and solution steps. The benchmark is designed to evaluate "
                "language models' ability to perform mathematical reasoning across different languages."
            ),
            citation="""@misc{shi2022language,
    title={Language Models are Multilingual Chain-of-Thought Reasoners},
    author={Freda Shi and Mirac Suzgun and Markus Freitag and Xuezhi Wang and Suraj Srivats and Soroush Vosoughi and Hyung Won Chung and Yi Tay and Sebastian Ruder and Denny Zhou and Dipanjan Das and Jason Wei},
    year={2022},
    eprint={2210.03057},
    archivePrefix={arXiv},
    primaryClass={cs.CL}
    }
@article{cobbe2021gsm8k,
    title={Training Verifiers to Solve Math Word Problems},
    author={Cobbe, Karl and Kosaraju, Vineet and Bavarian, Mohammad and Chen, Mark and Jun, Heewoo and Kaiser, Lukasz and Plappert, Matthias and Tworek, Jerry and Hilton, Jacob and Nakano, Reiichiro and Hesse, Christopher and Schulman, John},
    journal={arXiv preprint arXiv:2110.14168},
    year={2021}
}""",
            additional_info={
                "languages": [
                    "Bengali",
                    "German",
                    "English",
                    "Spanish",
                    "French",
                    "Japanese",
                    "Russian",
                    "Swahili",
                    "Telugu",
                    "Thai",
                    "Chinese",
                ],
                "size": "250 problems translated into each language",
                "base_dataset": "GSM8K (Grade School Math 8K)",
            },
        )

    def get_evaluation_metrics(self) -> list[EvaluationMetric]:
        """Returns the recommended evaluation metrics for MGSM dataset."""
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
                name="cross_lingual_consistency",
                type="comparison",
                description="Consistency of model performance across different language versions of the same problem",
                implementation="custom_language_comparator",
                primary=False,
            ),
        ]


if __name__ == "__main__":
    from pprint import pprint

    parser = MGSMDatasetParser()
    parser.load(task_name="en")  # Load French dataset
    parser.parse()

    parsed_data = parser.get_parsed_data
    pprint(parsed_data[0].prompt)
    pprint(parsed_data[0].answer)
    pprint(parsed_data[0].raw_question)
    pprint(parsed_data[0].numerical_answer)
    pprint(parsed_data[0].language)
