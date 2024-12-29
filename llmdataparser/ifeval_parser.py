from dataclasses import dataclass
from typing import Any, ClassVar, List

from llmdataparser.base_parser import (
    DatasetDescription,
    EvaluationMetric,
    HuggingFaceDatasetParser,
    HuggingFaceParseEntry,
)
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

    def get_dataset_description(self) -> DatasetDescription:
        """Returns description of the IFEval dataset."""
        return DatasetDescription.create(
            name="IFEval",
            purpose="Evaluate instruction following capabilities through verifiable instructions",
            source="Google Research",
            language="English (BCP-47 en)",
            format="Verifiable instruction prompts with automated evaluation criteria",
            characteristics=(
                "Collection of approximately 500 verifiable instructions designed to evaluate "
                "language models' instruction-following capabilities. Instructions include "
                "specific, measurable criteria like 'write in more than 400 words' or "
                "'mention the keyword AI at least 3 times' that can be verified through "
                "automated heuristics. Used as a core benchmark in the Open LLM Leaderboard "
                "for evaluating chat or instruction fine-tuned language models."
            ),
            citation="""@misc{zhou2023instructionfollowingevaluationlargelanguage,
                title={Instruction-Following Evaluation for Large Language Models},
                author={Jeffrey Zhou and Tianjian Lu and Swaroop Mishra and Siddhartha Brahma and Sujoy Basu and Yi Luan and Denny Zhou and Le Hou},
                year={2023},
                eprint={2311.07911},
                archivePrefix={arXiv},
                primaryClass={cs.CL},
                url={https://arxiv.org/abs/2311.07911}
            }""",
        )

    def get_evaluation_metrics(self) -> list[EvaluationMetric]:
        """Returns recommended evaluation metrics for IFEval."""
        return [
            EvaluationMetric.create(
                name="format_compliance",
                type="text",
                description="Verifies if the output follows specified formatting rules (e.g., highlighting, bullet points, sections)",
                implementation="custom_format_checker",
                primary=True,
            ),
            EvaluationMetric.create(
                name="length_constraints",
                type="text",
                description="Checks if the response meets word, sentence, or paragraph count requirements",
                implementation="custom_length_validator",
                primary=True,
            ),
            EvaluationMetric.create(
                name="punctuation_rules",
                type="text",
                description="Validates adherence to punctuation constraints (e.g., no commas, specific endings)",
                implementation="custom_punctuation_checker",
                primary=True,
            ),
            EvaluationMetric.create(
                name="keyword_usage",
                type="text",
                description="Verifies correct usage of required keywords or avoidance of forbidden words",
                implementation="custom_keyword_validator",
                primary=False,
            ),
            EvaluationMetric.create(
                name="structural_requirements",
                type="text",
                description="Checks for specific structural elements like sections, paragraphs, or formatting patterns",
                implementation="custom_structure_validator",
                primary=False,
            ),
        ]


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
