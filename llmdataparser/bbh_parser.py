from dataclasses import dataclass
from typing import Any, ClassVar, List

from llmdataparser.base_parser import (
    DatasetDescription,
    EvaluationMetric,
    HuggingFaceDatasetParser,
    HuggingFaceParseEntry,
)
from llmdataparser.prompts import BBH_SYSTEM_PROMPT  # You'll need to create this


@dataclass(frozen=True, kw_only=True, slots=True)
class BBHParseEntry(HuggingFaceParseEntry):
    """Custom entry class for BBH (Big Bench Hard), with fields specific to this dataset."""

    @classmethod
    def create(
        cls,
        prompt: str,
        answer: str,
        raw_question: str,
        raw_answer: str,
        task_name: str,
    ) -> "BBHParseEntry":
        return cls(
            prompt=prompt,
            answer=answer,
            raw_question=raw_question,
            raw_answer=raw_answer,
            task_name=task_name,
        )


class BBHDatasetParser(HuggingFaceDatasetParser[BBHParseEntry]):
    """Parser for the Big Bench Hard dataset."""

    _data_source: ClassVar[str] = "lukaemon/bbh"
    _task_names: ClassVar[list[str]] = [
        "boolean_expressions",
        "causal_judgement",
        "date_understanding",
        "disambiguation_qa",
        "dyck_languages",
        "formal_fallacies",
        "geometric_shapes",
        "hyperbaton",
        "logical_deduction_five_objects",
        "logical_deduction_seven_objects",
        "logical_deduction_three_objects",
        "movie_recommendation",
        "multistep_arithmetic_two",
        "navigate",
        "object_counting",
        "penguins_in_a_table",
        "reasoning_about_colored_objects",
        "ruin_names",
        "salient_translation_error_detection",
        "snarks",
        "sports_understanding",
        "temporal_sequences",
        "tracking_shuffled_objects_five_objects",
        "tracking_shuffled_objects_seven_objects",
        "tracking_shuffled_objects_three_objects",
        "web_of_lies",
        "word_sorting",
    ]
    _default_task: ClassVar[str] = "reasoning_about_colored_objects"
    _default_system_prompt: ClassVar[str] = BBH_SYSTEM_PROMPT

    def process_entry(
        self, row: dict[str, Any], task_name: str | None = None, **kwargs: Any
    ) -> BBHParseEntry:
        """Process a single BBH entry."""
        raw_question = row["input"]
        raw_answer = row["target"]

        # Remove parentheses from the answer
        clean_answer = raw_answer.strip("()")

        # Combine system prompt with the question
        prompt = f"{self._system_prompt}\n\n{raw_question}"

        # Use task_name if provided, otherwise use default
        task = task_name or self._get_current_task(row)

        return BBHParseEntry.create(
            prompt=prompt,
            answer=clean_answer,
            raw_question=raw_question,
            raw_answer=raw_answer,
            task_name=task,
        )

    def get_dataset_description(self) -> DatasetDescription:
        """Returns a description of the Big Bench Hard dataset."""
        return DatasetDescription.create(
            name="Big Bench Hard (BBH)",
            purpose="A curated subset of 23 challenging BIG-Bench tasks where language models initially performed below average human-rater performance",
            source="https://github.com/suzgunmirac/BIG-Bench-Hard",
            language="English",
            format="Multiple choice questions with single correct answers",
            characteristics=(
                "Tasks require complex multi-step reasoning and were selected based on "
                "initial model performance below human baseline. Performance can be "
                "significantly improved through chain-of-thought prompting. The dataset "
                "includes 23 core tasks plus additional related tasks."
            ),
            citation=(
                "@article{suzgun2022challenging,\n"
                "  title={Challenging BIG-Bench Tasks and Whether Chain-of-Thought Can Solve Them},\n"
                '  author={Suzgun, Mirac and Scales, Nathan and Sch{"a}rli, Nathanael and Gehrmann, Sebastian and Tay, Yi and Chung, Hyung Won and Chowdhery, Aakanksha and Le, Quoc V and Chi, Ed H and Zhou, Denny and Wei, Jason},\n'
                "  journal={arXiv preprint arXiv:2210.09261},\n"
                "  year={2022}\n"
                "}"
            ),
            additional_info={
                "model_performance": (
                    "With chain-of-thought prompting, PaLM surpassed human performance on "
                    "10/23 tasks, while Codex surpassed human performance on 17/23 tasks"
                ),
                "size": "6.5k examples across 27 tasks (23 core + 4 related)",
            },
        )

    def get_evaluation_metrics(self) -> List[EvaluationMetric]:
        """Returns the recommended evaluation metrics for BBH dataset."""
        return [
            EvaluationMetric.create(
                name="accuracy",
                type="classification",
                description="Proportion of exactly correct answers (after stripping parentheses)",
                implementation="evaluate.load('accuracy')",
                primary=True,
            ),
            EvaluationMetric.create(
                name="human_eval_delta",
                type="comparison",
                description="Difference between model accuracy and average human-rater performance baseline",
                implementation="custom_human_baseline_comparison",
                primary=True,
            ),
            EvaluationMetric.create(
                name="per_task_accuracy",
                type="classification",
                description="Accuracy broken down by individual reasoning tasks",
                implementation="custom_task_accuracy",
                primary=False,
            ),
            EvaluationMetric.create(
                name="exact_match",
                type="string_match",
                description="Strict exact match between predicted and target answers",
                implementation="evaluate.load('exact_match')",
                primary=False,
            ),
        ]


if __name__ == "__main__":
    # Example usage
    parser = BBHDatasetParser()

    # Load the dataset with a specific task
    parser.load(task_name="reasoning_about_colored_objects")

    # Parse all splits
    parser.parse()

    # Get parsed data
    parsed_data = parser.get_parsed_data

    # Print example entry
    if parsed_data:
        example = parsed_data[0]
        print("\nExample parsed entry:")
        print(f"Task: {example.task_name}")
        print(f"Question: {example.raw_question}")
        print(f"Answer: {example.answer}")
