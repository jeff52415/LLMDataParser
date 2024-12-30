from dataclasses import dataclass
from typing import Any, Final

from llmdataparser.base_parser import (
    DatasetDescription,
    EvaluationMetric,
    HuggingFaceDatasetParser,
    HuggingFaceParseEntry,
)
from llmdataparser.prompts import TMLU_SYSTEM_PROMPT

TMLU_VALID_ANSWERS: Final[set[str]] = {"A", "B", "C", "D"}
TMLU_VALID_ANSWER_STR: Final[str] = ", ".join(sorted(TMLU_VALID_ANSWERS))


@dataclass(frozen=True, kw_only=True, slots=True)
class TMLUParseEntry(HuggingFaceParseEntry):
    """Custom entry class for TMLU, with fields specific to this dataset parser."""

    raw_choices: list[str]
    explanation: str
    metadata: dict[str, Any]

    @classmethod
    def create(
        cls,
        prompt: str,
        answer: str,
        raw_question: str,
        raw_choices: list[str],
        raw_answer: str,
        task_name: str,
        explanation: str = "",
        metadata: dict[str, Any] = {},
    ) -> "TMLUParseEntry":
        if answer not in TMLU_VALID_ANSWERS:
            raise ValueError(
                f"Invalid answer_letter '{answer}'; must be one of {TMLU_VALID_ANSWER_STR}"
            )
        return cls(
            prompt=prompt,
            answer=answer,
            raw_question=raw_question,
            raw_answer=raw_answer,
            raw_choices=raw_choices,
            task_name=task_name,
            explanation=explanation,
            metadata=metadata,
        )


class TMLUDatasetParser(HuggingFaceDatasetParser[TMLUParseEntry]):
    """Parser for the TMLU dataset."""

    _data_source = "miulab/tmlu"
    _default_task = "AST_chinese"
    _task_names = [
        "AST_chinese",
        "AST_mathematics",
        "AST_biology",
        "AST_chemistry",
        "AST_physics",
        "AST_civics",
        "AST_geography",
        "AST_history",
        "GSAT_chinese",
        "GSAT_chemistry",
        "GSAT_biology",
        "GSAT_physics",
        "GSAT_earth_science",
        "GSAT_mathematics",
        "GSAT_geography",
        "GSAT_history",
        "GSAT_civics",
        "CAP_mathematics",
        "CAP_biology",
        "CAP_physics",
        "CAP_chemistry",
        "CAP_earth_science",
        "CAP_civics",
        "CAP_history",
        "CAP_geography",
        "CAP_chinese",
        "driving_rule",
        "basic_traditional_chinese_medicine",
        "clinical_traditional_chinese_medicine",
        "lawyer_qualification",
        "nutritionist",
        "tour_leader",
        "tour_guide",
        "taiwan_tourist_resources",
        "clinical_psychologist",
        "teacher_qualification",
        "accountant",
    ]
    _default_system_prompt = TMLU_SYSTEM_PROMPT

    def process_entry(
        self, row: dict[str, Any], task_name: str | None = None, **kwargs: Any
    ) -> TMLUParseEntry:
        """Process a single TMLU entry."""
        task = task_name or self._get_current_task(row)
        # Extract choices in order
        raw_choices = [row["A"], row["B"], row["C"], row["D"]]
        choices = "\n".join(
            f"{chr(65 + i)}. {choice}" for i, choice in enumerate(raw_choices)
        )
        raw_question = row["question"]
        raw_answer = row["answer"]
        explanation = row.get("explanation", "")
        metadata = row.get("metadata", {})

        prompt = f"{self._system_prompt}\nQuestion: {raw_question}\n{choices}\nAnswer:"

        return TMLUParseEntry.create(
            prompt=prompt,
            answer=raw_answer,
            raw_question=raw_question,
            raw_choices=raw_choices,
            raw_answer=raw_answer,
            task_name=task,
            explanation=explanation,
            metadata=metadata,
        )

    def get_dataset_description(self) -> DatasetDescription:
        """Returns description of the TMLU dataset."""
        return DatasetDescription.create(
            name="Taiwan Multiple-choice Language Understanding (TMLU)",
            language="Traditional Chinese",
            purpose="Evaluate models on Taiwan-specific educational and professional knowledge",
            source="Various Taiwan standardized tests and professional certifications",
            category=["Taiwan", "General Knowledge and Reasoning"],
            format="Multiple choice questions (A/B/C/D)",
            characteristics=(
                "Covers various subjects including Advanced Subjects Test (AST), "
                "General Scholastic Ability Test (GSAT), College Admission Practice (CAP), "
                "and professional certifications"
            ),
            citation="""@article{DBLP:journals/corr/abs-2403-20180,
    author       = {Po-Heng Chen and Sijia Cheng and Wei-Lin Chen and Yen-Ting Lin and Yun-Nung Chen},
    title        = {Measuring Taiwanese Mandarin Language Understanding},
    journal      = {CoRR},
    volume       = {abs/2403.20180},
    year         = {2024},
    url          = {https://doi.org/10.48550/arXiv.2403.20180},
    doi          = {10.48550/ARXIV.2403.20180},
    eprinttype   = {arXiv},
    eprint       = {2403.20180},
    timestamp    = {Wed, 10 Apr 2024 17:37:45 +0200},
    biburl       = {https://dblp.org/rec/journals/corr/abs-2403-20180.bib},
    bibsource    = {dblp computer science bibliography, https://dblp.org}
    }""",
        )

    def get_evaluation_metrics(self) -> list[EvaluationMetric]:
        """Returns recommended evaluation metrics for TMLU."""
        return [
            EvaluationMetric.create(
                name="accuracy",
                type="classification",
                description="Overall percentage of correctly answered questions",
                implementation="datasets.load_metric('accuracy')",
                primary=True,
            ),
            EvaluationMetric.create(
                name="per_subject_accuracy",
                type="classification",
                description="Accuracy broken down by subject areas (AST, GSAT, CAP, etc.)",
                implementation="custom_subject_accuracy",
                primary=True,
            ),
        ]


if __name__ == "__main__":
    # Example usage
    parser = TMLUDatasetParser()
    parser.load()
    parser.parse()

    # Get parsed data with correct type
    parsed_data = parser.get_parsed_data

    # Print example entry
    if parsed_data:
        example = parsed_data[0]
        print("\nExample parsed entry:")
        print(f"Task: {example.task_name}")
        print(f"Question: {example.raw_question}")
        print("Choices:")
        for i, choice in enumerate(example.raw_choices):
            print(f"{chr(65 + i)}. {choice}")
        print(f"Correct Answer: {example.answer}")
        if example.explanation:
            print(f"Explanation: {example.explanation}")
        print(f"Metadata: {example.metadata}")
