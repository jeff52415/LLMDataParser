from dataclasses import dataclass
from typing import Any, Final

from llmdataparser.base_parser import (
    DatasetDescription,
    EvaluationMetric,
    HuggingFaceDatasetParser,
    HuggingFaceParseEntry,
)
from llmdataparser.prompts import MMLU_PRO_SYSTEM_PROMPT, MMLU_SYSTEM_PROMPT

MMLU_VALID_ANSWERS: Final[set[str]] = {"A", "B", "C", "D"}
MMLU_PRO_VALID_ANSWERS: Final[set[str]] = {
    "A",
    "B",
    "C",
    "D",
    "E",
    "F",
    "G",
    "H",
    "I",
    "J",
}
MMLU_VALID_ANSWER_STR: Final[str] = ", ".join(sorted(MMLU_VALID_ANSWERS))
MMLU_PRO_VALID_ANSWER_STR: Final[str] = ", ".join(sorted(MMLU_PRO_VALID_ANSWERS))


@dataclass(frozen=True, kw_only=True, slots=True)
class MMLUParseEntry(HuggingFaceParseEntry):
    """Custom entry class for MMLU, with fields specific to this dataset parser."""

    raw_choices: list[str]
    task_name: str

    @classmethod
    def create(
        cls,
        prompt: str,
        answer: str,
        raw_question: str,
        raw_choices: list[str],
        raw_answer: str,
        task_name: str,
    ) -> "MMLUParseEntry":
        if answer not in MMLU_VALID_ANSWERS:
            raise ValueError(
                f"Invalid answer_letter '{answer}'; must be one of {MMLU_VALID_ANSWER_STR}"
            )
        if not task_name:
            raise ValueError("Task name cannot be empty")
        return cls(
            prompt=prompt,
            answer=answer,
            raw_question=raw_question,
            raw_answer=raw_answer,
            raw_choices=raw_choices,
            task_name=task_name,
        )


@dataclass(frozen=True, kw_only=True, slots=True)
class MMLUProParseEntry(HuggingFaceParseEntry):
    """Custom entry class for MMLU, with fields specific to this dataset parser."""

    raw_choices: list[str]
    task_name: str

    @classmethod
    def create(
        cls,
        prompt: str,
        answer: str,
        raw_question: str,
        raw_choices: list[str],
        raw_answer: str,
        task_name: str,
    ) -> "MMLUProParseEntry":
        if answer not in MMLU_PRO_VALID_ANSWERS:
            raise ValueError(
                f"Invalid answer_letter '{answer}'; must be one of {MMLU_PRO_VALID_ANSWER_STR}"
            )
        if not task_name:
            raise ValueError("Task name cannot be empty")
        return cls(
            prompt=prompt,
            answer=answer,
            raw_question=raw_question,
            raw_choices=raw_choices,
            raw_answer=raw_answer,
            task_name=task_name,
        )


class MMLUDatasetParser(HuggingFaceDatasetParser[MMLUParseEntry]):
    """Base class for MMLU dataset parsers with common functionality."""

    _default_system_prompt = MMLU_SYSTEM_PROMPT

    def _get_task_from_entry(self, data_entry: dict[str, Any]) -> str:
        """Get the task name from the data entry or default task name."""
        task_name: str = data_entry.get("subject", "")
        return task_name if task_name else (self._current_task or self._default_task)

    def process_entry(
        self, row: dict[str, Any], task_name: str | None = None, **kwargs: Any
    ) -> MMLUParseEntry:
        """
        Generate a prompt and expected answer from the given row.

        Args:
            row: A data point to be formatted.
            task_name: Optional task name for the entry.
            **kwargs: Additional keyword arguments.

        Returns:
            MMLUParseEntry: The formatted entry object.
        """
        task = task_name or self._get_current_task(row)
        # Ensure task is not None
        final_task = task or self._default_task

        choices = "\n".join(
            f"{chr(65 + i)}. {choice}" for i, choice in enumerate(row["choices"])
        )
        raw_question = row["question"]
        raw_choices = row["choices"]
        raw_answer = str(row["answer"])  # Ensure raw_answer is a string

        prompt = f"{self._system_prompt}\nQuestion: {raw_question}\n{choices}\nAnswer:"
        answer_letter = chr(65 + int(raw_answer))  # Convert index to 'A', 'B', 'C', 'D'

        return MMLUParseEntry.create(
            prompt=prompt,
            answer=answer_letter,
            raw_question=raw_question,
            raw_choices=raw_choices,
            raw_answer=raw_answer,
            task_name=final_task,
        )


class BaseMMLUDatasetParser(MMLUDatasetParser):
    """Parser for the original MMLU dataset."""

    _data_source = "cais/mmlu"
    _default_task = "all"
    _task_names = [
        "abstract_algebra",
        "anatomy",
        "astronomy",
        "business_ethics",
        "clinical_knowledge",
        "college_biology",
        "college_chemistry",
        "college_computer_science",
        "college_mathematics",
        "college_medicine",
        "college_physics",
        "computer_security",
        "conceptual_physics",
        "econometrics",
        "electrical_engineering",
        "elementary_mathematics",
        "formal_logic",
        "global_facts",
        "high_school_biology",
        "high_school_chemistry",
        "high_school_computer_science",
        "high_school_european_history",
        "high_school_geography",
        "high_school_government_and_politics",
        "high_school_macroeconomics",
        "high_school_mathematics",
        "high_school_microeconomics",
        "high_school_physics",
        "high_school_psychology",
        "high_school_statistics",
        "high_school_us_history",
        "high_school_world_history",
        "human_aging",
        "human_sexuality",
        "international_law",
        "jurisprudence",
        "logical_fallacies",
        "machine_learning",
        "management",
        "marketing",
        "medical_genetics",
        "miscellaneous",
        "moral_disputes",
        "moral_scenarios",
        "nutrition",
        "philosophy",
        "prehistory",
        "professional_accounting",
        "professional_law",
        "professional_medicine",
        "professional_psychology",
        "public_relations",
        "security_studies",
        "sociology",
        "us_foreign_policy",
        "virology",
        "world_religions",
    ]

    def get_dataset_description(self) -> DatasetDescription:
        """Returns a description of the MMLU dataset."""
        return DatasetDescription.create(
            name="Massive Multitask Language Understanding (MMLU)",
            purpose="Evaluate models' extensive world knowledge and problem-solving abilities across diverse branches of knowledge",
            source="https://huggingface.co/datasets/cais/mmlu",
            language="English",
            category=["General Knowledge and Reasoning"],
            format="Multiple choice questions with four options (A, B, C, D)",
            characteristics=(
                "Comprehensive evaluation benchmark spanning humanities, social sciences, hard sciences, "
                "and other essential areas of knowledge. The test includes 57 subjects such as "
                "elementary mathematics, US history, computer science, and law. Success on this test "
                "requires both extensive world knowledge and strong problem-solving capabilities."
            ),
            citation="""@article{hendryckstest2021,
    title={Measuring Massive Multitask Language Understanding},
    author={Dan Hendrycks and Collin Burns and Steven Basart and Andy Zou and Mantas Mazeika and Dawn Song and Jacob Steinhardt},
    journal={Proceedings of the International Conference on Learning Representations (ICLR)},
    year={2021}
    }
@article{hendrycks2021ethics,
    title={Aligning AI With Shared Human Values},
    author={Dan Hendrycks and Collin Burns and Steven Basart and Andrew Critch and Jerry Li and Dawn Song and Jacob Steinhardt},
    journal={Proceedings of the International Conference on Learning Representations (ICLR)},
    year={2021}
    }""",
            additional_info={
                "subjects": "57 tasks/subjects",
                "categories": [
                    "Humanities",
                    "Social Sciences",
                    "Hard Sciences",
                    "Other",
                ],
                "example_subjects": [
                    "Elementary Mathematics",
                    "US History",
                    "Computer Science",
                    "Law",
                ],
                "requirements": [
                    "Extensive world knowledge",
                    "Problem solving ability",
                ],
            },
        )

    def get_evaluation_metrics(self) -> list[EvaluationMetric]:
        """Returns the recommended evaluation metrics for MMLU dataset."""
        return [
            EvaluationMetric.create(
                name="accuracy",
                type="classification",
                description="Proportion of correctly answered multiple-choice questions (exact match with A, B, C, D)",
                implementation="evaluate.load('accuracy')",
                primary=True,
            ),
            EvaluationMetric.create(
                name="subject_accuracy",
                type="classification",
                description="Per-subject accuracy scores across all 57 tasks",
                implementation="custom_subject_accuracy",
                primary=True,
            ),
            EvaluationMetric.create(
                name="category_accuracy",
                type="classification",
                description="Accuracy grouped by major categories (Humanities, Social Sciences, Hard Sciences, Other)",
                implementation="custom_category_accuracy",
                primary=True,
            ),
            EvaluationMetric.create(
                name="task_correlation",
                type="analysis",
                description="Analysis of performance correlations between different subjects/tasks",
                implementation="custom_task_correlation",
                primary=False,
            ),
        ]


class MMLUReduxDatasetParser(MMLUDatasetParser):
    """Parser for the MMLU Redux dataset."""

    _data_source = "edinburgh-dawg/mmlu-redux"
    _default_task = "anatomy"
    _task_names = [
        "anatomy",
        "astronomy",
        "business_ethics",
        "clinical_knowledge",
        "college_chemistry",
        "college_computer_science",
        "college_mathematics",
        "college_medicine",
        "college_physics",
        "conceptual_physics",
        "econometrics",
        "electrical_engineering",
        "formal_logic",
        "global_facts",
        "high_school_chemistry",
        "high_school_geography",
        "high_school_macroeconomics",
        "high_school_mathematics",
        "high_school_physics",
        "high_school_statistics",
        "high_school_us_history",
        "human_aging",
        "logical_fallacies",
        "machine_learning",
        "miscellaneous",
        "philosophy",
        "professional_accounting",
        "professional_law",
        "public_relations",
        "virology",
    ]

    def get_dataset_description(self) -> DatasetDescription:
        """Returns description of the MMLU Redux dataset."""
        return DatasetDescription.create(
            name="MMLU Redux",
            purpose="Provide a manually re-annotated subset of MMLU with error analysis and corrections",
            source="https://huggingface.co/datasets/edinburgh-dawg/mmlu-redux",
            language="English",
            format="Multiple choice questions with four options (A, B, C, D)",
            category=["General Knowledge and Reasoning"],
            characteristics=(
                "A carefully curated subset of 3,000 questions across 30 MMLU subjects, "
                "manually re-annotated to identify and classify various types of errors. "
                "The dataset maintains the original questions but provides additional "
                "error annotations and corrections based on expert review and verification "
                "against credible sources."
            ),
            citation="""@misc{gema2024mmlu,
    title={Are We Done with MMLU?},
    author={Aryo Pradipta Gema and Joshua Ong Jun Leang and Giwon Hong and Alessio Devoto and Alberto Carlo Maria Mancino and Rohit Saxena and Xuanli He and Yu Zhao and Xiaotang Du and Mohammad Reza Ghasemi Madani and Claire Barale and Robert McHardy and Joshua Harris and Jean Kaddour and Emile van Krieken and Pasquale Minervini},
    year={2024},
    eprint={2406.04127},
    archivePrefix={arXiv},
    primaryClass={cs.CL}
    }""",
            additional_info={
                "size": "3,000 questions (100 per subject)",
                "subjects": "30 MMLU subjects",
                "license": "CC-BY-4.0",
                "error_types": {
                    "Question Assessment": [
                        "Bad Question Clarity",
                        "Bad Options Clarity",
                    ],
                    "Ground Truth Verification": [
                        "No Correct Answer",
                        "Multiple Correct Answers",
                        "Wrong Ground Truth",
                    ],
                },
                "verification_process": "Expert review with source verification",
                "base_dataset": "cais/mmlu",
            },
        )

    def get_evaluation_metrics(self) -> list[EvaluationMetric]:
        """Returns the recommended evaluation metrics for MMLU Redux dataset."""
        return [
            EvaluationMetric.create(
                name="accuracy",
                type="classification",
                description="Proportion of correctly answered multiple-choice questions (exact match with A, B, C, D)",
                implementation="evaluate.load('accuracy')",
                primary=True,
            ),
            EvaluationMetric.create(
                name="subject_accuracy",
                type="classification",
                description="Per-subject accuracy scores across 30 subjects (100 questions each)",
                implementation="custom_subject_accuracy",
                primary=True,
            ),
            EvaluationMetric.create(
                name="question_clarity",
                type="analysis",
                description="Analysis of performance on questions with different clarity issues",
                implementation="custom_clarity_analysis",
                primary=False,
            ),
        ]


class TMMLUPlusDatasetParser(MMLUDatasetParser):
    """Parser for the TMMLU+ dataset."""

    _data_source = "ikala/tmmluplus"
    _default_task = "taiwanese_hokkien"
    _task_names = [
        "engineering_math",
        "dentistry",
        "traditional_chinese_medicine_clinical_medicine",
        "clinical_psychology",
        "technical",
        "culinary_skills",
        "mechanical",
        "logic_reasoning",
        "real_estate",
        "general_principles_of_law",
        "finance_banking",
        "anti_money_laundering",
        "ttqav2",
        "marketing_management",
        "business_management",
        "organic_chemistry",
        "advance_chemistry",
        "physics",
        "secondary_physics",
        "human_behavior",
        "national_protection",
        "jce_humanities",
        "politic_science",
        "agriculture",
        "official_document_management",
        "financial_analysis",
        "pharmacy",
        "educational_psychology",
        "statistics_and_machine_learning",
        "management_accounting",
        "introduction_to_law",
        "computer_science",
        "veterinary_pathology",
        "accounting",
        "fire_science",
        "optometry",
        "insurance_studies",
        "pharmacology",
        "taxation",
        "trust_practice",
        "geography_of_taiwan",
        "physical_education",
        "auditing",
        "administrative_law",
        "education_(profession_level)",
        "economics",
        "veterinary_pharmacology",
        "nautical_science",
        "occupational_therapy_for_psychological_disorders",
        "basic_medical_science",
        "macroeconomics",
        "trade",
        "chinese_language_and_literature",
        "tve_design",
        "junior_science_exam",
        "junior_math_exam",
        "junior_chinese_exam",
        "junior_social_studies",
        "tve_mathematics",
        "tve_chinese_language",
        "tve_natural_sciences",
        "junior_chemistry",
        "music",
        "education",
        "three_principles_of_people",
        "taiwanese_hokkien",
    ]

    def process_entry(
        self, row: dict[str, Any], task_name: str | None = None, **kwargs: Any
    ) -> MMLUParseEntry:
        """Process a single TMMLU+ entry."""
        # Extract choices in order
        raw_choices = [row["A"], row["B"], row["C"], row["D"]]
        choices = "\n".join(
            f"{chr(65 + i)}. {choice}" for i, choice in enumerate(raw_choices)
        )
        raw_question = row["question"]
        raw_answer = row["answer"]

        prompt = f"{self._system_prompt}\nQuestion: {raw_question}\n{choices}\nAnswer:"
        task = task_name or self._get_current_task(row)

        return MMLUParseEntry.create(
            prompt, raw_answer, raw_question, raw_choices, raw_answer, task
        )

    def get_dataset_description(self) -> DatasetDescription:
        """Returns description of the TMMLU+ dataset."""
        return DatasetDescription.create(
            name="Traditional Chinese Massive Multitask Language Understanding Plus (TMMLU+)",
            purpose="Evaluate language models' understanding and reasoning capabilities in Traditional Chinese across diverse subjects",
            source="https://huggingface.co/datasets/ikala/tmmluplus",
            language="Traditional Chinese",
            category=["General Knowledge and Reasoning", "Taiwan"],
            format="Multiple choice questions with four options (A, B, C, D)",
            characteristics=(
                "A comprehensive evaluation benchmark featuring 66 subjects from elementary "
                "to professional level. The dataset is six times larger than the original TMMLU "
                "and provides more balanced subject coverage. Includes benchmark results from "
                "both closed-source models and 20 open-weight Chinese language models with "
                "parameters ranging from 1.8B to 72B."
            ),
            citation="""@article{ikala2024improved,
    title={An Improved Traditional Chinese Evaluation Suite for Foundation Model},
    author={Tam, Zhi-Rui and Pai, Ya-Ting and Lee, Yen-Wei and Cheng, Sega and Shuai, Hong-Han},
    journal={arXiv preprint arXiv:2403.01858},
    year={2024}
    }""",
            additional_info={
                "subjects": "66 diverse subjects",
                "difficulty_levels": ["Elementary", "Secondary", "Professional"],
                "model_benchmarks": {
                    "model_types": ["Closed-source models", "Open-weight Chinese LLMs"],
                    "parameter_range": "1.8B - 72B",
                },
                "comparison": "6x larger than original TMMLU",
                "script": "Traditional Chinese",
            },
        )

    def get_evaluation_metrics(self) -> list[EvaluationMetric]:
        """Returns the recommended evaluation metrics for TMMLU+ dataset."""
        return [
            EvaluationMetric.create(
                name="accuracy",
                type="classification",
                description="Overall percentage of correctly answered multiple-choice questions",
                implementation="evaluate.load('accuracy')",
                primary=True,
            ),
            EvaluationMetric.create(
                name="subject_accuracy",
                type="classification",
                description="Per-subject accuracy scores across all 66 subjects",
                implementation="custom_subject_accuracy",
                primary=True,
            ),
            EvaluationMetric.create(
                name="difficulty_analysis",
                type="classification",
                description="Performance analysis across different difficulty levels (elementary to professional)",
                implementation="custom_difficulty_analysis",
                primary=False,
            ),
        ]


class MMLUProDatasetParser(HuggingFaceDatasetParser[MMLUProParseEntry]):
    """Parser for the MMLU Pro dataset."""

    _data_source = "TIGER-Lab/MMLU-Pro"
    _default_task = "default"
    _task_names = ["default"]
    _hidden_task_names = [
        "math",
        "physics",
        "chemistry",
        "law",
        "engineering",
        "other",
        "economics",
        "health",
        "psychology",
        "business",
        "biology",
        "philosophy",
        "computer_science",
        "history",
    ]
    _default_system_prompt = MMLU_PRO_SYSTEM_PROMPT

    def _get_task_from_entry(self, data_entry: dict[str, Any]) -> str:
        """Get the task name from the data entry or default task name."""
        if data_entry is not None:
            task_name: str = data_entry.get("category", "")
            if task_name:
                return task_name
        return self._current_task or self._default_task

    def process_entry(
        self, row: dict[str, Any], task_name: str | None = None, **kwargs: Any
    ) -> MMLUProParseEntry:
        """
        Generate a prompt and expected answer from the given row.

        Args:
            row (dict[str, Any]): A data point to be formatted with MMLU Pro specific structure
                containing 'question', 'options', 'answer', and 'answer_index' keys.

        Returns:
            MMLUParseEntry: The formatted entry object.
        """
        task = task_name or self._get_current_task(row)
        # Ensure task is not None
        final_task = task or self._default_task

        # Extract choices in order
        raw_choices = row["options"]
        choices = "\n".join(
            f"{chr(65 + i)}. {choice}" for i, choice in enumerate(raw_choices)
        )
        raw_question = row["question"]
        raw_answer = row["answer"]
        answer_index = row["answer_index"]

        prompt = f"{self._system_prompt}\nQuestion: {raw_question}\n{choices}\nAnswer:"
        answer_letter = chr(
            65 + answer_index
        )  # Convert index to 'A', 'B', 'C', 'D', etc.

        return MMLUProParseEntry.create(
            prompt, answer_letter, raw_question, raw_choices, raw_answer, final_task
        )

    def get_dataset_description(self) -> DatasetDescription:
        """Returns description of the MMLU Pro dataset."""
        return DatasetDescription.create(
            name="MMLU Pro",
            purpose="Provide a more robust and challenging multi-task language understanding benchmark with enhanced reasoning requirements",
            source="https://huggingface.co/datasets/TIGER-Lab/MMLU-Pro",
            language="English",
            category=["General Knowledge and Reasoning", "Advanced Reasoning"],
            format="Multiple choice questions with up to 10 options (expanded from original 4)",
            characteristics=(
                "A more challenging version of MMLU containing 12K complex questions across various "
                "disciplines. Features increased number of options (up to 10), stronger focus on "
                "reasoning over pure knowledge, and reduced sensitivity to prompt variations. "
                "Questions are sourced from original MMLU, STEM websites, TheoremQA, and SciBench, "
                "with expert review and GPT-4 assisted distractor generation."
            ),
            citation="""@article{wang2024mmlu,
    title={Mmlu-pro: A more robust and challenging multi-task language understanding benchmark},
    author={Wang, Yubo and Ma, Xueguang and Zhang, Ge and Ni, Yuansheng and Chandra, Abhranil and Guo, Shiguang and Ren, Weiming and Arulraj, Aaran and He, Xuan and Jiang, Ziyan and others},
    journal={arXiv preprint arXiv:2406.01574},
    year={2024}
    }""",
            additional_info={
                "size": "12,000 complex questions",
                "options": "Up to 10 choices per question",
                "sources": [
                    "Original MMLU (filtered)",
                    "STEM Website",
                    "TheoremQA",
                    "SciBench",
                ],
                "enhanced_subjects": [
                    "Biology",
                    "Business",
                    "Chemistry",
                    "Computer Science",
                    "Economics",
                    "Engineering",
                    "Math",
                    "Physics",
                    "Psychology",
                ],
                "construction_process": [
                    "Initial MMLU filtering",
                    "Question collection from multiple sources",
                    "GPT-4 assisted option augmentation",
                    "Expert review by 10+ experts",
                ],
                "prompt_sensitivity": "2% (reduced from 4-5% in MMLU)",
                "reasoning_improvement": "20% higher CoT performance compared to PPL",
            },
        )

    def get_evaluation_metrics(self) -> list[EvaluationMetric]:
        """Returns the recommended evaluation metrics for MMLU Pro dataset."""
        return [
            EvaluationMetric.create(
                name="accuracy",
                type="classification",
                description="Proportion of correctly answered multiple-choice questions (exact match)",
                implementation="evaluate.load('accuracy')",
                primary=True,
            ),
            EvaluationMetric.create(
                name="subject_accuracy",
                type="classification",
                description="Per-subject accuracy scores with focus on enhanced subjects",
                implementation="custom_subject_accuracy",
                primary=True,
            ),
            EvaluationMetric.create(
                name="reasoning_analysis",
                type="analysis",
                description="Comparison of Chain-of-Thought vs standard PPL performance",
                implementation="custom_reasoning_analysis",
                primary=True,
            ),
            EvaluationMetric.create(
                name="prompt_robustness",
                type="analysis",
                description="Analysis of performance stability across different prompt variations",
                implementation="custom_prompt_sensitivity",
                primary=False,
            ),
        ]


if __name__ == "__main__":
    # Example usage of MMLU Pro parser
    parser = MMLUProDatasetParser()
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
