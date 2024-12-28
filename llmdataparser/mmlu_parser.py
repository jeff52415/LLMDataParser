from dataclasses import dataclass
from typing import Any, Final

from llmdataparser.base_parser import HuggingFaceDatasetParser, HuggingFaceParseEntry
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
        task_name = data_entry.get("subject")
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
            task_name = data_entry.get("category")
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
