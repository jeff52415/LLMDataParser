from dataclasses import dataclass
from typing import Any, ClassVar

from llmdataparser.base_parser import HuggingFaceDatasetParser, HuggingFaceParseEntry
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
