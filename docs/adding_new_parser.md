# Adding a New Dataset Parser

This guide explains how to add a new dataset parser to the llmdataparser library. The library is designed to make it easy to add support for new datasets while maintaining consistent interfaces and functionality.

## Step-by-Step Guide

### 1. Create a New Parser Class

Create a new file `your_dataset_parser.py` in the `llmdataparser` folder. Your parser should inherit from `HuggingFaceDatasetParser[T]` where T is your custom entry type.

```python
from llmdataparser.base_parser import (
    DatasetDescription,
    EvaluationMetric,
    HuggingFaceDatasetParser,
    HuggingFaceParseEntry,
)

@dataclass(frozen=True, kw_only=True, slots=True)
class YourDatasetParseEntry(HuggingFaceParseEntry):
    """Custom entry class for your dataset."""
    # Add any additional fields specific to your dataset
    custom_field: str

    @classmethod
    def create(cls, question: str, answer: str, raw_question: str,
               raw_answer: str, task_name: str, custom_field: str) -> "YourDatasetParseEntry":
        return cls(
            question=question,
            answer=answer,
            raw_question=raw_question,
            raw_answer=raw_answer,
            task_name=task_name,
            custom_field=custom_field
        )

class YourDatasetParser(HuggingFaceDatasetParser[YourDatasetParseEntry]):
    """Parser for your dataset."""

    # Required class variables
    _data_source = "huggingface/your-dataset"
    _default_task = "default"
    _task_names = ["task1", "task2", "task3"]
```

### 2. Implement Required Methods

Your parser needs to implement these key methods:

```python
def process_entry(
    self,
    row: dict[str, Any],
    task_name: str | None = None,
    **kwargs: Any
) -> YourDatasetParseEntry:
    """Process a single dataset entry."""
    # Extract data from the row
    raw_question = row["question"]
    raw_answer = row["answer"]
    task = task_name or self._get_current_task(row)

    question = f"Question: {raw_question}\nAnswer:"

    return YourDatasetParseEntry.create(
        question=question,
        answer=raw_answer,
        raw_question=raw_question,
        raw_answer=raw_answer,
        task_name=task,
        custom_field=row["custom_field"]
    )

def get_dataset_description(self) -> DatasetDescription:
    """Returns description of your dataset."""
    return DatasetDescription.create(
        name="Your Dataset Name",
        purpose="Purpose of the dataset",
        source="Dataset source/URL",
        language="Dataset language",
        format="Data format (e.g., multiple choice, free text)",
        characteristics="Key characteristics of the dataset",
        citation="Dataset citation if available"
    )

def get_evaluation_metrics(self) -> list[EvaluationMetric]:
    """Returns recommended evaluation metrics."""
    return [
        EvaluationMetric.create(
            name="metric_name",
            type="metric_type",
            description="Metric description",
            implementation="implementation_details",
            primary=True
        )
    ]
```

### 3. Add Example Usage

Add example usage at the bottom of your parser file:

```python
if __name__ == "__main__":
    # Example usage
    parser = YourDatasetParser()
    parser.load()
    parser.parse()

    # Get parsed data
    parsed_data = parser.get_parsed_data

    # Print example entry
    if parsed_data:
        example = parsed_data[0]
        print("\nExample parsed entry:")
        print(f"Question: {example.raw_question}")
        print(f"Answer: {example.answer}")
```

### 4. Create Tests

Create a test file `tests/test_your_dataset_parser.py`:

```python
import pytest
from llmdataparser.your_dataset_parser import YourDatasetParser, YourDatasetParseEntry

def test_parser_initialization():
    parser = YourDatasetParser()
    assert parser._data_source == "huggingface/your-dataset"
    assert parser._default_task == "default"
    assert "task1" in parser._task_names

def test_process_entry():
    parser = YourDatasetParser()
    sample_row = {
        "question": "Sample question",
        "answer": "Sample answer",
        "custom_field": "Custom value"
    }

    entry = parser.process_entry(sample_row)
    assert isinstance(entry, YourDatasetParseEntry)
    assert entry.raw_question == "Sample question"
    assert entry.custom_field == "Custom value"
```

## Best Practices

1. **Type Safety**: Use type hints consistently and ensure your parser is properly typed.
1. **Documentation**: Add clear docstrings and comments explaining your parser's functionality.
1. **Error Handling**: Include appropriate error checking and validation.
1. **Testing**: Write comprehensive tests covering different scenarios.

## Examples

Look at existing parsers for reference:

- `mmlu_parser.py` for multiple-choice questions
- `gsm8k_parser.py` for math word problems
- `humaneval_parser.py` for code generation tasks

## Common Patterns

1. **Parse Entry Class**: Create a custom parse entry class if you need additional fields.
1. **Task Names**: Define all available tasks in `_task_names`.
1. **Process Entry**: Handle data extraction and formatting in `process_entry`.
1. **Dataset Description**: Provide comprehensive dataset information.
1. **Evaluation Metrics**: Define appropriate metrics for your dataset.

## Testing Your Parser

1. Run the example usage code to verify basic functionality
1. Run pytest to execute your test cases
1. Try different dataset splits and tasks
1. Verify the parsed output format
1. Check error handling with invalid inputs
