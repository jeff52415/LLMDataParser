import pytest

from llmdataparser.mmlu_parser import (
    BaseMMLUDatasetParser,
    MMLUParseEntry,
    MMLUProDatasetParser,
    MMLUProParseEntry,
    MMLUReduxDatasetParser,
    TMMLUPlusDatasetParser,
)


@pytest.fixture
def base_parser():
    """Create a base MMLU parser instance."""
    return BaseMMLUDatasetParser()


@pytest.fixture
def redux_parser():
    """Create a MMLU Redux parser instance."""
    return MMLUReduxDatasetParser()


@pytest.fixture
def tmmlu_parser():
    """Create a TMMLU+ parser instance."""
    return TMMLUPlusDatasetParser()


@pytest.fixture
def mmlu_pro_parser():
    """Create a MMLU Pro parser instance."""
    return MMLUProDatasetParser()


@pytest.fixture
def sample_mmlu_entries():
    """Create sample MMLU dataset entries for testing."""
    return [
        {
            "question": "What is the capital of France?",
            "choices": ["London", "Paris", "Berlin", "Madrid"],
            "answer": 1,  # Paris
            "subject": "geography",
        },
        {
            "question": "Which of these is a primary color?",
            "choices": ["Green", "Purple", "Blue", "Orange"],
            "answer": 2,  # Blue
            "subject": "art",
        },
    ]


@pytest.fixture
def sample_mmlu_pro_entries():
    """Create sample MMLU Pro dataset entries for testing."""
    return [
        {
            "question": "What is the time complexity of quicksort?",
            "options": ["O(n)", "O(n log n)", "O(n²)", "O(2ⁿ)", "O(n!)", "O(1)"],
            "answer": "The average time complexity of quicksort is O(n log n)",
            "answer_index": 1,
            "category": "computer_science",
        }
    ]


def test_mmlu_parse_entry_creation_valid():
    """Test valid creation of MMLUParseEntry."""
    entry = MMLUParseEntry.create(
        prompt="Test prompt",
        answer="A",
        raw_question="Test question",
        raw_choices=["choice1", "choice2", "choice3", "choice4"],
        raw_answer="0",
        task_name="test_task",
    )
    assert isinstance(entry, MMLUParseEntry)
    assert entry.prompt == "Test prompt"
    assert entry.answer == "A"
    assert entry.raw_choices == ["choice1", "choice2", "choice3", "choice4"]
    assert entry.task_name == "test_task"


@pytest.mark.parametrize("invalid_answer", ["E", "F", "1", "", None])
def test_mmlu_parse_entry_creation_invalid(invalid_answer):
    """Test invalid answer handling in MMLUParseEntry creation."""
    with pytest.raises(
        ValueError, match="Invalid answer_letter.*must be one of A, B, C, D"
    ):
        MMLUParseEntry.create(
            prompt="Test prompt",
            answer=invalid_answer,
            raw_question="Test question",
            raw_choices=["choice1", "choice2", "choice3", "choice4"],
            raw_answer="4",
            task_name="test_task",
        )


def test_process_entry_base(base_parser, sample_mmlu_entries):
    """Test processing entries in base MMLU parser."""
    entry = base_parser.process_entry(sample_mmlu_entries[0], task_name="geography")

    assert isinstance(entry, MMLUParseEntry)
    assert entry.answer == "B"  # Index 1 maps to B
    assert "A. London" in entry.prompt
    assert "B. Paris" in entry.prompt
    assert "C. Berlin" in entry.prompt
    assert "D. Madrid" in entry.prompt
    assert entry.raw_question == "What is the capital of France?"
    assert entry.raw_choices == ["London", "Paris", "Berlin", "Madrid"]
    assert entry.raw_answer == "1"
    assert entry.task_name == "geography"


def test_mmlu_pro_parse_entry_creation_valid():
    """Test valid creation of MMLUProParseEntry."""
    entry = MMLUProParseEntry.create(
        prompt="Test prompt",
        answer="E",  # MMLU Pro supports up to J
        raw_question="Test question",
        raw_choices=["choice1", "choice2", "choice3", "choice4", "choice5"],
        raw_answer="4",
        task_name="test_task",
    )
    assert isinstance(entry, MMLUProParseEntry)
    assert entry.answer == "E"
    assert len(entry.raw_choices) == 5


def test_process_entry_mmlu_pro(mmlu_pro_parser, sample_mmlu_pro_entries):
    """Test processing entries in MMLU Pro parser."""
    entry = mmlu_pro_parser.process_entry(
        sample_mmlu_pro_entries[0], task_name="computer_science"
    )

    assert isinstance(entry, MMLUProParseEntry)
    assert entry.answer == "B"  # Index 1 maps to B
    assert "O(n log n)" in entry.prompt
    assert entry.task_name == "computer_science"
    assert len(entry.raw_choices) == 6


def test_tmmlu_process_entry(tmmlu_parser):
    """Test processing entries in TMMLU+ parser."""
    test_row = {
        "question": "什麼是台灣最高的山峰？",
        "A": "玉山",
        "B": "阿里山",
        "C": "合歡山",
        "D": "雪山",
        "answer": "A",
        "subject": "geography_of_taiwan",
    }

    entry = tmmlu_parser.process_entry(test_row, task_name="geography_of_taiwan")
    assert isinstance(entry, MMLUParseEntry)
    assert entry.answer == "A"
    assert entry.raw_choices == ["玉山", "阿里山", "合歡山", "雪山"]
    assert entry.task_name == "geography_of_taiwan"


@pytest.mark.parametrize(
    "parser_fixture,expected_tasks,expected_source",
    [
        ("base_parser", 57, "cais/mmlu"),
        ("redux_parser", 30, "edinburgh-dawg/mmlu-redux"),
        ("tmmlu_parser", 66, "ikala/tmmluplus"),
        ("mmlu_pro_parser", 1, "TIGER-Lab/MMLU-Pro"),
    ],
)
def test_parser_initialization(
    request, parser_fixture, expected_tasks, expected_source
):
    """Test initialization of different MMLU parser variants."""
    parser = request.getfixturevalue(parser_fixture)
    assert len(parser.task_names) == expected_tasks
    assert parser._data_source == expected_source
    assert (
        parser.get_huggingface_link
        == f"https://huggingface.co/datasets/{expected_source}"
    )


@pytest.mark.integration
def test_load_dataset(base_parser):
    """Test loading the MMLU dataset."""
    base_parser.load(task_name="anatomy", split="test")
    assert base_parser.raw_data is not None
    assert base_parser.split_names == ["test"]
    assert base_parser._current_task == "anatomy"


def test_parser_string_representation(base_parser):
    """Test string representation of MMLU parser."""
    repr_str = str(base_parser)
    assert "MMLUDatasetParser" in repr_str
    assert "cais/mmlu" in repr_str
    assert "not loaded" in repr_str


@pytest.mark.integration
def test_different_splits_parsing(base_parser):
    """Test parsing different splits of the dataset."""
    # Load and parse test split
    base_parser.load(task_name="anatomy", split="test")
    base_parser.parse(split_names="test", force=True)
    test_count = len(base_parser.get_parsed_data)

    # Load and parse validation split
    base_parser.load(task_name="anatomy", split="validation")
    base_parser.parse(split_names="validation", force=True)
    val_count = len(base_parser.get_parsed_data)

    assert test_count > 0
    assert val_count > 0
    assert test_count != val_count


def test_base_mmlu_dataset_description(base_parser):
    """Test dataset description for base MMLU."""
    description = base_parser.get_dataset_description()

    assert description.name == "Massive Multitask Language Understanding (MMLU)"
    assert "cais/mmlu" in description.source
    assert description.language == "English"

    # Check characteristics
    assert "57 subjects" in description.characteristics.lower()

    # Check citation
    assert "hendryckstest2021" in description.citation


def test_mmlu_redux_dataset_description(redux_parser):
    """Test dataset description for MMLU Redux."""
    description = redux_parser.get_dataset_description()

    assert description.name == "MMLU Redux"
    assert "manually re-annotated" in description.purpose.lower()
    assert "edinburgh-dawg/mmlu-redux" in description.source
    assert description.language == "English"

    # Check characteristics
    assert "3,000" in description.characteristics


def test_tmmlu_plus_dataset_description(tmmlu_parser):
    """Test dataset description for TMMLU+."""
    description = tmmlu_parser.get_dataset_description()

    assert "ikala/tmmluplus" in description.source
    assert description.language == "Traditional Chinese"

    # Check characteristics
    assert "66 subjects" in description.characteristics.lower()

    # Check citation
    assert "ikala2024improved" in description.citation


def test_mmlu_pro_dataset_description(mmlu_pro_parser):
    """Test dataset description for MMLU Pro."""
    description = mmlu_pro_parser.get_dataset_description()

    assert description.name == "MMLU Pro"
    assert "challenging" in description.purpose.lower()
    assert "TIGER-Lab/MMLU-Pro" in description.source
    assert description.language == "English"


def test_base_mmlu_evaluation_metrics(base_parser):
    """Test evaluation metrics for base MMLU."""
    metrics = base_parser.get_evaluation_metrics()

    assert len(metrics) >= 3
    metric_names = {m.name for m in metrics}

    assert "accuracy" in metric_names
    assert "subject_accuracy" in metric_names
    assert "category_accuracy" in metric_names

    accuracy_metric = next(m for m in metrics if m.name == "accuracy")
    assert accuracy_metric.type == "classification"
    assert accuracy_metric.primary is True
    assert "multiple-choice" in accuracy_metric.description.lower()


def test_mmlu_redux_evaluation_metrics(redux_parser):
    """Test evaluation metrics for MMLU Redux."""
    metrics = redux_parser.get_evaluation_metrics()

    metric_names = {m.name for m in metrics}
    assert "question_clarity" in metric_names


def test_tmmlu_plus_evaluation_metrics(tmmlu_parser):
    """Test evaluation metrics for TMMLU+."""
    metrics = tmmlu_parser.get_evaluation_metrics()

    metric_names = {m.name for m in metrics}
    assert "difficulty_analysis" in metric_names


def test_mmlu_pro_evaluation_metrics(mmlu_pro_parser):
    """Test evaluation metrics for MMLU Pro."""
    metrics = mmlu_pro_parser.get_evaluation_metrics()

    metric_names = {m.name for m in metrics}
    assert "reasoning_analysis" in metric_names
    assert "prompt_robustness" in metric_names
