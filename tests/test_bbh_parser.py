import pytest

from llmdataparser.bbh_parser import BBHDatasetParser, BBHParseEntry


@pytest.fixture
def bbh_parser():
    """Create a BBH parser instance for testing."""
    return BBHDatasetParser()


@pytest.fixture
def loaded_bbh_parser(bbh_parser):
    """Create and load a BBH parser instance for testing."""
    bbh_parser.load(task_name="reasoning_about_colored_objects", split="test")
    return bbh_parser


@pytest.fixture
def sample_row():
    """Create a sample BBH data row for testing."""
    return {
        "input": "What color is the sky on a clear day?\nA) Blue\nB) Green\nC) Red\nD) Yellow",
        "target": "(A)",
    }


def test_bbh_parse_entry_creation_valid():
    """Test valid creation of BBHParseEntry."""
    entry = BBHParseEntry.create(
        question="Test question",
        answer="A",
        raw_question="Test question",
        raw_answer="(A)",
        task_name="reasoning_about_colored_objects",
    )
    assert isinstance(entry, BBHParseEntry)
    assert entry.question == "Test question"
    assert entry.answer == "A"
    assert entry.raw_question == "Test question"
    assert entry.raw_answer == "(A)"
    assert entry.task_name == "reasoning_about_colored_objects"


def test_bbh_parser_initialization(bbh_parser):
    """Test BBH parser initialization."""
    assert bbh_parser._data_source == "lukaemon/bbh"
    assert bbh_parser._default_task == "reasoning_about_colored_objects"
    assert "boolean_expressions" in bbh_parser._task_names
    assert "word_sorting" in bbh_parser._task_names
    assert (
        bbh_parser.get_huggingface_link
        == "https://huggingface.co/datasets/lukaemon/bbh"
    )


def test_load_dataset(loaded_bbh_parser):
    """Test loading the dataset."""
    assert loaded_bbh_parser.raw_data is not None
    assert loaded_bbh_parser.split_names == ["test"]
    assert loaded_bbh_parser._current_task == "reasoning_about_colored_objects"


@pytest.mark.integration
def test_full_parse_workflow(loaded_bbh_parser):
    """Test the complete workflow of loading and parsing data."""
    # Parse the test split
    loaded_bbh_parser.parse(split_names="test", force=True)
    parsed_data = loaded_bbh_parser.get_parsed_data

    # Basic checks
    assert len(parsed_data) > 0

    # Check first entry structure
    first_entry = parsed_data[0]
    assert isinstance(first_entry, BBHParseEntry)
    assert first_entry.task_name == "reasoning_about_colored_objects"
    assert first_entry.answer.strip("()").isalpha()  # Should be a single letter


def test_process_entry(bbh_parser, sample_row):
    """Test processing of a single BBH entry."""
    entry = bbh_parser.process_entry(
        sample_row, task_name="reasoning_about_colored_objects"
    )

    assert isinstance(entry, BBHParseEntry)
    assert entry.answer == "A"  # Stripped from "(A)"
    assert "What color is the sky" in entry.question
    assert entry.raw_answer == "(A)"
    assert entry.task_name == "reasoning_about_colored_objects"


@pytest.mark.parametrize("split_name", ["invalid_split", "wrong_split"])
def test_parse_with_invalid_split(bbh_parser, split_name):
    """Test parsing with invalid split names."""
    bbh_parser.raw_data = {"train": [], "test": []}  # Mock data

    with pytest.raises(
        ValueError, match=f"Split '{split_name}' not found in the dataset"
    ):
        bbh_parser.parse(split_name)


def test_parse_without_loaded_data(bbh_parser):
    """Test parsing without loading data first."""
    with pytest.raises(
        ValueError, match="No data loaded. Please load the dataset first"
    ):
        bbh_parser.parse()


@pytest.mark.parametrize(
    "test_case",
    [
        {"input": "Test question", "target": "(A)"},
        {"input": "Test question", "target": "(B)"},
        {"input": "Test question", "target": "(C)"},
    ],
)
def test_answer_stripping(bbh_parser, test_case):
    """Test stripping of parentheses from answers."""
    entry = bbh_parser.process_entry(
        test_case, task_name="reasoning_about_colored_objects"
    )
    assert entry.answer == test_case["target"].strip("()")
    assert entry.raw_answer == test_case["target"]


def test_parser_properties(bbh_parser):
    """Test parser property getters."""
    assert len(bbh_parser.task_names) > 0
    assert bbh_parser.total_tasks == len(bbh_parser._task_names)
    assert all(isinstance(task, str) for task in bbh_parser.task_names)


def test_parser_string_representation(loaded_bbh_parser):
    """Test string representation of parser."""
    repr_str = str(loaded_bbh_parser)
    assert "BBHDatasetParser" in repr_str
    assert "lukaemon/bbh" in repr_str
    assert "reasoning_about_colored_objects" in repr_str
    assert "loaded" in repr_str


@pytest.mark.integration
@pytest.mark.parametrize(
    "task_name", ["boolean_expressions", "causal_judgement", "date_understanding"]
)
def test_different_tasks_parsing(bbh_parser, task_name):
    """Test parsing different tasks of the dataset."""
    bbh_parser.load(task_name=task_name, split="test")
    bbh_parser.parse(split_names="test", force=True)
    parsed_data = bbh_parser.get_parsed_data

    assert len(parsed_data) > 0
    assert all(entry.task_name == task_name for entry in parsed_data)
    assert all(isinstance(entry.answer, str) for entry in parsed_data)


def test_get_dataset_description(bbh_parser):
    """Test dataset description generation."""
    description = bbh_parser.get_dataset_description()

    assert description.name == "Big Bench Hard (BBH)"
    assert description.language == "English"
    assert description.format == "Multiple choice questions with single correct answers"
    assert "suzgun2022challenging" in description.citation


def test_get_evaluation_metrics(bbh_parser):
    """Test evaluation metrics generation."""
    metrics = bbh_parser.get_evaluation_metrics()

    assert len(metrics) == 4  # Check total number of metrics

    # Check primary metrics
    primary_metrics = [m for m in metrics if m.primary]
    assert len(primary_metrics) == 2
    assert any(m.name == "accuracy" for m in primary_metrics)
    assert any(m.name == "human_eval_delta" for m in primary_metrics)

    # Check specific metric properties
    accuracy_metric = next(m for m in metrics if m.name == "accuracy")
    assert accuracy_metric.type == "classification"
    assert "evaluate.load('accuracy')" in accuracy_metric.implementation

    # Check non-primary metrics
    assert any(m.name == "per_task_accuracy" and not m.primary for m in metrics)
    assert any(m.name == "exact_match" and not m.primary for m in metrics)
