import pytest

from llmdataparser.ifeval_parser import IFEvalDatasetParser, IFEvalParseEntry


@pytest.fixture
def sample_ifeval_entries():
    """Create sample IFEval dataset entries for testing."""
    return [
        {
            "key": 1,
            "prompt": "Write a function to calculate factorial.",
            "instruction_id_list": ["math_001", "programming_001"],
            "kwargs": {"difficulty": "medium", "category": "mathematics"},
        },
        {
            "key": 2,
            "prompt": "Explain quantum computing.",
            "instruction_id_list": ["physics_001"],
            "kwargs": {"difficulty": "hard", "category": "physics"},
        },
    ]


@pytest.fixture
def ifeval_parser():
    """Create an IFEval parser instance."""
    return IFEvalDatasetParser()


def test_ifeval_parse_entry_creation_valid():
    """Test valid creation of IFEvalParseEntry."""
    entry = IFEvalParseEntry.create(
        prompt="Test system prompt\n\nTest instruction",
        answer="",  # IFEval doesn't have answers
        raw_question="Test instruction",
        raw_answer="",
        key=1,
        instruction_id_list=["test_001", "test_002"],
        kwargs={"difficulty": "easy"},
        task_name="default",
    )

    assert isinstance(entry, IFEvalParseEntry)
    assert entry.prompt == "Test system prompt\n\nTest instruction"
    assert entry.answer == ""
    assert entry.key == 1
    assert entry.instruction_id_list == ["test_001", "test_002"]
    assert entry.kwargs == {"difficulty": "easy"}
    assert entry.task_name == "default"


def test_process_entry_ifeval(ifeval_parser, sample_ifeval_entries):
    """Test processing entries in IFEval parser."""
    entry = ifeval_parser.process_entry(sample_ifeval_entries[0])

    assert isinstance(entry, IFEvalParseEntry)
    assert entry.key == 1
    assert entry.instruction_id_list == ["math_001", "programming_001"]
    assert entry.kwargs == {"difficulty": "medium", "category": "mathematics"}
    assert entry.raw_question == "Write a function to calculate factorial."
    assert entry.answer == ""  # IFEval doesn't have answers
    assert entry.task_name == "default"


def test_parser_initialization(ifeval_parser):
    """Test initialization of IFEval parser."""
    assert ifeval_parser._data_source == "google/IFEval"
    assert ifeval_parser._default_task == "default"
    assert ifeval_parser.task_names == ["default"]
    assert (
        ifeval_parser.get_huggingface_link
        == "https://huggingface.co/datasets/google/IFEval"
    )


@pytest.mark.integration
def test_load_dataset(ifeval_parser):
    """Test loading the IFEval dataset."""
    ifeval_parser.load(split="train")
    assert ifeval_parser.raw_data is not None
    assert ifeval_parser.split_names == ["train"]
    assert ifeval_parser._current_task == "default"


def test_parser_string_representation(ifeval_parser):
    """Test string representation of IFEval parser."""
    repr_str = str(ifeval_parser)
    assert "IFEvalDatasetParser" in repr_str
    assert "google/IFEval" in repr_str
    assert "not loaded" in repr_str


def test_get_dataset_description(ifeval_parser):
    """Test dataset description generation for IFEval."""
    description = ifeval_parser.get_dataset_description()

    assert description.name == "IFEval"
    assert description.source == "Google Research"
    assert description.language == "English (BCP-47 en)"


def test_get_evaluation_metrics(ifeval_parser):
    """Test evaluation metrics generation for IFEval."""
    metrics = ifeval_parser.get_evaluation_metrics()

    # Should have 5 metrics total
    assert len(metrics) == 5

    # Check primary metrics
    primary_metrics = [m for m in metrics if m.primary]
    assert len(primary_metrics) == 3

    # Verify specific metrics exist and have correct properties
    metric_names = {m.name for m in metrics}
    assert "format_compliance" in metric_names
    assert "length_constraints" in metric_names
    assert "punctuation_rules" in metric_names
    assert "keyword_usage" in metric_names
    assert "structural_requirements" in metric_names
