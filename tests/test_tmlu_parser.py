import pytest

from llmdataparser.tmlu_parser import TMLUDatasetParser, TMLUParseEntry


@pytest.fixture
def tmlu_parser():
    """Create a TMLU parser instance for testing."""
    return TMLUDatasetParser()


@pytest.fixture
def sample_tmlu_entries():
    """Create sample TMLU dataset entries for testing."""
    return [
        {
            "question": "閱讀下文，選出依序最適合填入□內的選項：",
            "A": "張揚／綢繆未雨／奏疏",
            "B": "抽搐／煮繭抽絲／奏疏",
            "C": "張揚／煮繭抽絲／進貢",
            "D": "抽搐／綢繆未雨／進貢",
            "answer": "B",
            "explanation": "根據文意，選項B最為恰當。",
            "metadata": {
                "timestamp": "2023-10-09T18:27:20.304623",
                "source": "AST chinese - 108",
                "explanation_source": "",
            },
        },
        {
            "question": "下列何者是質數？",
            "A": "21",
            "B": "27",
            "C": "31",
            "D": "33",
            "answer": "C",
            "explanation": "31是質數，其他選項都是合數。",
            "metadata": {
                "timestamp": "2023-10-09T18:27:20.304623",
                "source": "AST mathematics - 108",
                "explanation_source": "",
            },
        },
    ]


def test_tmlu_parse_entry_creation_valid():
    """Test valid creation of TMLUParseEntry."""
    entry = TMLUParseEntry.create(
        prompt="Test prompt",
        answer="A",
        raw_question="Test question",
        raw_choices=["choice1", "choice2", "choice3", "choice4"],
        raw_answer="A",
        task_name="AST_chinese",
        explanation="Test explanation",
        metadata={"source": "test"},
    )
    assert isinstance(entry, TMLUParseEntry)
    assert entry.prompt == "Test prompt"
    assert entry.answer == "A"
    assert entry.raw_choices == ["choice1", "choice2", "choice3", "choice4"]
    assert entry.explanation == "Test explanation"
    assert entry.metadata == {"source": "test"}


@pytest.mark.parametrize("invalid_answer", ["E", "F", "1", "", None])
def test_tmlu_parse_entry_creation_invalid(invalid_answer):
    """Test invalid answer handling in TMLUParseEntry creation."""
    with pytest.raises(
        ValueError, match="Invalid answer_letter.*must be one of A, B, C, D"
    ):
        TMLUParseEntry.create(
            prompt="Test prompt",
            answer=invalid_answer,
            raw_question="Test question",
            raw_choices=["choice1", "choice2", "choice3", "choice4"],
            raw_answer=invalid_answer,
            task_name="AST_chinese",
        )


def test_process_entry(tmlu_parser, sample_tmlu_entries):
    """Test processing entries in TMLU parser."""
    entry = tmlu_parser.process_entry(sample_tmlu_entries[0], task_name="AST_chinese")

    assert isinstance(entry, TMLUParseEntry)
    assert entry.answer == "B"
    assert entry.task_name == "AST_chinese"
    assert len(entry.raw_choices) == 4
    assert entry.explanation == "根據文意，選項B最為恰當。"
    assert "AST chinese - 108" in entry.metadata["source"]


def test_tmlu_parser_initialization(tmlu_parser):
    """Test TMLU parser initialization and properties."""
    assert isinstance(tmlu_parser.task_names, list)
    assert len(tmlu_parser.task_names) == 37  # Total number of tasks
    assert tmlu_parser._data_source == "miulab/tmlu"
    assert tmlu_parser._default_task == "AST_chinese"
    assert "AST_chinese" in tmlu_parser.task_names
    assert "GSAT_mathematics" in tmlu_parser.task_names
    assert (
        tmlu_parser.get_huggingface_link
        == "https://huggingface.co/datasets/miulab/tmlu"
    )


@pytest.mark.integration
def test_load_dataset(tmlu_parser):
    """Test loading the TMLU dataset."""
    tmlu_parser.load(task_name="AST_chinese", split="test")
    assert tmlu_parser.raw_data is not None
    assert tmlu_parser.split_names == ["test"]
    assert tmlu_parser._current_task == "AST_chinese"


def test_parser_string_representation(tmlu_parser):
    """Test string representation of TMLU parser."""
    repr_str = str(tmlu_parser)
    assert "TMLUDatasetParser" in repr_str
    assert "miulab/tmlu" in repr_str
    assert "not loaded" in repr_str


@pytest.mark.integration
def test_different_tasks_parsing(tmlu_parser):
    """Test parsing different tasks of the dataset."""
    # Load and parse AST_chinese
    tmlu_parser.load(task_name="AST_chinese", split="test")
    tmlu_parser.parse(split_names="test", force=True)
    chinese_count = len(tmlu_parser.get_parsed_data)

    # Load and parse AST_mathematics
    tmlu_parser.load(task_name="AST_mathematics", split="test")
    tmlu_parser.parse(split_names="test", force=True)
    math_count = len(tmlu_parser.get_parsed_data)

    assert chinese_count > 0
    assert math_count > 0


def test_system_prompt_override(tmlu_parser):
    """Test overriding the default system prompt."""
    custom_prompt = "Custom system prompt for testing"
    parser = TMLUDatasetParser(system_prompt=custom_prompt)

    test_entry = {
        "question": "Test question",
        "A": "Choice A",
        "B": "Choice B",
        "C": "Choice C",
        "D": "Choice D",
        "answer": "A",
        "explanation": "Test explanation",
        "metadata": {"source": "test"},
    }

    entry = parser.process_entry(test_entry)
    assert custom_prompt in entry.prompt


def test_metadata_handling(tmlu_parser, sample_tmlu_entries):
    """Test proper handling of metadata in entries."""
    entry = tmlu_parser.process_entry(sample_tmlu_entries[0])

    assert "timestamp" in entry.metadata
    assert "source" in entry.metadata
    assert "explanation_source" in entry.metadata
    assert entry.metadata["source"] == "AST chinese - 108"


def test_get_dataset_description(tmlu_parser):
    """Test dataset description generation."""
    description = tmlu_parser.get_dataset_description()

    assert description.name == "Taiwan Multiple-choice Language Understanding (TMLU)"
    assert description.language == "Traditional Chinese"
    assert "Taiwan-specific educational" in description.purpose
    assert "Various Taiwan standardized tests" in description.source
    assert description.format == "Multiple choice questions (A/B/C/D)"
    assert "Advanced Subjects Test (AST)" in description.characteristics
    assert "DBLP:journals/corr/abs-2403-20180" in description.citation


def test_get_evaluation_metrics(tmlu_parser):
    """Test evaluation metrics generation."""
    metrics = tmlu_parser.get_evaluation_metrics()

    assert len(metrics) == 5  # Check total number of metrics

    # Check primary metrics
    primary_metrics = [m for m in metrics if m.primary]
    assert len(primary_metrics) == 2
    assert any(m.name == "accuracy" for m in primary_metrics)
    assert any(m.name == "per_subject_accuracy" for m in primary_metrics)

    # Check specific metric properties
    accuracy_metric = next(m for m in metrics if m.name == "accuracy")
    assert accuracy_metric.type == "classification"
    assert "datasets.load_metric('accuracy')" in accuracy_metric.implementation

    # Check non-primary metrics
    non_primary_metrics = {m.name for m in metrics if not m.primary}
    assert non_primary_metrics == {
        "per_difficulty_accuracy",
        "confusion_matrix",
        "explanation_quality",
    }
