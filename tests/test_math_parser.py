import pytest

from llmdataparser.math_parser import MATHDatasetParser, MATHParseEntry


@pytest.fixture
def math_parser():
    """Create a MATH parser instance for testing."""
    return MATHDatasetParser()


@pytest.fixture
def loaded_math_parser(math_parser):
    """Create and load a MATH parser instance with test split."""
    math_parser.load(task_name="algebra", split="test")
    return math_parser


@pytest.fixture
def sample_math_entries():
    """Create sample MATH dataset entries for testing."""
    return [
        {
            "problem": "Solve for x: 2x + 4 = 10",
            "level": "Level 3",
            "solution": "Let's solve step by step:\n1) Subtract 4 from both sides: 2x = 6\n2) Divide both sides by 2\n\nTherefore, x = 3",
            "type": "algebra",
        },
        {
            "problem": "Find the area of a circle with radius 5 units.",
            "level": "Level 2",
            "solution": "Area = πr²\nArea = π(5)²\nArea = 25π square units",
            "type": "geometry",
        },
        {
            "problem": "What is the limit of (x²-1)/(x-1) as x approaches 1?",
            "level": "Level 4",
            "solution": "Using L'Hôpital's rule:\nlim(x→1) (x²-1)/(x-1) = lim(x→1) (2x)/(1) = 2",
            "type": "calculus",
        },
    ]


def test_math_parse_entry_creation_valid():
    """Test valid creation of MATHParseEntry with all fields."""
    entry = MATHParseEntry.create(
        prompt="Test prompt",
        answer="Test answer",
        raw_question="Test question",
        raw_answer="Test solution",
        level="Level 5",
        task_name="algebra",
        solution="Test solution",
    )

    assert isinstance(entry, MATHParseEntry)
    assert entry.prompt == "Test prompt"
    assert entry.answer == "Test answer"
    assert entry.raw_question == "Test question"
    assert entry.raw_answer == "Test solution"
    assert entry.level == "Level 5"
    assert entry.task_name == "algebra"
    assert entry.solution == "Test solution"


@pytest.mark.parametrize(
    "test_case",
    [
        {
            "problem": "Solve for x: 2x + 4 = 10",
            "level": "Level 3",
            "solution": "x = 3",
            "type": "algebra",
        },
        {
            "problem": "Find the derivative of f(x) = x²",
            "level": "Level 4",
            "solution": "f'(x) = 2x",
            "type": "calculus",
        },
    ],
)
def test_process_entry(math_parser, test_case):
    """Test processing different types of MATH entries."""
    entry = math_parser.process_entry(test_case, task_name=test_case["type"])

    assert isinstance(entry, MATHParseEntry)
    assert (
        entry.prompt == f"{math_parser._default_system_prompt}\n{test_case['problem']}"
    )
    assert entry.answer == test_case["solution"]
    assert entry.raw_question == test_case["problem"]
    assert entry.raw_answer == test_case["solution"]
    assert entry.level == test_case["level"]
    assert entry.task_name == test_case["type"]
    assert entry.solution == test_case["solution"]


def test_math_parser_initialization(math_parser):
    """Test MATH parser initialization and properties."""
    assert isinstance(math_parser.task_names, list)
    assert len(math_parser.task_names) == 8
    assert math_parser._data_source == "lighteval/MATH"
    assert math_parser._default_task == "all"
    assert "algebra" in math_parser.task_names
    assert "geometry" in math_parser.task_names
    assert (
        math_parser.get_huggingface_link
        == "https://huggingface.co/datasets/lighteval/MATH"
    )
    assert "mathematics problem" in math_parser._default_system_prompt.lower()


def test_get_current_task(math_parser):
    """Test task name resolution in different scenarios."""
    # Test with valid type in data entry
    test_row_with_type = {"type": "algebra"}
    assert math_parser._get_current_task(test_row_with_type) == "algebra"

    # Test without type in data entry
    test_row_without_type = {}
    math_parser._current_task = "geometry"
    assert math_parser._get_current_task(test_row_without_type) == "geometry"

    # Test with invalid type - should return current task
    test_row_invalid_type = {"type": "invalid_type"}
    math_parser._current_task = "algebra"
    assert math_parser._get_current_task(test_row_invalid_type) == "algebra"


def test_valid_levels(math_parser):
    """Test handling of valid level values."""
    for i in range(1, 6):
        test_row = {
            "problem": "Test problem",
            "level": f"Level {i}",
            "solution": "Test solution",
            "type": "algebra",
        }
        entry = math_parser.process_entry(test_row, task_name="algebra")
        assert entry.level == f"Level {i}"


@pytest.mark.parametrize(
    "invalid_level",
    [
        "Level 0",  # Too low
        "Level 6",  # Too high
        "Invalid",  # Wrong format
        None,  # Missing
        "",  # Empty
        "level 1",  # Wrong capitalization
    ],
)
def test_invalid_level_handling(math_parser, invalid_level):
    """Test handling of invalid level values."""
    test_row = {
        "problem": "Test problem",
        "level": invalid_level,
        "solution": "Test solution",
        "type": "algebra",
    }

    entry = math_parser.process_entry(test_row, task_name="algebra")
    assert entry.level == "Unknown"


@pytest.mark.integration
def test_load_dataset(loaded_math_parser):
    """Test loading the MATH dataset."""
    assert loaded_math_parser.raw_data is not None
    assert loaded_math_parser.split_names == ["test"]
    assert loaded_math_parser._current_task == "algebra"


def test_parser_string_representation(loaded_math_parser):
    """Test string representation of MATH parser."""
    repr_str = str(loaded_math_parser)
    assert "MATHDatasetParser" in repr_str
    assert "lighteval/MATH" in repr_str
    assert "algebra" in repr_str
    assert "loaded" in repr_str


@pytest.mark.integration
def test_different_splits_parsing(math_parser):
    """Test parsing different splits of the dataset."""
    # Load and parse test split
    math_parser.load(task_name="algebra", split="test")
    math_parser.parse(split_names="test", force=True)
    test_count = len(math_parser.get_parsed_data)

    # Load and parse train split
    math_parser.load(task_name="algebra", split="train")
    math_parser.parse(split_names="train", force=True)
    train_count = len(math_parser.get_parsed_data)

    assert test_count > 0
    assert train_count > 0
    assert train_count != test_count


def test_get_dataset_description(math_parser):
    """Test dataset description generation."""
    description = math_parser.get_dataset_description()

    assert description.name == "MATH"
    assert "mathematical problem-solving" in description.purpose.lower()
    assert "Hendrycks" in description.source
    assert description.language == "English"
    assert "competition mathematics problems" in description.format.lower()
    assert "12,500" in description.characteristics
    assert "step-by-step solutions" in description.characteristics.lower()
    assert "hendrycksmath2021" in description.citation
    assert "NeurIPS" in description.citation

    # Check additional info
    assert description.additional_info is not None
    assert description.additional_info["difficulty_levels"] == "1-5"
    assert "algebra" in description.additional_info["topics"]
    assert "geometry" in description.additional_info["topics"]
    assert description.additional_info["size"] == "12,500 problems"
    assert "sympy" in description.additional_info["evaluation_note"].lower()
    assert "github.com/hendrycks/math" in description.additional_info["homepage"]


def test_get_evaluation_metrics(math_parser):
    """Test evaluation metrics generation."""
    metrics = math_parser.get_evaluation_metrics()

    # Check total number of metrics
    assert len(metrics) == 5

    # Check primary metrics
    primary_metrics = [m for m in metrics if m.primary]
    assert len(primary_metrics) == 3

    # Verify specific metrics exist with correct properties
    metric_names = {m.name for m in metrics}
    assert "symbolic_equivalence" in metric_names
    assert "solution_presence" in metric_names
    assert "reasoning_validity" in metric_names
    assert "mathematical_notation" in metric_names
    assert "solution_clarity" in metric_names

    # Check specific metric properties
    symbolic_metric = next(m for m in metrics if m.name == "symbolic_equivalence")
    assert symbolic_metric.type == "exact_match"
    assert symbolic_metric.primary is True
    assert "sympy" in symbolic_metric.description.lower()
    assert "equivalence" in symbolic_metric.description.lower()

    solution_metric = next(m for m in metrics if m.name == "solution_presence")
    assert solution_metric.type == "text"
    assert solution_metric.primary is True
    assert "step-by-step" in solution_metric.description.lower()

    reasoning_metric = next(m for m in metrics if m.name == "reasoning_validity")
    assert reasoning_metric.type == "text"
    assert reasoning_metric.primary is True
    assert "mathematical reasoning" in reasoning_metric.description.lower()

    # Check non-primary metrics
    non_primary_metrics = {m.name for m in metrics if not m.primary}
    assert non_primary_metrics == {"mathematical_notation", "solution_clarity"}
