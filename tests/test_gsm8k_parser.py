import pytest

from llmdataparser.gsm8k_parser import GSM8KDatasetParser, GSM8KParseEntry


@pytest.fixture
def gsm8k_parser():
    """Create a GSM8K parser instance for testing."""
    return GSM8KDatasetParser()


@pytest.fixture
def loaded_gsm8k_parser(gsm8k_parser):
    """Create and load a GSM8K parser instance for testing."""
    gsm8k_parser.load(
        task_name="main", split="test"
    )  # Using test split as it's smaller
    return gsm8k_parser


@pytest.fixture
def sample_row():
    """Create a sample GSM8K data row for testing."""
    return {
        "question": "Janet has 3 apples. She buys 2 more. How many apples does she have now?",
        "answer": "Let's solve this step by step:\n1) Initially, Janet has 3 apples\n2) She buys 2 more apples\n3) Total apples = 3 + 2\n#### 5",
    }


def test_gsm8k_parse_entry_creation_valid():
    """Test valid creation of GSM8KParseEntry."""
    entry = GSM8KParseEntry.create(
        prompt="Test prompt",
        answer="5",
        raw_question="Test question",
        raw_answer="Solution steps #### 5",
        solution="Solution steps",
        task_name="main",
        numerical_answer=5,
    )
    assert isinstance(entry, GSM8KParseEntry)
    assert entry.prompt == "Test prompt"
    assert entry.answer == "5"
    assert entry.solution == "Solution steps"
    assert entry.numerical_answer == 5
    assert entry.task_name == "main"


def test_gsm8k_parser_initialization(gsm8k_parser):
    """Test GSM8K parser initialization."""
    assert gsm8k_parser._data_source == "openai/gsm8k"
    assert gsm8k_parser._default_task == "main"
    assert gsm8k_parser._task_names == ["main", "socratic"]
    assert (
        gsm8k_parser.get_huggingface_link
        == "https://huggingface.co/datasets/openai/gsm8k"
    )


def test_load_dataset(loaded_gsm8k_parser):
    """Test loading the dataset."""
    assert loaded_gsm8k_parser.raw_data is not None
    assert loaded_gsm8k_parser.split_names == [
        "test"
    ]  # Since we specifically loaded the test split
    assert loaded_gsm8k_parser._current_task == "main"


@pytest.mark.integration
def test_full_parse_workflow(loaded_gsm8k_parser):
    """Test the complete workflow of loading and parsing data."""
    # Parse the test split
    loaded_gsm8k_parser.parse(split_names="test", force=True)
    parsed_data = loaded_gsm8k_parser.get_parsed_data

    # Basic checks
    assert len(parsed_data) > 0

    # Check first entry structure
    first_entry = parsed_data[0]
    assert isinstance(first_entry, GSM8KParseEntry)
    assert first_entry.task_name == "main"
    assert isinstance(first_entry.numerical_answer, (str, int, float))
    assert "####" in first_entry.raw_answer
    assert first_entry.solution
    assert first_entry.prompt.startswith(loaded_gsm8k_parser._system_prompt)


def test_process_entry(gsm8k_parser, sample_row):
    """Test processing of a single GSM8K entry."""
    entry = gsm8k_parser.process_entry(sample_row, task_name="main")

    assert isinstance(entry, GSM8KParseEntry)
    assert entry.numerical_answer == 5
    assert "Janet has 3 apples" in entry.raw_question
    assert "#### 5" in entry.raw_answer
    assert "Let's solve this step by step:" in entry.solution
    assert gsm8k_parser._system_prompt in entry.prompt
    assert entry.task_name == "main"


@pytest.mark.parametrize("split_name", ["invalid_split", "wrong_split"])
def test_parse_with_invalid_split(gsm8k_parser, split_name):
    """Test parsing with invalid split names."""
    gsm8k_parser.raw_data = {"train": [], "test": []}  # Mock data

    with pytest.raises(
        ValueError, match=f"Split '{split_name}' not found in the dataset"
    ):
        gsm8k_parser.parse(split_name)


def test_parse_without_loaded_data(gsm8k_parser):
    """Test parsing without loading data first."""
    with pytest.raises(
        ValueError, match="No data loaded. Please load the dataset first"
    ):
        gsm8k_parser.parse()


@pytest.mark.parametrize(
    "test_case",
    [
        {"question": "Test question", "answer": "Some solution steps #### 42"},
        {
            "question": "Test question",
            "answer": "Complex solution\nWith multiple lines\n#### 123.45",
        },
        {"question": "Test question", "answer": "No steps #### 0"},
    ],
)
def test_numerical_answer_extraction(gsm8k_parser, test_case):
    """Test extraction of numerical answers from different formats."""
    entry = gsm8k_parser.process_entry(test_case, task_name="main")
    assert str(entry.numerical_answer) == test_case["answer"].split("####")[
        -1
    ].strip().replace(",", "")


def test_solution_extraction(gsm8k_parser):
    """Test extraction of solution steps."""
    row = {
        "question": "Test question",
        "answer": "Step 1: Do this\nStep 2: Do that\n#### 42",
    }

    entry = gsm8k_parser.process_entry(row, task_name="main")
    assert entry.solution == "Step 1: Do this\nStep 2: Do that"
    assert entry.task_name == "main"
    assert "####" not in entry.solution


def test_parser_properties(gsm8k_parser):
    """Test parser property getters."""
    assert gsm8k_parser.task_names == ["main", "socratic"]
    assert gsm8k_parser.total_tasks == 2


def test_parser_string_representation(loaded_gsm8k_parser):
    """Test string representation of parser."""
    repr_str = str(loaded_gsm8k_parser)
    assert "GSM8KDatasetParser" in repr_str
    assert "openai/gsm8k" in repr_str
    assert "main" in repr_str
    assert "loaded" in repr_str


@pytest.mark.integration
def test_different_splits_parsing(gsm8k_parser):
    """Test parsing different splits of the dataset."""
    # Load and parse test split
    gsm8k_parser.load(task_name="main", split="test")
    gsm8k_parser.parse(split_names="test", force=True)
    test_count = len(gsm8k_parser.get_parsed_data)

    # Load and parse train split
    gsm8k_parser.load(task_name="main", split="train")
    gsm8k_parser.parse(split_names="train", force=True)
    train_count = len(gsm8k_parser.get_parsed_data)

    assert test_count > 0
    assert train_count > 0
    assert train_count != test_count
