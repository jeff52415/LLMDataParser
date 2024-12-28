import pytest

from llmdataparser.mbpp_parser import MBPPDatasetParser, MBPPParseEntry


@pytest.fixture
def sample_entry():
    return {
        "text": "Write a function to find the sum of numbers in a list.",
        "code": "def sum_list(lst):\n    return sum(lst)",
        "task_id": 42,
        "test_list": ["assert sum_list([1, 2, 3]) == 6"],
        "test_setup_code": "",
        "challenge_test_list": ["assert sum_list([4, 5, 6]) == 15"],
    }


@pytest.fixture
def parser():
    return MBPPDatasetParser()


def test_mbpp_parse_entry_creation():
    """Test creation of MBPPParseEntry"""
    entry = MBPPParseEntry.create(
        prompt="test prompt",
        answer="test answer",
        raw_question="raw question",
        task_id=42,
        test_list=["test1", "test2"],
        test_setup_code="setup code",
        challenge_test_list=["challenge1"],
        task_name="full",
        source_file="test.pdf",
    )

    assert entry.prompt == "test prompt"
    assert entry.answer == "test answer"
    assert entry.raw_question == "raw question"
    assert entry.raw_answer == "test answer"
    assert entry.task_id == 42
    assert entry.test_list == ["test1", "test2"]
    assert entry.test_setup_code == "setup code"
    assert entry.challenge_test_list == ["challenge1"]
    assert entry.task_name == "full"


def test_mbpp_parse_entry_validation():
    """Test validation of required fields"""
    with pytest.raises(ValueError, match="Task ID must be an integer"):
        MBPPParseEntry.create(
            prompt="test",
            answer="test",
            raw_question="test",
            task_id="not_an_int",  # Invalid task_id type
            test_list=[],
            test_setup_code="",
            challenge_test_list=[],
            task_name="full",
            source_file="test.pdf",
        )


def test_process_entry(parser, sample_entry):
    """Test processing of a single entry"""
    result = parser.process_entry(sample_entry, task_name="full")

    assert isinstance(result, MBPPParseEntry)
    assert result.task_id == 42
    assert result.raw_question == sample_entry["text"]
    assert result.answer == sample_entry["code"]
    assert result.test_list == sample_entry["test_list"]
    assert result.challenge_test_list == sample_entry["challenge_test_list"]
    expected_prompt = f"{parser._system_prompt}\n\nTask: {sample_entry['text']}"
    assert result.prompt == expected_prompt
    assert result.task_name == "full"


def test_parser_initialization(parser):
    """Test parser initialization and properties"""
    assert parser._data_source == "google-research-datasets/mbpp"
    assert parser._default_task == "full"
    assert parser._task_names == ["full", "sanitized"]
    assert (
        parser.get_huggingface_link
        == "https://huggingface.co/datasets/google-research-datasets/mbpp"
    )


@pytest.mark.integration
@pytest.mark.skip(reason="Requires access to HuggingFace MBPP dataset")
def test_parser_load_and_parse(parser):
    """Integration test for loading and parsing data"""
    parser.load(split="train")
    parser.parse(force=True)
    parsed_data = parser.get_parsed_data

    assert len(parsed_data) > 0
    assert all(isinstance(entry, MBPPParseEntry) for entry in parsed_data)


def test_get_current_task(parser, sample_entry):
    """Test _get_current_task method"""
    task = parser._get_current_task(sample_entry)
    assert task == parser._default_task


@pytest.mark.parametrize("task_name", ["full", "sanitized"])
@pytest.mark.skip(reason="Requires access to HuggingFace MBPP dataset")
def test_different_tasks_loading(parser, task_name):
    """Test loading different tasks of the dataset"""
    parser.load(task_name=task_name, split="train")
    assert parser._current_task == task_name


def test_parser_string_representation(parser):
    """Test string representation of parser"""
    repr_str = str(parser)
    assert "MBPPDatasetParser" in repr_str
    assert "google-research-datasets/mbpp" in repr_str
    assert "not loaded" in repr_str


def test_parse_without_loaded_data(parser):
    """Test parsing without loading data first"""
    with pytest.raises(
        ValueError, match="No data loaded. Please load the dataset first"
    ):
        parser.parse()


@pytest.mark.integration
@pytest.mark.skip(reason="Requires access to HuggingFace MBPP dataset")
def test_full_workflow_with_different_splits(parser):
    """Test the complete workflow with different splits"""
    parser.load(split="train")
    parser.parse(force=True)
    train_data = parser.get_parsed_data

    assert len(train_data) > 0
    assert all(isinstance(entry, MBPPParseEntry) for entry in train_data)
    assert all(entry.task_name == "full" for entry in train_data)


def test_custom_system_prompt():
    """Test parser initialization with custom system prompt"""
    custom_prompt = "Custom system prompt"
    parser = MBPPDatasetParser(system_prompt=custom_prompt)
    assert parser._system_prompt == custom_prompt


def test_default_system_prompt(parser):
    """Test parser uses default system prompt when none provided"""
    assert parser._system_prompt == parser._default_system_prompt
