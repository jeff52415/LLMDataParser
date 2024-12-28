import pytest

from llmdataparser.humaneval_parser import (
    HumanEvalDatasetParser,
    HumanEvalDatasetPlusParser,
    HumanEvalParseEntry,
)


@pytest.fixture
def sample_entry():
    return {
        "prompt": 'def add(a, b):\n    """Add two numbers."""\n',
        "canonical_solution": "def add(a, b):\n    return a + b\n",
        "task_id": "HumanEval/0",
        "entry_point": "add",
        "test": "def test_add(): assert add(2, 3) == 5",
    }


@pytest.fixture
def parser():
    return HumanEvalDatasetParser()


@pytest.fixture
def plus_parser():
    return HumanEvalDatasetPlusParser()


@pytest.fixture
def plus_sample_entry():
    return {
        "prompt": 'def add(a, b):\n    """Add two numbers."""\n',
        "canonical_solution": "def add(a, b):\n    return a + b\n",
        "task_id": "HumanEval/0",
        "entry_point": "add",
        "test": "def test_add(): assert add(2, 3) == 5",
    }


def test_humaneval_parse_entry_creation():
    """Test creation of HumanEvalParseEntry"""
    entry = HumanEvalParseEntry.create(
        prompt="test prompt",
        answer="test answer",
        raw_question="raw question",
        task_id="HumanEval/1",
        entry_point="test_func",
        test="test case",
        task_name="openai_humaneval",
    )

    assert entry.prompt == "test prompt"
    assert entry.answer == "test answer"
    assert entry.raw_question == "raw question"
    assert entry.raw_answer == "test answer"  # Should match answer
    assert entry.task_id == "HumanEval/1"
    assert entry.entry_point == "test_func"
    assert entry.test == "test case"
    assert entry.task_name == "openai_humaneval"


def test_humaneval_parse_entry_validation():
    """Test validation of required fields"""
    with pytest.raises(ValueError, match="Task ID cannot be empty"):
        HumanEvalParseEntry.create(
            prompt="test",
            answer="test",
            raw_question="test",
            task_id="",  # Empty task_id should raise error
            entry_point="test",
            test="test",
            task_name="test",
        )

    with pytest.raises(ValueError, match="Entry point cannot be empty"):
        HumanEvalParseEntry.create(
            prompt="test",
            answer="test",
            raw_question="test",
            task_id="test",
            entry_point="",  # Empty entry_point should raise error
            test="test",
            task_name="test",
        )


def test_process_entry(parser, sample_entry):
    """Test processing of a single entry"""
    result = parser.process_entry(sample_entry, task_name="openai_humaneval")

    assert isinstance(result, HumanEvalParseEntry)
    assert result.task_id == "HumanEval/0"
    assert result.entry_point == "add"
    assert (
        result.prompt == f"{parser._default_system_prompt}\n\n{sample_entry['prompt']}"
    )
    assert result.answer == sample_entry["canonical_solution"]
    assert result.test == sample_entry["test"]
    assert result.task_name == "openai_humaneval"


def test_parser_initialization(parser):
    """Test parser initialization and properties"""
    assert parser._data_source == "openai/openai_humaneval"
    assert parser._default_task == "openai_humaneval"
    assert parser._task_names == ["openai_humaneval"]
    assert (
        parser.get_huggingface_link
        == "https://huggingface.co/datasets/openai/openai_humaneval"
    )


@pytest.mark.integration
def test_parser_load_and_parse(parser):
    """Integration test for loading and parsing data"""
    parser.load()
    parser.parse()
    parsed_data = parser.get_parsed_data

    assert len(parsed_data) > 0
    assert all(isinstance(entry, HumanEvalParseEntry) for entry in parsed_data)


def test_get_current_task(parser, sample_entry):
    """Test _get_current_task method"""
    task = parser._get_current_task(sample_entry)
    assert task == parser._default_task


def test_plus_parser_initialization(plus_parser):
    """Test HumanEvalDatasetPlusParser initialization and properties"""
    assert plus_parser._data_source == "evalplus/humanevalplus"
    assert plus_parser._default_task == "default"
    assert plus_parser._task_names == ["default"]
    assert (
        plus_parser.get_huggingface_link
        == "https://huggingface.co/datasets/evalplus/humanevalplus"
    )


def test_plus_process_entry(plus_parser, plus_sample_entry):
    """Test processing of a single entry in HumanEvalDatasetPlusParser"""
    result = plus_parser.process_entry(plus_sample_entry, task_name="default")

    assert isinstance(result, HumanEvalParseEntry)
    assert result.task_id == "HumanEval/0"
    assert result.entry_point == "add"
    assert (
        result.prompt
        == f"{plus_parser._default_system_prompt}\n\n{plus_sample_entry['prompt']}"
    )
    assert result.answer == plus_sample_entry["canonical_solution"]
    assert result.test == plus_sample_entry["test"]
    assert result.task_name == "default"


@pytest.mark.integration
def test_plus_parser_load_and_parse(plus_parser):
    """Integration test for loading and parsing data with HumanEvalDatasetPlusParser"""
    plus_parser.load()
    plus_parser.parse()
    parsed_data = plus_parser.get_parsed_data

    assert len(parsed_data) > 0
    assert all(isinstance(entry, HumanEvalParseEntry) for entry in parsed_data)


def test_plus_get_current_task(plus_parser, plus_sample_entry):
    """Test _get_current_task method for HumanEvalDatasetPlusParser"""
    task = plus_parser._get_current_task(plus_sample_entry)
    assert task == plus_parser._default_task
