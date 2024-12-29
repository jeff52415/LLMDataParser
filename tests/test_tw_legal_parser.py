import pytest

from llmdataparser.tw_legal_parser import TWLegalDatasetParser, TWLegalParseEntry


@pytest.fixture
def tw_legal_parser():
    """Create a Taiwan Legal parser instance for testing."""
    return TWLegalDatasetParser()


@pytest.fixture
def sample_tw_legal_entries():
    """Create sample Taiwan Legal dataset entries for testing."""
    return [
        {
            "question": "依民法規定，下列關於法人之敘述，何者錯誤？",
            "A": "法人於法令限制內，有享受權利負擔義務之能力",
            "B": "法人因目的之達到而消滅",
            "C": "法人非依法律之規定，不得成立",
            "D": "法人於登記前，即取得權利能力",
            "answer": "D",
        },
        {
            "question": "關於刑法第321條第1項第4款之結夥三人以上而犯竊盜罪，下列敘述何者正確？",
            "A": "須行為人主觀上有結夥犯竊盜之認識",
            "B": "三人以上當場在場實施竊盜行為始足當之",
            "C": "三人以上已達成犯意聯絡即可成立",
            "D": "三人以上須全部在現場實施竊盜行為",
            "answer": "A",
        },
    ]


def test_tw_legal_parse_entry_creation_valid():
    """Test valid creation of TWLegalParseEntry."""
    entry = TWLegalParseEntry.create(
        prompt="Test prompt",
        answer="A",
        raw_question="Test question",
        raw_choices=["choice1", "choice2", "choice3", "choice4"],
        raw_answer="A",
        task_name="default",
    )
    assert isinstance(entry, TWLegalParseEntry)
    assert entry.prompt == "Test prompt"
    assert entry.answer == "A"
    assert entry.raw_choices == ["choice1", "choice2", "choice3", "choice4"]


@pytest.mark.parametrize("invalid_answer", ["E", "F", "1", "", None])
def test_tw_legal_parse_entry_creation_invalid(invalid_answer):
    """Test invalid answer handling in TWLegalParseEntry creation."""
    with pytest.raises(
        ValueError, match="Invalid answer_letter.*must be one of A, B, C, D"
    ):
        TWLegalParseEntry.create(
            prompt="Test prompt",
            answer=invalid_answer,
            raw_question="Test question",
            raw_choices=["choice1", "choice2", "choice3", "choice4"],
            raw_answer=invalid_answer,
            task_name="default",
        )


def test_process_entry(tw_legal_parser, sample_tw_legal_entries):
    """Test processing entries in Taiwan Legal parser."""
    entry = tw_legal_parser.process_entry(sample_tw_legal_entries[0])

    assert isinstance(entry, TWLegalParseEntry)
    assert entry.answer == "D"
    assert "A. 法人於法令限制內，有享受權利負擔義務之能力" in entry.prompt
    assert "B. 法人因目的之達到而消滅" in entry.prompt
    assert "C. 法人非依法律之規定，不得成立" in entry.prompt
    assert "D. 法人於登記前，即取得權利能力" in entry.prompt
    assert entry.raw_question == "依民法規定，下列關於法人之敘述，何者錯誤？"
    assert len(entry.raw_choices) == 4


def test_tw_legal_parser_initialization(tw_legal_parser):
    """Test Taiwan Legal parser initialization and properties."""
    assert isinstance(tw_legal_parser.task_names, list)
    assert len(tw_legal_parser.task_names) == 1  # Only default task
    assert tw_legal_parser._data_source == "lianghsun/tw-legal-benchmark-v1"
    assert tw_legal_parser._default_task == "default"
    assert (
        tw_legal_parser.get_huggingface_link
        == "https://huggingface.co/datasets/lianghsun/tw-legal-benchmark-v1"
    )


@pytest.mark.integration
def test_load_dataset(tw_legal_parser):
    """Test loading the Taiwan Legal dataset."""
    tw_legal_parser.load(split="train")
    assert tw_legal_parser.raw_data is not None
    assert tw_legal_parser.split_names == ["train"]
    assert tw_legal_parser._current_task == "default"


def test_parser_string_representation(tw_legal_parser):
    """Test string representation of Taiwan Legal parser."""
    repr_str = str(tw_legal_parser)
    assert "TWLegalDatasetParser" in repr_str
    assert "lianghsun/tw-legal-benchmark-v1" in repr_str
    assert "not loaded" in repr_str


@pytest.mark.integration
def test_data_parsing(tw_legal_parser):
    """Test parsing the dataset."""
    # Load and parse train split
    tw_legal_parser.load(split="train")
    tw_legal_parser.parse(split_names="train", force=True)
    train_count = len(tw_legal_parser.get_parsed_data)

    assert train_count > 0
    # Additional assertions about the parsed data
    parsed_data = tw_legal_parser.get_parsed_data
    assert all(isinstance(entry, TWLegalParseEntry) for entry in parsed_data)
    assert all(entry.answer in {"A", "B", "C", "D"} for entry in parsed_data)


def test_system_prompt_override(tw_legal_parser):
    """Test overriding the default system prompt."""
    custom_prompt = "Custom system prompt for testing"
    parser = TWLegalDatasetParser(system_prompt=custom_prompt)

    test_entry = {
        "question": "Test question",
        "A": "Choice A",
        "B": "Choice B",
        "C": "Choice C",
        "D": "Choice D",
        "answer": "A",
    }

    entry = parser.process_entry(test_entry)
    assert custom_prompt in entry.prompt


def test_get_dataset_description(tw_legal_parser):
    """Test getting dataset description for Taiwan Legal parser."""
    description = tw_legal_parser.get_dataset_description()

    assert description.name == "Taiwan Legal Benchmark"
    assert description.language == "Traditional Chinese"
    assert "Taiwan's legal system" in description.characteristics
    assert (
        "huggingface.co/datasets/lianghsun/tw-legal-benchmark-v1"
        in description.citation
    )


def test_get_evaluation_metrics(tw_legal_parser):
    """Test getting evaluation metrics for Taiwan Legal parser."""
    metrics = tw_legal_parser.get_evaluation_metrics()

    assert len(metrics) == 1
    metric = metrics[0]
    assert metric.name == "accuracy"
    assert metric.type == "classification"
    assert metric.primary is True
