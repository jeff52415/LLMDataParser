import pytest

from llmdataparser.mgsm_parser import MGSMDatasetParser, MGSMParseEntry


@pytest.fixture
def mgsm_parser():
    """Create a MGSM parser instance for testing."""
    return MGSMDatasetParser()


@pytest.fixture
def loaded_mgsm_parser(mgsm_parser):
    """Create and load a MGSM parser instance with test split."""
    mgsm_parser.load(task_name="en", split="test")
    return mgsm_parser


@pytest.fixture
def sample_mgsm_entries():
    """Create sample MGSM dataset entries for testing."""
    return [
        {
            "question": "John has 5 apples and buys 3 more. How many apples does he have now?",
            "answer": "Let's solve step by step:\n1) Initial apples = 5\n2) Bought apples = 3\n3) Total = 5 + 3 = 8\nJohn has 8 apples now.",
            "answer_number": 8,
            "equation_solution": "5 + 3 = 8",
            "language": "en",
        },
        {
            "question": "Juan tiene 5 manzanas y compra 3 más. ¿Cuántas manzanas tiene ahora?",
            "answer": "Resolvamos paso a paso:\n1) Manzanas iniciales = 5\n2) Manzanas compradas = 3\n3) Total = 5 + 3 = 8\nJuan tiene 8 manzanas ahora.",
            "answer_number": 8,
            "equation_solution": "5 + 3 = 8",
            "language": "es",
        },
        {
            "question": "ジョンはリンゴを5個持っていて、さらに3個買います。今何個持っていますか？",
            "answer": None,  # Testing case with missing detailed answer
            "answer_number": 8,
            "equation_solution": "5 + 3 = 8",
            "language": "ja",
        },
    ]


def test_mgsm_parse_entry_creation_valid():
    """Test valid creation of MGSMParseEntry with all fields."""
    entry = MGSMParseEntry.create(
        question="Test question",
        answer="Test answer",
        raw_question="Test question",
        raw_answer="Test answer",
        numerical_answer=42,
        equation_solution="21 * 2 = 42",
        task_name="en",
        language="en",
    )

    assert isinstance(entry, MGSMParseEntry)
    assert entry.question == "Test question"
    assert entry.answer == "Test answer"
    assert entry.raw_question == "Test question"
    assert entry.raw_answer == "Test answer"
    assert entry.numerical_answer == 42
    assert entry.equation_solution == "21 * 2 = 42"
    assert entry.task_name == "en"
    assert entry.language == "en"


def test_process_entry_with_detailed_answer(mgsm_parser, sample_mgsm_entries):
    """Test processing entry with detailed answer in English."""
    entry = mgsm_parser.process_entry(sample_mgsm_entries[0], task_name="en")

    assert isinstance(entry, MGSMParseEntry)
    assert entry.numerical_answer == 8
    assert entry.equation_solution == "5 + 3 = 8"
    assert "step by step" in entry.answer
    assert entry.language == "en"
    assert entry.task_name == "en"


def test_process_entry_without_detailed_answer(mgsm_parser, sample_mgsm_entries):
    """Test processing entry without detailed answer (Japanese)."""
    entry = mgsm_parser.process_entry(sample_mgsm_entries[2], task_name="ja")

    assert isinstance(entry, MGSMParseEntry)
    assert entry.numerical_answer == 8
    assert entry.equation_solution == "5 + 3 = 8"
    assert entry.answer == "8"  # Should use numerical_answer as string
    assert entry.language == "ja"
    assert entry.task_name == "ja"


def test_process_entry_spanish(mgsm_parser, sample_mgsm_entries):
    """Test processing Spanish entry."""
    entry = mgsm_parser.process_entry(sample_mgsm_entries[1], task_name="es")

    assert isinstance(entry, MGSMParseEntry)
    assert entry.numerical_answer == 8
    assert entry.equation_solution == "5 + 3 = 8"
    assert "paso a paso" in entry.answer  # Spanish for "step by step"
    assert entry.language == "es"
    assert entry.task_name == "es"


def test_mgsm_parser_initialization(mgsm_parser):
    """Test MGSM parser initialization and properties."""
    assert isinstance(mgsm_parser.task_names, list)
    assert len(mgsm_parser.task_names) == 11  # 11 supported languages
    assert mgsm_parser._data_source == "juletxara/mgsm"
    assert mgsm_parser._default_task == "en"
    assert all(lang in mgsm_parser.task_names for lang in ["en", "es", "ja", "zh"])
    assert (
        mgsm_parser.get_huggingface_link
        == "https://huggingface.co/datasets/juletxara/mgsm"
    )


@pytest.mark.integration
def test_load_dataset(loaded_mgsm_parser):
    """Test loading the MGSM dataset."""
    assert loaded_mgsm_parser.raw_data is not None
    assert loaded_mgsm_parser.split_names == ["test"]
    assert loaded_mgsm_parser._current_task == "en"


def test_parser_string_representation(loaded_mgsm_parser):
    """Test string representation of MGSM parser."""
    repr_str = str(loaded_mgsm_parser)
    assert "MGSMDatasetParser" in repr_str
    assert "juletxara/mgsm" in repr_str
    assert "en" in repr_str
    assert "loaded" in repr_str


@pytest.mark.integration
def test_different_languages_parsing(mgsm_parser):
    """Test parsing different language versions."""
    # Load and parse English
    mgsm_parser.load(task_name="en", split="test")
    mgsm_parser.parse(split_names="test", force=True)
    en_count = len(mgsm_parser.get_parsed_data)

    # Load and parse Spanish
    mgsm_parser.load(task_name="es", split="test")
    mgsm_parser.parse(split_names="test", force=True)
    es_count = len(mgsm_parser.get_parsed_data)

    assert en_count > 0
    assert es_count > 0
    assert en_count == es_count  # Should have same number of problems in each language


@pytest.mark.parametrize("language", ["en", "es", "ja", "zh", "ru"])
def test_supported_languages(mgsm_parser, language):
    """Test that each supported language can be processed."""
    test_entry = {
        "question": f"Test question in {language}",
        "answer": f"Test answer in {language}",
        "answer_number": 42,
        "equation_solution": "21 * 2 = 42",
    }

    entry = mgsm_parser.process_entry(test_entry, task_name=language)
    assert entry.language == language
    assert entry.task_name == language
    assert entry.numerical_answer == 42


def test_get_dataset_description(mgsm_parser):
    """Test dataset description generation."""
    description = mgsm_parser.get_dataset_description()

    assert description.name == "Multilingual Grade School Math (MGSM)"
    assert "multilingual chain-of-thought reasoning" in description.purpose.lower()
    assert "juletxara/mgsm" in description.source
    assert description.language == "Multilingual (11 languages)"

    assert "mathematical reasoning" in description.characteristics.lower()

    # Check citations
    assert "shi2022language" in description.citation
    assert "cobbe2021gsm8k" in description.citation

    # Check additional info
    assert description.additional_info is not None
    assert len(description.additional_info["languages"]) == 11
    assert "English" in description.additional_info["languages"]
    assert "Chinese" in description.additional_info["languages"]


def test_get_evaluation_metrics(mgsm_parser):
    """Test evaluation metrics generation."""
    metrics = mgsm_parser.get_evaluation_metrics()

    # Check total number of metrics
    assert len(metrics) == 4

    # Check primary metrics
    primary_metrics = [m for m in metrics if m.primary]
    assert len(primary_metrics) == 3

    # Verify specific metrics exist with correct properties
    metric_names = {m.name for m in metrics}
    assert "exact_match" in metric_names
    assert "solution_validity" in metric_names
    assert "step_accuracy" in metric_names
    assert "cross_lingual_consistency" in metric_names

    # Check specific metric properties
    exact_match_metric = next(m for m in metrics if m.name == "exact_match")
    assert exact_match_metric.type == "string"
    assert exact_match_metric.primary is True
    assert "numerical answers" in exact_match_metric.description.lower()
    assert "custom_exact_match" in exact_match_metric.implementation

    solution_metric = next(m for m in metrics if m.name == "solution_validity")
    assert solution_metric.type == "text"
    assert solution_metric.primary is True
    assert "mathematically valid" in solution_metric.description.lower()
    assert "custom_solution_validator" in solution_metric.implementation

    step_metric = next(m for m in metrics if m.name == "step_accuracy")
    assert step_metric.type == "numerical"
    assert step_metric.primary is True
    assert "calculation steps" in step_metric.description.lower()
    assert "custom_step_accuracy" in step_metric.implementation
