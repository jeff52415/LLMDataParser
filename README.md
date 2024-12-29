# LLMDataParser

**LLMDataParser** is a Python library that provides parsers for benchmark datasets used in evaluating Large Language Models (LLMs). It offers a unified interface for loading and parsing datasets like **MMLU**, **GSM8k**, and others, streamlining dataset preparation for LLM evaluation. The library aims to simplify the process of working with common LLM benchmark datasets through a consistent API.

## Features

- **Unified Interface**: Consistent `DatasetParser` for all datasets.
- **LLM-Agnostic**: Independent of any specific language model.
- **Easy to Use**: Simple methods and built-in Python types.
- **Extensible**: Easily add support for new datasets.
- **Gradio**: Built-in Gradio interface for interactive dataset exploration and testing.

## Installation

### Option 1: Using pip

You can install the package directly using `pip`. Even with only a `pyproject.toml` file, this method works for standard installations.

1. **Clone the Repository**:

   ```bash
   git clone https://github.com/jeff52415/LLMDataParser.git
   cd LLMDataParser
   ```

1. **Install Dependencies with pip**:

   ```bash
   pip install .
   ```

### Option 2: Using Poetry

Poetry manages the virtual environment and dependencies automatically, so you don't need to create a conda environment first.

1. **Install Dependencies with Poetry**:

   ```bash
   poetry install
   ```

1. **Activate the Virtual Environment**:

   ```bash
   poetry shell
   ```

## Available Parsers

- **MMLUDatasetParser**
- **MMLUProDatasetParser**
- **MMLUReduxDatasetParser**
- **TMMLUPlusDatasetParser**
- **GSM8KDatasetParser**
- **MATHDatasetParser**
- **MGSMDatasetParser**
- **HumanEvalDatasetParser**
- **HumanEvalDatasetPlusParser**
- **BBHDatasetParser**
- **MBPPDatasetParser**
- **IFEvalDatasetParser**
- **TWLegalDatasetParser**
- **TMLUDatasetParser**

## Adding New Dataset Parsers

To add support for a new dataset, please refer to our detailed guide in [docs/adding_new_parser.md](docs/adding_new_parser.md). The guide includes:

- Step-by-step instructions for creating a new parser
- Code examples and templates
- Best practices and common patterns
- Testing guidelines

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Contact

For questions or support, please open an issue on GitHub or contact [jeff52415@gmail.com](mailto:jeff52415@gmail.com).
