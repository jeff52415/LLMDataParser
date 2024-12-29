# llmdataparser/__init__.py
from typing import Type

from .base_parser import DatasetParser
from .bbh_parser import BBHDatasetParser
from .gsm8k_parser import GSM8KDatasetParser
from .humaneval_parser import HumanEvalDatasetParser, HumanEvalDatasetPlusParser
from .ifeval_parser import IFEvalDatasetParser
from .math_parser import MATHDatasetParser
from .mbpp_parser import MBPPDatasetParser
from .mgsm_parser import MGSMDatasetParser
from .mmlu_parser import (
    BaseMMLUDatasetParser,
    MMLUProDatasetParser,
    MMLUReduxDatasetParser,
    TMMLUPlusDatasetParser,
)
from .tmlu_parser import TMLUDatasetParser
from .tw_legal_parser import TWLegalDatasetParser


class ParserRegistry:
    """
    Registry to keep track of available parsers and provide them on request.
    """

    _registry: dict = {}

    @classmethod
    def register_parser(cls, name: str, parser_class: Type[DatasetParser]) -> None:
        cls._registry[name.lower()] = parser_class

    @classmethod
    def get_parser(cls, name: str, **kwargs) -> Type[DatasetParser]:
        parser_class = cls._registry.get(name.lower())
        if parser_class is None:
            raise ValueError(f"Parser '{name}' is not registered.")
        return parser_class(**kwargs)

    @classmethod
    def list_parsers(cls) -> list[str]:
        """Returns a list of available parser names."""
        return list(cls._registry.keys())


# Register parsers
ParserRegistry.register_parser("mmlu", BaseMMLUDatasetParser)
ParserRegistry.register_parser("mmlupro", MMLUProDatasetParser)
ParserRegistry.register_parser("mmluredux", MMLUReduxDatasetParser)
ParserRegistry.register_parser("tmmluplus", TMMLUPlusDatasetParser)
ParserRegistry.register_parser("gsm8k", GSM8KDatasetParser)
ParserRegistry.register_parser("math", MATHDatasetParser)
ParserRegistry.register_parser("mgsm", MGSMDatasetParser)
ParserRegistry.register_parser("humaneval", HumanEvalDatasetParser)
ParserRegistry.register_parser("humanevalplus", HumanEvalDatasetPlusParser)
ParserRegistry.register_parser("bbh", BBHDatasetParser)
ParserRegistry.register_parser("mbpp", MBPPDatasetParser)
ParserRegistry.register_parser("ifeval", IFEvalDatasetParser)
ParserRegistry.register_parser("twlegal", TWLegalDatasetParser)
ParserRegistry.register_parser("tmlu", TMLUDatasetParser)
