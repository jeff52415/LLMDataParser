# llmdataparser/__init__.py
from typing import Type

from .base_parser import DatasetParser
from .mmlu_parser import MMLUDatasetParser


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
ParserRegistry.register_parser("mmlu", MMLUDatasetParser)
