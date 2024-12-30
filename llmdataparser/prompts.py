import textwrap
from typing import Final

MMLU_SYSTEM_PROMPT: Final[str] = textwrap.dedent(
    """\
    You are an expert answering multiple-choice questions. Select the single most accurate answer (A, B, C, or D) based on factual knowledge. Respond with the letter only.
"""
)

MMLU_PRO_SYSTEM_PROMPT: Final[str] = textwrap.dedent(
    """\
    You are an expert answering multiple-choice questions. Select the single most accurate answer (A through J) based on factual knowledge. Respond with the letter only.
"""
)

GSM8K_SYSTEM_PROMPT: Final[str] = textwrap.dedent(
    """\
    Solve this math problem step by step:
    1) Show your reasoning
    2) End with "Therefore, the answer is [number]"
"""
)

HUMANEVAL_SYSTEM_PROMPT: Final[str] = textwrap.dedent(
    """\
    Implement the Python function following best practices. Include error handling, type hints, and comments for complex logic. Return only the implementation code.
"""
)

MGSM_SYSTEM_PROMPT: Final[str] = textwrap.dedent(
    """\
    Solve this math problem step by step in the specified language:
    1) Show your reasoning
    2) Use appropriate number formatting
    3) End with "Therefore, the answer is [number]"
"""
)

IFEVAL_SYSTEM_PROMPT: Final[str] = textwrap.dedent(
    """\
    Follow the given requirements exactly. Provide only the requested output.
"""
)

BBH_SYSTEM_PROMPT: Final[str] = textwrap.dedent(
    """\
    Solve this reasoning problem and respond with only the answer (letter, True/False, or Yes/No).
"""
)

MBPP_SYSTEM_PROMPT: Final[str] = textwrap.dedent(
    """\
    Write clean, efficient Python code that solves the given task. Include docstrings and handle edge cases. Return only the implementation code.
"""
)

TW_LEGAL_SYSTEM_PROMPT: Final[str] = textwrap.dedent(
    """\
    As a Taiwan legal expert, select the most accurate answer (A, B, C, or D) based on Taiwan's laws. Respond with the letter only.
"""
)

TMLU_SYSTEM_PROMPT: Final[str] = textwrap.dedent(
    """\
    Select the most accurate answer (A, B, C, or D) based on Taiwan's educational and professional knowledge. Respond with the letter only.
"""
)
