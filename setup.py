"""Legacy setuptools entry point for offline editable installs."""
from __future__ import annotations

from setuptools import setup


setup(
    name="openbench",
    version="0.1.0",
    description="Interactive CLI for running vLLM inference containers and benchmarks.",
    long_description=open("README.md", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    packages=["benchcli"],
    python_requires=">=3.10",
    install_requires=[
        "typer>=0.12.0",
        "rich>=13.7.0",
        "questionary>=2.0.1",
        "docker>=7.1.0",
        "pydantic>=2.6.0",
    ],
    entry_points={
        "console_scripts": [
            "openbench=benchcli.cli:app",
            "benchcli=benchcli.cli:app",
        ]
    },
)
