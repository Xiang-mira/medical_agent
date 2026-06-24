# Agent Harness

Installable Python package for the MedAI command-line interface and core
pipeline.

## Package structure

| Path | Content |
| --- | --- |
| `cli_anything/medai/medai_cli.py` | CLI command definitions |
| `cli_anything/medai/core/` | inference, routing, QC, fusion, M-step, and reporting modules |
| `cli_anything/medai/skills/` | CLI skill metadata |
| `tests/` | unit and integration tests |
| `requirements.txt` | base dependencies |
| `requirements-real-inference.txt` | real-inference dependencies |
| `setup.py` | package configuration |

## Installation

```bash
pip install -e agent-harness
```

## Tests

```bash
PYTHONPATH=agent-harness pytest -q agent-harness/tests
```
