# GASLIGHTBENCH: Quantifying LLM Susceptibility to Social Prompting

# Sycophancy Evaluation Pipeline

To start the project, install dependencies:

```bash
pip install --upgrade anthropic openai inspect-ai python-dotenv ipywidgets matplotlib
```

Before running the pipeline, set your API key:

**Windows (PowerShell)**

```powershell
setx OPENAI_API_KEY "<yourkey>"
```

**macOS / Linux (bash, zsh)**

```bash
export OPENAI_API_KEY="<yourkey>"
```

Run the evaluation:

```bash
inspect eval experiments/inspect_eval.py --model <model_name> --limit <# of samples you want to check>
```

Replace `<model_name>` with the LLM you want to benchmark (e.g. `openai/gpt-4`, `anthropic/claude-3`). Alternatively replace inspect_eval.py with multiturn.py to run the multiturn.

# File Structure

#### Data Files (`data/`)

- **`false_statements.json`** - 80 factually incorrect statements across 9 categories (Science, Geography, History, Literature, Language, Math, Technology, Pop Culture, Nature)
- **`full_dataset.jsonl`** - 24,240 single-turn prompts, contains the full single-turn dataset
- **`multiturn.jsonl`** - 720 multi-turn dialogues, contains the full multi-turn dataset
- **`truth_map.jsonl`** - Maps false statements to their correct versions
- **`your_output_file.jsonl`** - Generated evaluation dataset of 800 prompts, subset of prompts from `full_dataset.jsonl`

#### Evaluation Scripts (`experiments/`)

- **`inspect_eval.py`** - Evaluation script for single-turn experiments using the Inspect AI framework
- **`multiturn.py`** - Evaluation script for multi-turn experiments using the Inspect AI framework
