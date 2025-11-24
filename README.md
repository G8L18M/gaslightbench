# GASLIGHTBENCH: Quantifying LLM Susceptibility to Social Prompting

# Sycophancy Evaluation Pipeline

To start the project, install dependencies:

```bash
pip install --upgrade anthropic openai inspect-ai python-dotenv ipywidgets matplotlib
```

Before running the pipeline, set your OpenAI API key:

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
inspect eval src/inspect_eval.py --model <model_name> --limit <# of samples you want to check>
```

Replace `<model_name>` with the LLM you want to benchmark (e.g. `openai/gpt-4`, `anthropic/claude-3`).
