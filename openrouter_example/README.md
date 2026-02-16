# Clinical Knowledge Graph Extraction with OpenRouter

Simple and clear tools for extracting clinical knowledge graphs from medical transcripts using OpenRouter API.

## Setup

```bash
# Set your OpenRouter API key
export OPENROUTER_API_KEY="your-key-here"

# Install dependencies
uv add openai pyvis
```

## Scripts

### 1. Single-Agent Extraction (Simple)

Extract knowledge graph using one LLM call:

```bash
uv run src/open_router_example/transcript_open_router_poc.py \
    -t /path/to/transcript.txt \
    -o ./output
```

**Output:** `output/RES0198_kg.json`

### 2. Multi-Agent Extraction (Critique-Refine)

Extract knowledge graph using three agents (Doctor → Critic → Refiner):

```bash
uv run src/open_router_example/multi_agent_kg_extraction.py \
    -t /path/to/transcript.txt \
    -o ./output
```

**Outputs:**
- `RES0198_initial.json` - Doctor's initial extraction
- `RES0198_critique.txt` - Critic's feedback
- `RES0198_refined.json` - Refiner's improved version
- `RES0198_summary.json` - Statistics

### 3. Visualize Knowledge Graph

Create interactive HTML visualization:

```bash
# Basic usage
uv run src/open_router_example/visualize_kg.py /path/to/kg.json

# With custom output
uv run src/open_router_example/visualize_kg.py /path/to/kg.json -o viz.html
```

**Output:** `RES0198_kg_viz.html` (open in browser)

## Configuration

Edit the model in each script (**must use approved models only**):

```python
# Approved models:
MODEL = "zhipu-ai/glm-4.7-flash"           # Baseline (default)
MODEL = "qwen/qwen3-14b"
MODEL = "nvidia/nemotron-3-nano-30b-a3b"
MODEL = "openai/gpt-oss-20b"
MODEL = "deepseek/deepseek-r1-distill-qwen-32b"
```

Adjust token limits:

```python
MAX_TOKENS = 4000  # Increase for longer outputs
```

## Features

- **Validation:** Automatically removes invalid edges (references to non-existent nodes)
- **JSON Parsing:** Handles markdown code blocks, trailing commas, and malformed JSON
- **Interactive Viz:** Color-coded nodes, hover tooltips, draggable layout

## Files

```
src/open_router_example/
├── transcript_open_router_poc.py    # Single-agent extraction
├── multi_agent_kg_extraction.py     # Multi-agent critique-refine
├── visualize_kg.py                  # Interactive visualization
├── schema_prompt.py                 # Extraction prompt & schema
└── agent_prompts.py                 # Critic & refiner prompts
```

## Troubleshooting

**JSON parsing errors:**
- Check `debug_raw_response.txt` and `debug_invalid_json.txt`
- Try a different approved model
- Increase `max_tokens`

**402 Payment errors:**
- Reduce `MAX_TOKENS` (e.g., to 2000-3000)
- Check your OpenRouter credit balance
