# Clinical GraphRAG Evaluation

Evaluate Knowledge Graph extraction methods for clinical transcripts using GraphRAG QA.

## Pipeline Overview

![Pipeline](figures/pipeline_overview.png)

**Stage 1: KG Construction**
- Extract entities & relations from clinical transcripts (per-patient)
- Merge into unified graph with entity resolution (BGE-M3 embedding, 0.85 threshold)

**Stage 2: Evaluation**
- GraphRAG retrieves relevant KG triples to answer clinical questions
- 4 LLM judges (GPT/Claude/Gemini/Grok) × 3 trials score each answer

---

## Results

![Comparison](figures/comparison_all_methods.png)

| Method | Score | vs Curated | Cost |
|--------|-------|------------|------|
| Curated (Gold) | 3.49 | 100% | $40 |
| Self-Critic GLM | 3.24 | 93% | $0.12 |
| Naive | 3.08 | 88% | $0.05 |
| Text RAG | 2.14 | 61% | $0 |

**Takeaway**: Self-Critic achieves 93% quality at 0.3% cost.

---

## Quick Start

```bash
# Step 1: Setup
cp api_keys_example.json api_keys.json  # Add your API keys

# Step 2: KG Extraction
python kg_extraction.py --input evaluation_bundle/transcripts --output my_kg/ --method self-critic

# Step 3: Entity Resolution
python dump_graph.py --input my_kg/ --output my_unified_graph.json

# Step 4: GraphRAG QA
python graphrag_qa_pipeline.py --kg my_unified_graph.json --bundle evaluation_bundle/ --output my_results/

# Step 5: LLM Judge
python llm_judge_batch_parallel.py --results my_results/ --output my_scores/

# Step 6: Compare to baseline
python kg_similarity_scorer.py --student my_unified_graph.json --baseline baseline_curated/unified_graph_curated.json
```

## Structure

```
├── evaluation_bundle/     # 20 patients × 7 question types
├── baseline_*/            # Pre-computed baselines with scores
├── figures/               # Visualizations
└── *.py                   # Pipeline scripts
```
