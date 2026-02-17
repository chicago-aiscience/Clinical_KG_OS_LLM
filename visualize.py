"""
Unified Visualization Script
============================
Generates all figures for the GraphRAG evaluation project.

Usage:
    python visualize.py                    # Generate all visualizations
    python visualize.py --baseline curated # Generate for specific baseline
    python visualize.py --comparison       # Generate only comparison figures

Output:
    baseline_*/figures/llm_judge_dashboard.png  - Per-baseline dashboards

    Comparison Figures (figures/):
    - comparison_all_methods.png     - 5-method overall comparison
    - comparison_radar.png           - 4-dimension radar chart
    - comparison_question_types.png  - Per question type heatmap
    - comparison_distributions.png   - Score distribution overlay
    - comparison_judge_agreement.png - Cross-method judge consistency
    - ablation_glm_vs_gemini.png     - Ablation study
    - cost_vs_quality.png            - Cost-effectiveness analysis
    - kg_similarity.png              - KG quality comparison
    - kg_stats.png                   - KG size comparison
    - summary_table.png              - Comprehensive summary table
"""

import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from collections import defaultdict

# === Configuration ===
BASELINES = {
    "curated": {
        "name": "Curated (Gold)",
        "short": "Curated",
        "judge_dir": "baseline_curated/judge_scores",
        "kg_file": "baseline_curated/unified_graph_curated.json",
        "color": "#2ECC71",
        "cost": 40.00,
    },
    "self_critic_glm": {
        "name": "Self-Critic (GLM)",
        "short": "SC-GLM",
        "judge_dir": "baseline_naive_self_critic/judge_scores",
        "kg_file": "baseline_naive_self_critic/unified_graph_naive_self_critic.json",
        "color": "#3498DB",
        "cost": 0.12,
    },
    "self_critic_gemini": {
        "name": "Self-Critic (Gemini)",
        "short": "SC-Gemini",
        "judge_dir": "baseline_naive_self_critic_gemini/judge_scores",
        "kg_file": "baseline_naive_self_critic_gemini/unified_graph_naive_self_critic_gemini.json",
        "color": "#9B59B6",
        "cost": 0.50,
    },
    "naive": {
        "name": "Naive (GLM)",
        "short": "Naive",
        "judge_dir": "baseline_naive/judge_scores",
        "kg_file": "baseline_naive/unified_graph_naive.json",
        "color": "#E74C3C",
        "cost": 0.05,
    },
    "text_rag": {
        "name": "Text RAG",
        "short": "TextRAG",
        "judge_dir": "baseline_text_rag/judge_scores",
        "kg_file": None,
        "color": "#F39C12",
        "cost": 0.00,
    },
    "3_agent": {
        "name": "3-Agent (GLM)",
        "short": "3-Agent",
        "judge_dir": "baseline_3_agent/judge_scores",
        "kg_file": "baseline_3_agent/unified_graph_3_agent.json",
        "color": "#1ABC9C",
        "cost": 0.18,
    },
}

DIMS = ["correctness", "completeness", "faithfulness", "relevance"]
DIM_LABELS = ["Correctness", "Completeness", "Faithfulness", "Relevance"]

QUESTION_TYPES = [
    "symptom_recall", "diagnostic_reasoning", "treatment_inventory",
    "clinical_plan", "risk_factors", "ruled_out", "temporal_progression"
]
QUESTION_TYPE_LABELS = [
    "Symptom\nRecall", "Diagnostic\nReasoning", "Treatment\nInventory",
    "Clinical\nPlan", "Risk\nFactors", "Ruled\nOut", "Temporal\nProgression"
]

FIGURES_DIR = Path("figures")


def load_ensemble_results(judge_dir):
    """Load all ensemble results from a judge directory."""
    results = []
    judge_path = Path(judge_dir)
    if not judge_path.exists():
        return results
    for ef in judge_path.glob("*_ensemble_result.json"):
        with open(ef) as f:
            results.append(json.load(f))
    return results


MODELS = ["gpt", "claude", "gemini", "grok"]
MODEL_LABELS = {"gpt": "GPT", "claude": "Claude", "gemini": "Gemini", "grok": "Grok"}


def compute_stats(results):
    """Compute statistics for a baseline."""
    if not results:
        return None

    dim_scores = {d: [] for d in DIMS}
    avg_scores = []
    type_scores = defaultdict(list)

    # Per-model scores
    model_scores = {m: [] for m in MODELS}
    model_dim_scores = {m: {d: [] for d in DIMS} for m in MODELS}

    for r in results:
        if r.get("ensemble"):
            for d in DIMS:
                if d in r["ensemble"]:
                    dim_scores[d].append(r["ensemble"][d])
            if "average" in r["ensemble"]:
                avg_scores.append(r["ensemble"]["average"])
                if "question_type" in r:
                    type_scores[r["question_type"]].append(r["ensemble"]["average"])

        # Collect per-model scores
        if r.get("per_model"):
            for m in MODELS:
                if m in r["per_model"] and "average" in r["per_model"][m]:
                    model_scores[m].append(r["per_model"][m]["average"])
                    for d in DIMS:
                        if d in r["per_model"][m]:
                            model_dim_scores[m][d].append(r["per_model"][m][d])

    if not avg_scores:
        return None

    return {
        "dims": {d: np.mean(dim_scores[d]) if dim_scores[d] else 0 for d in DIMS},
        "dim_std": {d: np.std(dim_scores[d]) if dim_scores[d] else 0 for d in DIMS},
        "avg": np.mean(avg_scores),
        "std": np.std(avg_scores),
        "scores": avg_scores,
        "type_scores": dict(type_scores),
        "n": len(avg_scores),
        "model_scores": {m: model_scores[m] for m in MODELS},
        "model_avgs": {m: np.mean(model_scores[m]) if model_scores[m] else 0 for m in MODELS},
        "model_dim_scores": model_dim_scores,
    }


def load_kg_stats(kg_file):
    """Load KG statistics."""
    if not kg_file or not Path(kg_file).exists():
        return None
    with open(kg_file) as f:
        kg = json.load(f)
    return {
        "nodes": len(kg.get("nodes", [])),
        "edges": len(kg.get("edges", [])),
    }


def generate_baseline_dashboard(key, config, stats, output_dir):
    """Generate dashboard for a single baseline."""
    from scipy import stats as sp_stats

    if not stats:
        print(f"  Skipping {key}: no stats available")
        return

    fig = plt.figure(figsize=(16, 12))
    fig.suptitle(f"{config['name']} - LLM Judge Dashboard", fontsize=14, fontweight='bold')

    # Layout: 2 rows × 3 cols
    # Row 1: Score Distribution, Judge×Dim Heatmap, Per-Judge Bar
    # Row 2: Pearson Correlation, Spearman Correlation, Summary

    # Panel 1: Score Distribution
    ax1 = fig.add_subplot(2, 3, 1)
    ax1.hist(stats["scores"], bins=20, color=config["color"], alpha=0.7, edgecolor='black')
    ax1.axvline(x=stats["avg"], color='red', linestyle='--', linewidth=2, label=f'Mean: {stats["avg"]:.2f}')
    ax1.set_xlabel("Score", fontsize=11)
    ax1.set_ylabel("Count", fontsize=11)
    ax1.set_title("Score Distribution", fontsize=12, fontweight='bold')
    ax1.legend()
    ax1.grid(axis='y', alpha=0.3)

    # Panel 2: Per-Model × Per-Dimension Heatmap
    ax2 = fig.add_subplot(2, 3, 2)
    model_dim_matrix = np.zeros((len(MODELS), len(DIMS)))
    for i, m in enumerate(MODELS):
        for j, d in enumerate(DIMS):
            scores = stats["model_dim_scores"][m][d]
            model_dim_matrix[i, j] = np.mean(scores) if scores else 0

    im = ax2.imshow(model_dim_matrix, cmap='RdYlGn', aspect='auto', vmin=1, vmax=5)
    ax2.set_xticks(range(len(DIMS)))
    ax2.set_xticklabels(DIM_LABELS, fontsize=9)
    ax2.set_yticks(range(len(MODELS)))
    ax2.set_yticklabels([MODEL_LABELS[m] for m in MODELS], fontsize=10)
    ax2.set_title("Judge × Dimension Scores", fontsize=12, fontweight='bold')

    for i in range(len(MODELS)):
        for j in range(len(DIMS)):
            val = model_dim_matrix[i, j]
            color = 'white' if val < 2.5 or val > 4 else 'black'
            ax2.text(j, i, f'{val:.2f}', ha='center', va='center', color=color, fontsize=9)
    plt.colorbar(im, ax=ax2, shrink=0.8)

    # Panel 3: Per-Judge Average Bar
    ax3 = fig.add_subplot(2, 3, 3)
    model_avgs = [stats["model_avgs"][m] for m in MODELS]
    model_colors = ['#10a37f', '#cc785c', '#4285f4', '#1da1f2']
    bars = ax3.bar([MODEL_LABELS[m] for m in MODELS], model_avgs, color=model_colors, alpha=0.8, edgecolor='black')
    ax3.axhline(y=stats["avg"], color='red', linestyle='--', linewidth=2, label=f'Ensemble: {stats["avg"]:.2f}')
    ax3.set_ylabel("Average Score", fontsize=11)
    ax3.set_title("Per-Judge Average", fontsize=12, fontweight='bold')
    ax3.set_ylim(0, 5)
    ax3.legend(loc='upper right', fontsize=9)
    ax3.grid(axis='y', alpha=0.3)
    for bar, val in zip(bars, model_avgs):
        ax3.annotate(f'{val:.2f}', xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                     xytext=(0, 3), textcoords='offset points', ha='center', va='bottom', fontsize=10, fontweight='bold')

    # Compute correlation matrices
    q_scores = {m: np.array(stats["model_scores"][m]) for m in MODELS}
    pearson_matrix = np.zeros((len(MODELS), len(MODELS)))
    spearman_matrix = np.zeros((len(MODELS), len(MODELS)))

    for i, m1 in enumerate(MODELS):
        for j, m2 in enumerate(MODELS):
            if len(q_scores[m1]) > 2 and len(q_scores[m2]) > 2:
                min_len = min(len(q_scores[m1]), len(q_scores[m2]))
                pearson_matrix[i, j], _ = sp_stats.pearsonr(q_scores[m1][:min_len], q_scores[m2][:min_len])
                spearman_matrix[i, j], _ = sp_stats.spearmanr(q_scores[m1][:min_len], q_scores[m2][:min_len])

    # Panel 4: Pearson Correlation Matrix (Judge vs Judge)
    ax4 = fig.add_subplot(2, 3, 4)
    im4 = ax4.imshow(pearson_matrix, cmap='RdYlGn', vmin=0.85, vmax=1.0)
    ax4.set_xticks(range(len(MODELS)))
    ax4.set_yticks(range(len(MODELS)))
    ax4.set_xticklabels([MODEL_LABELS[m] for m in MODELS], fontsize=9, rotation=15, ha='right')
    ax4.set_yticklabels([MODEL_LABELS[m] for m in MODELS], fontsize=9)
    ax4.set_title("Pearson r (Score Agreement)", fontsize=12, fontweight='bold')

    for i in range(len(MODELS)):
        for j in range(len(MODELS)):
            color = 'white' if pearson_matrix[i, j] > 0.95 else 'black'
            ax4.text(j, i, f'{pearson_matrix[i, j]:.3f}', ha='center', va='center',
                     fontsize=10, fontweight='bold', color=color)
    plt.colorbar(im4, ax=ax4, shrink=0.8)

    # Panel 5: Spearman Rank Correlation Matrix (Judge vs Judge)
    ax5 = fig.add_subplot(2, 3, 5)
    im5 = ax5.imshow(spearman_matrix, cmap='RdYlGn', vmin=0.85, vmax=1.0)
    ax5.set_xticks(range(len(MODELS)))
    ax5.set_yticks(range(len(MODELS)))
    ax5.set_xticklabels([MODEL_LABELS[m] for m in MODELS], fontsize=9, rotation=15, ha='right')
    ax5.set_yticklabels([MODEL_LABELS[m] for m in MODELS], fontsize=9)
    ax5.set_title("Spearman ρ (Rank Agreement)", fontsize=12, fontweight='bold')

    for i in range(len(MODELS)):
        for j in range(len(MODELS)):
            color = 'white' if spearman_matrix[i, j] > 0.95 else 'black'
            ax5.text(j, i, f'{spearman_matrix[i, j]:.3f}', ha='center', va='center',
                     fontsize=10, fontweight='bold', color=color)
    plt.colorbar(im5, ax=ax5, shrink=0.8)

    # Panel 6: Summary Stats
    ax6 = fig.add_subplot(2, 3, 6)
    ax6.axis('off')

    # Calculate average inter-judge correlation
    off_diag_pearson = (np.sum(pearson_matrix) - len(MODELS)) / (len(MODELS) * (len(MODELS) - 1))
    off_diag_spearman = (np.sum(spearman_matrix) - len(MODELS)) / (len(MODELS) * (len(MODELS) - 1))

    summary = f"""
    {config['name']}
    {'─' * 36}

    Questions: {stats['n']}
    Score: {stats['avg']:.2f} ± {stats['std']:.2f}

    Dimensions:
      Correctness:   {stats['dims']['correctness']:.2f}
      Completeness:  {stats['dims']['completeness']:.2f}
      Faithfulness:  {stats['dims']['faithfulness']:.2f}
      Relevance:     {stats['dims']['relevance']:.2f}

    Inter-Judge Agreement:
      Pearson r:  {off_diag_pearson:.3f}
      Spearman ρ: {off_diag_spearman:.3f}

    Cost: ${config['cost']:.2f}
    """
    ax6.text(0.1, 0.5, summary, transform=ax6.transAxes, fontsize=10,
             verticalalignment='center', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='lavender', alpha=0.8))

    plt.tight_layout()

    # Save
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "llm_judge_dashboard.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_path}")


def generate_comparison_figures(all_stats):
    """Generate comparison figures across all baselines."""
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    # Filter baselines with valid stats
    valid = [(k, v) for k, v in all_stats.items() if v is not None]
    if not valid:
        print("No valid stats for comparison")
        return

    # === Figure 1: All Methods Comparison ===
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle("GraphRAG QA Evaluation - Method Comparison", fontsize=14, fontweight='bold')

    names = [BASELINES[k]["name"] for k, _ in valid]
    avgs = [v["avg"] for _, v in valid]
    stds = [v["std"] for _, v in valid]
    colors = [BASELINES[k]["color"] for k, _ in valid]

    # Panel 1: Overall Score
    ax1 = axes[0]
    bars = ax1.bar(range(len(names)), avgs, yerr=stds, color=colors, alpha=0.85, capsize=5, edgecolor='black')
    ax1.set_xticks(range(len(names)))
    ax1.set_xticklabels([n.replace(" ", "\n") for n in names], fontsize=9)
    ax1.set_ylabel("LLM Judge Score (1-5)", fontsize=11)
    ax1.set_title("Overall Average Score", fontsize=12, fontweight='bold')
    ax1.set_ylim(0, 5)
    ax1.axhline(y=3, color='gray', linestyle='--', alpha=0.5)
    ax1.grid(axis='y', alpha=0.3)
    for bar, avg in zip(bars, avgs):
        ax1.annotate(f'{avg:.2f}', xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                     xytext=(0, 5), textcoords='offset points', ha='center', va='bottom', fontsize=10, fontweight='bold')

    # Panel 2: Per-Dimension
    ax2 = axes[1]
    x = np.arange(len(DIMS))
    width = 0.8 / len(valid)
    for i, (k, v) in enumerate(valid):
        dim_vals = [v["dims"][d] for d in DIMS]
        ax2.bar(x + i*width - 0.4 + width/2, dim_vals, width, label=BASELINES[k]["short"],
                color=BASELINES[k]["color"], alpha=0.85)
    ax2.set_xticks(x)
    ax2.set_xticklabels(DIM_LABELS, fontsize=10)
    ax2.set_ylabel("Score (1-5)", fontsize=11)
    ax2.set_title("Per-Dimension Scores", fontsize=12, fontweight='bold')
    ax2.set_ylim(0, 5)
    ax2.legend(loc='upper right', fontsize=8)
    ax2.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "comparison_all_methods.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {FIGURES_DIR / 'comparison_all_methods.png'}")

    # === Figure 2: Ablation GLM vs Gemini ===
    if "self_critic_glm" in all_stats and "self_critic_gemini" in all_stats:
        glm = all_stats["self_critic_glm"]
        gemini = all_stats["self_critic_gemini"]

        fig2, ax = plt.subplots(figsize=(10, 6))
        fig2.suptitle("Ablation: GLM-4.7-flash vs Gemini 2.0 Flash", fontsize=14, fontweight='bold')

        x = np.arange(len(DIMS) + 1)
        width = 0.35

        glm_vals = [glm["dims"][d] for d in DIMS] + [glm["avg"]]
        gemini_vals = [gemini["dims"][d] for d in DIMS] + [gemini["avg"]]

        bars1 = ax.bar(x - width/2, glm_vals, width, label='GLM ($0.12)', color='#3498DB', alpha=0.85)
        bars2 = ax.bar(x + width/2, gemini_vals, width, label='Gemini ($0.50)', color='#9B59B6', alpha=0.85)

        ax.set_xticks(x)
        ax.set_xticklabels(DIM_LABELS + ["Overall"], fontsize=10)
        ax.set_ylabel("Score (1-5)", fontsize=11)
        ax.set_ylim(0, 5)
        ax.legend(loc='upper right', fontsize=10)
        ax.grid(axis='y', alpha=0.3)

        for bar, val in zip(bars1, glm_vals):
            ax.annotate(f'{val:.2f}', xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                        xytext=(0, 3), textcoords='offset points', ha='center', va='bottom', fontsize=9)
        for bar, val in zip(bars2, gemini_vals):
            ax.annotate(f'{val:.2f}', xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                        xytext=(0, 3), textcoords='offset points', ha='center', va='bottom', fontsize=9)

        ax.annotate(f"GLM: +{(glm['avg']-gemini['avg']):.2f}\n4x cheaper", xy=(4.5, 2.5),
                    fontsize=10, ha='center', style='italic',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        plt.tight_layout()
        plt.savefig(FIGURES_DIR / "ablation_glm_vs_gemini.png", dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  Saved: {FIGURES_DIR / 'ablation_glm_vs_gemini.png'}")

    # === Figure 3: Cost vs Quality ===
    fig3, axes3 = plt.subplots(1, 2, figsize=(14, 6))
    fig3.suptitle("Cost vs Quality Trade-off", fontsize=14, fontweight='bold')

    costs = [BASELINES[k]["cost"] for k, _ in valid]
    costs_plot = [c if c > 0 else 0.001 for c in costs]

    # Panel 1: Cost Bar
    ax3a = axes3[0]
    bars = ax3a.bar(range(len(names)), costs_plot, color=colors, alpha=0.85, edgecolor='black')
    ax3a.set_xticks(range(len(names)))
    ax3a.set_xticklabels([n.replace(" ", "\n") for n in names], fontsize=9)
    ax3a.set_ylabel("KG Extraction Cost ($)", fontsize=11)
    ax3a.set_title("KG Extraction Cost", fontsize=12, fontweight='bold')
    ax3a.set_yscale('log')
    ax3a.set_ylim(0.001, 100)
    ax3a.grid(axis='y', alpha=0.3)
    for bar, cost in zip(bars, costs):
        label = f'${cost:.2f}' if cost < 1 else f'${cost:.0f}'
        if cost == 0:
            label = '$0'
        ax3a.annotate(label, xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                      xytext=(0, 5), textcoords='offset points', ha='center', va='bottom', fontsize=9, fontweight='bold')

    # Panel 2: Scatter
    ax3b = axes3[1]
    for (k, v), cost, color in zip(valid, costs_plot, colors):
        ax3b.scatter(cost, v["avg"], s=300, c=color, alpha=0.85, edgecolors='black', linewidth=2, zorder=3)
        ax3b.annotate(BASELINES[k]["short"], xy=(cost, v["avg"]), xytext=(5, 5),
                      textcoords='offset points', fontsize=9)
    ax3b.set_xlabel("KG Extraction Cost ($) [Log Scale]", fontsize=11)
    ax3b.set_ylabel("QA Quality (LLM Judge Score)", fontsize=11)
    ax3b.set_title("Quality vs Cost", fontsize=12, fontweight='bold')
    ax3b.set_xscale('log')
    ax3b.set_xlim(0.001, 100)
    ax3b.set_ylim(1.5, 4.5)
    ax3b.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "cost_vs_quality.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {FIGURES_DIR / 'cost_vs_quality.png'}")

    # === Figure 4: KG Similarity ===
    # Load similarity reports if they exist
    sim_files = {
        "naive": FIGURES_DIR / "kg_similarity_naive_vs_curated.json",
        "self_critic_glm": FIGURES_DIR / "kg_similarity_self_critic_vs_curated.json",
        "self_critic_gemini": FIGURES_DIR / "kg_similarity_gemini_vs_curated.json",
    }
    sim_data = {}
    for key, fpath in sim_files.items():
        if fpath.exists():
            with open(fpath) as f:
                sim_data[key] = json.load(f)

    if sim_data:
        fig4, ax4 = plt.subplots(figsize=(10, 6))
        fig4.suptitle("KG Quality vs Curated Baseline", fontsize=14, fontweight='bold')

        metrics = ['Overall', 'Node F1', 'Edge Jaccard', 'Patient Coverage']
        x4 = np.arange(len(metrics))
        width4 = 0.25

        for i, (key, data) in enumerate(sim_data.items()):
            vals = [
                data['overall_similarity'],
                data['node_overlap']['f1'],
                data['edge_jaccard'],
                data['per_patient_coverage']['avg_recall']
            ]
            ax4.bar(x4 + i*width4 - width4, vals, width4, label=BASELINES[key]["name"],
                    color=BASELINES[key]["color"], alpha=0.85)

        ax4.set_xticks(x4)
        ax4.set_xticklabels(metrics, fontsize=10)
        ax4.set_ylabel("Similarity Score", fontsize=11)
        ax4.set_ylim(0, 1)
        ax4.legend(loc='upper right')
        ax4.grid(axis='y', alpha=0.3)

        plt.tight_layout()
        plt.savefig(FIGURES_DIR / "kg_similarity.png", dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  Saved: {FIGURES_DIR / 'kg_similarity.png'}")

    # === Figure 5: Radar Chart (4 Dimensions) ===
    fig5 = plt.figure(figsize=(10, 10))
    ax5 = fig5.add_subplot(111, polar=True)

    angles = np.linspace(0, 2*np.pi, len(DIMS), endpoint=False).tolist()
    angles += angles[:1]  # close the polygon

    for k, v in valid:
        values = [v["dims"][d] for d in DIMS]
        values += values[:1]
        ax5.plot(angles, values, 'o-', linewidth=2, label=BASELINES[k]["short"], color=BASELINES[k]["color"])
        ax5.fill(angles, values, alpha=0.15, color=BASELINES[k]["color"])

    ax5.set_xticks(angles[:-1])
    ax5.set_xticklabels(DIM_LABELS, fontsize=11)
    ax5.set_ylim(0, 5)
    ax5.set_yticks([1, 2, 3, 4, 5])
    ax5.set_yticklabels(['1', '2', '3', '4', '5'], fontsize=9)
    ax5.set_title("Dimension Comparison (Radar)", fontsize=14, fontweight='bold', pad=20)
    ax5.legend(loc='upper right', bbox_to_anchor=(1.15, 1.1), fontsize=9)
    ax5.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "comparison_radar.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {FIGURES_DIR / 'comparison_radar.png'}")

    # === Figure 6: Question Type Heatmap (Methods × Question Types) ===
    fig6, ax6 = plt.subplots(figsize=(14, 8))

    type_matrix = np.zeros((len(valid), len(QUESTION_TYPES)))
    method_labels = []
    for i, (k, v) in enumerate(valid):
        method_labels.append(BASELINES[k]["short"])
        for j, qt in enumerate(QUESTION_TYPES):
            if qt in v["type_scores"] and v["type_scores"][qt]:
                type_matrix[i, j] = np.mean(v["type_scores"][qt])

    im6 = ax6.imshow(type_matrix, cmap='RdYlGn', aspect='auto', vmin=1.5, vmax=4.5)
    ax6.set_xticks(range(len(QUESTION_TYPES)))
    ax6.set_xticklabels(QUESTION_TYPE_LABELS, fontsize=10, rotation=0)
    ax6.set_yticks(range(len(valid)))
    ax6.set_yticklabels(method_labels, fontsize=11)
    ax6.set_title("Performance by Question Type", fontsize=14, fontweight='bold')

    for i in range(len(valid)):
        for j in range(len(QUESTION_TYPES)):
            val = type_matrix[i, j]
            if val > 0:
                color = 'white' if val < 2.5 or val > 3.8 else 'black'
                ax6.text(j, i, f'{val:.2f}', ha='center', va='center', color=color, fontsize=10, fontweight='bold')

    plt.colorbar(im6, ax=ax6, shrink=0.6, label='Score (1-5)')
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "comparison_question_types.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {FIGURES_DIR / 'comparison_question_types.png'}")

    # === Figure 7: Score Distribution Overlay ===
    fig7, ax7 = plt.subplots(figsize=(12, 6))

    for k, v in valid:
        ax7.hist(v["scores"], bins=20, alpha=0.4, label=BASELINES[k]["short"],
                 color=BASELINES[k]["color"], edgecolor='black', linewidth=0.5)
        ax7.axvline(x=v["avg"], color=BASELINES[k]["color"], linestyle='--', linewidth=2, alpha=0.8)

    ax7.set_xlabel("LLM Judge Score", fontsize=12)
    ax7.set_ylabel("Count", fontsize=12)
    ax7.set_title("Score Distribution Comparison", fontsize=14, fontweight='bold')
    ax7.legend(loc='upper left', fontsize=10)
    ax7.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "comparison_distributions.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {FIGURES_DIR / 'comparison_distributions.png'}")

    # === Figure 8: Cross-Method Judge Agreement ===
    from scipy import stats as sp_stats

    fig8, axes8 = plt.subplots(1, 2, figsize=(14, 6))
    fig8.suptitle("Inter-Judge Agreement Across Methods", fontsize=14, fontweight='bold')

    pearson_by_method = []
    spearman_by_method = []
    method_labels_short = []

    for k, v in valid:
        method_labels_short.append(BASELINES[k]["short"])
        q_scores = {m: np.array(v["model_scores"][m]) for m in MODELS}

        pearson_vals = []
        spearman_vals = []
        for i, m1 in enumerate(MODELS):
            for j, m2 in enumerate(MODELS):
                if i < j and len(q_scores[m1]) > 2 and len(q_scores[m2]) > 2:
                    min_len = min(len(q_scores[m1]), len(q_scores[m2]))
                    r, _ = sp_stats.pearsonr(q_scores[m1][:min_len], q_scores[m2][:min_len])
                    rho, _ = sp_stats.spearmanr(q_scores[m1][:min_len], q_scores[m2][:min_len])
                    pearson_vals.append(r)
                    spearman_vals.append(rho)

        pearson_by_method.append(np.mean(pearson_vals) if pearson_vals else 0)
        spearman_by_method.append(np.mean(spearman_vals) if spearman_vals else 0)

    # Panel 1: Pearson
    bars1 = axes8[0].bar(range(len(valid)), pearson_by_method, color=colors, alpha=0.85, edgecolor='black')
    axes8[0].set_xticks(range(len(valid)))
    axes8[0].set_xticklabels(method_labels_short, fontsize=10)
    axes8[0].set_ylabel("Average Pearson r", fontsize=11)
    axes8[0].set_title("Score Agreement (Pearson)", fontsize=12, fontweight='bold')
    axes8[0].set_ylim(0.8, 1.0)
    axes8[0].axhline(y=0.9, color='gray', linestyle='--', alpha=0.5)
    axes8[0].grid(axis='y', alpha=0.3)
    for bar, val in zip(bars1, pearson_by_method):
        axes8[0].annotate(f'{val:.3f}', xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                          xytext=(0, 3), textcoords='offset points', ha='center', va='bottom', fontsize=9)

    # Panel 2: Spearman
    bars2 = axes8[1].bar(range(len(valid)), spearman_by_method, color=colors, alpha=0.85, edgecolor='black')
    axes8[1].set_xticks(range(len(valid)))
    axes8[1].set_xticklabels(method_labels_short, fontsize=10)
    axes8[1].set_ylabel("Average Spearman ρ", fontsize=11)
    axes8[1].set_title("Rank Agreement (Spearman)", fontsize=12, fontweight='bold')
    axes8[1].set_ylim(0.8, 1.0)
    axes8[1].axhline(y=0.9, color='gray', linestyle='--', alpha=0.5)
    axes8[1].grid(axis='y', alpha=0.3)
    for bar, val in zip(bars2, spearman_by_method):
        axes8[1].annotate(f'{val:.3f}', xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                          xytext=(0, 3), textcoords='offset points', ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "comparison_judge_agreement.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {FIGURES_DIR / 'comparison_judge_agreement.png'}")

    # === Figure 9: KG Stats Comparison ===
    kg_nodes = []
    kg_edges = []
    kg_labels = []

    for k, config in BASELINES.items():
        kg_stat = load_kg_stats(config.get("kg_file"))
        if kg_stat:
            kg_nodes.append(kg_stat["nodes"])
            kg_edges.append(kg_stat["edges"])
            kg_labels.append(BASELINES[k]["short"])

    if kg_nodes:
        fig9, axes9 = plt.subplots(1, 2, figsize=(12, 5))
        fig9.suptitle("Knowledge Graph Statistics", fontsize=14, fontweight='bold')

        kg_colors = [BASELINES[k]["color"] for k in BASELINES if BASELINES[k].get("kg_file") and load_kg_stats(BASELINES[k].get("kg_file"))]

        # Nodes
        bars1 = axes9[0].bar(range(len(kg_labels)), kg_nodes, color=kg_colors[:len(kg_labels)], alpha=0.85, edgecolor='black')
        axes9[0].set_xticks(range(len(kg_labels)))
        axes9[0].set_xticklabels(kg_labels, fontsize=10)
        axes9[0].set_ylabel("Node Count", fontsize=11)
        axes9[0].set_title("Nodes per KG", fontsize=12, fontweight='bold')
        axes9[0].grid(axis='y', alpha=0.3)
        for bar, val in zip(bars1, kg_nodes):
            axes9[0].annotate(f'{val}', xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                              xytext=(0, 3), textcoords='offset points', ha='center', va='bottom', fontsize=10, fontweight='bold')

        # Edges
        bars2 = axes9[1].bar(range(len(kg_labels)), kg_edges, color=kg_colors[:len(kg_labels)], alpha=0.85, edgecolor='black')
        axes9[1].set_xticks(range(len(kg_labels)))
        axes9[1].set_xticklabels(kg_labels, fontsize=10)
        axes9[1].set_ylabel("Edge Count", fontsize=11)
        axes9[1].set_title("Edges per KG", fontsize=12, fontweight='bold')
        axes9[1].grid(axis='y', alpha=0.3)
        for bar, val in zip(bars2, kg_edges):
            axes9[1].annotate(f'{val}', xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                              xytext=(0, 3), textcoords='offset points', ha='center', va='bottom', fontsize=10, fontweight='bold')

        plt.tight_layout()
        plt.savefig(FIGURES_DIR / "kg_stats.png", dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  Saved: {FIGURES_DIR / 'kg_stats.png'}")

    # === Figure 10: Comprehensive Summary Table ===
    fig10, ax10 = plt.subplots(figsize=(16, 8))
    ax10.axis('off')

    # Build table data
    col_labels = ["Method", "Score", "Correct", "Complete", "Faithful", "Relevant", "Std", "vs Curated", "Cost", "ROI"]
    curated_avg = all_stats.get("curated", {})
    curated_avg = curated_avg["avg"] if curated_avg else 1

    table_data = []
    for k, v in valid:
        pct = v["avg"] / curated_avg * 100 if curated_avg else 0
        cost = BASELINES[k]["cost"]
        roi = v["avg"] / max(cost, 0.01)  # ROI = score per dollar
        table_data.append([
            BASELINES[k]["name"],
            f"{v['avg']:.2f}",
            f"{v['dims']['correctness']:.2f}",
            f"{v['dims']['completeness']:.2f}",
            f"{v['dims']['faithfulness']:.2f}",
            f"{v['dims']['relevance']:.2f}",
            f"{v['std']:.2f}",
            f"{pct:.1f}%",
            f"${cost:.2f}",
            f"{roi:.1f}" if cost > 0 else "∞"
        ])

    table = ax10.table(cellText=table_data, colLabels=col_labels,
                       loc='center', cellLoc='center',
                       colColours=['#e6e6e6'] * len(col_labels))
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 2)

    # Color code cells
    for i, (k, v) in enumerate(valid):
        # Score cell - color by value
        score = v["avg"]
        if score >= 3.5:
            table[(i+1, 1)].set_facecolor('#90EE90')
        elif score >= 3.0:
            table[(i+1, 1)].set_facecolor('#FFFFE0')
        else:
            table[(i+1, 1)].set_facecolor('#FFB6C1')

        # Method cell - use baseline color
        table[(i+1, 0)].set_facecolor(BASELINES[k]["color"])
        table[(i+1, 0)].set_text_props(color='white', fontweight='bold')

    ax10.set_title("Comprehensive Evaluation Summary", fontsize=14, fontweight='bold', pad=20)

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "summary_table.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {FIGURES_DIR / 'summary_table.png'}")


def main():
    parser = argparse.ArgumentParser(description="Generate visualizations")
    parser.add_argument("--baseline", type=str, default=None, help="Generate for specific baseline only")
    parser.add_argument("--comparison", action="store_true", help="Generate only comparison figures")
    args = parser.parse_args()

    print("=" * 60)
    print("Visualization Generator")
    print("=" * 60)

    # Load all stats
    all_stats = {}
    for key, config in BASELINES.items():
        results = load_ensemble_results(config["judge_dir"])
        all_stats[key] = compute_stats(results)
        if all_stats[key]:
            print(f"{config['name']}: {all_stats[key]['avg']:.2f} ± {all_stats[key]['std']:.2f} (n={all_stats[key]['n']})")

    print()

    # Generate baseline dashboards
    if not args.comparison:
        print("Generating baseline dashboards...")
        for key, config in BASELINES.items():
            if args.baseline and key != args.baseline:
                continue
            output_dir = Path(config["judge_dir"]).parent / "figures"
            generate_baseline_dashboard(key, config, all_stats.get(key), output_dir)

    # Generate comparison figures
    if not args.baseline:
        print("\nGenerating comparison figures...")
        generate_comparison_figures(all_stats)

    # Print summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    curated_avg = all_stats.get("curated", {})
    curated_avg = curated_avg["avg"] if curated_avg else 1

    print(f"\n{'Method':<25} {'Score':<12} {'vs Curated':<12} {'Cost':<10}")
    print("-" * 60)
    for key, config in BASELINES.items():
        if all_stats.get(key):
            score = all_stats[key]["avg"]
            pct = score / curated_avg * 100
            print(f"{config['name']:<25} {score:.2f}/5{'':>5} {pct:.1f}%{'':>5} ${config['cost']:.2f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
