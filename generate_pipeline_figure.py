#!/usr/bin/env python3
"""Generate improved pipeline overview figure with dual evaluation paths."""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np

def create_pipeline_figure():
    fig, ax = plt.subplots(1, 1, figsize=(16, 10))
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 10)
    ax.axis('off')

    # Colors
    colors = {
        'input': '#D6EAF8',      # Light blue
        'process': '#FCF3CF',    # Light yellow/orange
        'output': '#D5F5E3',     # Light green
        'eval_dev': '#E8DAEF',   # Light purple - Dev evaluation (cheap)
        'eval_final': '#FADBD8', # Light pink - Final evaluation (expensive)
        'arrow': '#5D6D7E',
        'correlation': '#27AE60', # Green for correlation line
    }

    def draw_box(x, y, w, h, color, text, subtext=None, border_color=None):
        box = FancyBboxPatch((x, y), w, h,
                             boxstyle="round,pad=0.02,rounding_size=0.15",
                             facecolor=color,
                             edgecolor=border_color or '#555555',
                             linewidth=2 if border_color else 1.5)
        ax.add_patch(box)

        if subtext:
            ax.text(x + w/2, y + h/2 + 0.15, text, ha='center', va='center',
                   fontsize=11, fontweight='bold')
            ax.text(x + w/2, y + h/2 - 0.25, subtext, ha='center', va='center',
                   fontsize=9, style='italic', color='#555')
        else:
            ax.text(x + w/2, y + h/2, text, ha='center', va='center',
                   fontsize=11, fontweight='bold')

    def draw_arrow(x1, y1, x2, y2, color='#5D6D7E', style='->', linewidth=2):
        ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                   arrowprops=dict(arrowstyle=style, color=color, lw=linewidth))

    # Title
    ax.text(8, 9.5, 'Clinical GraphRAG Evaluation Pipeline',
           ha='center', va='center', fontsize=18, fontweight='bold')

    # ===== TOP ROW: KG Construction =====
    ax.text(8, 8.5, 'Stage 1: KG Construction (Your Focus)',
           ha='center', va='center', fontsize=13, fontweight='bold', color='#2C3E50')

    # Transcripts
    draw_box(0.5, 6.8, 2.2, 1.2, colors['input'], 'Transcripts', '(20 patients)')

    # KG Extraction
    draw_box(3.5, 6.8, 2.5, 1.2, colors['process'], 'KG Extraction', '(kg_extraction.py)')
    ax.text(4.75, 8.15, 'Step 1', ha='center', fontsize=9, color='#888')

    # Entity Resolution
    draw_box(6.8, 6.8, 2.5, 1.2, colors['process'], 'Entity Resolution', '(dump_graph.py)')
    ax.text(8.05, 8.15, 'Step 2', ha='center', fontsize=9, color='#888')

    # Unified KG
    draw_box(10.1, 6.8, 2.2, 1.2, colors['output'], 'Unified KG', '(.json)')

    # Arrows for top row
    draw_arrow(2.7, 7.4, 3.5, 7.4)
    draw_arrow(6.0, 7.4, 6.8, 7.4)
    draw_arrow(9.3, 7.4, 10.1, 7.4)

    # ===== EVALUATION SPLIT =====

    # Vertical arrow down from Unified KG
    draw_arrow(11.2, 6.8, 11.2, 5.8)

    # Split point text
    ax.text(11.2, 5.5, 'Evaluate', ha='center', va='center', fontsize=10,
           fontweight='bold', color='#2C3E50')

    # ===== LEFT BRANCH: Dev Evaluation (Cheap/Fast) =====
    ax.text(4, 5.0, 'Development Path', ha='center', va='center',
           fontsize=12, fontweight='bold', color='#8E44AD')
    ax.text(4, 4.6, '(Fast & Free)', ha='center', va='center',
           fontsize=10, color='#8E44AD')

    # Curated KG (baseline)
    draw_box(1.0, 3.0, 2.2, 1.0, colors['input'], 'Curated KG', '(baseline)')

    # KG Similarity Scorer
    draw_box(4.0, 2.7, 3.0, 1.5, colors['eval_dev'], 'KG Similarity', '(kg_similarity_scorer.py)',
            border_color='#8E44AD')
    ax.text(5.5, 4.35, 'Step 3a', ha='center', fontsize=9, color='#8E44AD')

    # KG Score output
    draw_box(4.2, 0.8, 2.6, 1.2, colors['eval_dev'], 'Composite Score', '(r=0.997 with QA)',
            border_color='#8E44AD')

    # Arrows for left branch
    draw_arrow(11.2, 5.2, 7.0, 4.2, color='#8E44AD')  # From split to scorer
    draw_arrow(3.2, 3.5, 4.0, 3.5, color='#8E44AD')   # Curated to scorer
    draw_arrow(5.5, 2.7, 5.5, 2.0, color='#8E44AD')   # Scorer to score

    # ===== RIGHT BRANCH: Final Evaluation (Expensive) =====
    ax.text(13, 5.0, 'Final Evaluation', ha='center', va='center',
           fontsize=12, fontweight='bold', color='#C0392B')
    ax.text(13, 4.6, '(~$8-10 per run)', ha='center', va='center',
           fontsize=10, color='#C0392B')

    # Questions input
    draw_box(8.5, 3.0, 2.0, 1.0, colors['input'], 'Questions', '(7 types × 20)')

    # GraphRAG QA
    draw_box(11.0, 2.7, 2.5, 1.5, colors['process'], 'GraphRAG QA', '(graphrag_qa_pipeline.py)')
    ax.text(12.25, 4.35, 'Step 3b', ha='center', fontsize=9, color='#C0392B')

    # LLM Judge
    draw_box(14.0, 2.7, 2.3, 1.5, colors['eval_final'], 'LLM Judge', '(4 judges × 3 trials)',
            border_color='#C0392B')
    ax.text(15.15, 4.35, 'Step 4', ha='center', fontsize=9, color='#C0392B')

    # QA Scores output
    draw_box(13.5, 0.8, 2.4, 1.2, colors['eval_final'], 'QA Scores', '(4 dims ensemble)',
            border_color='#C0392B')

    # Arrows for right branch
    draw_arrow(11.2, 5.2, 12.25, 4.2, color='#C0392B')  # From split to QA
    draw_arrow(10.5, 3.5, 11.0, 3.5, color='#C0392B')   # Questions to QA
    draw_arrow(13.5, 3.45, 14.0, 3.45, color='#C0392B') # QA to Judge
    draw_arrow(15.15, 2.7, 14.7, 2.0, color='#C0392B')  # Judge to scores

    # ===== Correlation indicator =====
    # Dashed line showing correlation between two scores
    ax.annotate('', xy=(13.5, 1.4), xytext=(6.8, 1.4),
               arrowprops=dict(arrowstyle='<->', color=colors['correlation'],
                              lw=2, linestyle='--'))
    ax.text(10.15, 1.7, 'r = 0.997', ha='center', va='center', fontsize=11,
           fontweight='bold', color=colors['correlation'])
    ax.text(10.15, 1.15, 'high correlation', ha='center', va='center', fontsize=9,
           color=colors['correlation'])

    # ===== Legend =====
    legend_y = 9.2
    legend_items = [
        (colors['input'], 'Input Data'),
        (colors['process'], 'Processing'),
        (colors['output'], 'Output'),
        (colors['eval_dev'], 'Dev Eval (Free)'),
        (colors['eval_final'], 'Final Eval ($$$)'),
    ]

    for i, (color, label) in enumerate(legend_items):
        x = 1.5 + i * 2.8
        box = FancyBboxPatch((x, legend_y), 0.4, 0.25,
                             boxstyle="round,pad=0.01,rounding_size=0.05",
                             facecolor=color, edgecolor='#555555', linewidth=1)
        ax.add_patch(box)
        ax.text(x + 0.55, legend_y + 0.12, label, ha='left', va='center', fontsize=9)

    # ===== Recommendation box =====
    rec_box = FancyBboxPatch((0.5, 0.2), 7.5, 0.5,
                             boxstyle="round,pad=0.02,rounding_size=0.1",
                             facecolor='#EAFAF1', edgecolor='#27AE60', linewidth=2)
    ax.add_patch(rec_box)
    ax.text(4.25, 0.45, 'Tip: Use KG Similarity Score during development for fast iteration',
           ha='center', va='center', fontsize=10, color='#1E8449', fontweight='bold')

    plt.tight_layout()
    plt.savefig('figures/pipeline_overview.png', dpi=150, bbox_inches='tight',
               facecolor='white', edgecolor='none')
    print("Saved: figures/pipeline_overview.png")
    plt.close()

if __name__ == '__main__':
    create_pipeline_figure()
