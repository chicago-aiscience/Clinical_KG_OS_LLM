#!/usr/bin/env python3
"""Generate improved pipeline overview figure with dual evaluation paths."""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np

def create_pipeline_figure():
    fig, ax = plt.subplots(1, 1, figsize=(18, 11))
    ax.set_xlim(0, 18)
    ax.set_ylim(0, 11)
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
                             linewidth=2.5 if border_color else 1.5)
        ax.add_patch(box)

        if subtext:
            ax.text(x + w/2, y + h/2 + 0.2, text, ha='center', va='center',
                   fontsize=12, fontweight='bold')
            ax.text(x + w/2, y + h/2 - 0.3, subtext, ha='center', va='center',
                   fontsize=10, style='italic', color='#555')
        else:
            ax.text(x + w/2, y + h/2, text, ha='center', va='center',
                   fontsize=12, fontweight='bold')

    def draw_arrow(x1, y1, x2, y2, color='#5D6D7E', style='->', linewidth=2):
        ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                   arrowprops=dict(arrowstyle=style, color=color, lw=linewidth))

    # Title
    ax.text(9, 10.5, 'Clinical GraphRAG Evaluation Pipeline',
           ha='center', va='center', fontsize=20, fontweight='bold')

    # ===== Legend (below title) =====
    legend_y = 9.8
    legend_items = [
        (colors['input'], 'Input Data'),
        (colors['process'], 'Processing'),
        (colors['output'], 'Output'),
        (colors['eval_dev'], 'Dev Eval (Free)'),
        (colors['eval_final'], 'Final Eval ($$$)'),
    ]

    for i, (color, label) in enumerate(legend_items):
        x = 2.5 + i * 3.2
        box = FancyBboxPatch((x, legend_y), 0.5, 0.3,
                             boxstyle="round,pad=0.01,rounding_size=0.05",
                             facecolor=color, edgecolor='#555555', linewidth=1)
        ax.add_patch(box)
        ax.text(x + 0.65, legend_y + 0.15, label, ha='left', va='center', fontsize=11)

    # ===== TOP ROW: KG Construction =====
    ax.text(9, 8.8, 'Stage 1: KG Construction (Your Focus)',
           ha='center', va='center', fontsize=14, fontweight='bold', color='#2C3E50')

    # Transcripts
    draw_box(1.5, 7.0, 2.4, 1.3, colors['input'], 'Transcripts', '(20 patients)')

    # KG Extraction
    draw_box(4.8, 7.0, 2.8, 1.3, colors['process'], 'KG Extraction', '(kg_extraction.py)')
    ax.text(6.2, 8.45, 'Step 1', ha='center', fontsize=10, color='#888')

    # Entity Resolution
    draw_box(8.5, 7.0, 2.8, 1.3, colors['process'], 'Entity Resolution', '(dump_graph.py)')
    ax.text(9.9, 8.45, 'Step 2', ha='center', fontsize=10, color='#888')

    # Unified KG
    draw_box(12.2, 7.0, 2.4, 1.3, colors['output'], 'Unified KG', '(.json)')

    # Arrows for top row
    draw_arrow(3.9, 7.65, 4.8, 7.65)
    draw_arrow(7.6, 7.65, 8.5, 7.65)
    draw_arrow(11.3, 7.65, 12.2, 7.65)

    # ===== EVALUATION SPLIT =====

    # Vertical arrow down from Unified KG
    draw_arrow(13.4, 7.0, 13.4, 6.0)

    # Split point text
    ax.text(13.4, 5.7, 'Evaluate', ha='center', va='center', fontsize=11,
           fontweight='bold', color='#2C3E50')

    # ===== LEFT BRANCH: Dev Evaluation (Cheap/Fast) =====
    ax.text(5, 5.2, 'Development Path', ha='center', va='center',
           fontsize=13, fontweight='bold', color='#8E44AD')
    ax.text(5, 4.75, '(Your Team - Fast & Free)', ha='center', va='center',
           fontsize=11, color='#8E44AD')

    # Curated KG (baseline)
    draw_box(1.5, 3.0, 2.4, 1.1, colors['input'], 'Curated KG', '(baseline)')

    # KG Similarity Scorer
    draw_box(5.0, 2.7, 3.2, 1.6, colors['eval_dev'], 'KG Similarity', '(kg_similarity_scorer.py)',
            border_color='#8E44AD')
    ax.text(6.6, 4.45, 'Step 3a', ha='center', fontsize=10, color='#8E44AD')

    # KG Score output
    draw_box(5.2, 0.8, 2.8, 1.3, colors['eval_dev'], 'Composite Score', '(r=0.94 with QA)',
            border_color='#8E44AD')

    # Arrows for left branch
    draw_arrow(13.4, 5.4, 8.2, 4.3, color='#8E44AD')  # From split to scorer
    draw_arrow(3.9, 3.55, 5.0, 3.55, color='#8E44AD')   # Curated to scorer
    draw_arrow(6.6, 2.7, 6.6, 2.1, color='#8E44AD')   # Scorer to score

    # ===== RIGHT BRANCH: Final Evaluation (Expensive) =====
    ax.text(14.5, 5.2, 'Final Evaluation', ha='center', va='center',
           fontsize=13, fontweight='bold', color='#C0392B')
    ax.text(14.5, 4.75, '(Organizers - ~$8-10/run)', ha='center', va='center',
           fontsize=11, color='#C0392B')

    # Questions input
    draw_box(9.5, 3.0, 2.2, 1.1, colors['input'], 'Questions', '(7 types × 20)')

    # GraphRAG QA
    draw_box(12.2, 2.7, 2.8, 1.6, colors['process'], 'GraphRAG QA', '(graphrag_qa_pipeline.py)')
    ax.text(13.6, 4.45, 'Step 3b', ha='center', fontsize=10, color='#C0392B')

    # LLM Judge
    draw_box(15.5, 2.7, 2.4, 1.6, colors['eval_final'], 'LLM Judge', '(4 judges × 3 trials)',
            border_color='#C0392B')
    ax.text(16.7, 4.45, 'Step 4', ha='center', fontsize=10, color='#C0392B')

    # QA Scores output
    draw_box(14.0, 0.8, 2.6, 1.3, colors['eval_final'], 'QA Scores', '(4 dims ensemble)',
            border_color='#C0392B')

    # Arrows for right branch
    draw_arrow(13.4, 5.4, 13.6, 4.3, color='#C0392B')  # From split to QA
    draw_arrow(11.7, 3.55, 12.2, 3.55, color='#C0392B')   # Questions to QA
    draw_arrow(15.0, 3.5, 15.5, 3.5, color='#C0392B') # QA to Judge
    draw_arrow(16.7, 2.7, 15.3, 2.1, color='#C0392B')  # Judge to scores

    # ===== Correlation indicator =====
    # Dashed line showing correlation between two scores
    ax.annotate('', xy=(14.0, 1.45), xytext=(8.0, 1.45),
               arrowprops=dict(arrowstyle='<->', color=colors['correlation'],
                              lw=2.5, linestyle='--'))
    ax.text(11.0, 1.8, 'r = 0.94', ha='center', va='center', fontsize=13,
           fontweight='bold', color=colors['correlation'])
    ax.text(11.0, 1.1, 'high correlation', ha='center', va='center', fontsize=11,
           color=colors['correlation'])

    # ===== Recommendation box =====
    rec_box = FancyBboxPatch((1.5, 0.15), 8.0, 0.55,
                             boxstyle="round,pad=0.02,rounding_size=0.1",
                             facecolor='#EAFAF1', edgecolor='#27AE60', linewidth=2)
    ax.add_patch(rec_box)
    ax.text(5.5, 0.42, 'Tip: Use KG Similarity Score during development for fast iteration',
           ha='center', va='center', fontsize=11, color='#1E8449', fontweight='bold')

    plt.tight_layout()
    plt.savefig('figures/pipeline_overview.png', dpi=150, bbox_inches='tight',
               facecolor='white', edgecolor='none')
    print("Saved: figures/pipeline_overview.png")
    plt.close()

if __name__ == '__main__':
    create_pipeline_figure()
