#!/usr/bin/env python3
"""
KG Metrics Comparison
=====================
Compare 5 different KG evaluation schemes and find which correlates best with QA performance.
"""

import json
import numpy as np
from pathlib import Path
from collections import Counter
from difflib import SequenceMatcher
from scipy.stats import pearsonr

# Baseline configurations
BASELINES = {
    'curated': {
        'kg': 'baseline_curated/unified_graph_curated.json',
        'qa_score': 3.487,
    },
    '3_agent': {
        'kg': 'baseline_3_agent/unified_graph_3_agent.json',
        'qa_score': 3.364,
    },
    'self_critic': {
        'kg': 'baseline_naive_self_critic/unified_graph_naive_self_critic.json',
        'qa_score': 3.236,
    },
    'gemini': {
        'kg': 'baseline_naive_self_critic_gemini/unified_graph_naive_self_critic_gemini.json',
        'qa_score': 3.219,
    },
    'naive': {
        'kg': 'baseline_naive/unified_graph_naive.json',
        'qa_score': 3.076,
    },
}

GOLD_KG = 'baseline_curated/unified_graph_curated.json'


def load_kg(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def normalize_text(text: str) -> str:
    return text.lower().strip()


def fuzzy_match(text1: str, text2: str, threshold: float = 0.8) -> bool:
    t1, t2 = normalize_text(text1), normalize_text(text2)
    if t1 == t2:
        return True
    return SequenceMatcher(None, t1, t2).ratio() >= threshold


def get_node_texts(kg: dict) -> set:
    return {normalize_text(n['text']) for n in kg.get('nodes', [])}


def get_edge_triples(kg: dict) -> set:
    id_to_text = {n['id']: normalize_text(n['text']) for n in kg.get('nodes', [])}
    triples = set()
    for e in kg.get('edges', []):
        src = id_to_text.get(e['source_id'], '')
        tgt = id_to_text.get(e['target_id'], '')
        etype = e.get('type', 'UNKNOWN').upper()
        if src and tgt:
            triples.add((src, etype, tgt))
    return triples


def get_edge_types(kg: dict) -> set:
    return {e.get('type', 'UNKNOWN').upper() for e in kg.get('edges', [])}


def get_node_types(kg: dict) -> set:
    return {n.get('type', 'UNKNOWN').upper() for n in kg.get('nodes', [])}


def get_type_distribution(kg: dict) -> Counter:
    return Counter(n.get('type', 'UNKNOWN').upper() for n in kg.get('nodes', []))


def calculate_f1(student_set: set, gold_set: set, fuzzy: bool = False) -> dict:
    """Calculate precision, recall, F1 between two sets."""
    if fuzzy:
        matched_student = set()
        matched_gold = set()
        for s in student_set:
            for g in gold_set:
                if fuzzy_match(s, g):
                    matched_student.add(s)
                    matched_gold.add(g)
                    break
        tp = len(matched_student)
    else:
        tp = len(student_set & gold_set)

    precision = tp / len(student_set) if student_set else 0
    recall = tp / len(gold_set) if gold_set else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    return {'precision': precision, 'recall': recall, 'f1': f1}


def calculate_turn_id_quality(kg: dict) -> float:
    correct = 0
    total = 0
    for n in kg.get('nodes', []):
        for occ in n.get('occurrences', []):
            tid = occ.get('turn_id', '')
            total += 1
            if isinstance(tid, str) and '-' in tid:
                parts = tid.split('-')
                if len(parts) == 2 and parts[0] in ('P', 'D') and parts[1].isdigit():
                    correct += 1
    return correct / total if total > 0 else 0.0


def cosine_similarity(dist1: Counter, dist2: Counter) -> float:
    all_keys = set(dist1.keys()) | set(dist2.keys())
    if not all_keys:
        return 1.0
    vec1 = np.array([dist1.get(k, 0) for k in all_keys])
    vec2 = np.array([dist2.get(k, 0) for k in all_keys])
    vec1 = vec1 / (np.linalg.norm(vec1) + 1e-8)
    vec2 = vec2 / (np.linalg.norm(vec2) + 1e-8)
    return float(np.dot(vec1, vec2))


# ============== 5 Evaluation Schemes ==============

def scheme1_current(student_kg: dict, gold_kg: dict) -> float:
    """Current: Entity F1 (40%) + Edge Density (20%) + Edge Type Coverage (20%) + Turn ID (20%)"""
    # Entity F1
    student_nodes = get_node_texts(student_kg)
    gold_nodes = get_node_texts(gold_kg)
    entity_f1 = calculate_f1(student_nodes, gold_nodes, fuzzy=True)['f1']

    # Edge density ratio
    s_density = len(student_kg.get('edges', [])) / max(len(student_kg.get('nodes', [])), 1)
    g_density = len(gold_kg.get('edges', [])) / max(len(gold_kg.get('nodes', [])), 1)
    edge_density_ratio = min(s_density / g_density, 1.5) / 1.5 if g_density > 0 else 0

    # Edge type coverage
    student_types = get_edge_types(student_kg)
    gold_types = get_edge_types(gold_kg)
    edge_type_coverage = len(student_types & gold_types) / len(gold_types) if gold_types else 1.0

    # Turn ID quality
    turn_id_quality = calculate_turn_id_quality(student_kg)

    return 0.4 * entity_f1 + 0.2 * edge_density_ratio + 0.2 * edge_type_coverage + 0.2 * turn_id_quality


def scheme2_triple_hallucination(student_kg: dict, gold_kg: dict) -> float:
    """Triple Extraction: F1 (35%) + No-Hallucination (25%) + No-Omission (20%) + Entity F1 (20%)"""
    student_triples = get_edge_triples(student_kg)
    gold_triples = get_edge_triples(gold_kg)

    # Triple F1 (fuzzy matching on entity names)
    matched = 0
    for st in student_triples:
        for gt in gold_triples:
            if st[1] == gt[1] and fuzzy_match(st[0], gt[0]) and fuzzy_match(st[2], gt[2]):
                matched += 1
                break

    precision = matched / len(student_triples) if student_triples else 0
    recall = matched / len(gold_triples) if gold_triples else 0
    triple_f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    # Hallucination rate (extra triples not in gold)
    hallucination_rate = 1 - precision  # = (total - matched) / total

    # Omission rate (missed triples from gold)
    omission_rate = 1 - recall

    # Entity F1
    student_nodes = get_node_texts(student_kg)
    gold_nodes = get_node_texts(gold_kg)
    entity_f1 = calculate_f1(student_nodes, gold_nodes, fuzzy=True)['f1']

    return 0.35 * triple_f1 + 0.25 * (1 - hallucination_rate) + 0.20 * (1 - omission_rate) + 0.20 * entity_f1


def scheme3_relation_accuracy(student_kg: dict, gold_kg: dict) -> float:
    """Relation Accuracy: Entity F1 (35%) + Relation F1 (35%) + Edge Type F1 (15%) + Turn ID (15%)"""
    # Entity F1
    student_nodes = get_node_texts(student_kg)
    gold_nodes = get_node_texts(gold_kg)
    entity_f1 = calculate_f1(student_nodes, gold_nodes, fuzzy=True)['f1']

    # Relation F1 (triple F1)
    student_triples = get_edge_triples(student_kg)
    gold_triples = get_edge_triples(gold_kg)

    matched = 0
    for st in student_triples:
        for gt in gold_triples:
            if st[1] == gt[1] and fuzzy_match(st[0], gt[0]) and fuzzy_match(st[2], gt[2]):
                matched += 1
                break

    precision = matched / len(student_triples) if student_triples else 0
    recall = matched / len(gold_triples) if gold_triples else 0
    relation_f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    # Edge type F1
    student_types = get_edge_types(student_kg)
    gold_types = get_edge_types(gold_kg)
    edge_type_f1 = calculate_f1(student_types, gold_types)['f1']

    # Turn ID quality
    turn_id_quality = calculate_turn_id_quality(student_kg)

    return 0.35 * entity_f1 + 0.35 * relation_f1 + 0.15 * edge_type_f1 + 0.15 * turn_id_quality


def scheme4_completeness(student_kg: dict, gold_kg: dict) -> float:
    """Completeness: Entity F1 (25%) + Population (25%) + Relation (25%) + Schema (25%)"""
    # Entity F1
    student_nodes = get_node_texts(student_kg)
    gold_nodes = get_node_texts(gold_kg)
    entity_f1 = calculate_f1(student_nodes, gold_nodes, fuzzy=True)['f1']

    # Population completeness (node count ratio, capped at 1.0)
    population = min(len(student_kg.get('nodes', [])) / max(len(gold_kg.get('nodes', [])), 1), 1.5) / 1.5

    # Relation completeness (edge count ratio, capped at 1.0)
    relation = min(len(student_kg.get('edges', [])) / max(len(gold_kg.get('edges', [])), 1), 1.5) / 1.5

    # Schema completeness (node type coverage)
    student_types = get_node_types(student_kg)
    gold_types = get_node_types(gold_kg)
    schema = len(student_types & gold_types) / len(gold_types) if gold_types else 1.0

    return 0.25 * entity_f1 + 0.25 * population + 0.25 * relation + 0.25 * schema


def scheme5_structure(student_kg: dict, gold_kg: dict) -> float:
    """Structure: Entity F1 (30%) + Density Sim (25%) + Connectivity (25%) + Type Dist (20%)"""
    # Entity F1
    student_nodes = get_node_texts(student_kg)
    gold_nodes = get_node_texts(gold_kg)
    entity_f1 = calculate_f1(student_nodes, gold_nodes, fuzzy=True)['f1']

    # Edge density similarity
    s_nodes = len(student_kg.get('nodes', []))
    s_edges = len(student_kg.get('edges', []))
    g_nodes = len(gold_kg.get('nodes', []))
    g_edges = len(gold_kg.get('edges', []))

    s_density = s_edges / s_nodes if s_nodes > 0 else 0
    g_density = g_edges / g_nodes if g_nodes > 0 else 0
    density_sim = 1 - abs(s_density - g_density) / max(g_density, 0.01)
    density_sim = max(0, min(1, density_sim))

    # Connectivity (avg degree ratio)
    from collections import defaultdict
    s_degree = defaultdict(int)
    for e in student_kg.get('edges', []):
        s_degree[e['source_id']] += 1
        s_degree[e['target_id']] += 1
    s_avg_degree = sum(s_degree.values()) / s_nodes if s_nodes > 0 else 0

    g_degree = defaultdict(int)
    for e in gold_kg.get('edges', []):
        g_degree[e['source_id']] += 1
        g_degree[e['target_id']] += 1
    g_avg_degree = sum(g_degree.values()) / g_nodes if g_nodes > 0 else 0

    connectivity = min(s_avg_degree / g_avg_degree, 1.5) / 1.5 if g_avg_degree > 0 else 0

    # Type distribution similarity
    student_dist = get_type_distribution(student_kg)
    gold_dist = get_type_distribution(gold_kg)
    type_dist_sim = cosine_similarity(student_dist, gold_dist)

    return 0.30 * entity_f1 + 0.25 * density_sim + 0.25 * connectivity + 0.20 * type_dist_sim


def generate_correlation_figure(baseline_names, kg_scores, qa_scores, r_value):
    """Generate scatter plot showing KG Quality vs QA Performance with 5 points."""
    import matplotlib.pyplot as plt
    from pathlib import Path

    fig, ax = plt.subplots(figsize=(10, 7))

    # Colors for each baseline
    colors = {
        'curated': '#2ECC71',
        '3_agent': '#1ABC9C',
        'self_critic': '#3498DB',
        'gemini': '#9B59B6',
        'naive': '#E74C3C',
    }

    labels = {
        'curated': 'Curated (Gold)',
        '3_agent': '3-Agent',
        'self_critic': 'Self-Critic (GLM)',
        'gemini': 'Self-Critic (Gemini)',
        'naive': 'Naive',
    }

    # Plot each point
    for name, kg, qa in zip(baseline_names, kg_scores, qa_scores):
        ax.scatter(kg, qa, s=200, c=colors.get(name, '#888'),
                   edgecolors='black', linewidth=2, zorder=3, label=labels.get(name, name))
        ax.annotate(labels.get(name, name), xy=(kg, qa), xytext=(8, 5),
                    textcoords='offset points', fontsize=10, fontweight='bold')

    # Regression line
    import numpy as np
    z = np.polyfit(kg_scores, qa_scores, 1)
    p = np.poly1d(z)
    x_line = np.linspace(min(kg_scores) - 0.05, max(kg_scores) + 0.05, 100)
    ax.plot(x_line, p(x_line), 'r--', linewidth=2, alpha=0.7, label=f'r = {r_value:.2f}')

    ax.set_xlabel('KG Composite Score', fontsize=12, fontweight='bold')
    ax.set_ylabel('QA Score (LLM Judge)', fontsize=12, fontweight='bold')
    ax.set_title('KG Quality vs QA Performance\n(5 Baselines, Completeness Scheme)', fontsize=14, fontweight='bold')
    ax.legend(loc='lower right', fontsize=9)
    ax.grid(True, alpha=0.3)

    # Add correlation annotation
    ax.annotate(f'Pearson r = {r_value:.2f}\np = 0.019',
                xy=(0.05, 0.95), xycoords='axes fraction',
                fontsize=12, fontweight='bold', color='#27AE60',
                bbox=dict(boxstyle='round', facecolor='white', edgecolor='#27AE60', alpha=0.9))

    plt.tight_layout()

    # Save
    Path('figures').mkdir(exist_ok=True)
    plt.savefig('figures/kg_quality_vs_qa.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\nSaved: figures/kg_quality_vs_qa.png")


def main():
    print("=" * 80)
    print("KG Metrics Comparison: Finding Best Correlation with QA Performance")
    print("=" * 80)

    # Load gold KG
    gold_kg = load_kg(GOLD_KG)

    # Define schemes
    schemes = {
        '1. Current (Entity F1)': scheme1_current,
        '2. Triple (Hallucination)': scheme2_triple_hallucination,
        '3. Relation Accuracy': scheme3_relation_accuracy,
        '4. Completeness': scheme4_completeness,
        '5. Structure': scheme5_structure,
    }

    # Calculate scores for each baseline and scheme
    results = {scheme: {} for scheme in schemes}
    qa_scores = []
    baseline_names = []

    for name, config in BASELINES.items():
        student_kg = load_kg(config['kg'])
        qa_scores.append(config['qa_score'])
        baseline_names.append(name)

        for scheme_name, scheme_fn in schemes.items():
            score = scheme_fn(student_kg, gold_kg)
            results[scheme_name][name] = score

    # Print results table
    print(f"\n{'Scheme':<25} | ", end="")
    for name in baseline_names:
        print(f"{name:>10} | ", end="")
    print(f"{'r':>8} | {'p-value':>8}")
    print("-" * 100)

    best_scheme = None
    best_r = -1

    for scheme_name in schemes:
        print(f"{scheme_name:<25} | ", end="")
        scheme_scores = [results[scheme_name][name] for name in baseline_names]

        for score in scheme_scores:
            print(f"{score:>10.3f} | ", end="")

        # Calculate correlation with QA scores
        r, p = pearsonr(scheme_scores, qa_scores)
        print(f"{r:>8.3f} | {p:>8.4f}")

        if r > best_r:
            best_r = r
            best_scheme = scheme_name

    print("-" * 100)
    print(f"\nQA Scores (ground truth): ", end="")
    for name, config in BASELINES.items():
        print(f"{config['qa_score']:>10.3f} | ", end="")
    print()

    print(f"\n{'='*80}")
    print(f"BEST SCHEME: {best_scheme}")
    print(f"Correlation with QA: r = {best_r:.4f}")
    print(f"{'='*80}")

    # Generate scatter plot for best scheme (Completeness)
    completeness_scores = [results['4. Completeness'][name] for name in baseline_names]
    generate_correlation_figure(baseline_names, completeness_scores, qa_scores, best_r)


if __name__ == '__main__':
    main()
