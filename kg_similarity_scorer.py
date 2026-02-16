"""
KG Similarity Scorer
====================
Compare student KG against curated baseline without running expensive QA pipeline.
Provides quick feedback on KG quality.

Usage:
    python kg_similarity_scorer.py --student path/to/student_kg.json
    python kg_similarity_scorer.py --student student.json --baseline baseline.json --output report.json

Output metrics:
    - Node overlap (Jaccard with fuzzy matching)
    - Edge overlap (triple matching)
    - Per-patient coverage
    - Type distribution similarity
    - Structural metrics (degree, connectivity)
"""

import json
import argparse
import numpy as np
from pathlib import Path
from collections import Counter, defaultdict
from difflib import SequenceMatcher

# Default baseline
DEFAULT_BASELINE = "baseline_curated/unified_graph_curated.json"


def load_kg(path: str) -> dict:
    """Load KG from JSON file."""
    with open(path) as f:
        return json.load(f)


def normalize_text(text: str) -> str:
    """Normalize text for comparison."""
    return text.lower().strip()


def fuzzy_match(text1: str, text2: str, threshold: float = 0.8) -> bool:
    """Check if two texts are similar enough."""
    t1, t2 = normalize_text(text1), normalize_text(text2)
    if t1 == t2:
        return True
    ratio = SequenceMatcher(None, t1, t2).ratio()
    return ratio >= threshold


def get_node_texts(kg: dict) -> set:
    """Extract normalized node texts."""
    return {normalize_text(n['text']) for n in kg.get('nodes', [])}


def get_node_by_res(kg: dict) -> dict:
    """Group node texts by res_id."""
    res_nodes = defaultdict(set)
    for n in kg.get('nodes', []):
        for occ in n.get('occurrences', []):
            res_nodes[occ['res_id']].add(normalize_text(n['text']))
    return dict(res_nodes)


def get_edge_triples(kg: dict) -> set:
    """Extract (source_text, edge_type, target_text) triples."""
    # Build node id -> text mapping
    id_to_text = {n['id']: normalize_text(n['text']) for n in kg.get('nodes', [])}

    triples = set()
    for e in kg.get('edges', []):
        src = id_to_text.get(e['source_id'], '')
        tgt = id_to_text.get(e['target_id'], '')
        etype = e.get('type', 'UNKNOWN').upper()
        if src and tgt:
            triples.add((src, etype, tgt))
    return triples


def get_type_distribution(kg: dict) -> Counter:
    """Get distribution of node types."""
    types = [n.get('type', 'UNKNOWN').upper() for n in kg.get('nodes', [])]
    return Counter(types)


def jaccard_similarity(set1: set, set2: set) -> float:
    """Calculate Jaccard similarity between two sets."""
    if not set1 and not set2:
        return 1.0
    if not set1 or not set2:
        return 0.0
    intersection = len(set1 & set2)
    union = len(set1 | set2)
    return intersection / union if union > 0 else 0.0


def fuzzy_node_overlap(student_nodes: set, baseline_nodes: set, threshold: float = 0.8) -> dict:
    """Calculate fuzzy node overlap."""
    matched_student = set()
    matched_baseline = set()

    for sn in student_nodes:
        for bn in baseline_nodes:
            if fuzzy_match(sn, bn, threshold):
                matched_student.add(sn)
                matched_baseline.add(bn)
                break

    precision = len(matched_student) / len(student_nodes) if student_nodes else 0
    recall = len(matched_baseline) / len(baseline_nodes) if baseline_nodes else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    return {
        'matched_count': len(matched_student),
        'student_total': len(student_nodes),
        'baseline_total': len(baseline_nodes),
        'precision': round(precision, 4),
        'recall': round(recall, 4),
        'f1': round(f1, 4)
    }


def type_distribution_similarity(student_dist: Counter, baseline_dist: Counter) -> float:
    """Calculate cosine similarity between type distributions."""
    all_types = set(student_dist.keys()) | set(baseline_dist.keys())
    if not all_types:
        return 1.0

    vec1 = np.array([student_dist.get(t, 0) for t in all_types])
    vec2 = np.array([baseline_dist.get(t, 0) for t in all_types])

    # Normalize
    vec1 = vec1 / (np.linalg.norm(vec1) + 1e-8)
    vec2 = vec2 / (np.linalg.norm(vec2) + 1e-8)

    return float(np.dot(vec1, vec2))


def calculate_structural_metrics(kg: dict) -> dict:
    """Calculate structural metrics like average degree."""
    nodes = kg.get('nodes', [])
    edges = kg.get('edges', [])

    if not nodes:
        return {'avg_degree': 0, 'node_count': 0, 'edge_count': 0}

    # Calculate degree
    degree = defaultdict(int)
    for e in edges:
        degree[e['source_id']] += 1
        degree[e['target_id']] += 1

    avg_degree = sum(degree.values()) / len(nodes) if nodes else 0

    return {
        'node_count': len(nodes),
        'edge_count': len(edges),
        'avg_degree': round(avg_degree, 2)
    }


def per_patient_coverage(student_kg: dict, baseline_kg: dict) -> dict:
    """Calculate coverage for each patient."""
    student_by_res = get_node_by_res(student_kg)
    baseline_by_res = get_node_by_res(baseline_kg)

    all_res = set(student_by_res.keys()) | set(baseline_by_res.keys())

    coverage = {}
    for res_id in sorted(all_res):
        s_nodes = student_by_res.get(res_id, set())
        b_nodes = baseline_by_res.get(res_id, set())

        if not b_nodes:
            coverage[res_id] = {'recall': 1.0 if not s_nodes else 0.0, 'baseline_count': 0}
        else:
            matched = sum(1 for sn in s_nodes if any(fuzzy_match(sn, bn) for bn in b_nodes))
            coverage[res_id] = {
                'recall': round(matched / len(b_nodes), 4),
                'matched': matched,
                'student_count': len(s_nodes),
                'baseline_count': len(b_nodes)
            }

    avg_recall = np.mean([c['recall'] for c in coverage.values()])

    return {
        'per_patient': coverage,
        'avg_recall': round(avg_recall, 4)
    }


def compute_similarity(student_kg: dict, baseline_kg: dict) -> dict:
    """Compute all similarity metrics."""

    # Node metrics
    student_nodes = get_node_texts(student_kg)
    baseline_nodes = get_node_texts(baseline_kg)
    node_overlap = fuzzy_node_overlap(student_nodes, baseline_nodes)
    node_jaccard = jaccard_similarity(student_nodes, baseline_nodes)

    # Edge metrics
    student_edges = get_edge_triples(student_kg)
    baseline_edges = get_edge_triples(baseline_kg)
    edge_jaccard = jaccard_similarity(student_edges, baseline_edges)

    # Type distribution
    student_types = get_type_distribution(student_kg)
    baseline_types = get_type_distribution(baseline_kg)
    type_sim = type_distribution_similarity(student_types, baseline_types)

    # Structural metrics
    student_struct = calculate_structural_metrics(student_kg)
    baseline_struct = calculate_structural_metrics(baseline_kg)

    # Per-patient coverage
    patient_coverage = per_patient_coverage(student_kg, baseline_kg)

    # Overall score (weighted average)
    overall_score = (
        0.3 * node_overlap['f1'] +
        0.2 * edge_jaccard +
        0.2 * type_sim +
        0.3 * patient_coverage['avg_recall']
    )

    return {
        'overall_similarity': round(overall_score, 4),
        'node_overlap': node_overlap,
        'node_jaccard': round(node_jaccard, 4),
        'edge_jaccard': round(edge_jaccard, 4),
        'type_distribution_similarity': round(type_sim, 4),
        'student_types': dict(student_types),
        'baseline_types': dict(baseline_types),
        'structural': {
            'student': student_struct,
            'baseline': baseline_struct
        },
        'per_patient_coverage': patient_coverage
    }


def print_report(result: dict, student_path: str, baseline_path: str):
    """Print formatted report."""
    print('=' * 70)
    print('KG Similarity Report')
    print('=' * 70)
    print(f'Student KG:   {student_path}')
    print(f'Baseline KG:  {baseline_path}')
    print()

    print(f"Overall Similarity Score: {result['overall_similarity']:.2%}")
    print()

    print('Node Overlap:')
    no = result['node_overlap']
    print(f"  Precision: {no['precision']:.2%} ({no['matched_count']}/{no['student_total']} student nodes matched)")
    print(f"  Recall:    {no['recall']:.2%} ({no['matched_count']}/{no['baseline_total']} baseline nodes covered)")
    print(f"  F1 Score:  {no['f1']:.2%}")
    print()

    print(f"Edge Jaccard Similarity: {result['edge_jaccard']:.2%}")
    print(f"Type Distribution Similarity: {result['type_distribution_similarity']:.2%}")
    print()

    print('Structural Comparison:')
    ss, bs = result['structural']['student'], result['structural']['baseline']
    print(f"  Student:   {ss['node_count']} nodes, {ss['edge_count']} edges, avg degree {ss['avg_degree']:.2f}")
    print(f"  Baseline:  {bs['node_count']} nodes, {bs['edge_count']} edges, avg degree {bs['avg_degree']:.2f}")
    print()

    print(f"Per-Patient Average Coverage: {result['per_patient_coverage']['avg_recall']:.2%}")
    print()

    # Grade
    score = result['overall_similarity']
    if score >= 0.8:
        grade = 'A (Excellent)'
    elif score >= 0.6:
        grade = 'B (Good)'
    elif score >= 0.4:
        grade = 'C (Fair)'
    elif score >= 0.2:
        grade = 'D (Needs Improvement)'
    else:
        grade = 'F (Poor)'

    print(f'Grade: {grade}')
    print('=' * 70)


def main():
    parser = argparse.ArgumentParser(description='KG Similarity Scorer')
    parser.add_argument('--student', type=str, required=True, help='Path to student KG JSON')
    parser.add_argument('--baseline', type=str, default=DEFAULT_BASELINE, help='Path to baseline KG JSON')
    parser.add_argument('--output', type=str, default=None, help='Output JSON report path')
    args = parser.parse_args()

    # Load KGs
    student_kg = load_kg(args.student)
    baseline_kg = load_kg(args.baseline)

    # Compute similarity
    result = compute_similarity(student_kg, baseline_kg)

    # Print report
    print_report(result, args.student, args.baseline)

    # Save JSON if requested
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(result, f, indent=2)
        print(f'\nDetailed report saved to: {args.output}')


if __name__ == '__main__':
    main()
