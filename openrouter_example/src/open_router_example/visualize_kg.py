"""
Visualize Clinical Knowledge Graphs using PyVis
Creates an interactive HTML network visualization
"""

import argparse
import json
from pathlib import Path
from pyvis.network import Network


# Node type color scheme
NODE_COLORS = {
    'SYMPTOM': '#ff6b6b',          # Red
    'DIAGNOSIS': '#4ecdc4',         # Teal
    'TREATMENT': '#45b7d1',         # Blue
    'PROCEDURE': '#96ceb4',         # Green
    'LOCATION': '#ffeaa7',          # Yellow
    'MEDICAL_HISTORY': '#dfe6e9',   # Gray
    'LAB_RESULT': '#a29bfe'         # Purple
}

def load_knowledge_graph(file_path: str) -> dict:
    """Load knowledge graph from JSON file"""
    with open(file_path, 'r') as f:
        return json.load(f)

def visualize_knowledge_graph(kg: dict, output_file: str = "kg_visualization.html"):
    """Create interactive HTML visualization of knowledge graph

    Args:
        kg: Knowledge graph dictionary with nodes and edges
        output_file: Path to save HTML visualization
    """
    # Create network
    net = Network(
        height="800px",
        width="100%",
        directed=True,
        notebook=False,
        bgcolor="#ffffff",
        font_color="#000000"
    )

    # Configure physics for better layout
    net.set_options("""
    {
        "physics": {
            "forceAtlas2Based": {
                "gravitationalConstant": -50,
                "centralGravity": 0.01,
                "springLength": 200,
                "springConstant": 0.08
            },
            "maxVelocity": 50,
            "solver": "forceAtlas2Based",
            "timestep": 0.35,
            "stabilization": {"iterations": 150}
        }
    }
    """)

    # Add nodes and track valid node IDs
    node_ids = set()
    for node in kg.get("nodes", []):
        node_id = node['id']
        node_label = node['text']
        node_type = node['type']
        node_color = NODE_COLORS.get(node_type, '#95a5a6')

        # Create hover tooltip with all node information
        title = f"<b>{node_label}</b><br>"
        title += f"Type: {node_type}<br>"
        title += f"Evidence: {node['evidence']}<br>"
        title += f"Turn: {node['turn_id']}"

        net.add_node(
            node_id,
            label=node_label,
            title=title,
            color=node_color,
            size=25,
            font={'size': 14}
        )
        node_ids.add(node_id)

    # Add edges (skip edges with non-existent nodes)
    skipped_edges = []
    for edge in kg.get("edges", []):
        source = edge['source_id']
        target = edge['target_id']
        edge_type = edge['type']

        # Check if both nodes exist
        if source not in node_ids:
            skipped_edges.append(f"{source} -> {target} (source missing)")
            continue
        if target not in node_ids:
            skipped_edges.append(f"{source} -> {target} (target missing)")
            continue

        # Create hover tooltip for edge
        title = f"{edge_type}<br>"
        title += f"Evidence: {edge['evidence']}<br>"
        title += f"Turn: {edge['turn_id']}"

        net.add_edge(
            source,
            target,
            label=edge_type,
            title=title,
            arrows='to',
            color={'color': '#848484'},
            font={'size': 10, 'align': 'middle'}
        )

    # Report skipped edges
    if skipped_edges:
        print(f"\n⚠ Warning: Skipped {len(skipped_edges)} edges with missing nodes:")
        for edge in skipped_edges[:5]:  # Show first 5
            print(f"  - {edge}")
        if len(skipped_edges) > 5:
            print(f"  ... and {len(skipped_edges) - 5} more")

    # Add legend
    legend_html = """
    <div style="position: fixed; top: 10px; right: 10px; background: white;
                padding: 10px; border: 1px solid #ccc; border-radius: 5px;
                font-family: Arial; font-size: 12px; z-index: 1000;">
        <b>Node Types:</b><br>
    """
    for node_type, color in NODE_COLORS.items():
        legend_html += f'<span style="color: {color};">●</span> {node_type}<br>'
    legend_html += "</div>"

    # Save visualization
    net.save_graph(output_file)

    # Add legend to HTML file
    with open(output_file, 'r') as f:
        html_content = f.read()

    # Insert legend before closing body tag
    html_content = html_content.replace('</body>', f'{legend_html}</body>')

    with open(output_file, 'w') as f:
        f.write(html_content)

    print(f"✓ Interactive visualization saved to {output_file}")
    print(f"  Open in browser: file://{Path(output_file).absolute()}")

def parse_arguments():
    """Parse command-line arguments"""
    parser = argparse.ArgumentParser(
        description="Visualize clinical knowledge graphs as interactive HTML"
    )
    parser.add_argument(
        "kg_file",
        help="Path to knowledge graph JSON file"
    )
    parser.add_argument(
        "-o", "--output",
        help="Output HTML file (default: <kg_filename>_viz.html)"
    )
    return parser.parse_args()

# === Main Execution ===
if __name__ == "__main__":
    args = parse_arguments()

    print("Knowledge Graph Visualization")
    print("=" * 50)

    # Load knowledge graph
    print(f"\nLoading: {args.kg_file}")
    kg = load_knowledge_graph(args.kg_file)

    # Determine output file
    if args.output:
        output_file = args.output
    else:
        kg_path = Path(args.kg_file)
        output_file = kg_path.parent / f"{kg_path.stem}_viz.html"

    # Create visualization
    print(f"Creating visualization...")
    visualize_knowledge_graph(kg, str(output_file))
