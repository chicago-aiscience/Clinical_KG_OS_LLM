"""
Simple OpenRouter POC for Clinical Knowledge Graph Extraction
Adapted from lecture_1_openrouter_api_testing.ipynb
"""

import argparse
import json
import os
import re
from pathlib import Path

from openai import OpenAI

from schema_prompt import BASE_EXTRACTION_PROMPT

# === Configuration ===
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "your-api-key-here")
MODEL = "google/gemini-2.5-flash"  # Free model - alternatives: "deepseek/deepseek-chat-v3.1", "openrouter/free"
MAX_TOKENS = 4000  # Limit response size to stay within free tier (reduce to 2000-3000 for smaller responses)

# === OpenRouter API Client ===
client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=OPENROUTER_API_KEY,
)

def read_transcript(file_path: str) -> str:
    """Read transcript from file"""
    with open(file_path, 'r') as f:
        return f.read()

def validate_knowledge_graph(kg: dict, fix: bool = True) -> dict:
    """Validate knowledge graph integrity and optionally fix issues"""
    if not kg or 'nodes' not in kg or 'edges' not in kg:
        return kg

    # Build set of valid node IDs
    node_ids = {node['id'] for node in kg.get('nodes', [])}

    # Filter out invalid edges
    valid_edges = []
    invalid_count = 0

    for edge in kg.get('edges', []):
        source = edge.get('source_id')
        target = edge.get('target_id')

        if source in node_ids and target in node_ids:
            valid_edges.append(edge)
        else:
            invalid_count += 1

    if invalid_count > 0 and fix:
        print(f"\n⚠ Removed {invalid_count} invalid edges (referencing non-existent nodes)")
        kg['edges'] = valid_edges

    return kg

def extract_knowledge_graph(transcript: str, model: str = MODEL) -> dict:
    """Extract clinical knowledge graph from transcript using OpenRouter API"""

    # Format the prompt with the transcript
    prompt = BASE_EXTRACTION_PROMPT.format(transcript=transcript)

    # Make API call
    print(f"Calling {model}...")
    completion = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "user", "content": prompt}
        ],
        temperature=0.1,  # Low temperature for structured output
        max_tokens=MAX_TOKENS,
    )

    # Extract response
    response_text = completion.choices[0].message.content

    # Parse JSON - handle markdown code blocks
    result = None
    try:
        # Try direct JSON parse first
        result = json.loads(response_text)
    except json.JSONDecodeError:
        # Try extracting JSON from markdown code blocks
        json_match = re.search(r'```(?:json)?\s*(\{.*\})\s*```', response_text, re.DOTALL)
        if json_match:
            try:
                result = json.loads(json_match.group(1))
            except json.JSONDecodeError as e:
                print(f"Failed to parse extracted JSON: {e}")

        # Try finding JSON object directly
        if not result:
            json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
            if json_match:
                try:
                    result = json.loads(json_match.group(0))
                except json.JSONDecodeError as e:
                    print(f"Failed to parse found JSON: {e}")

    if not result:
        print(f"Could not extract valid JSON from response")
        print(f"Full response:\n{response_text}")
        return None

    # Validate and fix the knowledge graph
    result = validate_knowledge_graph(result, fix=True)
    return result

def print_knowledge_graph(kg: dict):
    """Pretty print the knowledge graph"""
    if not kg:
        print("No knowledge graph to display")
        return

    print("\n=== NODES ===")
    for node in kg.get("nodes", []):
        print(f"{node['id']}: {node['text']} ({node['type']})")
        print(f"  Evidence: '{node['evidence']}' [{node['turn_id']}]")

    print("\n=== EDGES ===")
    for edge in kg.get("edges", []):
        print(f"{edge['source_id']} --[{edge['type']}]--> {edge['target_id']}")
        print(f"  Evidence: '{edge['evidence']}' [{edge['turn_id']}]")

def save_knowledge_graph(kg: dict, transcript_path: str, output_dir: str = ".") -> Path:
    """Save knowledge graph to JSON file

    Args:
        kg: Knowledge graph dictionary
        transcript_path: Path to the input transcript file
        output_dir: Directory to save output (default: current directory)

    Returns:
        Path to the saved output file
    """
    # Generate output filename from transcript filename
    transcript_name = Path(transcript_path).stem  # Get filename without extension
    output_dir_path = Path(output_dir)
    output_dir_path.mkdir(parents=True, exist_ok=True)  # Create output dir if needed
    output_file = output_dir_path / f"{transcript_name}_kg.json"

    # Save to file
    with open(output_file, "w") as f:
        json.dump(kg, f, indent=2)

    print(f"\n✓ Saved to {output_file}")
    return output_file

def parse_arguments():
    """Parse command-line arguments

    Returns:
        argparse.Namespace: Parsed arguments with transcript and output_dir attributes
    """
    parser = argparse.ArgumentParser(
        description="Extract clinical knowledge graphs from medical transcripts using OpenRouter API"
    )
    parser.add_argument(
        "-t", "--transcript",
        nargs="?",
        help=f"Path to transcript file"
    )
    parser.add_argument(
        "-o", "--output-dir",
        default=".",
        help="Output directory for knowledge graph JSON (default: current directory)"
    )
    return parser.parse_args()

# === Main Execution ===
if __name__ == "__main__":
    args = parse_arguments()

    print("Clinical Knowledge Graph Extraction Proof of Concept")
    print("=" * 80)

    # Read transcript
    transcript_path = args.transcript
    print(f"\nReading transcript: {transcript_path}")
    transcript = read_transcript(transcript_path)
    print(f"Transcript length: {len(transcript)} characters\n")

    # Extract knowledge graph
    kg = extract_knowledge_graph(transcript)

    # Display and save results
    if kg:
        print_knowledge_graph(kg)
        save_knowledge_graph(kg, transcript_path, args.output_dir)
