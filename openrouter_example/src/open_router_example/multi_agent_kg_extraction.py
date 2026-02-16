"""
Multi-Agent Knowledge Graph Extraction with Critique-Refine Pattern
Uses three agents: Doctor (Extractor), Critic, and Refiner
"""

import argparse
import json
import os
import re
from pathlib import Path

from openai import OpenAI

from schema_prompt import BASE_EXTRACTION_PROMPT
from agent_prompts import CRITIC_PROMPT, REFINE_PROMPT

# === Configuration ===
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "your-api-key-here")
MODEL = "zhipu-ai/glm-4.7-flash"  # Baseline. Alternatives: qwen/qwen3-14b, deepseek/deepseek-r1-distill-qwen-32b
MAX_TOKENS = 4000

# === OpenRouter API Client ===
client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=OPENROUTER_API_KEY,
)

def read_transcript(file_path: str) -> str:
    """Read transcript from file"""
    with open(file_path, 'r') as f:
        return f.read()

def call_llm(prompt: str, agent_name: str = "Agent") -> str:
    """Call OpenRouter LLM with a prompt"""
    print(f"\n[{agent_name}] Calling {MODEL}...")

    completion = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.1,
        max_tokens=MAX_TOKENS,
    )

    response = completion.choices[0].message.content
    print(f"[{agent_name}] Response received ({len(response)} chars)")
    return response

def extract_json_from_response(response_text: str, save_raw: bool = False) -> dict:
    """Extract JSON from LLM response, handling markdown code blocks"""

    # Optionally save raw response for debugging
    if save_raw:
        with open("debug_raw_response.txt", "w") as f:
            f.write(response_text)

    # Try direct JSON parse first
    try:
        result = json.loads(response_text)
        print(f"  ✓ JSON parsed directly")
        return result
    except json.JSONDecodeError:
        pass  # Try other methods

    # Try extracting JSON from markdown code blocks
    json_match = re.search(r'```(?:json)?\s*(\{.*\})\s*```', response_text, re.DOTALL)
    if json_match:
        try:
            result = json.loads(json_match.group(1))
            print(f"  ✓ JSON extracted from markdown code block")
            return result
        except json.JSONDecodeError:
            pass  # Try next method

    # Try finding JSON object directly (greedy match)
    json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
    if json_match:
        json_str = json_match.group(0)

        # Try parsing as-is
        try:
            result = json.loads(json_str)
            print(f"  ✓ JSON extracted from response")
            return result
        except json.JSONDecodeError:
            pass  # Try fixing

        # Try to fix common issues
        print(f"  ⚠ JSON has errors, attempting to fix...")

        # Remove trailing commas before closing braces/brackets
        fixed_json = re.sub(r',(\s*[}\]])', r'\1', json_str)

        # Try parsing fixed version
        try:
            result = json.loads(fixed_json)
            print(f"  ✓ JSON fixed and parsed successfully")
            return result
        except json.JSONDecodeError as e:
            print(f"  ✗ Could not fix JSON: {e}")

            # Save problematic JSON for inspection
            with open("debug_invalid_json.txt", "w") as f:
                f.write(json_str)
            print(f"  ✗ Invalid JSON saved to debug_invalid_json.txt")
            if save_raw:
                print(f"  ✗ Raw response saved to debug_raw_response.txt")

    print(f"  ✗ Could not extract valid JSON from response")
    return None

def validate_knowledge_graph(kg: dict, fix: bool = True) -> tuple[dict, list[str]]:
    """Validate knowledge graph integrity and optionally fix issues

    Args:
        kg: Knowledge graph dictionary
        fix: If True, remove invalid edges; if False, just report

    Returns:
        Tuple of (fixed_kg, issues_list)
    """
    issues = []

    if not kg or 'nodes' not in kg or 'edges' not in kg:
        issues.append("Missing 'nodes' or 'edges' key in knowledge graph")
        return kg, issues

    # Build set of valid node IDs
    node_ids = {node['id'] for node in kg.get('nodes', [])}

    # Check edges
    valid_edges = []
    invalid_edges = []

    for edge in kg.get('edges', []):
        source = edge.get('source_id')
        target = edge.get('target_id')

        if not source or not target:
            issues.append(f"Edge missing source_id or target_id: {edge}")
            invalid_edges.append(edge)
            continue

        if source not in node_ids:
            issues.append(f"Edge references non-existent source node '{source}' -> '{target}'")
            invalid_edges.append(edge)
            continue

        if target not in node_ids:
            issues.append(f"Edge references non-existent target node '{source}' -> '{target}'")
            invalid_edges.append(edge)
            continue

        valid_edges.append(edge)

    # Report summary
    if invalid_edges:
        print(f"\n⚠ Validation found {len(invalid_edges)} invalid edges out of {len(kg['edges'])} total")
        if fix:
            print(f"  ✓ Removed {len(invalid_edges)} invalid edges")
            kg['edges'] = valid_edges
        else:
            print(f"  Use fix=True to remove invalid edges")

    return kg, issues

def doctor_agent(transcript: str) -> dict:
    """Doctor Agent: Initial knowledge graph extraction"""
    print("\n" + "="*80)
    print("DOCTOR AGENT: Extracting initial knowledge graph...")
    print("="*80)

    prompt = BASE_EXTRACTION_PROMPT.format(transcript=transcript)
    response = call_llm(prompt, "DOCTOR")
    kg = extract_json_from_response(response)

    if kg:
        print(f"✓ Extracted {len(kg.get('nodes', []))} nodes and {len(kg.get('edges', []))} edges")

        # Validate and fix
        kg, issues = validate_knowledge_graph(kg, fix=True)

    return kg

def critic_agent(transcript: str, kg: dict) -> str:
    """Critic Agent: Review and critique the knowledge graph"""
    print("\n" + "="*80)
    print("CRITIC AGENT: Reviewing knowledge graph...")
    print("="*80)

    kg_json = json.dumps(kg, indent=2)
    prompt = CRITIC_PROMPT.format(transcript=transcript, kg_json=kg_json)
    critique = call_llm(prompt, "CRITIC")

    print(f"\n[CRITIC] Feedback:\n{critique[:500]}...")

    return critique

def refiner_agent(transcript: str, kg: dict, critique: str) -> dict:
    """Refiner Agent: Improve knowledge graph based on critique"""
    print("\n" + "="*80)
    print("REFINER AGENT: Refining knowledge graph...")
    print("="*80)

    kg_json = json.dumps(kg, indent=2)
    prompt = REFINE_PROMPT.format(transcript=transcript, kg_json=kg_json, critique=critique)

    # Use higher token limit for refiner (needs to output full graph)
    print(f"[REFINER] Calling {MODEL} with increased token limit...")
    completion = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.1,
        max_tokens=6000,  # Increased for full graph output
    )

    response = completion.choices[0].message.content
    print(f"[REFINER] Response received ({len(response)} chars)")

    # Enable debug saving for refiner
    refined_kg = extract_json_from_response(response, save_raw=False)

    if refined_kg:
        print(f"✓ Refined graph has {len(refined_kg.get('nodes', []))} nodes and {len(refined_kg.get('edges', []))} edges")

        # Validate and fix
        refined_kg, _ = validate_knowledge_graph(refined_kg, fix=True)

        # Show improvement (after validation)
        initial_nodes = len(kg.get('nodes', []))
        initial_edges = len(kg.get('edges', []))
        refined_nodes = len(refined_kg.get('nodes', []))
        refined_edges = len(refined_kg.get('edges', []))

        print(f"  Improvement: +{refined_nodes - initial_nodes} nodes, +{refined_edges - initial_edges} edges")
    else:
        print("✗ Failed to parse refined JSON. Check debug_raw_response.txt and debug_invalid_json.txt")

    return refined_kg

def save_results(initial_kg: dict, critique: str, refined_kg: dict,
                 transcript_path: str, output_dir: str):
    """Save all results to files"""
    transcript_name = Path(transcript_path).stem
    output_dir_path = Path(output_dir)
    output_dir_path.mkdir(parents=True, exist_ok=True)

    # Save initial extraction
    initial_file = output_dir_path / f"{transcript_name}_initial.json"
    with open(initial_file, 'w') as f:
        json.dump(initial_kg, f, indent=2)
    print(f"\n✓ Initial KG saved to {initial_file}")

    # Save critique
    critique_file = output_dir_path / f"{transcript_name}_critique.txt"
    with open(critique_file, 'w') as f:
        f.write(critique)
    print(f"✓ Critique saved to {critique_file}")

    # Save refined extraction
    refined_file = output_dir_path / f"{transcript_name}_refined.json"
    with open(refined_file, 'w') as f:
        json.dump(refined_kg, f, indent=2)
    print(f"✓ Refined KG saved to {refined_file}")

    # Save summary
    summary_file = output_dir_path / f"{transcript_name}_summary.json"
    summary = {
        "transcript": transcript_path,
        "initial_stats": {
            "nodes": len(initial_kg.get('nodes', [])),
            "edges": len(initial_kg.get('edges', []))
        },
        "refined_stats": {
            "nodes": len(refined_kg.get('nodes', [])),
            "edges": len(refined_kg.get('edges', []))
        },
        "improvement": {
            "nodes_added": len(refined_kg.get('nodes', [])) - len(initial_kg.get('nodes', [])),
            "edges_added": len(refined_kg.get('edges', [])) - len(initial_kg.get('edges', []))
        }
    }
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"✓ Summary saved to {summary_file}")

def parse_arguments():
    """Parse command-line arguments"""
    parser = argparse.ArgumentParser(
        description="Multi-agent knowledge graph extraction with critique-refine pattern"
    )
    parser.add_argument(
        "-t", "--transcript",
        required=True,
        help="Path to transcript file"
    )
    parser.add_argument(
        "-o", "--output-dir",
        default=".",
        help="Output directory for results (default: current directory)"
    )
    return parser.parse_args()

# === Main Execution ===
if __name__ == "__main__":
    args = parse_arguments()

    print("="*80)
    print("MULTI-AGENT KNOWLEDGE GRAPH EXTRACTION")
    print("Architecture: Doctor (Extractor) → Critic → Refiner")
    print("="*80)

    # Read transcript
    transcript_path = args.transcript
    print(f"\nReading transcript: {transcript_path}")
    transcript = read_transcript(transcript_path)
    print(f"Transcript length: {len(transcript)} characters")

    # Agent 1: Doctor (Initial Extraction)
    initial_kg = doctor_agent(transcript)

    if not initial_kg:
        print("\n✗ Initial extraction failed. Exiting.")
        exit(1)

    # Agent 2: Critic (Review)
    critique = critic_agent(transcript, initial_kg)

    # Agent 3: Refiner (Improve)
    refined_kg = refiner_agent(transcript, initial_kg, critique)

    if not refined_kg:
        print("\n✗ Refinement failed. Using initial extraction.")
        refined_kg = initial_kg

    # Save all results
    print("\n" + "="*80)
    print("SAVING RESULTS")
    print("="*80)
    save_results(initial_kg, critique, refined_kg, transcript_path, args.output_dir)

    print("\n" + "="*80)
    print("✓ MULTI-AGENT EXTRACTION COMPLETE")
    print("="*80)
