"""
Unified KG Extraction Pipeline
==============================
Extract clinical knowledge graphs from transcripts using LLM.

Supports multiple methods and models:
  - Method: naive (1-pass), self-critic (2-pass), or 3-agent (3-pass with critic+refiner)
  - Model: glm-4.7-flash (via OpenRouter), gemini-2.0-flash, etc.

Usage:
    python kg_extraction.py --method naive --model glm --output baseline_naive/sub_kgs
    python kg_extraction.py --method self-critic --model glm --output baseline_self_critic/sub_kgs
    python kg_extraction.py --method 3-agent --model glm --output baseline_3_agent/sub_kgs

After extraction, merge with:
    python dump_graph.py --input baseline_naive/sub_kgs --output baseline_naive/
"""

import json
import re
import argparse
import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

# === Configuration ===
TRANSCRIPT_DIR = Path("src", "Clinical_KG_OS_LLM", "transcripts_1")
MAX_RETRIES = 3

# === Prompts ===
EXTRACTION_PROMPT = """Extract clinical knowledge graph from transcript.

## NODE TYPES:
- SYMPTOM: Patient-reported or observed symptoms (chest pain, shortness of breath)
- DIAGNOSIS: Active or suspected conditions (COPD exacerbation, pneumonia)
- TREATMENT: Medications, therapies, interventions (Aspirin, Metformin, DASH diet)
- PROCEDURE: Tests, exams, surgeries (ECG, stress test, CT angiography)
- LOCATION: Body parts and anatomical locations (chest, left arm, heart)
- MEDICAL_HISTORY: Pre-existing conditions, risk factors (diabetes, smoking)
- LAB_RESULT: Lab values and vital signs (A1C 7.2%, BP 148/90, BNP elevated)

## EDGE TYPES:
- CAUSES: Risk factor causes condition (smoking CAUSES heart disease)
- INDICATES: Symptom indicates diagnosis (chest pain INDICATES angina)
- LOCATED_AT: Symptom at body location (pain LOCATED_AT chest)
- RULES_OUT: Test rules out condition (ECG RULES_OUT arrhythmia)
- TAKEN_FOR: Treatment for condition (Aspirin TAKEN_FOR angina)
- CONFIRMS: Lab/test confirms diagnosis (elevated BNP CONFIRMS heart failure)

TRANSCRIPT:
{transcript}

## FORMAT REQUIREMENTS:
- Node IDs: "N_001", "N_002", etc.
- turn_id: String format "P-X" or "D-X" (P=Patient, D=Doctor, X=turn number)
  Example: "P-1", "D-39" (from [P-1], [D-39] in transcript)

Output JSON with nodes (id, text, type, evidence, turn_id) and edges (source_id, target_id, type, evidence, turn_id).
Output ONLY valid JSON."""

SELF_CRITIC_PROMPT = """Review and improve this clinical knowledge graph.

ORIGINAL TRANSCRIPT:
{transcript}

EXTRACTED KG:
{kg_json}

REVIEW CHECKLIST:
1. Are all symptoms, diagnoses, treatments, procedures, lab results captured?
2. Are node types correct? (SYMPTOM, DIAGNOSIS, TREATMENT, PROCEDURE, LOCATION, MEDICAL_HISTORY, LAB_RESULT)
3. Are edge types accurate? (CAUSES, INDICATES, LOCATED_AT, RULES_OUT, TAKEN_FOR, CONFIRMS)
4. Are there missing relationships that should be added?
5. Are there incorrect or hallucinated nodes/edges to remove?
6. Is evidence properly cited with turn_ids?

Output the IMPROVED knowledge graph. Add missing nodes/edges. Remove incorrect ones. Fix wrong types.

Output ONLY valid JSON:
{{
  "nodes": [...],
  "edges": [...]
}}"""

# === 3-Agent Prompts ===
CRITIC_PROMPT = """You are a Clinical Knowledge Graph Critic. Review this extraction thoroughly.

## SCHEMA REMINDER:
NODE TYPES: SYMPTOM, DIAGNOSIS, TREATMENT, PROCEDURE, LOCATION, MEDICAL_HISTORY, LAB_RESULT
EDGE TYPES: CAUSES, INDICATES, LOCATED_AT, RULES_OUT, TAKEN_FOR, CONFIRMS

## TRANSCRIPT:
{transcript}

## EXTRACTED KG:
{kg_json}

## REVIEW CHECKLIST:
1. MISSING ENTITIES - Check for:
   - Symptoms mentioned but not captured
   - Lab values (A1C, BP, BNP, LDL, etc.) → should be LAB_RESULT
   - Pre-existing conditions → should be MEDICAL_HISTORY (not DIAGNOSIS)
   - Family history → should be MEDICAL_HISTORY
   - Procedures/tests mentioned

2. TYPE ERRORS - Common mistakes:
   - MEDICAL_HISTORY vs DIAGNOSIS (chronic vs acute)
   - SYMPTOM vs DIAGNOSIS (patient complaint vs doctor assessment)
   - TREATMENT vs PROCEDURE (medication vs test)

3. MISSING RELATIONSHIPS - Check for:
   - Risk factors → CAUSES → conditions
   - Symptoms → INDICATES → diagnoses
   - Tests → RULES_OUT or CONFIRMS → conditions
   - Treatments → TAKEN_FOR → conditions

4. HALLUCINATIONS - Information not in transcript

5. ORPHAN NODES - Nodes without any edges

Provide specific critique with quotes from transcript. Do NOT output JSON."""

REFINER_PROMPT = """You are a Clinical Knowledge Graph Refiner. Improve the KG based on critique.

## TRANSCRIPT:
{transcript}

## CURRENT KG:
{kg_json}

## CRITIQUE:
{critique}

Fix all issues mentioned in critique:
- Add missing nodes with proper types
- Add missing edges
- Fix incorrect node/edge types
- Remove hallucinated information

## CRITICAL FORMAT REQUIREMENTS:
- Node IDs: Use format "N_001", "N_002", etc.
- turn_id: MUST be strings in format "P-X" or "D-X" where:
  - P = Patient utterance, D = Doctor utterance
  - X = turn number from transcript (e.g., [P-1], [D-39])
  - Example: "P-1", "P-5", "D-39" (NOT integers like 1, 5, 39)

Output ONLY valid JSON:
{{
  "nodes": [
    {{"id": "N_001", "text": "...", "type": "SYMPTOM", "turn_id": "P-1", "evidence": "..."}},
    ...
  ],
  "edges": [
    {{"source_id": "N_001", "target_id": "N_002", "type": "INDICATES", "turn_id": "D-15", "evidence": "..."}},
    ...
  ]
}}"""


# === Model Clients ===
class OpenRouterClient:
    """Client for OpenRouter API (GLM, etc.)"""

    def __init__(self, api_key: str, model: str = "z-ai/glm-4.7-flash"):
        from openai import OpenAI
        self.client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=api_key
        )
        self.model = model

    def generate(self, prompt: str) -> tuple:
        """Generate response. Returns (content, usage_dict)."""
        for attempt in range(MAX_RETRIES):
            try:
                stream = self.client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    stream=True
                )

                content = ""
                last_chunk = None
                for chunk in stream:
                    last_chunk = chunk
                    delta = chunk.choices[0].delta
                    if delta.content:
                        content += delta.content

                usage = None
                if last_chunk and hasattr(last_chunk, 'usage') and last_chunk.usage:
                    u = last_chunk.usage
                    usage = {
                        "prompt_tokens": u.prompt_tokens,
                        "completion_tokens": u.completion_tokens,
                    }

                if content:
                    return content, usage

            except Exception as e:
                print(f"(error: {e}, retry {attempt + 1})", end=" ", flush=True)
                time.sleep(2 ** attempt)

        return "", None


class GeminiClient:
    """Client for Google Gemini API."""

    def __init__(self, api_key: str, model: str = "gemini-2.0-flash"):
        from google import genai
        from google.genai import types
        self.client = genai.Client(api_key=api_key)
        self.model = model
        self.types = types

    def generate(self, prompt: str) -> tuple:
        """Generate response. Returns (content, usage_dict)."""
        for attempt in range(MAX_RETRIES):
            try:
                response = self.client.models.generate_content(
                    model=self.model,
                    contents=prompt,
                    config=self.types.GenerateContentConfig(
                        temperature=0.1,
                        max_output_tokens=4096,
                    )
                )

                content = response.text if response.text else ""

                usage = None
                if hasattr(response, 'usage_metadata') and response.usage_metadata:
                    um = response.usage_metadata
                    usage = {
                        "prompt_tokens": getattr(um, 'prompt_token_count', 0),
                        "completion_tokens": getattr(um, 'candidates_token_count', 0),
                    }

                if content:
                    return content, usage

            except Exception as e:
                print(f"(error: {e}, retry {attempt + 1})", end=" ", flush=True)
                time.sleep(2 ** attempt)

        return "", None


def get_client(model_name: str, api_keys: dict):
    """Get the appropriate client for the model."""
    if model_name in ("glm", "glm-4.7-flash"):
        return OpenRouterClient(api_keys["openrouter"], "z-ai/glm-4.7-flash")
    elif model_name in ("gemini", "gemini-2.0-flash"):
        return GeminiClient(api_keys["gemini"], "gemini-2.0-flash")
    else:
        raise ValueError(f"Unknown model: {model_name}")


# === Utilities ===
def read_transcript(file_path: Path) -> str:
    with open(file_path, 'r') as f:
        return f.read()


def extract_json_from_response(response_text: str) -> dict:
    """Extract JSON from LLM response."""
    try:
        return json.loads(response_text)
    except json.JSONDecodeError:
        pass

    if response_text.strip().startswith('```'):
        parts = response_text.split('```')
        if len(parts) >= 2:
            inner = parts[1]
            if inner.startswith('json'):
                inner = inner[4:]
            inner = inner.strip()
            try:
                return json.loads(inner)
            except json.JSONDecodeError:
                pass

    json_match = re.search(r'\{[\s\S]*\}', response_text)
    if json_match:
        json_str = json_match.group(0)
        try:
            return json.loads(json_str)
        except json.JSONDecodeError:
            fixed_json = re.sub(r',(\s*[}\]])', r'\1', json_str)
            try:
                return json.loads(fixed_json)
            except json.JSONDecodeError:
                pass

    return None


def validate_knowledge_graph(kg: dict) -> dict:
    """Validate and fix knowledge graph integrity."""
    if not kg or 'nodes' not in kg or 'edges' not in kg:
        return kg

    node_ids = {node['id'] for node in kg.get('nodes', [])}

    valid_edges = []
    invalid_count = 0

    for edge in kg.get('edges', []):
        source = edge.get('source_id')
        target = edge.get('target_id')
        if source in node_ids and target in node_ids:
            valid_edges.append(edge)
        else:
            invalid_count += 1

    if invalid_count > 0:
        print(f"    Removed {invalid_count} invalid edges")
        kg['edges'] = valid_edges

    return kg


def get_transcript_files():
    """Get all transcript files."""
    files = []
    for res_dir in sorted(TRANSCRIPT_DIR.glob("RES*")):
        if res_dir.is_dir():
            txt_file = res_dir / f"{res_dir.name}.txt"
            if txt_file.exists():
                files.append(txt_file)
    return files


# === Extraction Functions ===
def extract_naive(transcript: str, client) -> tuple:
    """Single-pass naive extraction."""
    prompt = EXTRACTION_PROMPT.format(transcript=transcript)
    content, usage = client.generate(prompt)

    if content:
        kg = extract_json_from_response(content)
        if kg:
            kg = validate_knowledge_graph(kg)
        return kg, usage
    return None, usage


def extract_self_critic(transcript: str, client) -> tuple:
    """Two-pass extraction with self-critic review."""
    # Pass 1: Initial extraction
    kg1, usage1 = extract_naive(transcript, client)
    if not kg1:
        return None, usage1

    n1, e1 = len(kg1.get('nodes', [])), len(kg1.get('edges', []))

    # Pass 2: Self-critic review
    kg_json = json.dumps(kg1, indent=2, ensure_ascii=False)
    prompt = SELF_CRITIC_PROMPT.format(transcript=transcript, kg_json=kg_json)
    content, usage2 = client.generate(prompt)

    kg2 = kg1  # Default to pass1 if pass2 fails
    if content:
        improved_kg = extract_json_from_response(content)
        if improved_kg:
            kg2 = validate_knowledge_graph(improved_kg)

    n2, e2 = len(kg2.get('nodes', [])), len(kg2.get('edges', []))

    # Combine usage
    total_usage = {
        "prompt_tokens": (usage1 or {}).get("prompt_tokens", 0) + (usage2 or {}).get("prompt_tokens", 0),
        "completion_tokens": (usage1 or {}).get("completion_tokens", 0) + (usage2 or {}).get("completion_tokens", 0),
        "pass1_nodes": n1,
        "pass1_edges": e1,
        "pass2_nodes": n2,
        "pass2_edges": e2,
    }

    # Add metadata
    kg2['_meta'] = {
        "pass1": {"nodes": n1, "edges": e1},
        "pass2": {"nodes": n2, "edges": e2},
    }

    return kg2, total_usage


def extract_3_agent(transcript: str, client) -> tuple:
    """Three-pass extraction: Extract → Critic → Refine."""
    # Pass 1: Initial extraction
    kg1, usage1 = extract_naive(transcript, client)
    if not kg1:
        return None, usage1

    n1, e1 = len(kg1.get('nodes', [])), len(kg1.get('edges', []))

    # Pass 2: Critic (outputs text critique, not JSON)
    kg_json = json.dumps(kg1, indent=2, ensure_ascii=False)
    prompt2 = CRITIC_PROMPT.format(transcript=transcript, kg_json=kg_json)
    critique, usage2 = client.generate(prompt2)

    if not critique:
        # If critic fails, return pass1 result
        return kg1, usage1

    # Pass 3: Refiner (uses critique to improve KG)
    prompt3 = REFINER_PROMPT.format(transcript=transcript, kg_json=kg_json, critique=critique)
    content3, usage3 = client.generate(prompt3)

    kg3 = kg1  # Default to pass1 if refiner fails
    if content3:
        refined_kg = extract_json_from_response(content3)
        if refined_kg:
            kg3 = validate_knowledge_graph(refined_kg)

    n3, e3 = len(kg3.get('nodes', [])), len(kg3.get('edges', []))

    # Combine usage
    total_usage = {
        "prompt_tokens": sum((u or {}).get("prompt_tokens", 0) for u in [usage1, usage2, usage3]),
        "completion_tokens": sum((u or {}).get("completion_tokens", 0) for u in [usage1, usage2, usage3]),
        "pass1_nodes": n1,
        "pass1_edges": e1,
        "pass3_nodes": n3,
        "pass3_edges": e3,
    }

    # Add metadata
    kg3['_meta'] = {
        "pass1": {"nodes": n1, "edges": e1},
        "pass3": {"nodes": n3, "edges": e3},
        "critique_length": len(critique),
    }

    return kg3, total_usage


def process_one(txt_path: Path, client, method: str, output_dir: Path, suffix: str) -> tuple:
    """Process single transcript."""
    res_id = txt_path.stem
    output_file = output_dir / f"{res_id}_{suffix}.json"

    # Skip if already exists
    if output_file.exists():
        return res_id, "SKIP", 0, 0, None

    try:
        transcript = read_transcript(txt_path)

        if method == "naive":
            print(f"  {res_id}...", end=" ", flush=True)
            kg, usage = extract_naive(transcript, client)
        elif method == "self-critic":
            print(f"  {res_id} pass1...", end=" ", flush=True)
            kg, usage = extract_self_critic(transcript, client)
        else:  # 3-agent
            print(f"  {res_id} extract...", end=" ", flush=True)
            kg, usage = extract_3_agent(transcript, client)

        if not kg:
            print("FAILED")
            return res_id, "FAILED", 0, 0, None

        n, e = len(kg.get('nodes', [])), len(kg.get('edges', []))

        # Add usage to kg
        kg['_usage'] = usage

        with open(output_file, 'w') as f:
            json.dump(kg, f, indent=2, ensure_ascii=False)

        if method == "self-critic" and usage:
            delta_n = usage.get('pass2_nodes', n) - usage.get('pass1_nodes', 0)
            delta_e = usage.get('pass2_edges', e) - usage.get('pass1_edges', 0)
            print(f"({usage.get('pass1_nodes', 0)}n/{usage.get('pass1_edges', 0)}e) pass2... → ({n}n/{e}e) Δ{delta_n:+d}n/{delta_e:+d}e")
        elif method == "3-agent" and usage:
            delta_n = usage.get('pass3_nodes', n) - usage.get('pass1_nodes', 0)
            delta_e = usage.get('pass3_edges', e) - usage.get('pass1_edges', 0)
            print(f"critic... refine... ({usage.get('pass1_nodes', 0)}n/{usage.get('pass1_edges', 0)}e) → ({n}n/{e}e) Δ{delta_n:+d}n/{delta_e:+d}e")
        else:
            print(f"({n}n/{e}e)")

        return res_id, "OK", n, e, usage

    except Exception as ex:
        print(f"ERROR: {ex}")
        return res_id, f"ERROR: {ex}", 0, 0, None


def main():
    parser = argparse.ArgumentParser(description="KG Extraction Pipeline")
    parser.add_argument("--method", type=str, choices=["naive", "self-critic", "3-agent"], required=True,
                        help="Extraction method: naive (1-pass), self-critic (2-pass), or 3-agent (3-pass)")
    parser.add_argument("--model", type=str, default="glm",
                        help="Model to use: glm, gemini (default: glm)")
    parser.add_argument("--output", type=str, required=True,
                        help="Output directory for sub-KG JSON files")
    parser.add_argument("--workers", type=int, default=1,
                        help="Number of parallel workers (default: 1)")
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load API keys
    with open("api_keys.json") as f:
        api_keys = json.load(f)

    # Get client
    client = get_client(args.model, api_keys)

    # Derive suffix for output files
    suffix = f"{args.method.replace('-', '_')}_{args.model}"

    transcript_files = get_transcript_files()
    print(f"KG Extraction Pipeline")
    print(f"Method: {args.method}")
    print(f"Model:  {args.model}")
    print(f"Output: {output_dir}")
    print(f"Processing {len(transcript_files)} transcripts")
    print("=" * 60)

    success = 0
    failed = 0
    total_tokens = {"prompt": 0, "completion": 0}
    all_stats = []

    for txt_path in transcript_files:
        res_id, status, nodes, edges, usage = process_one(txt_path, client, args.method, output_dir, suffix)

        if status == "OK":
            success += 1
            if usage:
                total_tokens["prompt"] += usage.get("prompt_tokens", 0)
                total_tokens["completion"] += usage.get("completion_tokens", 0)
                all_stats.append({"res_id": res_id, "nodes": nodes, "edges": edges, **usage})
        elif status == "SKIP":
            print(f"  {res_id}: SKIP (exists)")
            success += 1
        else:
            failed += 1

        # Small delay between requests
        time.sleep(0.3)

    # Save stats
    stats_file = output_dir / "_stats.json"
    with open(stats_file, 'w') as f:
        json.dump({
            "method": args.method,
            "model": args.model,
            "total_tokens": total_tokens,
            "success": success,
            "failed": failed,
            "details": all_stats
        }, f, indent=2)

    print("=" * 60)
    print(f"Done! Success: {success}, Failed: {failed}")
    print(f"Total tokens: {total_tokens['prompt'] + total_tokens['completion']}")
    print(f"Output: {output_dir}/")


if __name__ == "__main__":
    main()
