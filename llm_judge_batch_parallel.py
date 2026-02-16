"""
LLM-as-Judge Batch Evaluation - PARALLEL VERSION
=================================================
4 Models × 3 Trials with concurrent API calls.

Usage:
    # Default (unified_graph_curated)
    python llm_judge_batch_parallel.py

    # Custom directories
    python llm_judge_batch_parallel.py \
        --results-dir baseline_naive/test_with_type \
        --output-dir baseline_naive/test_with_type/judge_scores

    # Specific samples, models, trials
    python llm_judge_batch_parallel.py \
        --results-dir baseline_naive/test_with_type \
        --res-ids RES0198 RES0207 \
        --questions Q1 Q3 \
        --models gpt claude \
        --trials 1
"""

import json
import time
import argparse
import concurrent.futures
from pathlib import Path
from pydantic import BaseModel, Field
from typing import Literal
from threading import Lock

# ── Config ──
API_KEYS_PATH = Path("api_keys.json")
DEFAULT_BUNDLE_PATH = Path("evaluation_bundle")
DEFAULT_RESULTS_PATH = DEFAULT_BUNDLE_PATH / "unified_graph_curated" / "results_unified_graph_curated"
DEFAULT_OUTPUT_DIR = DEFAULT_BUNDLE_PATH / "unified_graph_curated" / "LLM_judge_score_unified_graph_curated"

DEFAULT_TRIALS = 3
MAX_WORKERS = 6  # Concurrent API calls (conservative for rate limits)
API_TIMEOUT = 90  # Seconds per API call

print_lock = Lock()

# ── Load API Keys ──
with open(API_KEYS_PATH) as f:
    api_keys = json.load(f)

# ── Pydantic Schema ──
class ReasoningDetails(BaseModel):
    correctness: str = Field(description="Step-by-step analysis of factual accuracy vs Gold Answer")
    completeness: str = Field(description="List key points covered vs missed from Gold Answer")
    faithfulness: str = Field(description="Verify claims against transcript, check for hallucination")
    relevance: str = Field(description="Assess how well answer addresses the question")

class EvaluationScores(BaseModel):
    correctness: Literal[1, 2, 3, 4, 5] = Field(description="Factual accuracy score 1-5")
    completeness: Literal[1, 2, 3, 4, 5] = Field(description="Coverage of key points 1-5")
    faithfulness: Literal[1, 2, 3, 4, 5] = Field(description="Grounded in transcript 1-5")
    relevance: Literal[1, 2, 3, 4, 5] = Field(description="Addresses question 1-5")

class EvaluationResult(BaseModel):
    reasoning: ReasoningDetails
    scores: EvaluationScores
    average_score: float = Field(description="Average of 4 dimension scores")
    summary: str = Field(description="One sentence overall assessment")

# ── Rubric ──
SYSTEM_PROMPT = """You are an expert clinical QA evaluator. Your task is to evaluate a Generated Answer against a Gold Answer for a clinical question.

## Evaluation Dimensions

Rate each dimension from 1-5 using these criteria:

### 1. CORRECTNESS (Factual accuracy compared to Gold Answer)
- 5: All facts fully consistent with Gold Answer, no errors
- 4: Minor discrepancies, core facts correct
- 3: Some factual errors but main point correct
- 2: Significant factual errors or contradictions
- 1: Mostly incorrect or contradicts Gold Answer

### 2. COMPLETENESS (Coverage of key points from Gold Answer)
- 5: Covers 100% of Gold Answer key points
- 4: Covers 80-99% of key points
- 3: Covers 50-79% of key points
- 2: Covers 20-49% of key points
- 1: Covers <20% of key points

### 3. FAITHFULNESS (Grounded in transcript, no hallucination)
- 5: All claims verifiable in transcript, citations accurate
- 4: Minor unverifiable details, mostly grounded
- 3: Some claims not in transcript but plausible
- 2: Multiple hallucinated facts
- 1: Largely fabricated content

### 4. RELEVANCE (Directly addresses the question)
- 5: Perfectly focused, no irrelevant content
- 4: Mostly relevant, minor tangents
- 3: Partially relevant, some off-topic content
- 2: Largely off-topic
- 1: Does not address the question

Think step-by-step for each dimension before scoring. Calculate average_score as the mean of all 4 scores."""

CLAUDE_SCHEMA = {
    "type": "object",
    "properties": {
        "reasoning": {
            "type": "object",
            "properties": {
                "correctness": {"type": "string"},
                "completeness": {"type": "string"},
                "faithfulness": {"type": "string"},
                "relevance": {"type": "string"}
            },
            "required": ["correctness", "completeness", "faithfulness", "relevance"],
            "additionalProperties": False
        },
        "scores": {
            "type": "object",
            "properties": {
                "correctness": {"type": "integer", "enum": [1, 2, 3, 4, 5]},
                "completeness": {"type": "integer", "enum": [1, 2, 3, 4, 5]},
                "faithfulness": {"type": "integer", "enum": [1, 2, 3, 4, 5]},
                "relevance": {"type": "integer", "enum": [1, 2, 3, 4, 5]}
            },
            "required": ["correctness", "completeness", "faithfulness", "relevance"],
            "additionalProperties": False
        },
        "average_score": {"type": "number"},
        "summary": {"type": "string"}
    },
    "required": ["reasoning", "scores", "average_score", "summary"],
    "additionalProperties": False
}


def build_user_prompt(question: str, transcript: str, gold_answer: str, generated_answer: str) -> str:
    return f"""## Question
{question}

## Source Transcript
{transcript}

## Gold Answer
{gold_answer}

## Generated Answer
{generated_answer}

Please evaluate the Generated Answer according to the rubric."""


def safe_print(*args, **kwargs):
    with print_lock:
        print(*args, **kwargs, flush=True)


# ── Individual Model Evaluators ──
def evaluate_gpt(user_prompt: str) -> dict:
    from openai import OpenAI
    client = OpenAI(api_key=api_keys["openai"])
    response = client.responses.parse(
        model="gpt-5.2",
        input=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        text_format=EvaluationResult,
        reasoning={"effort": "high"},
    )
    return response.output_parsed.model_dump()


def evaluate_claude(user_prompt: str) -> dict:
    import anthropic
    client = anthropic.Anthropic(api_key=api_keys["anthropic"])
    response = client.messages.create(
        model="claude-sonnet-4-5-20250929",
        max_tokens=4096,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_prompt}],
        output_config={"format": {"type": "json_schema", "schema": CLAUDE_SCHEMA}}
    )
    return json.loads(response.content[0].text)


def evaluate_gemini(user_prompt: str) -> dict:
    from google import genai
    client = genai.Client(api_key=api_keys["gemini"])
    full_prompt = SYSTEM_PROMPT + "\n\n" + user_prompt
    response = client.models.generate_content(
        model="gemini-3-flash-preview",
        contents=full_prompt,
        config={
            "response_mime_type": "application/json",
            "response_json_schema": EvaluationResult.model_json_schema(),
        },
    )
    return EvaluationResult.model_validate_json(response.text).model_dump()


def evaluate_grok(user_prompt: str) -> dict:
    from xai_sdk import Client as XAIClient
    from xai_sdk.chat import system, user
    client = XAIClient(api_key=api_keys["xai"])
    chat = client.chat.create(model="grok-4-1-fast-reasoning")
    chat.append(system(SYSTEM_PROMPT))
    chat.append(user(user_prompt))
    _, result = chat.parse(EvaluationResult)
    return result.model_dump()


EVALUATORS = {
    "gpt": evaluate_gpt,
    "claude": evaluate_claude,
    "gemini": evaluate_gemini,
    "grok": evaluate_grok,
}


def run_single_trial(args):
    """Run a single (model, trial) evaluation. Returns (model, trial, result)."""
    import signal

    res_id, qid, model_name, trial, user_prompt, output_dir = args

    trial_path = output_dir / f"{res_id}_{qid}_{model_name}_trial{trial}_result.json"

    # Check cache
    if trial_path.exists():
        with open(trial_path) as f:
            result = json.load(f)
        safe_print(f"    {model_name} trial{trial}: {result['average_score']:.2f} (cached)")
        return (model_name, trial, result)

    # Run evaluation with timeout
    try:
        # Use a simple approach: wrap in a try with overall timeout
        import threading
        result_container = [None]
        error_container = [None]

        def call_api():
            try:
                result_container[0] = EVALUATORS[model_name](user_prompt)
            except Exception as e:
                error_container[0] = e

        thread = threading.Thread(target=call_api)
        thread.start()
        thread.join(timeout=API_TIMEOUT)

        if thread.is_alive():
            safe_print(f"    {model_name} trial{trial}: TIMEOUT ({API_TIMEOUT}s)")
            return (model_name, trial, None)

        if error_container[0]:
            raise error_container[0]

        result = result_container[0]
        if result:
            with open(trial_path, "w", encoding="utf-8") as f:
                json.dump(result, f, indent=2, ensure_ascii=False)
            safe_print(f"    {model_name} trial{trial}: {result['average_score']:.2f}")
            return (model_name, trial, result)
        else:
            safe_print(f"    {model_name} trial{trial}: NO RESULT")
            return (model_name, trial, None)

    except Exception as e:
        safe_print(f"    {model_name} trial{trial}: ERROR - {e}")
        return (model_name, trial, None)


def evaluate_question_parallel(res_id: str, qid: str, question: dict,
                               generated: dict, transcript: str,
                               output_dir: Path, models: list, num_trials: int):
    """Evaluate one question with parallel API calls."""

    user_prompt = build_user_prompt(
        question=question["question"],
        transcript=transcript,
        gold_answer=question["gold_answer"],
        generated_answer=generated["answer"]
    )

    # Build tasks based on selected models and trials
    tasks = []
    for model_name in models:
        for trial in range(1, num_trials + 1):
            tasks.append((res_id, qid, model_name, trial, user_prompt, output_dir))

    # Run in parallel
    results = {m: {"trials": [], "avg_scores": {}} for m in models}

    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(run_single_trial, task): task for task in tasks}
        for future in concurrent.futures.as_completed(futures):
            model_name, trial, result = future.result()
            if result:
                results[model_name]["trials"].append(result)

    # Calculate averages
    for model_name in models:
        valid_trials = results[model_name]["trials"]
        if valid_trials:
            for dim in ["correctness", "completeness", "faithfulness", "relevance"]:
                results[model_name]["avg_scores"][dim] = sum(t["scores"][dim] for t in valid_trials) / len(valid_trials)
            results[model_name]["avg_scores"]["average"] = sum(t["average_score"] for t in valid_trials) / len(valid_trials)

    return results


def main():
    parser = argparse.ArgumentParser(
        description="LLM-as-Judge Batch Evaluation with parallel API calls"
    )
    parser.add_argument(
        "--results-dir", type=Path, default=DEFAULT_RESULTS_PATH,
        help="Directory containing generated answers (RES*/RES*_answers.json)"
    )
    parser.add_argument(
        "--bundle-dir", type=Path, default=DEFAULT_BUNDLE_PATH,
        help="Evaluation bundle directory (gold answers, transcripts)"
    )
    parser.add_argument(
        "--output-dir", type=Path, default=None,
        help="Output directory for judge scores (default: results-dir/../judge_scores)"
    )
    parser.add_argument(
        "--res-ids", nargs="+", default=None,
        help="Specific patient IDs to evaluate (default: all)"
    )
    parser.add_argument(
        "--questions", nargs="+", default=None,
        help="Specific question IDs to evaluate (default: all)"
    )
    parser.add_argument(
        "--models", nargs="+", default=list(EVALUATORS.keys()),
        choices=list(EVALUATORS.keys()),
        help="Models to use for evaluation (default: all)"
    )
    parser.add_argument(
        "--trials", type=int, default=DEFAULT_TRIALS,
        help=f"Number of trials per model (default: {DEFAULT_TRIALS})"
    )
    parser.add_argument(
        "--no-cache", action="store_true",
        help="Ignore cached results and re-evaluate"
    )
    args = parser.parse_args()

    # Setup paths
    results_path = args.results_dir
    bundle_path = args.bundle_dir
    output_dir = args.output_dir or results_path.parent / "judge_scores"
    output_dir.mkdir(exist_ok=True, parents=True)

    models = args.models
    num_trials = args.trials

    print("=" * 70, flush=True)
    print("LLM-as-Judge PARALLEL Batch Evaluation", flush=True)
    print(f"{len(models)} Models × {num_trials} Trials, {MAX_WORKERS} concurrent workers", flush=True)
    print(f"Results: {results_path}", flush=True)
    print(f"Output:  {output_dir}", flush=True)
    print("=" * 70, flush=True)

    # Find patient directories
    patient_dirs = sorted([d for d in results_path.iterdir() if d.is_dir()])
    if args.res_ids:
        patient_dirs = [d for d in patient_dirs if d.name in args.res_ids]

    total_questions = 0
    completed = 0

    # Count total
    for patient_dir in patient_dirs:
        answers_path = patient_dir / f"{patient_dir.name}_answers.json"
        if answers_path.exists():
            with open(answers_path) as f:
                answers = json.load(f)
                if args.questions:
                    answers = [a for a in answers if a["id"] in args.questions]
                total_questions += len(answers)

    print(f"\nTotal questions: {total_questions}", flush=True)

    for patient_dir in patient_dirs:
        res_id = patient_dir.name
        answers_path = patient_dir / f"{res_id}_answers.json"

        if not answers_path.exists():
            continue

        with open(answers_path) as f:
            generated_answers = json.load(f)

        # Filter questions if specified
        if args.questions:
            generated_answers = [a for a in generated_answers if a["id"] in args.questions]

        qa_path = bundle_path / res_id / f"{res_id}_standard_answer.json"
        with open(qa_path) as f:
            gold_questions = json.load(f)

        tx_path = bundle_path / res_id / f"{res_id}.txt"
        transcript = tx_path.read_text(encoding="utf-8")

        gold_by_id = {q["id"]: q for q in gold_questions}

        print(f"\n{'='*60}", flush=True)
        print(f"{res_id} ({len(generated_answers)} questions)", flush=True)
        print("=" * 60, flush=True)

        for gen_a in generated_answers:
            qid = gen_a["id"]
            gold_q = gold_by_id.get(qid)

            if not gold_q:
                continue

            # Check if already fully evaluated
            ensemble_path = output_dir / f"{res_id}_{qid}_ensemble_result.json"
            if ensemble_path.exists() and not args.no_cache:
                completed += 1
                print(f"  [{qid}] Already complete ({completed}/{total_questions})", flush=True)
                continue

            print(f"\n  [{qid}] {gold_q['question'][:50]}...", flush=True)

            # ── GUARD: Empty answer → automatic 0 score ──
            generated_answer = gen_a.get("answer", "").strip()
            if not generated_answer:
                print(f"    ⚠ EMPTY ANSWER DETECTED → Auto 0 score", flush=True)
                zero_result = {
                    "reasoning": {
                        "correctness": "Generated answer is empty. No content to evaluate.",
                        "completeness": "Generated answer is empty. 0% coverage.",
                        "faithfulness": "Generated answer is empty. Cannot assess faithfulness.",
                        "relevance": "Generated answer is empty. Does not address the question."
                    },
                    "scores": {"correctness": 1, "completeness": 1, "faithfulness": 1, "relevance": 1},
                    "average_score": 1.0,
                    "summary": "EMPTY ANSWER: The generated answer was empty or whitespace-only. Automatic minimum score assigned."
                }
                # Save trial results for each model
                for model_name in models:
                    for trial in range(1, num_trials + 1):
                        trial_path = output_dir / f"{res_id}_{qid}_{model_name}_trial{trial}_result.json"
                        with open(trial_path, "w", encoding="utf-8") as f:
                            json.dump(zero_result, f, indent=2, ensure_ascii=False)
                # Save ensemble
                ensemble_result = {
                    "res_id": res_id,
                    "qid": qid,
                    "question_type": gold_q.get("type", ""),
                    "per_model": {m: {"correctness": 1.0, "completeness": 1.0, "faithfulness": 1.0, "relevance": 1.0, "average": 1.0} for m in models},
                    "ensemble": {"correctness": 1.0, "completeness": 1.0, "faithfulness": 1.0, "relevance": 1.0, "average": 1.0}
                }
                with open(ensemble_path, "w", encoding="utf-8") as f:
                    json.dump(ensemble_result, f, indent=2, ensure_ascii=False)
                completed += 1
                print(f"  → Ensemble: 1.00/5 (empty answer) [{completed}/{total_questions}]", flush=True)
                continue

            t0 = time.time()
            results = evaluate_question_parallel(
                res_id, qid, gold_q, gen_a, transcript,
                output_dir, models, num_trials
            )
            elapsed = time.time() - t0

            # Calculate ensemble
            ensemble_scores = {}
            for dim in ["correctness", "completeness", "faithfulness", "relevance", "average"]:
                model_avgs = [results[m]["avg_scores"].get(dim, 0) for m in models
                             if results[m]["avg_scores"]]
                if model_avgs:
                    ensemble_scores[dim] = sum(model_avgs) / len(model_avgs)

            # Save ensemble
            ensemble_result = {
                "res_id": res_id,
                "qid": qid,
                "question_type": gold_q.get("type", ""),
                "per_model": {m: results[m]["avg_scores"] for m in models},
                "ensemble": ensemble_scores
            }
            with open(ensemble_path, "w", encoding="utf-8") as f:
                json.dump(ensemble_result, f, indent=2, ensure_ascii=False)

            completed += 1
            print(f"  → Ensemble: {ensemble_scores.get('average', 0):.2f}/5 ({elapsed:.1f}s) [{completed}/{total_questions}]", flush=True)

    print(f"\n{'='*70}", flush=True)
    print("BATCH EVALUATION COMPLETE", flush=True)
    print(f"{'='*70}", flush=True)
    print(f"Questions evaluated: {completed}/{total_questions}", flush=True)


if __name__ == "__main__":
    main()
