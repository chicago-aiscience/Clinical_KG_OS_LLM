"""
Naive RAG Pipeline (Text-only, no KG structure)
===============================================
Baseline comparison for GraphRAG. Uses simple text chunking + embedding retrieval.

Difference from GraphRAG:
- GraphRAG: Retrieves structured KG triples (entity-relation-entity)
- Naive RAG: Retrieves raw transcript chunks (no structure)

Usage:
    python naive_rag_pipeline.py --bundle evaluation_bundle/ --output baseline_naive_rag/results
"""

import json
import re
import argparse
import time
import requests
import numpy as np
from pathlib import Path

from llama_index.embeddings.huggingface import HuggingFaceEmbedding

# ── Config ──
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
OPENROUTER_MODEL = "z-ai/glm-4.7-flash"
OPENROUTER_TIMEOUT = 120
SIMILARITY_TOP_K = 10
CHUNK_SIZE = 3  # Number of turns per chunk

# Load API key
with open("api_keys.json") as f:
    OPENROUTER_KEY = json.load(f).get("openrouter")


def parse_transcript(txt_path: str) -> list[dict]:
    """Parse transcript into turns."""
    turns = []
    text = Path(txt_path).read_text(encoding="utf-8")
    for m in re.finditer(
        r"\[(D-\d+|P-\d+)\]\s*([DP]):\s*(.*?)(?=\n\[|\n*$)", text, re.DOTALL
    ):
        turns.append({
            "turn_id": m.group(1),
            "speaker": m.group(2),
            "text": m.group(3).strip()
        })
    return turns


def chunk_transcript(turns: list[dict], chunk_size: int = CHUNK_SIZE) -> list[dict]:
    """Split transcript into overlapping chunks."""
    chunks = []
    for i in range(0, len(turns), chunk_size - 1):  # Overlap by 1
        chunk_turns = turns[i:i + chunk_size]
        if not chunk_turns:
            continue

        chunk_text = "\n".join([
            f"[{t['turn_id']}] {t['speaker']}: {t['text']}"
            for t in chunk_turns
        ])

        chunks.append({
            "text": chunk_text,
            "turn_ids": [t["turn_id"] for t in chunk_turns],
            "start_idx": i
        })

    return chunks


def embed_texts(texts: list[str], embed_model) -> np.ndarray:
    """Embed a list of texts."""
    embeddings = []
    for text in texts:
        emb = embed_model.get_text_embedding(text)
        embeddings.append(emb)
    return np.array(embeddings)


def retrieve_chunks(question: str, chunks: list[dict], chunk_embeddings: np.ndarray,
                    embed_model, top_k: int = SIMILARITY_TOP_K) -> list[dict]:
    """Retrieve most relevant chunks using embedding similarity."""
    q_emb = np.array(embed_model.get_text_embedding(question))

    # Cosine similarity
    similarities = []
    for i, emb in enumerate(chunk_embeddings):
        sim = np.dot(q_emb, emb) / (np.linalg.norm(q_emb) * np.linalg.norm(emb) + 1e-8)
        similarities.append((i, sim))

    # Sort by similarity
    similarities.sort(key=lambda x: -x[1])

    # Get top-k chunks
    retrieved = []
    for idx, sim in similarities[:top_k]:
        chunk = chunks[idx].copy()
        chunk["similarity"] = float(sim)
        retrieved.append(chunk)

    return retrieved


def build_naive_rag_prompt(question: str, retrieved_chunks: list[dict]) -> str:
    """Build prompt with retrieved transcript chunks (no KG structure)."""

    # Sort chunks by turn order
    retrieved_chunks.sort(key=lambda c: c["start_idx"])

    context = "\n\n".join([
        f"--- Chunk (turns {c['turn_ids'][0]} to {c['turn_ids'][-1]}) ---\n{c['text']}"
        for c in retrieved_chunks
    ])

    prompt = f"""You are a clinical QA assistant. Answer the question based ONLY on the provided transcript excerpts.

TRANSCRIPT EXCERPTS:
{context}

QUESTION: {question}

INSTRUCTIONS:
- Answer based only on information in the transcript excerpts above
- Cite relevant turn IDs (e.g., P-1, D-3) to support your answer
- If information is not in the excerpts, say "Information not found in provided excerpts"
- Be concise and factual

ANSWER:"""

    return prompt


def generate_answer(prompt: str) -> str:
    """Generate answer using OpenRouter API."""
    headers = {
        "Authorization": f"Bearer {OPENROUTER_KEY}",
        "Content-Type": "application/json"
    }

    payload = {
        "model": OPENROUTER_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 1024,
        "temperature": 0.1
    }

    try:
        resp = requests.post(OPENROUTER_URL, headers=headers, json=payload, timeout=OPENROUTER_TIMEOUT)
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"].strip()
    except Exception as e:
        return f"[ERROR: {e}]"


def main():
    parser = argparse.ArgumentParser(description="Naive RAG Pipeline")
    parser.add_argument("--bundle", type=str, default="evaluation_bundle", help="Evaluation bundle path")
    parser.add_argument("--output", type=str, default="baseline_naive_rag/results", help="Output directory")
    parser.add_argument("--res-ids", nargs="+", default=None, help="Specific RES IDs to process")
    args = parser.parse_args()

    bundle_path = Path(args.bundle)
    output_path = Path(args.output)
    output_path.mkdir(parents=True, exist_ok=True)

    # Initialize embedding model
    print("Loading embedding model...")
    embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-m3")

    # Find all RES directories
    if args.res_ids:
        res_dirs = [bundle_path / rid for rid in args.res_ids]
    else:
        res_dirs = sorted([d for d in bundle_path.iterdir() if d.is_dir() and d.name.startswith("RES")])

    print(f"Processing {len(res_dirs)} patients...")

    total_questions = 0
    total_time = 0

    for res_dir in res_dirs:
        res_id = res_dir.name
        print(f"\n{'='*60}")
        print(f"Processing {res_id}")
        print(f"{'='*60}")

        # Load transcript
        transcript_file = res_dir / f"{res_id}.txt"
        if not transcript_file.exists():
            # Try alternative name
            transcript_file = res_dir / f"{res_id}_dialogue_clean.txt"
        if not transcript_file.exists():
            print(f"  Skipping: no transcript found")
            continue

        turns = parse_transcript(str(transcript_file))
        chunks = chunk_transcript(turns)
        print(f"  Transcript: {len(turns)} turns -> {len(chunks)} chunks")

        # Embed chunks
        chunk_texts = [c["text"] for c in chunks]
        chunk_embeddings = embed_texts(chunk_texts, embed_model)

        # Load questions
        qa_file = res_dir / f"{res_id}_standard_answer.json"
        if not qa_file.exists():
            print(f"  Skipping: no QA file found")
            continue

        with open(qa_file) as f:
            qa_data = json.load(f)

        # Handle both formats: list or dict with "questions" key
        if isinstance(qa_data, list):
            questions = qa_data
        else:
            questions = qa_data.get("questions", [])
        print(f"  Questions: {len(questions)}")

        # Process each question
        results = []
        for q in questions:
            qid = q["id"]
            question_text = q["question"]

            start_time = time.time()

            # Retrieve relevant chunks
            retrieved = retrieve_chunks(question_text, chunks, chunk_embeddings, embed_model)

            # Build prompt
            prompt = build_naive_rag_prompt(question_text, retrieved)

            # Generate answer
            answer = generate_answer(prompt)

            elapsed = time.time() - start_time
            total_time += elapsed
            total_questions += 1

            results.append({
                "id": qid,
                "question": question_text,
                "question_type": q.get("type", "unknown"),
                "gold_answer": q.get("gold_answer", ""),
                "answer": answer,
                "prompt": prompt,
                "retrieved_chunks": len(retrieved),
                "time_seconds": round(elapsed, 2)
            })

            print(f"  {qid}: {len(answer)} chars, {elapsed:.1f}s")

        # Save results
        res_output_dir = output_path / res_id
        res_output_dir.mkdir(parents=True, exist_ok=True)

        output_file = res_output_dir / f"{res_id}_answers.json"
        with open(output_file, "w") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

        print(f"  Saved: {output_file}")

    print(f"\n{'='*60}")
    print(f"Naive RAG Complete")
    print(f"{'='*60}")
    print(f"Total questions: {total_questions}")
    print(f"Total time: {total_time:.1f}s")
    print(f"Avg time/question: {total_time/total_questions:.1f}s")


if __name__ == "__main__":
    main()
