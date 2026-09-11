"""
Stage 7 — BGE Evidence Retrieval

Retrieves the Top-K most semantically similar source sentences
for every generated document sentence.

Model:
    BAAI/bge-small-en-v1.5

This stage retrieves evidence only.
It does NOT determine factual correctness.
"""

import csv
import json
import re
from pathlib import Path

import numpy as np
import torch
from sentence_transformers import SentenceTransformer


ROOT = Path(__file__).resolve().parents[1]

SOURCE_CSV = ROOT / "data" / "raw" / "cnn_dailymail_100.csv"
DOCUMENTS_JSONL = ROOT / "training_datasets" / "documents_full.jsonl"
OUTPUT_JSONL = ROOT / "training_datasets" / "evidence_full.jsonl"

MODEL_NAME = "BAAI/bge-small-en-v1.5"
TOP_K = 5

TARGET_FORMATS = {
    "press_advisory",
    "technical_brief",
}

DOCUMENT_FIELDS = (
    "overview",
    "key_findings",
    "technical_details",
    "implications",
)

SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")


def split_source_sentences(text):
    """Split source article into deterministic sentence IDs."""
    sentences = [
        sentence.strip()
        for sentence in SENTENCE_SPLIT_RE.split(text.strip())
        if sentence.strip()
    ]

    return [
        {
            "sentence_id": f"S{i:03d}",
            "text": sentence,
        }
        for i, sentence in enumerate(sentences, 1)
    ]


def split_generated_sentences(text):
    """Split a generated field into sentences."""
    return [
        sentence.strip()
        for sentence in SENTENCE_SPLIT_RE.split(text.strip())
        if sentence.strip()
    ]


def load_sources():
    if not SOURCE_CSV.exists():
        raise FileNotFoundError(
            f"Source file not found: {SOURCE_CSV}"
        )

    with SOURCE_CSV.open(
        "r",
        encoding="utf-8-sig",
        newline="",
    ) as f:
        rows = list(csv.DictReader(f))

    required_columns = {"article", "highlights", "id"}

    if not rows:
        raise ValueError("train.csv contains no rows.")

    if not required_columns.issubset(rows[0].keys()):
        raise ValueError(
            "train.csv is missing required columns. "
            f"Expected: {sorted(required_columns)}; "
            f"Found: {list(rows[0].keys())}"
        )

    return {
        row["id"]: row["article"]
        for row in rows
    }


def load_documents():
    if not DOCUMENTS_JSONL.exists():
        raise FileNotFoundError(
            f"Generated documents not found: {DOCUMENTS_JSONL}"
        )

    documents = []

    with DOCUMENTS_JSONL.open(
        "r",
        encoding="utf-8",
    ) as f:
        for line_number, line in enumerate(f, 1):
            if not line.strip():
                continue

            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON on line {line_number}: {exc}"
                ) from exc

            documents.append(record)

    return documents


def collect_generated_sentences(record):
    """
    Convert one generated document into sentence-level records.
    """

    document = record["document"]

    generated = []

    for field in DOCUMENT_FIELDS:
        value = document[field]

        if isinstance(value, str):
            sentences = split_generated_sentences(value)
        elif isinstance(value, list):
            sentences = []

            for item in value:
                if isinstance(item, str):
                    sentences.extend(
                        split_generated_sentences(item)
                    )
        else:
            continue

        for sentence_index, sentence in enumerate(sentences, 1):
            generated.append(
                {
                    "field": field,
                    "sentence_index": sentence_index,
                    "text": sentence,
                }
            )

    return generated


def retrieve_top_k(
    generated_embedding,
    source_embeddings,
    source_sentences,
):
    """
    Return Top-K source sentences by cosine similarity.

    SentenceTransformer.encode(normalize_embeddings=True)
    produces normalized vectors, so dot product equals cosine similarity.
    """

    similarities = np.dot(
        source_embeddings,
        generated_embedding,
    )

    top_k = min(TOP_K, len(source_sentences))

    indices = np.argsort(-similarities)[:top_k]

    evidence = []

    for index in indices:
        evidence.append(
            {
                "sentence_id": source_sentences[index]["sentence_id"],
                "similarity": round(
                    float(similarities[index]),
                    6,
                ),
                "text": source_sentences[index]["text"],
            }
        )

    return evidence


def main():
    OUTPUT_JSONL.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    print(f"Loading model: {MODEL_NAME}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Embedding device: {device}")
    model = SentenceTransformer(MODEL_NAME, device=device)

    print("Loading source articles...")
    sources = load_sources()

    print("Loading generated documents...")
    documents = load_documents()

    expected_ids = list(sources.keys())

    full_documents = [
        record
        for record in documents
        if (
            record.get("article_id") in expected_ids
            and record.get("target_format") in TARGET_FORMATS
        )
    ]

    if len(full_documents) != 200:
        raise ValueError(
            f"Expected exactly 200 full documents, "
            f"found {len(full_documents)}."
        )

    source_cache = {}

    print("Embedding source sentences...")

    for article_id in expected_ids:
        source_sentences = split_source_sentences(
            sources[article_id]
        )

        source_texts = [
            item["text"]
            for item in source_sentences
        ]

        embeddings = model.encode(
            source_texts,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )

        source_cache[article_id] = {
            "sentences": source_sentences,
            "embeddings": embeddings,
        }

        print(
            f"  {article_id}: "
            f"{len(source_sentences)} source sentences"
        )

    # Canonical output is recreated deterministically on each run.
    with OUTPUT_JSONL.open(
        "w",
        encoding="utf-8",
    ) as output:

        total_generated_sentences = 0

        for document_number, record in enumerate(
            full_documents,
            1,
        ):
            article_id = record["article_id"]
            target_format = record["target_format"]

            generated_sentences = collect_generated_sentences(
                record
            )

            generated_texts = [
                item["text"]
                for item in generated_sentences
            ]

            if not generated_texts:
                raise ValueError(
                    f"No generated sentences found for "
                    f"{article_id} / {target_format}"
                )

            generated_embeddings = model.encode(
                generated_texts,
                normalize_embeddings=True,
                convert_to_numpy=True,
                show_progress_bar=False,
            )

            source_sentences = source_cache[article_id]["sentences"]
            source_embeddings = source_cache[article_id]["embeddings"]

            for item, embedding in zip(
                generated_sentences,
                generated_embeddings,
            ):
                evidence = retrieve_top_k(
                    embedding,
                    source_embeddings,
                    source_sentences,
                )

                result = {
                    "article_id": article_id,
                    "target_format": target_format,
                    "field": item["field"],
                    "sentence_index": item["sentence_index"],
                    "generated_sentence": item["text"],
                    "evidence": evidence,
                }

                output.write(
                    json.dumps(
                        result,
                        ensure_ascii=False,
                    )
                    + "\n"
                )

                output.flush()

                total_generated_sentences += 1

            print(
                f"[{document_number}/200] "
                f"{article_id} / {target_format}: "
                f"{len(generated_sentences)} sentences retrieved"
            )

    print()
    print("Evidence retrieval completed.")
    print(
        f"Generated sentences processed: "
        f"{total_generated_sentences}"
    )
    print(f"Top-K evidence per sentence: {TOP_K}")
    print(f"Output: {OUTPUT_JSONL}")


if __name__ == "__main__":
    main()