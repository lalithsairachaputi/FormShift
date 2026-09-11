"""
Stage 8 — DeBERTa NLI factual verification

Input:
    training_datasets/evidence_pilot.jsonl

Output:
    training_datasets/nli_verification_pilot.jsonl

Model:
    MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli

This stage:
    - takes every generated sentence
    - evaluates it against its Top-5 retrieved evidence
    - produces entailment / contradiction / neutral results
    - aggregates the Top-5 evidence
    - preserves all individual NLI results for auditing

This stage does NOT repair or modify documents.
"""

import json
from pathlib import Path

import torch
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
)


# ============================================================
# PATHS
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

INPUT_JSONL = (
    ROOT
    / "training_datasets"
    / "evidence_full.jsonl"
)

OUTPUT_JSONL = (
    ROOT
    / "training_datasets"
    / "nli_verification_full.jsonl"
)


# ============================================================
# MODEL
# ============================================================

MODEL_NAME = (
    "MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli"
)

BATCH_SIZE = 8


# ============================================================
# LOAD INPUT
# ============================================================

def load_records():
    """Load BGE evidence records."""

    if not INPUT_JSONL.exists():
        raise FileNotFoundError(
            f"Evidence file not found:\n{INPUT_JSONL}"
        )

    records = []

    with INPUT_JSONL.open(
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
                    f"Invalid JSON on line "
                    f"{line_number}: {exc}"
                ) from exc

            records.append(record)

    return records


# ============================================================
# LABEL HANDLING
# ============================================================

def get_label_map(model):
    """
    Convert the model's numeric label IDs into
    readable labels.
    """

    return {
        int(index): str(label).lower()
        for index, label in model.config.id2label.items()
    }


def normalize_label(label):
    """
    Normalize model label names into:

        entailment
        contradiction
        neutral
    """

    label = label.lower().strip()

    if "entail" in label:
        return "entailment"

    if "contrad" in label:
        return "contradiction"

    if "neutral" in label:
        return "neutral"

    return label


# ============================================================
# NLI CLASSIFICATION
# ============================================================

def classify_pairs(
    model,
    tokenizer,
    pairs,
    label_map,
    device,
):
    """
    Run NLI on a list of:

        premise = source evidence
        hypothesis = generated sentence
    """

    if not pairs:
        return []

    results = []

    for start in range(
        0,
        len(pairs),
        BATCH_SIZE,
    ):

        batch = pairs[
            start : start + BATCH_SIZE
        ]

        premises = [
            item["evidence"]
            for item in batch
        ]

        hypotheses = [
            item["claim"]
            for item in batch
        ]

        encoded = tokenizer(
            premises,
            hypotheses,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt",
        )

        encoded = {
            key: value.to(device)
            for key, value in encoded.items()
        }

        with torch.inference_mode():

            outputs = model(
                **encoded
            )

            probabilities = torch.softmax(
                outputs.logits,
                dim=-1,
            )

        for row in probabilities:

            scores = {}

            for index in range(len(row)):

                label = normalize_label(
                    label_map[index]
                )

                scores[label] = float(
                    row[index]
                )

            predicted_index = int(
                torch.argmax(row).item()
            )

            predicted_label = normalize_label(
                label_map[predicted_index]
            )

            results.append(
                {
                    "label": predicted_label,
                    "scores": {
                        "entailment": round(
                            scores.get(
                                "entailment",
                                0.0,
                            ),
                            6,
                        ),
                        "contradiction": round(
                            scores.get(
                                "contradiction",
                                0.0,
                            ),
                            6,
                        ),
                        "neutral": round(
                            scores.get(
                                "neutral",
                                0.0,
                            ),
                            6,
                        ),
                    },
                }
            )

    return results


# ============================================================
# TOP-K AGGREGATION
# ============================================================

def aggregate_evidence(
    evidence_results
):
    """
    Aggregate the NLI results from the Top-K evidence.

    Priority:

        1. contradiction if contradictory evidence
           is at least as strong as the strongest
           entailment evidence

        2. entailment if supporting evidence exists

        3. neutral otherwise

    All individual evidence results are preserved.
    """

    contradiction_count = sum(
        item["nli"]["label"]
        == "contradiction"
        for item in evidence_results
    )

    entailment_count = sum(
        item["nli"]["label"]
        == "entailment"
        for item in evidence_results
    )

    neutral_count = sum(
        item["nli"]["label"]
        == "neutral"
        for item in evidence_results
    )

    max_contradiction = max(
        (
            item["nli"]["scores"][
                "contradiction"
            ]
            for item in evidence_results
        ),
        default=0.0,
    )

    max_entailment = max(
        (
            item["nli"]["scores"][
                "entailment"
            ]
            for item in evidence_results
        ),
        default=0.0,
    )

    if (
        contradiction_count > 0
        and max_contradiction
        >= max_entailment
    ):

        verdict = "contradiction"

    elif entailment_count > 0:

        verdict = "entailment"

    else:

        verdict = "neutral"

    return {
        "verdict": verdict,
        "entailment_count": entailment_count,
        "contradiction_count": contradiction_count,
        "neutral_count": neutral_count,
        "max_entailment": round(
            max_entailment,
            6,
        ),
        "max_contradiction": round(
            max_contradiction,
            6,
        ),
    }


# ============================================================
# MAIN
# ============================================================

def main():

    OUTPUT_JSONL.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------------
    # Load model
    # --------------------------------------------------------

    print(
        f"Loading model: {MODEL_NAME}"
    )

    tokenizer = (
        AutoTokenizer.from_pretrained(
            MODEL_NAME
        )
    )

    model = (
        AutoModelForSequenceClassification
        .from_pretrained(
            MODEL_NAME
        )
    )

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    model.to(device)
    model.eval()

    print(
        f"Device: {device}"
    )

    # --------------------------------------------------------
    # Load evidence records
    # --------------------------------------------------------

    records = load_records()

    if not records:
        raise ValueError(
            "Evidence file contains no records."
        )

    print(
        f"Evidence records loaded: "
        f"{len(records)}"
    )

    # --------------------------------------------------------
    # Label map
    # --------------------------------------------------------

    label_map = get_label_map(model)

    print(
        f"Model labels: {label_map}"
    )

    # --------------------------------------------------------
    # Process records
    # --------------------------------------------------------

    output_records = []

    for record_number, record in enumerate(
        records,
        1,
    ):

        generated_sentence = (
            record["generated_sentence"]
        )

        evidence = record["evidence"]

        if not evidence:

            raise ValueError(
                "No evidence found for "
                f"record {record_number}: "
                f"{record.get('article_id')}"
            )

        # ----------------------------------------------------
        # Create NLI pairs
        #
        # Premise    = source evidence
        # Hypothesis = generated sentence
        # ----------------------------------------------------

        pairs = [
            {
                "claim": generated_sentence,
                "evidence": item["text"],
            }
            for item in evidence
        ]

        # ----------------------------------------------------
        # Run NLI
        # ----------------------------------------------------

        nli_results = classify_pairs(
            model=model,
            tokenizer=tokenizer,
            pairs=pairs,
            label_map=label_map,
            device=device,
        )

        # ----------------------------------------------------
        # Attach NLI results to evidence
        # ----------------------------------------------------

        evidence_results = []

        for (
            evidence_item,
            nli_result,
        ) in zip(
            evidence,
            nli_results,
        ):

            evidence_results.append(
                {
                    "sentence_id": (
                        evidence_item[
                            "sentence_id"
                        ]
                    ),
                    "similarity": (
                        evidence_item[
                            "similarity"
                        ]
                    ),
                    "text": (
                        evidence_item[
                            "text"
                        ]
                    ),
                    "nli": nli_result,
                }
            )

        # ----------------------------------------------------
        # Aggregate Top-K
        # ----------------------------------------------------

        aggregate = (
            aggregate_evidence(
                evidence_results
            )
        )

        # ----------------------------------------------------
        # Final record
        # ----------------------------------------------------

        output_record = {
            "article_id": record[
                "article_id"
            ],
            "target_format": record[
                "target_format"
            ],
            "field": record["field"],
            "sentence_index": record[
                "sentence_index"
            ],
            "generated_sentence": (
                generated_sentence
            ),
            "evidence": evidence_results,
            "aggregate": aggregate,
        }

        output_records.append(
            output_record
        )

        if (
            record_number % 10 == 0
            or record_number == len(records)
        ):

            print(
                f"Processed "
                f"{record_number}/"
                f"{len(records)} "
                f"generated sentences"
            )

    # --------------------------------------------------------
    # Save output incrementally
    # --------------------------------------------------------

    with OUTPUT_JSONL.open(
        "w",
        encoding="utf-8",
    ) as f:

        for record in output_records:

            f.write(
                json.dumps(
                    record,
                    ensure_ascii=False,
                )
                + "\n"
            )

            f.flush()

    # --------------------------------------------------------
    # Count verdicts
    # --------------------------------------------------------

    verdict_counts = {
        "entailment": 0,
        "contradiction": 0,
        "neutral": 0,
    }

    for record in output_records:

        verdict = record[
            "aggregate"
        ]["verdict"]

        if verdict not in verdict_counts:

            raise ValueError(
                f"Unexpected NLI verdict: "
                f"{verdict}"
            )

        verdict_counts[
            verdict
        ] += 1

    # --------------------------------------------------------
    # Final report
    # --------------------------------------------------------

    print()
    print(
        "NLI verification completed."
    )

    print(
        "Generated sentences verified: "
        f"{len(output_records)}"
    )

    print(
        "Entailment: "
        f"{verdict_counts['entailment']}"
    )

    print(
        "Contradiction: "
        f"{verdict_counts['contradiction']}"
    )

    print(
        "Neutral: "
        f"{verdict_counts['neutral']}"
    )

    print(
        f"Output: {OUTPUT_JSONL}"
    )


if __name__ == "__main__":
    main()