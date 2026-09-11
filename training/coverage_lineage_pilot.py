"""
Stage 9 — Source Coverage + Claim Lineage

Inputs:
    data/splits/train.csv
    training_datasets/claim_ledgers_pilot.jsonl
    training_datasets/documents_pilot.jsonl
    training_datasets/nli_verification_pilot.jsonl

Output:
    training_datasets/coverage_lineage_pilot.jsonl

This stage connects:
    Claim Ledger -> generated sentences -> retrieved evidence -> NLI

It calculates important-claim coverage:
    COVERED
    PARTIALLY_COVERED
    OMITTED

No models are used in this stage.
"""

import csv
import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

SOURCE_CSV = ROOT / "data" / "raw" / "cnn_dailymail_100.csv"
LEDGER_JSONL = ROOT / "training_datasets" / "claim_ledgers_full.jsonl"
DOCUMENTS_JSONL = ROOT / "training_datasets" / "documents_full.jsonl"
NLI_JSONL = ROOT / "training_datasets" / "nli_verification_full.jsonl"
OUTPUT_JSONL = ROOT / "training_datasets" / "coverage_lineage_full.jsonl"

IMPORTANT_THRESHOLD = 0.5

# Conservative token matching for deterministic lineage.
# This does NOT claim semantic equivalence; it provides an auditable
# candidate mapping that later stages can inspect.
TOKEN_RE = re.compile(r"[A-Za-z0-9]+")


def normalize(text):
    return " ".join(
        TOKEN_RE.findall(text.lower())
    )


def tokens(text):
    return set(TOKEN_RE.findall(text.lower()))


def load_jsonl(path):
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    records = []

    with path.open("r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, 1):
            if not line.strip():
                continue

            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON on line {line_number} in {path}: {exc}"
                ) from exc

    return records


def load_sources():
    if not SOURCE_CSV.exists():
        raise FileNotFoundError(f"Source file not found: {SOURCE_CSV}")

    with SOURCE_CSV.open(
        "r",
        encoding="utf-8-sig",
        newline="",
    ) as f:
        rows = list(csv.DictReader(f))

    return {
        row["id"]: row["article"]
        for row in rows
    }


def build_ledger_map(records):
    result = {}

    for record in records:
        article_id = record["article_id"]

        if article_id in result:
            raise ValueError(
                f"Duplicate claim ledger for article: {article_id}"
            )

        result[article_id] = record

    return result


def build_document_map(records):
    result = {}

    for record in records:
        key = (
            record["article_id"],
            record["target_format"],
        )

        if key in result:
            raise ValueError(
                f"Duplicate document: {key}"
            )

        result[key] = record

    return result


def build_nli_map(records):
    result = {}

    for record in records:
        key = (
            record["article_id"],
            record["target_format"],
            record["field"],
            record["sentence_index"],
        )

        if key in result:
            raise ValueError(
                f"Duplicate NLI record: {key}"
            )

        result[key] = record

    return result


def sentence_claim_match_score(
    claim_text,
    generated_text,
):
    """
    Deterministic lexical overlap score.

    Returns:
        overlap_ratio:
            fraction of claim tokens present in generated sentence.

    This is used only to create auditable candidate lineage.
    Semantic truth remains the responsibility of NLI.
    """

    claim_tokens = tokens(claim_text)
    generated_tokens = tokens(generated_text)

    if not claim_tokens:
        return 0.0

    return len(
        claim_tokens & generated_tokens
    ) / len(claim_tokens)


def classify_claim(
    claim,
    generated_matches,
):
    """
    Classify one ledger claim.

    COVERED:
        Strong lexical candidate match and at least one
        supporting/entailing NLI result.

    PARTIALLY_COVERED:
        Candidate generated match exists, but NLI does not
        establish full support.

    OMITTED:
        No generated sentence has a sufficient candidate match.
    """

    if not generated_matches:
        return "OMITTED"

    best_match = max(
        generated_matches,
        key=lambda item: item["match_score"],
    )

    if best_match["match_score"] >= 0.80:
        supporting = (
            best_match["aggregate_verdict"]
            == "entailment"
        )

        if supporting:
            return "COVERED"

        return "PARTIALLY_COVERED"

    if best_match["match_score"] >= 0.40:
        return "PARTIALLY_COVERED"

    return "OMITTED"


def main():
    OUTPUT_JSONL.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    sources = load_sources()
    ledgers = build_ledger_map(
        load_jsonl(LEDGER_JSONL)
    )
    documents = build_document_map(
        load_jsonl(DOCUMENTS_JSONL)
    )
    nli_records = load_jsonl(NLI_JSONL)
    nli_map = build_nli_map(nli_records)

    pilot_ids = list(sources.keys())

    expected_ledger_ids = set(pilot_ids)

    if not expected_ledger_ids.issubset(ledgers):
        missing = sorted(
            expected_ledger_ids - set(ledgers)
        )
        raise ValueError(
            f"Missing claim ledgers: {missing}"
        )

    expected_document_keys = {
        (article_id, target_format)
        for article_id in pilot_ids
        for target_format in (
            "press_advisory",
            "technical_brief",
        )
    }

    if not expected_document_keys.issubset(
        documents
    ):
        missing = sorted(
            expected_document_keys - set(documents)
        )
        raise ValueError(
            f"Missing documents: {missing}"
        )

    output_records = []

    for article_id in pilot_ids:

        ledger = ledgers[article_id]

        claims = ledger.get("claims", [])

        if not claims:
            raise ValueError(
                f"No claims found for {article_id}"
            )

        important_claims = [
            claim
            for claim in claims
            if float(
                claim["importance"]
            ) >= IMPORTANT_THRESHOLD
        ]

        for target_format in (
            "press_advisory",
            "technical_brief",
        ):

            document_key = (
                article_id,
                target_format,
            )

            # ------------------------------------------------
            # Collect generated sentence + NLI information
            # ------------------------------------------------

            generated_records = []

            for (
                key,
                nli_record,
            ) in nli_map.items():

                if (
                    key[0] != article_id
                    or key[1] != target_format
                ):
                    continue

                generated_records.append(
                    {
                        "field": nli_record["field"],
                        "sentence_index": nli_record[
                            "sentence_index"
                        ],
                        "text": nli_record[
                            "generated_sentence"
                        ],
                        "aggregate_verdict": (
                            nli_record[
                                "aggregate"
                            ]["verdict"]
                        ),
                        "evidence": [
                            {
                                "sentence_id": item[
                                    "sentence_id"
                                ],
                                "similarity": item[
                                    "similarity"
                                ],
                                "nli_label": item[
                                    "nli"
                                ]["label"],
                            }
                            for item in nli_record[
                                "evidence"
                            ]
                        ],
                    }
                )

            if not generated_records:
                raise ValueError(
                    f"No NLI records for "
                    f"{document_key}"
                )

            # ------------------------------------------------
            # Build claim -> generated sentence candidates
            # ------------------------------------------------

            claim_results = []

            for claim in claims:

                candidates = []

                for generated in generated_records:

                    score = (
                        sentence_claim_match_score(
                            claim["text"],
                            generated["text"],
                        )
                    )

                    # Only retain meaningful candidate matches.
                    if score >= 0.40:

                        candidates.append(
                            {
                                "field": generated[
                                    "field"
                                ],
                                "sentence_index": (
                                    generated[
                                        "sentence_index"
                                    ]
                                ),
                                "generated_sentence": (
                                    generated["text"]
                                ),
                                "match_score": round(
                                    score,
                                    4,
                                ),
                                "aggregate_verdict": (
                                    generated[
                                        "aggregate_verdict"
                                    ]
                                ),
                                "evidence_sentence_ids": [
                                    item[
                                        "sentence_id"
                                    ]
                                    for item in generated[
                                        "evidence"
                                    ]
                                ],
                            }
                        )

                status = classify_claim(
                    claim,
                    candidates,
                )

                claim_results.append(
                    {
                        "claim_id": claim["id"],
                        "claim_text": claim["text"],
                        "importance": claim[
                            "importance"
                        ],
                        "claim_type": claim["type"],
                        "source_sentence_id": claim[
                            "source_sentence_id"
                        ],
                        "coverage": status,
                        "candidate_lineage": sorted(
                            candidates,
                            key=lambda item: (
                                -item["match_score"],
                                item["field"],
                                item["sentence_index"],
                            ),
                        )[:5],
                    }
                )

            # ------------------------------------------------
            # Important claim coverage
            # ------------------------------------------------

            important_results = [
                result
                for result in claim_results
                if float(
                    result["importance"]
                ) >= IMPORTANT_THRESHOLD
            ]

            covered = sum(
                result["coverage"] == "COVERED"
                for result in important_results
            )

            partially_covered = sum(
                result["coverage"]
                == "PARTIALLY_COVERED"
                for result in important_results
            )

            omitted = sum(
                result["coverage"] == "OMITTED"
                for result in important_results
            )

            total_important = len(
                important_results
            )

            coverage_percentage = (
                round(
                    (
                        covered
                        + 0.5 * partially_covered
                    )
                    / total_important
                    * 100,
                    2,
                )
                if total_important
                else 100.0
            )

            # ------------------------------------------------
            # Document-level sentence lineage
            # ------------------------------------------------

            document_lineage = []

            for generated in generated_records:

                related_claims = []

                for claim in claims:

                    score = (
                        sentence_claim_match_score(
                            claim["text"],
                            generated["text"],
                        )
                    )

                    if score >= 0.40:

                        related_claims.append(
                            {
                                "claim_id": claim[
                                    "id"
                                ],
                                "match_score": round(
                                    score,
                                    4,
                                ),
                                "source_sentence_id": (
                                    claim[
                                        "source_sentence_id"
                                    ]
                                ),
                            }
                        )

                related_claims.sort(
                    key=lambda item: -item[
                        "match_score"
                    ]
                )

                document_lineage.append(
                    {
                        "field": generated[
                            "field"
                        ],
                        "sentence_index": (
                            generated[
                                "sentence_index"
                            ]
                        ),
                        "generated_sentence": (
                            generated["text"]
                        ),
                        "claim_ids": [
                            item["claim_id"]
                            for item in related_claims[:5]
                        ],
                        "source_sentence_ids": [
                            item[
                                "source_sentence_id"
                            ]
                            for item in related_claims[:5]
                        ],
                        "evidence_sentence_ids": [
                            item["sentence_id"]
                            for item in generated[
                                "evidence"
                            ]
                        ],
                        "nli_verdict": (
                            generated[
                                "aggregate_verdict"
                            ]
                        ),
                    }
                )

            result = {
                "article_id": article_id,
                "target_format": target_format,
                "important_claims": {
                    "total": total_important,
                    "covered": covered,
                    "partially_covered": (
                        partially_covered
                    ),
                    "omitted": omitted,
                    "coverage_percentage": (
                        coverage_percentage
                    ),
                },
                "claim_coverage": claim_results,
                "document_lineage": document_lineage,
            }

            output_records.append(result)

    # --------------------------------------------------------
    # Save incrementally
    # --------------------------------------------------------

    with OUTPUT_JSONL.open(
        "w",
        encoding="utf-8",
    ) as f:

        for result in output_records:

            f.write(
                json.dumps(
                    result,
                    ensure_ascii=False,
                )
                + "\n"
            )

            f.flush()

    print()
    print(
        "Source coverage + claim lineage completed."
    )
    print(
        f"Documents processed: "
        f"{len(output_records)}"
    )

    total_important = 0
    total_covered = 0
    total_partial = 0
    total_omitted = 0

    for result in output_records:

        stats = result["important_claims"]

        total_important += stats["total"]
        total_covered += stats["covered"]
        total_partial += stats[
            "partially_covered"
        ]
        total_omitted += stats["omitted"]

    print(
        f"Important claims: {total_important}"
    )
    print(
        f"Covered: {total_covered}"
    )
    print(
        f"Partially covered: {total_partial}"
    )
    print(
        f"Omitted: {total_omitted}"
    )

    overall_coverage = (
        (
            total_covered
            + 0.5 * total_partial
        )
        / total_important
        * 100
        if total_important
        else 100.0
    )

    print(
        "Weighted important-claim coverage: "
        f"{overall_coverage:.2f}%"
    )

    print(
        f"Output: {OUTPUT_JSONL}"
    )


if __name__ == "__main__":
    main()
