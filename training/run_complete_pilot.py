"""
Stage 11 — Complete 10-document end-to-end pilot.

This stage re-verifies the REPAIRED documents through the complete
deterministic + retrieval + NLI + coverage/lineage chain.

It does not modify:
    data/raw/*
    data/splits/*
    training_datasets/documents_pilot.jsonl

It creates Stage-11-specific reports so the original Stage 6–10
artifacts remain available for comparison.
"""

import importlib.util
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TRAINING_DIR = ROOT / "training"
DATASETS_DIR = ROOT / "training_datasets"

REPAIRED_DOCUMENTS = DATASETS_DIR / "repaired_documents_pilot.jsonl"
REPAIR_AUDIT = DATASETS_DIR / "repair_audit_pilot.jsonl"

STAGE11_VALIDATION = DATASETS_DIR / "stage11_document_validation.jsonl"
STAGE11_EVIDENCE = DATASETS_DIR / "stage11_evidence.jsonl"
STAGE11_NLI = DATASETS_DIR / "stage11_nli_verification.jsonl"
STAGE11_COVERAGE = DATASETS_DIR / "stage11_coverage_lineage.jsonl"
STAGE11_AUDIT = DATASETS_DIR / "stage11_final_audit.jsonl"


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load module: {path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


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
                    f"Invalid JSON on line {line_number}: {exc}"
                ) from exc

    return records


def write_jsonl(path, records):
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(
                json.dumps(
                    record,
                    ensure_ascii=False,
                )
                + "\n"
            )
            f.flush()


def configure_validation():
    module = load_module(
        TRAINING_DIR / "validate_documents.py",
        "stage11_validate_documents",
    )

    module.DOCUMENTS_JSONL = REPAIRED_DOCUMENTS
    module.OUTPUT_JSONL = STAGE11_VALIDATION

    return module


def configure_retrieval():
    module = load_module(
        TRAINING_DIR / "retrieve_evidence.py",
        "stage11_retrieve_evidence",
    )

    module.DOCUMENTS_JSONL = REPAIRED_DOCUMENTS
    module.OUTPUT_JSONL = STAGE11_EVIDENCE

    return module


def configure_nli():
    module = load_module(
        TRAINING_DIR / "nli_verify.py",
        "stage11_nli_verify",
    )

    # nli_verify.py uses INPUT_JSONL (not EVIDENCE_JSONL).
    module.INPUT_JSONL = STAGE11_EVIDENCE
    module.OUTPUT_JSONL = STAGE11_NLI

    return module


def configure_coverage():
    module = load_module(
        TRAINING_DIR / "coverage_lineage_pilot.py",
        "stage11_coverage_lineage",
    )

    module.DOCUMENTS_JSONL = REPAIRED_DOCUMENTS
    module.NLI_JSONL = STAGE11_NLI
    module.OUTPUT_JSONL = STAGE11_COVERAGE

    return module


def validate_repair_audit():
    records = load_jsonl(REPAIR_AUDIT)

    required = {
        "article_id",
        "target_format",
        "field",
        "sentence_index",
        "accepted",
        "status",
    }

    failures = []

    for index, record in enumerate(records, 1):
        missing = required - set(record.keys())

        if missing:
            failures.append(
                f"repair audit record {index} missing {sorted(missing)}"
            )

        if record.get("status") not in {
            "REPAIRED",
            "ORIGINAL_PRESERVED",
            "SKIPPED_INVALID_LOCATION",
        }:
            failures.append(
                f"repair audit record {index} has invalid status"
            )

    return records, failures


def build_final_audit(
    validation_records,
    evidence_records,
    nli_records,
    coverage_records,
    repair_records,
):
    validation_map = {
        (
            record["article_id"],
            record["target_format"],
        ): record
        for record in validation_records
        if record.get("article_id") != "__PILOT_SUMMARY__"
    }

    evidence_counts = {}

    for record in evidence_records:
        key = (
            record["article_id"],
            record["target_format"],
        )
        evidence_counts[key] = (
            evidence_counts.get(key, 0) + 1
        )

    nli_map = {}

    for record in nli_records:
        key = (
            record["article_id"],
            record["target_format"],
        )
        nli_map[key] = nli_map.get(key, 0) + 1

    coverage_map = {
        (
            record["article_id"],
            record["target_format"],
        ): record
        for record in coverage_records
    }

    repair_counts = {}

    for record in repair_records:
        key = (
            record["article_id"],
            record["target_format"],
        )

        repair_counts.setdefault(
            key,
            {
                "repaired": 0,
                "preserved": 0,
            },
        )

        if record.get("status") == "REPAIRED":
            repair_counts[key]["repaired"] += 1

        elif record.get("status") == "ORIGINAL_PRESERVED":
            repair_counts[key]["preserved"] += 1

    expected_keys = sorted(
        validation_map.keys()
    )

    audits = []

    for key in expected_keys:
        article_id, target_format = key

        validation = validation_map[key]
        coverage = coverage_map.get(key, {})
        repairs = repair_counts.get(
            key,
            {
                "repaired": 0,
                "preserved": 0,
            },
        )

        nli_count = nli_map.get(key, 0)
        evidence_count = evidence_counts.get(key, 0)

        nli_verdict_counts = {
            "entailment": 0,
            "contradiction": 0,
            "neutral": 0,
        }

        for record in nli_records:
            if (
                record.get("article_id"),
                record.get("target_format"),
            ) != key:
                continue

            verdict = record.get(
                "aggregate",
                {},
            ).get("verdict")

            if verdict in nli_verdict_counts:
                nli_verdict_counts[verdict] += 1

        structural_status = validation.get(
            "structural_status",
            "FAIL",
        )

        number_status = validation.get(
            "number_validation",
            "FAIL",
        )

        date_status = validation.get(
            "date_validation",
            "FAIL",
        )

        coverage_info = coverage.get(
            "important_claims",
            {},
        )

        coverage_percentage = coverage_info.get(
            "coverage_percentage"
        )

        contradictions = nli_verdict_counts[
            "contradiction"
        ]

        unresolved_claims = (
            coverage_info.get("omitted", 0)
            if isinstance(coverage_info, dict)
            else None
        )

        factual_support = (
            "PASS"
            if contradictions == 0
            else "REVIEW"
        )

        if (
            structural_status == "PASS"
            and number_status == "PASS"
            and date_status == "PASS"
            and evidence_count == nli_count
            and nli_count > 0
            and coverage
        ):
            final_status = "PASS_WITH_NLI_REVIEW"
        else:
            final_status = "FAIL"

        audits.append(
            {
                "article_id": article_id,
                "target_format": target_format,
                "structural_status": structural_status,
                "number_validation": number_status,
                "date_validation": date_status,
                "factual_support": factual_support,
                "contradictions": contradictions,
                "nli_sentence_counts": nli_verdict_counts,
                "evidence_sentence_records": evidence_count,
                "nli_sentence_records": nli_count,
                "unresolved_claims": unresolved_claims,
                "source_coverage": coverage_percentage,
                "repairs_performed": repairs["repaired"],
                "repairs_preserved": repairs["preserved"],
                "final_status": final_status,
            }
        )

    return audits


def main():
    DATASETS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    if not REPAIRED_DOCUMENTS.exists():
        raise FileNotFoundError(
            f"Stage 10 repaired documents not found: "
            f"{REPAIRED_DOCUMENTS}"
        )

    if not REPAIR_AUDIT.exists():
        raise FileNotFoundError(
            f"Stage 10 repair audit not found: "
            f"{REPAIR_AUDIT}"
        )

    print("=" * 60)
    print("STAGE 11 — COMPLETE 10-DOCUMENT PILOT")
    print("=" * 60)

    # ---------------------------------------------------------
    # 1. Validate repaired documents
    # ---------------------------------------------------------

    print("\n[1/5] Structural + deterministic validation")

    validation_module = configure_validation()
    validation_module.main()

    # ---------------------------------------------------------
    # 2. Re-retrieve evidence from repaired documents
    # ---------------------------------------------------------

    print("\n[2/5] Evidence retrieval on repaired documents")

    retrieval_module = configure_retrieval()
    retrieval_module.main()

    # ---------------------------------------------------------
    # 3. Re-run NLI on repaired documents' evidence
    # ---------------------------------------------------------

    print("\n[3/5] NLI verification on repaired documents")

    nli_module = configure_nli()
    nli_module.main()

    # ---------------------------------------------------------
    # 4. Recalculate coverage + lineage
    # ---------------------------------------------------------

    print("\n[4/5] Source coverage + claim lineage")

    coverage_module = configure_coverage()
    coverage_module.main()

    # ---------------------------------------------------------
    # 5. Build final auditable report
    # ---------------------------------------------------------

    print("\n[5/5] Final audit")

    validation_records = load_jsonl(
        STAGE11_VALIDATION
    )

    evidence_records = load_jsonl(
        STAGE11_EVIDENCE
    )

    nli_records = load_jsonl(
        STAGE11_NLI
    )

    coverage_records = load_jsonl(
        STAGE11_COVERAGE
    )

    repair_records, repair_failures = (
        validate_repair_audit()
    )

    if repair_failures:
        raise ValueError(
            "Repair audit validation failed:\n"
            + "\n".join(repair_failures)
        )

    audits = build_final_audit(
        validation_records=validation_records,
        evidence_records=evidence_records,
        nli_records=nli_records,
        coverage_records=coverage_records,
        repair_records=repair_records,
    )

    write_jsonl(
        STAGE11_AUDIT,
        audits,
    )

    # ---------------------------------------------------------
    # Summary
    # ---------------------------------------------------------

    total = len(audits)
    structural_pass = sum(
        a["structural_status"] == "PASS"
        for a in audits
    )
    number_pass = sum(
        a["number_validation"] == "PASS"
        for a in audits
    )
    date_pass = sum(
        a["date_validation"] == "PASS"
        for a in audits
    )

    total_evidence = sum(
        a["evidence_sentence_records"]
        for a in audits
    )

    total_nli = sum(
        a["nli_sentence_records"]
        for a in audits
    )

    total_contradictions = sum(
        a["contradictions"]
        for a in audits
    )

    total_repairs = sum(
        a["repairs_performed"]
        for a in audits
    )

    print("\n" + "=" * 60)
    print("STAGE 11 COMPLETE")
    print("=" * 60)

    print(f"Pilot documents: {total}")
    print(
        f"Structural PASS: "
        f"{structural_pass}/{total}"
    )
    print(
        f"Number validation PASS: "
        f"{number_pass}/{total}"
    )
    print(
        f"Date validation PASS: "
        f"{date_pass}/{total}"
    )
    print(
        f"Evidence sentence records: "
        f"{total_evidence}"
    )
    print(
        f"NLI sentence records: "
        f"{total_nli}"
    )
    print(
        f"Aggregate contradictions after repair: "
        f"{total_contradictions}"
    )
    print(
        f"Repairs accepted in Stage 10: "
        f"{total_repairs}"
    )

    print("\nStage 11 reports:")
    print(f"  Validation: {STAGE11_VALIDATION}")
    print(f"  Evidence:   {STAGE11_EVIDENCE}")
    print(f"  NLI:        {STAGE11_NLI}")
    print(f"  Coverage:   {STAGE11_COVERAGE}")
    print(f"  Final audit:{STAGE11_AUDIT}")

    # Stage 11 is considered technically complete only when all
    # 10 repaired documents survive structural/deterministic checks
    # and every sentence has evidence + NLI verification.
    if (
        total != 10
        or structural_pass != 10
        or number_pass != 10
        or date_pass != 10
        or total_evidence != total_nli
        or total_nli == 0
    ):
        print("\nSTAGE 11 STATUS: FAIL")
        raise SystemExit(1)

    print("\nSTAGE 11 STATUS: PASS")
    print(
        "Note: remaining NLI contradictions are reported for review; "
        "they are not silently treated as confirmed factual errors."
    )


if __name__ == "__main__":
    main()
