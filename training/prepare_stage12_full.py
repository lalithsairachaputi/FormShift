"""
Stage 12 — final preparation for the full 100-article pipeline.

This is based on the current training scripts supplied by the user.
It is idempotent: already-correct settings are accepted and only remaining
settings are changed. It does NOT run any generation/verification stage.

Run from project root:
    python training/prepare_stage12_full_final.py
"""

from pathlib import Path
import ast
import re

ROOT = Path(__file__).resolve().parents[1]
TRAINING = ROOT / "training"

FILES = [
    "generate_transformation_contracts.py",
    "generate_claim_ledgers.py",
    "generate_documents.py",
    "validate_documents.py",
    "retrieve_evidence.py",
    "nli_verify.py",
    "coverage_lineage_pilot.py",
    "targeted_repair.py",
]


def replace_once(text, old, new, filename):
    count = text.count(old)
    if count == 1:
        return text.replace(old, new, 1), True
    if count == 0:
        return text, False
    raise RuntimeError(
        f"{filename}: expected one occurrence of {old!r}, found {count}."
    )


def replace_all_if_present(text, old, new):
    if old not in text:
        return text, False
    return text.replace(old, new), True


def patch_file(name):
    path = TRAINING / name
    if not path.exists():
        raise FileNotFoundError(path)

    text = path.read_text(encoding="utf-8")
    original = text

    if name == "generate_transformation_contracts.py":
        text, _ = replace_all_if_present(
            text,
            'OUTPUT_FILE = OUTPUT_DIR / "transformation_contracts_pilot.jsonl"',
            'OUTPUT_FILE = OUTPUT_DIR / "transformation_contracts_full.jsonl"',
        )
        text, _ = replace_all_if_present(
            text, "PILOT_ARTICLES = 5", "PILOT_ARTICLES = 100"
        )

    elif name == "generate_claim_ledgers.py":
        text, _ = replace_all_if_present(
            text,
            'OUTPUT_FILE = OUTPUT_DIR / "claim_ledgers_pilot.jsonl"',
            'OUTPUT_FILE = OUTPUT_DIR / "claim_ledgers_full.jsonl"',
        )
        text, _ = replace_all_if_present(
            text, "PILOT_ARTICLES = 5", "PILOT_ARTICLES = 100"
        )

    elif name == "generate_documents.py":
        for old, new in [
            ('"transformation_contracts_pilot.jsonl"',
             '"transformation_contracts_full.jsonl"'),
            ('"claim_ledgers_pilot.jsonl"',
             '"claim_ledgers_full.jsonl"'),
            ('"documents_pilot.jsonl"',
             '"documents_full.jsonl"'),
            ("pilot_ids = list(ledgers.keys())[:5]",
             "pilot_ids = list(ledgers.keys())"),
            ('print(f"Pilot articles : {len(pilot_ids)}")',
             'print(f"Full articles  : {len(pilot_ids)}")'),
        ]:
            text, _ = replace_all_if_present(text, old, new)

    elif name == "validate_documents.py":
        for old, new in [
            ('DOCUMENTS_JSONL = ROOT / "training_datasets" / "documents_pilot.jsonl"',
             'DOCUMENTS_JSONL = ROOT / "training_datasets" / "documents_full.jsonl"'),
            ('OUTPUT_JSONL = ROOT / "training_datasets" / "document_validation_pilot.jsonl"',
             'OUTPUT_JSONL = ROOT / "training_datasets" / "document_validation_full.jsonl"'),
            ("expected_ids = list(sources.keys())[:5]",
             "expected_ids = list(sources.keys())"),
            ('"__PILOT_SUMMARY__"', '"__FULL_SUMMARY__"'),
            ('"Missing expected pilot pairs: "',
             '"Missing expected full pairs: "'),
            ('print(f"Expected pilot pairs: {len(expected_pairs)}")',
             'print(f"Expected full pairs: {len(expected_pairs)}")'),
            ('print(f"Unique pilot pairs found: {len(seen_pairs)}")',
             'print(f"Unique full pairs found: {len(seen_pairs)}")'),
            ('len(record_results) != 10',
             'len(record_results) != len(expected_pairs)'),
        ]:
            text, _ = replace_all_if_present(text, old, new)

    elif name == "retrieve_evidence.py":
        for old, new in [
            ('DOCUMENTS_JSONL = ROOT / "training_datasets" / "documents_pilot.jsonl"',
             'DOCUMENTS_JSONL = ROOT / "training_datasets" / "documents_full.jsonl"'),
            ('OUTPUT_JSONL = ROOT / "training_datasets" / "evidence_pilot.jsonl"',
             'OUTPUT_JSONL = ROOT / "training_datasets" / "evidence_full.jsonl"'),
            ("expected_ids = list(sources.keys())[:5]",
             "expected_ids = list(sources.keys())"),
            ("pilot_documents = [",
             "full_documents = ["),
            ("if len(pilot_documents) != 10:",
             "if len(full_documents) != 200:"),
            ('f"Expected exactly 10 pilot documents, "',
             'f"Expected exactly 200 full documents, "'),
            ('f"found {len(pilot_documents)}."',
             'f"found {len(full_documents)}."'),
            ("            pilot_documents,", "            full_documents,"),
            ('f"[{document_number}/10] "', 'f"[{document_number}/200] "'),
        ]:
            text, _ = replace_all_if_present(text, old, new)

        # Ensure SentenceTransformer is explicitly placed on CUDA when available.
        if "import torch\nfrom sentence_transformers import SentenceTransformer" not in text:
            text, changed = replace_once(
                text,
                "from sentence_transformers import SentenceTransformer",
                "import torch\nfrom sentence_transformers import SentenceTransformer",
                name,
            )

        desired_model_block = (
            '    device = "cuda" if torch.cuda.is_available() else "cpu"\n'
            '    print(f"Embedding device: {device}")\n'
            '    model = SentenceTransformer(MODEL_NAME, device=device)'
        )
        cpu_block = "    model = SentenceTransformer(MODEL_NAME)"
        if desired_model_block not in text:
            text, changed = replace_once(
                text, cpu_block, desired_model_block, name
            )

    elif name == "nli_verify.py":
        text, _ = replace_all_if_present(
            text, '/ "evidence_pilot.jsonl"', '/ "evidence_full.jsonl"'
        )
        text, _ = replace_all_if_present(
            text,
            '/ "nli_verification_pilot.jsonl"',
            '/ "nli_verification_full.jsonl"',
        )

    elif name == "coverage_lineage_pilot.py":
        for old, new in [
            ('/ "claim_ledgers_pilot.jsonl"', '/ "claim_ledgers_full.jsonl"'),
            ('/ "documents_pilot.jsonl"', '/ "documents_full.jsonl"'),
            ('/ "nli_verification_pilot.jsonl"',
             '/ "nli_verification_full.jsonl"'),
            ('/ "coverage_lineage_pilot.jsonl"',
             '/ "coverage_lineage_full.jsonl"'),
            ("pilot_ids = list(sources.keys())[:5]",
             "pilot_ids = list(sources.keys())"),
            ('"__PILOT_SUMMARY__"', '"__FULL_SUMMARY__"'),
        ]:
            text, _ = replace_all_if_present(text, old, new)

    elif name == "targeted_repair.py":
        for old, new in [
            ('/ "documents_pilot.jsonl"', '/ "documents_full.jsonl"'),
            ('/ "evidence_pilot.jsonl"', '/ "evidence_full.jsonl"'),
            ('/ "nli_verification_pilot.jsonl"',
             '/ "nli_verification_full.jsonl"'),
            ('/ "repaired_documents_pilot.jsonl"',
             '/ "repaired_documents_full.jsonl"'),
            ('/ "repair_audit_pilot.jsonl"',
             '/ "repair_audit_full.jsonl"'),
        ]:
            text, _ = replace_all_if_present(text, old, new)

    ast.parse(text, filename=str(path))
    if text != original:
        path.write_text(text, encoding="utf-8")
        return "patched"
    return "already"


def assert_contains(name, needles):
    text = (TRAINING / name).read_text(encoding="utf-8")
    for needle in needles:
        if needle not in text:
            raise RuntimeError(
                f"Final configuration check failed in {name}: {needle!r}"
            )


def main():
    print("=" * 68)
    print("STAGE 12 — FINAL FULL PIPELINE PREPARATION")
    print("=" * 68)

    for name in FILES:
        print(f"{name}: {patch_file(name)}")

    print("\nSyntax verification:")
    for name in FILES:
        path = TRAINING / name
        ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        print(f"  PASS: {name}")

    print("\nConfiguration verification:")

    assert_contains(
        "generate_transformation_contracts.py",
        ["transformation_contracts_full.jsonl", "PILOT_ARTICLES = 100"],
    )
    print("  PASS: transformation contracts -> 100 articles")

    assert_contains(
        "generate_claim_ledgers.py",
        ["claim_ledgers_full.jsonl", "PILOT_ARTICLES = 100"],
    )
    print("  PASS: claim ledgers -> 100 articles")

    assert_contains(
        "generate_documents.py",
        [
            "transformation_contracts_full.jsonl",
            "claim_ledgers_full.jsonl",
            "documents_full.jsonl",
            "pilot_ids = list(ledgers.keys())",
        ],
    )
    print("  PASS: documents -> 200 documents")

    assert_contains(
        "validate_documents.py",
        [
            "documents_full.jsonl",
            "document_validation_full.jsonl",
            "expected_ids = list(sources.keys())",
            "__FULL_SUMMARY__",
            "len(record_results) != len(expected_pairs)",
        ],
    )
    print("  PASS: structural validation -> 200 pairs")

    assert_contains(
        "retrieve_evidence.py",
        [
            "documents_full.jsonl",
            "evidence_full.jsonl",
            "expected_ids = list(sources.keys())",
            "full_documents",
            "!= 200",
            'device = "cuda" if torch.cuda.is_available() else "cpu"',
            "SentenceTransformer(MODEL_NAME, device=device)",
        ],
    )
    if "pilot_documents" in (
        TRAINING / "retrieve_evidence.py"
    ).read_text(encoding="utf-8"):
        raise RuntimeError(
            "retrieve_evidence.py still contains pilot_documents."
        )
    print("  PASS: BGE retrieval -> 200 documents + CUDA selection")

    assert_contains(
        "nli_verify.py",
        ["evidence_full.jsonl", "nli_verification_full.jsonl"],
    )
    print("  PASS: NLI -> full evidence input/output")

    assert_contains(
        "coverage_lineage_pilot.py",
        [
            "claim_ledgers_full.jsonl",
            "documents_full.jsonl",
            "nli_verification_full.jsonl",
            "coverage_lineage_full.jsonl",
            "pilot_ids = list(sources.keys())",
        ],
    )
    print("  PASS: coverage/lineage -> 100 articles")

    assert_contains(
        "targeted_repair.py",
        [
            "documents_full.jsonl",
            "evidence_full.jsonl",
            "nli_verification_full.jsonl",
            "repaired_documents_full.jsonl",
            "repair_audit_full.jsonl",
        ],
    )
    print("  PASS: targeted repair -> full outputs")

    print("\nSTAGE 12 PREPARATION STATUS: PASS")
    print("Target: 100 articles × 2 formats = 200 documents.")
    print("Pilot outputs are not used by the full pipeline.")
    print("Raw CSV is not modified.")
    print("No generation or verification stage was started.")
    print("\nNext: perform Stage 12 pre-flight checks before generation.")


if __name__ == "__main__":
    main()
