"""
Stage 10 (corrected) — Targeted Repair

Purpose:
    Repair only sentences with an explicit NLI contradiction.

Inputs:
    data/splits/train.csv
    training_datasets/documents_pilot.jsonl
    training_datasets/evidence_pilot.jsonl
    training_datasets/nli_verification_pilot.jsonl

Outputs:
    training_datasets/repaired_documents_pilot.jsonl
    training_datasets/repair_audit_pilot.jsonl

Model:
    qwen2.5:7b-instruct through local Ollama

Important design correction:
    - NEUTRAL is NOT automatically repaired.
    - PARTIALLY_COVERED / OMITTED claims are NOT automatically repaired.
    - Only explicit sentence-level NLI CONTRADICTIONS are repaired.
    - A sentence is repaired at most once.
    - If a safe replacement cannot be produced, the ORIGINAL sentence is
      preserved rather than replacing it with "Not provided in source."
    - Original documents are never modified.
"""

import csv
import json
import re
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

SOURCE_CSV = ROOT / "data" / "raw" / "cnn_dailymail_100.csv"
DOCUMENTS_JSONL = ROOT / "training_datasets" / "documents_full.jsonl"
EVIDENCE_JSONL = ROOT / "training_datasets" / "evidence_full.jsonl"
NLI_JSONL = ROOT / "training_datasets" / "nli_verification_full.jsonl"

REPAIRED_DOCUMENTS_JSONL = (
    ROOT / "training_datasets" / "repaired_documents_full.jsonl"
)
REPAIR_AUDIT_JSONL = (
    ROOT / "training_datasets" / "repair_audit_full.jsonl"
)

OLLAMA_MODEL = "qwen2.5:7b-instruct"
MAX_REPAIR_ATTEMPTS = 2

FALLBACK = "Not provided in source."

DOCUMENT_FIELDS = (
    "overview",
    "key_findings",
    "technical_details",
    "implications",
)

SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")

NUMBER_RE = re.compile(
    r"(?<![\w.])"
    r"(?:\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+(?:\.\d+)?)"
    r"(?:%|[KMB])?"
    r"(?![\w.])",
    re.IGNORECASE,
)

YEAR_RE = re.compile(r"(?<!\d)(?:18|19|20|21)\d{2}(?!\d)")

DATE_PATTERNS = [
    re.compile(
        r"\b(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|"
        r"Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:t(?:ember)?)?|"
        r"Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)"
        r"\s+\d{1,2}(?:,\s*|\s+)\d{4}\b",
        re.IGNORECASE,
    ),
    re.compile(r"\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b"),
    re.compile(r"\b\d{4}-\d{1,2}-\d{1,2}\b"),
]


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

    required = {"article", "highlights", "id"}

    if not rows:
        raise ValueError("train.csv contains no rows.")

    if not required.issubset(rows[0].keys()):
        raise ValueError(
            f"Missing source columns. Required: {sorted(required)}"
        )

    return {
        row["id"]: row["article"]
        for row in rows
    }


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
            raise ValueError(f"Duplicate NLI record: {key}")

        result[key] = record

    return result


def build_evidence_map(records):
    result = {}

    for record in records:
        key = (
            record["article_id"],
            record["target_format"],
            record["field"],
            record["sentence_index"],
        )

        if key in result:
            raise ValueError(f"Duplicate evidence record: {key}")

        result[key] = record

    return result


def normalize_number(token):
    token = token.strip().replace(",", "")
    return re.sub(
        r"[KMB%]$",
        "",
        token,
        flags=re.IGNORECASE,
    )


def extract_numbers(text):
    return {
        normalize_number(match.group(0))
        for match in NUMBER_RE.finditer(text)
    }


def extract_dates(text):
    dates = set()

    for pattern in DATE_PATTERNS:
        dates.update(
            match.group(0).lower()
            for match in pattern.finditer(text)
        )

    dates.update(YEAR_RE.findall(text))

    return dates


def deterministic_safety_check(
    replacement,
    source_text,
):
    """
    Check only deterministic source-closed constraints that can be
    safely established without another model.

    Returns:
        (True, None) when safe
        (False, reason) otherwise
    """

    if not isinstance(replacement, str):
        return False, "Replacement is not a string."

    replacement = replacement.strip()

    if not replacement:
        return False, "Replacement is empty."

    if replacement == FALLBACK:
        return True, None

    if "```" in replacement:
        return False, "Replacement contains a Markdown fence."

    unsupported_numbers = sorted(
        extract_numbers(replacement)
        - extract_numbers(source_text)
    )

    if unsupported_numbers:
        return (
            False,
            "Replacement contains unsupported numbers: "
            + ", ".join(unsupported_numbers),
        )

    unsupported_dates = sorted(
        extract_dates(replacement)
        - extract_dates(source_text)
    )

    if unsupported_dates:
        return (
            False,
            "Replacement contains unsupported dates/years: "
            + ", ".join(unsupported_dates),
        )

    return True, None


def extract_json_object(text):
    text = text.strip()

    if text.startswith("```"):
        text = re.sub(
            r"^```(?:json)?\s*",
            "",
            text,
            flags=re.IGNORECASE,
        )
        text = re.sub(
            r"\s*```$",
            "",
            text,
        )

    start = text.find("{")
    end = text.rfind("}")

    if start == -1 or end == -1 or end <= start:
        raise ValueError(
            "Model response does not contain a JSON object."
        )

    return json.loads(
        text[start:end + 1]
    )


def call_ollama(prompt):
    result = subprocess.run(
        [
            "ollama",
            "run",
            OLLAMA_MODEL,
            prompt,
        ],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )

    if result.returncode != 0:
        raise RuntimeError(
            "Ollama failed:\n"
            + result.stderr.strip()
        )

    return result.stdout.strip()


def build_repair_prompt(
    article_id,
    target_format,
    field,
    original_sentence,
    evidence,
):
    evidence_text = "\n".join(
        f"- {item['sentence_id']}: {item['text']}"
        for item in evidence
    )

    return f"""
You are performing a targeted factual repair.

Article ID:
{article_id}

Target format:
{target_format}

Document field:
{field}

Original generated sentence:
{original_sentence}

The sentence was classified as CONTRADICTION by the factual
verification system.

Source evidence:
{evidence_text}

Rules:
1. Return exactly one replacement sentence.
2. Correct only the factual contradiction.
3. Use ONLY information supported by the supplied source evidence.
4. Do not use outside knowledge.
5. Do not invent or add facts, names, organizations, numbers, dates,
   measurements, methods, quotes, causes, outcomes, or technical details.
6. Preserve useful wording from the original sentence when it remains true.
7. If the evidence cannot safely support a replacement, return exactly:
   "Not provided in source."
8. Do not use Markdown.
9. Do not explain your answer.

Return JSON only:
{{"replacement":"..."}}
"""

def repair_sentence(
    article_id,
    target_format,
    field,
    original_sentence,
    evidence,
    source_text,
):
    prompt = build_repair_prompt(
        article_id=article_id,
        target_format=target_format,
        field=field,
        original_sentence=original_sentence,
        evidence=evidence,
    )

    last_error = None

    for attempt in range(1, MAX_REPAIR_ATTEMPTS + 1):

        raw = call_ollama(prompt)

        try:
            parsed = extract_json_object(raw)

            replacement = parsed.get("replacement")

            if not isinstance(replacement, str):
                raise ValueError(
                    "JSON field 'replacement' must be a string."
                )

            replacement = replacement.strip()

            safe, reason = deterministic_safety_check(
                replacement,
                source_text,
            )

            if safe:
                return {
                    "replacement": replacement,
                    "attempts": attempt,
                    "accepted": True,
                    "reason": None,
                }

            last_error = reason

            prompt += f"""

The previous replacement was rejected because:
{reason}

Return a new replacement that obeys all rules.
"""

        except Exception as exc:
            last_error = str(exc)

            prompt += f"""

The previous response was invalid:
{exc}

Return valid JSON with exactly one replacement sentence.
"""

    # Critical correction:
    # Preserve the original instead of destroying it when repair is unsafe.
    return {
        "replacement": original_sentence,
        "attempts": MAX_REPAIR_ATTEMPTS,
        "accepted": False,
        "reason": (
            "Repair rejected after maximum attempts. "
            f"Original sentence preserved. Last reason: {last_error}"
        ),
    }


def split_string_sentences(value):
    return [
        item.strip()
        for item in SENTENCE_SPLIT_RE.split(value.strip())
        if item.strip()
    ]


def collect_sentence_locations(document, field):
    """
    Return references to sentence positions inside a document field.

    For string fields:
        ("string", sentence_index)

    For list fields:
        ("list", list_index, sentence_index_within_item)
    """

    value = document["document"][field]
    locations = []

    if isinstance(value, str):

        sentences = split_string_sentences(value)

        for index in range(len(sentences)):
            locations.append(
                ("string", index)
            )

    elif isinstance(value, list):

        for list_index, item in enumerate(value):

            if not isinstance(item, str):
                continue

            sentences = split_string_sentences(item)

            for sentence_index in range(len(sentences)):
                locations.append(
                    (
                        "list",
                        list_index,
                        sentence_index,
                    )
                )

    return locations


def get_sentence_at(
    document,
    field,
    sentence_index,
):
    value = document["document"][field]

    if isinstance(value, str):

        sentences = split_string_sentences(value)

        if sentence_index < 1 or sentence_index > len(sentences):
            return None

        return sentences[sentence_index - 1]

    if isinstance(value, list):

        flattened = []

        for item in value:
            if isinstance(item, str):
                flattened.extend(
                    split_string_sentences(item)
                )

        if sentence_index < 1 or sentence_index > len(flattened):
            return None

        return flattened[sentence_index - 1]

    return None


def replace_sentence(
    document,
    field,
    sentence_index,
    replacement,
):
    value = document["document"][field]

    if isinstance(value, str):

        sentences = split_string_sentences(value)

        if sentence_index < 1 or sentence_index > len(sentences):
            raise IndexError(
                f"Sentence index {sentence_index} out of range."
            )

        sentences[sentence_index - 1] = replacement

        document["document"][field] = " ".join(
            sentences
        )
        return

    if isinstance(value, list):

        counter = 0

        for list_index, item in enumerate(value):

            if not isinstance(item, str):
                continue

            sentences = split_string_sentences(item)

            for local_index in range(len(sentences)):

                counter += 1

                if counter == sentence_index:

                    sentences[local_index] = replacement

                    document["document"][field][
                        list_index
                    ] = " ".join(sentences)

                    return

        raise IndexError(
            f"Sentence index {sentence_index} out of range."
        )

    raise TypeError(
        f"Unsupported field type: {type(value).__name__}"
    )


def main():

    REPAIRED_DOCUMENTS_JSONL.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    sources = load_sources()

    documents = load_jsonl(
        DOCUMENTS_JSONL
    )

    evidence_records = load_jsonl(
        EVIDENCE_JSONL
    )

    nli_records = load_jsonl(
        NLI_JSONL
    )

    nli_map = build_nli_map(
        nli_records
    )

    evidence_map = build_evidence_map(
        evidence_records
    )

    # --------------------------------------------------------
    # Deep-copy the original generated documents.
    # --------------------------------------------------------

    repaired_documents = [
        json.loads(
            json.dumps(document)
        )
        for document in documents
    ]

    document_lookup = {
        (
            document["article_id"],
            document["target_format"],
        ): document
        for document in repaired_documents
    }

    # --------------------------------------------------------
    # CORRECTED TARGET SELECTION
    #
    # Only explicit contradiction.
    # Neutral and coverage-only signals do NOT trigger repair.
    # --------------------------------------------------------

    repair_targets = []

    for key, nli_record in nli_map.items():

        verdict = nli_record[
            "aggregate"
        ]["verdict"]

        if verdict != "contradiction":
            continue

        repair_targets.append(
            {
                "key": key,
                "reason": (
                    "NLI aggregate verdict: contradiction"
                ),
            }
        )

    repair_targets.sort(
        key=lambda item: item["key"]
    )

    print(
        f"Documents loaded: {len(documents)}"
    )

    print(
        f"NLI sentence records: {len(nli_records)}"
    )

    print(
        "Explicit contradiction targets: "
        f"{len(repair_targets)}"
    )

    audit_records = []

    repaired_count = 0
    preserved_count = 0

    # --------------------------------------------------------
    # Repair contradiction targets only.
    # --------------------------------------------------------

    for target_number, target in enumerate(
        repair_targets,
        1,
    ):

        (
            article_id,
            target_format,
            field,
            sentence_index,
        ) = target["key"]

        document = document_lookup.get(
            (
                article_id,
                target_format,
            )
        )

        nli_record = nli_map.get(
            target["key"]
        )

        evidence_record = evidence_map.get(
            target["key"]
        )

        if document is None:
            continue

        if nli_record is None:
            continue

        if evidence_record is None:
            evidence = nli_record["evidence"]
        else:
            evidence = evidence_record["evidence"]

        original_sentence = get_sentence_at(
            document,
            field,
            sentence_index,
        )

        if original_sentence is None:
            audit_records.append(
                {
                    "article_id": article_id,
                    "target_format": target_format,
                    "field": field,
                    "sentence_index": sentence_index,
                    "original_sentence": None,
                    "replacement_sentence": None,
                    "failure_reason": target["reason"],
                    "accepted": False,
                    "repair_attempts": 0,
                    "fallback_used": False,
                    "status": "SKIPPED_INVALID_LOCATION",
                }
            )
            continue

        result = repair_sentence(
            article_id=article_id,
            target_format=target_format,
            field=field,
            original_sentence=original_sentence,
            evidence=evidence,
            source_text=sources[article_id],
        )

        replacement = result[
            "replacement"
        ]

        accepted = result[
            "accepted"
        ]

        if accepted:

            replace_sentence(
                document=document,
                field=field,
                sentence_index=sentence_index,
                replacement=replacement,
            )

            repaired_count += 1
            status = "REPAIRED"

        else:

            # Preserve the original sentence.
            preserved_count += 1
            status = "ORIGINAL_PRESERVED"

        audit_records.append(
            {
                "article_id": article_id,
                "target_format": target_format,
                "field": field,
                "sentence_index": sentence_index,
                "original_sentence": original_sentence,
                "replacement_sentence": (
                    replacement
                    if accepted
                    else original_sentence
                ),
                "failure_reason": target["reason"],
                "evidence_sentence_ids": [
                    item["sentence_id"]
                    for item in evidence
                ],
                "repair_attempts": result[
                    "attempts"
                ],
                "accepted": accepted,
                "fallback_used": (
                    accepted
                    and replacement == FALLBACK
                ),
                "status": status,
                "repair_rejection_reason": (
                    None
                    if accepted
                    else result["reason"]
                ),
            }
        )

        print(
            f"[{target_number}/{len(repair_targets)}] "
            f"{article_id} / {target_format} / "
            f"{field}[{sentence_index}] -> "
            f"{status}"
        )

    # --------------------------------------------------------
    # Save repaired documents.
    # --------------------------------------------------------

    with REPAIRED_DOCUMENTS_JSONL.open(
        "w",
        encoding="utf-8",
    ) as f:

        for document in repaired_documents:

            f.write(
                json.dumps(
                    document,
                    ensure_ascii=False,
                )
                + "\n"
            )

            f.flush()

    # --------------------------------------------------------
    # Save repair audit.
    # --------------------------------------------------------

    with REPAIR_AUDIT_JSONL.open(
        "w",
        encoding="utf-8",
    ) as f:

        for audit in audit_records:

            f.write(
                json.dumps(
                    audit,
                    ensure_ascii=False,
                )
                + "\n"
            )

            f.flush()

    print()
    print(
        "Corrected targeted repair completed."
    )

    print(
        f"Documents preserved: "
        f"{len(repaired_documents)}"
    )

    print(
        f"Explicit contradiction targets: "
        f"{len(repair_targets)}"
    )

    print(
        f"Actual repairs accepted: "
        f"{repaired_count}"
    )

    print(
        f"Original sentences preserved: "
        f"{preserved_count}"
    )

    print(
        "Neutral sentences automatically repaired: 0"
    )

    print(
        "Coverage-only sentences automatically repaired: 0"
    )

    print(
        f"Repaired documents: "
        f"{REPAIRED_DOCUMENTS_JSONL}"
    )

    print(
        f"Repair audit: "
        f"{REPAIR_AUDIT_JSONL}"
    )


if __name__ == "__main__":
    main()
