"""
Stage 6 — deterministic document validation.

Full Stage 12 configuration:
  Input source:
    data/raw/cnn_dailymail_100.csv

  Generated documents:
    training_datasets/documents_full.jsonl

  Validation report:
    training_datasets/document_validation_full.jsonl

No AI models are used in this stage.
Raw data and generated documents are never modified.
"""

import csv
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

SOURCE_CSV = ROOT / "data" / "raw" / "cnn_dailymail_100.csv"
DOCUMENTS_JSONL = ROOT / "training_datasets" / "documents_full.jsonl"
OUTPUT_JSONL = ROOT / "training_datasets" / "document_validation_full.jsonl"

TARGET_FORMATS = {"press_advisory", "technical_brief"}

REQUIRED_DOCUMENT_KEYS = {
    "title",
    "overview",
    "key_findings",
    "technical_details",
    "implications",
}

REQUIRED_TOP_LEVEL_KEYS = {
    "article_id",
    "target_format",
    "document",
}


# ---------------------------------------------------------------------------
# Numeric/date normalization
# ---------------------------------------------------------------------------
#
# The validator is intentionally deterministic. It checks that explicit
# numeric/date information in a generated document is represented in the
# source. Semantic factual checking is deferred to BGE + NLI.
#
# Important:
# - Commas are treated as thousands separators.
# - Decimal values are preserved.
# - Percent signs and K/M/B suffixes are normalized.
# - Numeric ranges/scores such as 2-2 are handled as separate numeric values.
# - Date formatting differences such as "Sept 14, 2006" vs
#   "September 14 2006" are normalized.
#

NUMBER_RE = re.compile(
    r"(?<![\w.])"
    r"(?:"
    r"\d{1,3}(?:,\d{3})+(?:\.\d+)?"
    r"|"
    r"\d+(?:\.\d+)?"
    r")"
    r"(?:%|[KMB])?"
    r"(?!\w)",
    re.IGNORECASE,
)

YEAR_RE = re.compile(r"(?<!\d)(?:18|19|20|21)\d{2}(?!\d)")

DATE_PATTERNS = [
    re.compile(
        r"\b(?:"
        r"Jan(?:uary)?|"
        r"Feb(?:ruary)?|"
        r"Mar(?:ch)?|"
        r"Apr(?:il)?|"
        r"May|"
        r"Jun(?:e)?|"
        r"Jul(?:y)?|"
        r"Aug(?:ust)?|"
        r"Sep(?:t(?:ember)?)?|"
        r"Oct(?:ober)?|"
        r"Nov(?:ember)?|"
        r"Dec(?:ember)?"
        r")"
        r"\s+\d{1,2}"
        r"(?:,\s*|\s+)"
        r"\d{4}\b",
        re.IGNORECASE,
    ),
    re.compile(r"\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b"),
    re.compile(r"\b\d{4}-\d{1,2}-\d{1,2}\b"),
]

MONTHS = {
    "jan": "january",
    "january": "january",
    "feb": "february",
    "february": "february",
    "mar": "march",
    "march": "march",
    "apr": "april",
    "april": "april",
    "may": "may",
    "jun": "june",
    "june": "june",
    "jul": "july",
    "july": "july",
    "aug": "august",
    "august": "august",
    "sep": "september",
    "sept": "september",
    "september": "september",
    "oct": "october",
    "october": "october",
    "nov": "november",
    "november": "november",
    "dec": "december",
    "december": "december",
}


def normalize_number(token: str) -> str:
    """
    Convert equivalent numeric formatting to one canonical representation.

    Examples:
      "2,000" -> "2000"
      "$250,000" -> "250000"
      "70%" -> "70"
      "2.5" -> "2.5"
    """
    token = token.strip().replace(",", "")
    token = re.sub(r"[KMB%]$", "", token, flags=re.IGNORECASE)
    return token


def extract_numbers(text: str) -> set[str]:
    return {
        normalize_number(match.group(0))
        for match in NUMBER_RE.finditer(text)
    }


def normalize_date(value: str) -> str:
    """Normalize equivalent full-date representations to YYYY-MM-DD."""
    value = re.sub(r"\s+", " ", value.strip().lower())
    value = re.sub(r"\s*,\s*", ", ", value)

    match = re.match(
        r"^([a-z]+)\s+(\d{1,2}),?\s+(\d{4})$",
        value,
        re.IGNORECASE,
    )
    if match:
        month, day, year = match.groups()
        month_name = MONTHS.get(month.lower())
        if month_name is None:
            return value
        month_num = {
            "january": 1, "february": 2, "march": 3, "april": 4,
            "may": 5, "june": 6, "july": 7, "august": 8,
            "september": 9, "october": 10, "november": 11, "december": 12,
        }[month_name]
        return f"{year}-{month_num:02d}-{int(day):02d}"

    match = re.match(r"^(\d{1,2})[/-](\d{1,2})[/-](\d{2,4})$", value)
    if match:
        month, day, year = match.groups()
        if len(year) == 2:
            year = "20" + year
        return f"{year}-{int(month):02d}-{int(day):02d}"

    match = re.match(r"^(\d{4})-(\d{1,2})-(\d{1,2})$", value)
    if match:
        year, month, day = match.groups()
        return f"{year}-{int(month):02d}-{int(day):02d}"

    return value


def extract_dates(text: str) -> set[str]:
    found = set()

    for pattern in DATE_PATTERNS:
        for match in pattern.finditer(text):
            found.add(normalize_date(match.group(0)))

    # Years are tracked independently so a year appearing by itself is also
    # checked. Full dates are normalized above, while the year remains useful
    # as an independent source fact.
    found.update(YEAR_RE.findall(text))

    return found


# ---------------------------------------------------------------------------
# Document helpers
# ---------------------------------------------------------------------------

def contains_markdown_wrapper(value: str) -> bool:
    stripped = value.strip()
    return stripped.startswith("```") or stripped.endswith("```")


def iter_document_text(document: dict):
    yield document["title"]
    yield document["overview"]

    for field in (
        "key_findings",
        "technical_details",
        "implications",
    ):
        for item in document[field]:
            yield item


# ---------------------------------------------------------------------------
# Input loading
# ---------------------------------------------------------------------------

def load_source_articles() -> dict[str, str]:
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

    required = {"article", "highlights", "id"}

    if not rows:
        raise ValueError(
            f"{SOURCE_CSV.name} contains no rows"
        )

    if not required.issubset(rows[0].keys()):
        raise ValueError(
            f"{SOURCE_CSV.name} must contain columns "
            f"{sorted(required)}; found {list(rows[0].keys())}"
        )

    return {
        row["id"]: row["article"]
        for row in rows
    }


def load_documents():
    if not DOCUMENTS_JSONL.exists():
        raise FileNotFoundError(
            f"Document file not found: {DOCUMENTS_JSONL}"
        )

    records = []

    with DOCUMENTS_JSONL.open(
        "r",
        encoding="utf-8",
    ) as f:
        for line_number, line in enumerate(f, 1):
            if not line.strip():
                continue

            try:
                records.append(
                    (line_number, json.loads(line))
                )
            except json.JSONDecodeError as exc:
                records.append(
                    (
                        line_number,
                        {
                            "__parse_error__": str(exc),
                            "__raw_line__": line.rstrip("\n"),
                        },
                    )
                )

    return records


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def validate_record(record, sources: dict[str, str]):
    errors = []
    warnings = []

    if "__parse_error__" in record:
        return {
            "structural_status": "FAIL",
            "number_validation": "NOT_CHECKED",
            "date_validation": "NOT_CHECKED",
            "errors": [
                f"Invalid JSON: {record['__parse_error__']}"
            ],
            "warnings": [],
        }

    # Exact top-level schema.
    if set(record.keys()) != REQUIRED_TOP_LEVEL_KEYS:
        errors.append(
            "Top-level keys must be exactly "
            f"{sorted(REQUIRED_TOP_LEVEL_KEYS)}; "
            f"found {sorted(record.keys())}"
        )

    article_id = record.get("article_id")
    target_format = record.get("target_format")
    document = record.get("document")

    # Article ID.
    if not isinstance(article_id, str) or not article_id:
        errors.append(
            "article_id must be a non-empty string"
        )
    elif article_id not in sources:
        errors.append(
            f"article_id not found in raw dataset: {article_id}"
        )

    # Format.
    if target_format not in TARGET_FORMATS:
        errors.append(
            "target_format must be one of "
            f"{sorted(TARGET_FORMATS)}"
        )

    # Document object.
    if not isinstance(document, dict):
        errors.append("document must be an object")
        return {
            "structural_status": "FAIL",
            "number_validation": "NOT_CHECKED",
            "date_validation": "NOT_CHECKED",
            "errors": errors,
            "warnings": warnings,
        }

    # Exact document schema.
    if set(document.keys()) != REQUIRED_DOCUMENT_KEYS:
        errors.append(
            "Document keys must be exactly "
            f"{sorted(REQUIRED_DOCUMENT_KEYS)}; "
            f"found {sorted(document.keys())}"
        )

    # String fields.
    for field in ("title", "overview"):
        value = document.get(field)

        if not isinstance(value, str):
            errors.append(
                f"{field} must be a string"
            )
        elif not value.strip():
            errors.append(
                f"{field} must not be empty"
            )
        elif contains_markdown_wrapper(value):
            errors.append(
                f"{field} contains a markdown fence/wrapper"
            )

    # List fields.
    for field in (
        "key_findings",
        "technical_details",
        "implications",
    ):
        value = document.get(field)

        if not isinstance(value, list):
            errors.append(
                f"{field} must be a list"
            )
            continue

        if not value:
            errors.append(
                f"{field} must not be empty"
            )

        for index, item in enumerate(value):
            if not isinstance(item, str):
                errors.append(
                    f"{field}[{index}] must be a string"
                )
            elif not item.strip():
                errors.append(
                    f"{field}[{index}] must not be empty"
                )
            elif contains_markdown_wrapper(item):
                errors.append(
                    f"{field}[{index}] contains a markdown fence/wrapper"
                )

    number_status = "NOT_CHECKED"
    date_status = "NOT_CHECKED"

    # Deterministic source-closed numeric/date validation.
    if article_id in sources and isinstance(document, dict):
        source_text = sources[article_id]
        generated_text = "\n".join(
            iter_document_text(document)
        )

        source_numbers = extract_numbers(source_text)
        generated_numbers = extract_numbers(generated_text)

        unsupported_numbers = sorted(
            generated_numbers - source_numbers,
            key=lambda value: (
                float(value)
                if re.fullmatch(r"\d+(?:\.\d+)?", value)
                else float("inf"),
                value,
            ),
        )

        source_dates = extract_dates(source_text)
        generated_dates = extract_dates(generated_text)

        unsupported_dates = sorted(
            generated_dates - source_dates
        )

        if unsupported_numbers:
            number_status = "FAIL"
            errors.append(
                "Generated numeric tokens not found in source: "
                + ", ".join(unsupported_numbers)
            )
        else:
            number_status = "PASS"

        if unsupported_dates:
            date_status = "FAIL"
            errors.append(
                "Generated dates/years not found in source: "
                + ", ".join(unsupported_dates)
            )
        else:
            date_status = "PASS"

    return {
        "structural_status": (
            "FAIL" if errors else "PASS"
        ),
        "number_validation": number_status,
        "date_validation": date_status,
        "errors": errors,
        "warnings": warnings,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    OUTPUT_JSONL.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    sources = load_source_articles()
    raw_records = load_documents()

    # Full Stage 12 expects every raw article in both formats.
    expected_ids = list(sources.keys())

    expected_pairs = {
        (article_id, target_format)
        for article_id in expected_ids
        for target_format in sorted(TARGET_FORMATS)
    }

    seen_pairs = set()
    results = []

    for line_number, record in raw_records:
        article_id = (
            record.get("article_id")
            if isinstance(record, dict)
            else None
        )

        target_format = (
            record.get("target_format")
            if isinstance(record, dict)
            else None
        )

        pair = (article_id, target_format)

        result = validate_record(
            record,
            sources,
        )

        # Duplicate pair detection.
        if pair in seen_pairs:
            result["structural_status"] = "FAIL"
            result["errors"].append(
                "Duplicate article_id/target_format pair: "
                f"{pair}"
            )
        elif (
            article_id in expected_ids
            and target_format in TARGET_FORMATS
        ):
            seen_pairs.add(pair)

        result_record = {
            "article_id": article_id,
            "target_format": target_format,
            **result,
        }

        results.append(result_record)

    # Completeness check.
    missing_pairs = sorted(
        expected_pairs - seen_pairs
    )

    if missing_pairs:
        results.append(
            {
                "article_id": "__FULL_SUMMARY__",
                "target_format": "all",
                "structural_status": "FAIL",
                "number_validation": "NOT_CHECKED",
                "date_validation": "NOT_CHECKED",
                "errors": [
                    "Missing expected full pairs: "
                    + ", ".join(
                        f"{article_id}:{target_format}"
                        for article_id, target_format
                        in missing_pairs
                    )
                ],
                "warnings": [],
            }
        )

    # Recreate the canonical validation report deterministically.
    with OUTPUT_JSONL.open(
        "w",
        encoding="utf-8",
    ) as f:
        for result in results:
            f.write(
                json.dumps(
                    result,
                    ensure_ascii=False,
                )
                + "\n"
            )

    record_results = [
        result
        for result in results
        if result["article_id"] != "__FULL_SUMMARY__"
    ]

    passed = sum(
        result["structural_status"] == "PASS"
        for result in record_results
    )

    failed = (
        len(record_results) - passed
    )

    print(
        f"Validated records: {len(record_results)}"
    )
    print(f"PASS: {passed}")
    print(f"FAIL: {failed}")
    print(
        f"Expected full pairs: {len(expected_pairs)}"
    )
    print(
        f"Unique full pairs found: {len(seen_pairs)}"
    )
    print(
        f"Report: {OUTPUT_JSONL}"
    )

    if (
        failed
        or missing_pairs
        or len(record_results) != len(expected_pairs)
    ):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
