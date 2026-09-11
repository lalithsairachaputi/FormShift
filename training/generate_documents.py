import csv
import json
import os
import re
import urllib.request

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

RAW_CSV = os.path.join(
    BASE_DIR, "data", "raw", "cnn_dailymail_100.csv"
)

CONTRACTS_FILE = os.path.join(
    BASE_DIR, "training_datasets",
    "transformation_contracts_full.jsonl"
)

LEDGERS_FILE = os.path.join(
    BASE_DIR, "training_datasets",
    "claim_ledgers_full.jsonl"
)

OUTPUT_FILE = os.path.join(
    BASE_DIR, "training_datasets",
    "documents_full.jsonl"
)

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL = "qwen2.5:7b-instruct"

TARGET_FORMATS = [
    "press_advisory",
    "technical_brief"
]

REQUIRED_FIELDS = [
    "title",
    "overview",
    "key_findings",
    "technical_details",
    "implications"
]


def read_csv_articles(path):
    articles = {}

    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)

        for row in reader:
            articles[row["id"]] = row["article"]

    return articles


def read_jsonl(path):
    records = []

    if not os.path.exists(path):
        return records

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()

            if line:
                records.append(json.loads(line))

    return records


def load_contracts(path):
    records = read_jsonl(path)

    result = {}

    for record in records:
        key = (
            record["article_id"],
            record["target_format"]
        )

        result[key] = record

    return result


def load_ledgers(path):
    records = read_jsonl(path)

    result = {}

    for record in records:
        result[record["article_id"]] = record

    return result


def clean_model_output(text):
    text = text.strip()

    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)

    return text.strip()


def call_ollama(prompt):
    payload = {
        "model": MODEL,
        "prompt": prompt,
        "stream": False,
        "temperature": 0,
        "format": "json"
    }

    data = json.dumps(payload).encode("utf-8")

    request = urllib.request.Request(
        OLLAMA_URL,
        data=data,
        headers={
            "Content-Type": "application/json"
        },
        method="POST"
    )

    with urllib.request.urlopen(request, timeout=300) as response:
        result = json.loads(response.read().decode("utf-8"))

    return result["response"]


def validate_document(document):
    if not isinstance(document, dict):
        return False, "output is not a JSON object"

    actual_keys = set(document.keys())
    expected_keys = set(REQUIRED_FIELDS)

    if actual_keys != expected_keys:
        return False, (
            f"wrong fields: "
            f"expected={sorted(expected_keys)}, "
            f"got={sorted(actual_keys)}"
        )

    for field in REQUIRED_FIELDS:
        value = document[field]

        if isinstance(value, str):
            if not value.strip():
                return False, f"{field} is empty"

        elif isinstance(value, list):
            if not value:
                return False, f"{field} is empty"

            for item in value:
                if not isinstance(item, str):
                    return False, f"{field} contains non-string item"

                if not item.strip():
                    return False, f"{field} contains empty item"

        else:
            return False, f"{field} has invalid type"

    return True, "valid"


def build_prompt(article_id, article, contract, ledger, target_format):
    if target_format == "Press Advisory":
        format_rules = """
PRESS ADVISORY PRIORITIES:
- Focus on the event, announcement, or development.
- Clearly communicate who, what, when, and where when provided.
- Preserve public-facing facts.
- Preserve attributed statements and quotes when present.
- Explain immediate implications only when supported by the source.
- Do not introduce technical details unless they are relevant to the source event.
"""

    else:
        format_rules = """
TECHNICAL BRIEF PRIORITIES:
- Focus on technical findings and mechanisms.
- Preserve measurements and numerical information.
- Preserve methods and technical details when provided.
- Preserve limitations when provided.
- Explain technical implications only when supported by the source.
- Do not invent technical explanations.
"""

    schema = """
{
  "title": "string",
  "overview": "string",
  "key_findings": ["string"],
  "technical_details": ["string"],
  "implications": ["string"]
}
"""

    return f"""
You are generating a source-closed document.

ARTICLE ID:
{article_id}

TARGET FORMAT:
{target_format}

SOURCE ARTICLE:
{article}

TRANSFORMATION CONTRACT:
{json.dumps(contract, ensure_ascii=False, indent=2)}

CLAIM LEDGER:
{json.dumps(ledger, ensure_ascii=False, indent=2)}

{format_rules}

GLOBAL RULES:
1. Use ONLY information contained in the source article.
2. Do not use outside knowledge.
3. Do not invent facts.
4. Do not invent people, organizations, locations, dates, numbers, measurements, methods, causes, outcomes, or quotations.
5. Do not strengthen or exaggerate claims.
6. Preserve important numerical information exactly.
7. Preserve dates exactly when they are relevant.
8. Preserve named entities accurately.
9. Paraphrasing is allowed.
10. Compression is allowed.
11. Reordering information is allowed.
12. Grouping related facts is allowed.
13. If a requested type of information is not present in the source, write:
   "Not provided in source."
14. Do not add Markdown.
15. Return ONLY valid JSON.
16. Return exactly the five fields shown below.
17. Do not add any other fields.

OUTPUT SCHEMA:
{schema}
"""


def main():
    os.makedirs(
        os.path.dirname(OUTPUT_FILE),
        exist_ok=True
    )

    articles = read_csv_articles(RAW_CSV)
    contracts = load_contracts(CONTRACTS_FILE)
    ledgers = load_ledgers(LEDGERS_FILE)

    pilot_ids = list(ledgers.keys())

    existing = {}

    if os.path.exists(OUTPUT_FILE):
        with open(OUTPUT_FILE, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()

                if not line:
                    continue

                try:
                    record = json.loads(line)

                    key = (
                        record["article_id"],
                        record["target_format"]
                    )

                    existing[key] = record

                except Exception:
                    continue

    expected = len(pilot_ids) * len(TARGET_FORMATS)

    print("=" * 55)
    print("STAGE 5 - DOCUMENT GENERATION")
    print("=" * 55)

    print(f"Input articles : {RAW_CSV}")
    print(f"Contracts      : {CONTRACTS_FILE}")
    print(f"Ledgers        : {LEDGERS_FILE}")
    print(f"Output         : {OUTPUT_FILE}")
    print(f"Model          : {MODEL}")
    print()
    print(f"Full articles  : {len(pilot_ids)}")
    print(f"Target formats : {len(TARGET_FORMATS)}")
    print(f"Expected docs  : {expected}")
    print(f"Existing docs  : {len(existing)}")
    print()

    generated = 0
    skipped = 0

    with open(OUTPUT_FILE, "a", encoding="utf-8") as out:
        for article_id in pilot_ids:

            article = articles.get(article_id)

            if article is None:
                print(f"FAIL {article_id} | article not found")
                continue

            ledger = ledgers[article_id]

            for target_format in TARGET_FORMATS:

                key = (article_id, target_format)

                if key in existing:
                    skipped += 1
                    print(
                        f"SKIP {article_id} | {target_format}"
                    )
                    continue

                contract = contracts.get(key)

                if contract is None:
                    print(
                        f"FAIL {article_id} | {target_format} "
                        f"| contract not found"
                    )
                    continue

                print()
                print(
                    f"GENERATING {article_id} | {target_format}"
                )

                prompt = build_prompt(
                    article_id,
                    article,
                    contract,
                    ledger,
                    target_format
                )

                try:
                    raw = call_ollama(prompt)
                    cleaned = clean_model_output(raw)
                    document = json.loads(cleaned)

                except Exception as e:
                    print(
                        f"FAIL {article_id} | {target_format} "
                        f"| model/JSON error: {e}"
                    )
                    continue

                valid, reason = validate_document(document)

                if not valid:
                    print(
                        f"FAIL {article_id} | {target_format} "
                        f"| {reason}"
                    )
                    continue

                record = {
                    "article_id": article_id,
                    "target_format": target_format,
                    "document": document
                }

                out.write(
                    json.dumps(
                        record,
                        ensure_ascii=False
                    ) + "\n"
                )

                out.flush()

                existing[key] = record
                generated += 1

                print(
                    f"PASS {article_id} | "
                    f"{target_format}"
                )

    total_valid = len(existing)

    print()
    print("=" * 55)
    print("STAGE 5 RESULT")
    print("=" * 55)

    print(f"Expected documents : {expected}")
    print(f"Generated this run : {generated}")
    print(f"Skipped existing   : {skipped}")
    print(f"Total valid docs   : {total_valid}")

    if total_valid == expected:
        print()
        print("STAGE 5 PASS")
    else:
        print()
        print("STAGE 5 FAIL")
        print(
            "Fix failed documents before continuing."
        )


if __name__ == "__main__":
    main()