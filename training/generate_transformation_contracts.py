import csv
import json
import urllib.request
from pathlib import Path


# ============================================================
# Configuration
# ============================================================
def build_validation_retry_prompt(original_prompt: str, validation_error: str) -> str:
    return (
        "The previous JSON response failed local validation.\n\n"
        "Validation error:\n"
        + validation_error
        + "\n\n"
        "Return a corrected JSON object for the SAME task.\n\n"
        "The JSON object MUST contain EXACTLY these top-level fields:\n"
        "- article_id\n"
        "- target_format\n"
        "- required_sections\n"
        "- information_priorities\n"
        "- allowed_transformations\n"
        "- forbidden_transformations\n"
        "- protected_information\n"
        "- validation_priorities\n\n"
        "The validation_priorities object MUST contain EXACTLY these keys:\n"
        "- numbers\n"
        "- dates\n"
        "- named_entities\n"
        "- attributed_claims\n"
        "- general_paraphrases\n\n"
        "Do not add, remove, rename, or nest these keys differently.\n"
        "Keep article_id and target_format exactly as specified in the original task.\n"
        "Return ONLY the corrected JSON object. No Markdown and no explanation.\n\n"
        "Original task:\n"
        + original_prompt
    ).strip()
PROJECT_ROOT = Path(__file__).resolve().parent.parent

INPUT_FILE = PROJECT_ROOT / "data" / "raw" / "cnn_dailymail_100.csv"
OUTPUT_DIR = PROJECT_ROOT / "training_datasets"
OUTPUT_FILE = OUTPUT_DIR / "transformation_contracts_full.jsonl"

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL = "qwen2.5:1.5b-instruct"

PILOT_ARTICLES = 100

TARGET_FORMATS = [
    "press_advisory",
    "technical_brief",
]

REQUIRED_FIELDS = {
    "article_id",
    "target_format",
    "required_sections",
    "information_priorities",
    "allowed_transformations",
    "forbidden_transformations",
    "protected_information",
    "validation_priorities",
}

REQUIRED_SECTIONS = [
    "title",
    "overview",
    "key_findings",
    "technical_details",
    "implications",
]


# ============================================================
# Transformation Contract schema
# ============================================================

CONTRACT_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": [
        "article_id",
        "target_format",
        "required_sections",
        "information_priorities",
        "allowed_transformations",
        "forbidden_transformations",
        "protected_information",
        "validation_priorities",
    ],
    "properties": {
        "article_id": {
            "type": "string"
        },
        "target_format": {
            "type": "string",
            "enum": TARGET_FORMATS,
        },
        "required_sections": {
            "type": "array",
            "items": {
                "type": "string"
            }
        },
        "information_priorities": {
            "type": "array",
            "items": {
                "type": "string"
            }
        },
        "allowed_transformations": {
            "type": "array",
            "items": {
                "type": "string"
            }
        },
        "forbidden_transformations": {
            "type": "array",
            "items": {
                "type": "string"
            }
        },
        "protected_information": {
            "type": "array",
            "items": {
                "type": "string"
            }
        },
        "validation_priorities": {
            "type": "object",
            "additionalProperties": {
                "type": "string"
            }
        },
    },
}


# ============================================================
# Prompt
# ============================================================

def build_prompt(article_id: str, article: str, target_format: str) -> str:
    format_name = (
        "Press Advisory"
        if target_format == "press_advisory"
        else "Technical Brief"
    )

    if target_format == "press_advisory":
        priorities = """
Prioritize:
- event
- announcement
- who
- what
- when
- where
- important public-facing facts
- relevant attributed statements
- immediate implications
"""
    else:
        priorities = """
Prioritize:
- technical findings
- measurements
- methods
- mechanisms
- numbers
- limitations
- technical implications
"""

    return f"""
You are creating a Transformation Contract for an auditable document
transformation system.

SOURCE ARTICLE ID:
{article_id}

TARGET FORMAT:
{format_name}

The source article is the ONLY source of information.

Your task is to create rules describing how this specific source article
may be transformed into the requested target format.

{priorities}

Allowed transformations:
- paraphrase
- compress
- reorder
- group_related_facts

Forbidden transformations:
- invent_facts
- invent_numbers
- invent_dates
- invent_names
- invent_methods
- invent_limitations

Protected information:
- dates
- numbers
- named_entities
- technical_measurements
- technical_terms

Validation priorities:
- numbers: strict
- dates: strict
- named_entities: strict
- attributed_claims: strict
- general_paraphrases: normal

Required document sections:
- title
- overview
- key_findings
- technical_details
- implications

Important rules:

1. Return ONLY the JSON object.
2. Do not use Markdown.
3. Do not add explanations.
4. article_id MUST be exactly "{article_id}".
5. target_format MUST be exactly "{target_format}".
6. Do not introduce facts from your general knowledge.
7. The contract defines transformation rules; it must not contain
   unsupported facts about the article.

SOURCE ARTICLE:
{article}
""".strip()


# ============================================================
# Ollama call
# ============================================================

def call_ollama(prompt: str) -> dict:
    payload = {
        "model": MODEL,
        "prompt": prompt,
        "stream": False,
        "temperature": 0,
        "format": CONTRACT_SCHEMA,
    }

    request_data = json.dumps(payload).encode("utf-8")

    request = urllib.request.Request(
        OLLAMA_URL,
        data=request_data,
        headers={
            "Content-Type": "application/json"
        },
        method="POST",
    )

    with urllib.request.urlopen(request, timeout=300) as response:
        response_data = response.read().decode("utf-8")

    result = json.loads(response_data)

    if "response" not in result:
        raise RuntimeError("Ollama response did not contain 'response'.")

    return json.loads(result["response"])


# ============================================================
# Local validation
# ============================================================

def validate_contract(
    contract: dict,
    expected_article_id: str,
    expected_format: str,
) -> None:

    if not isinstance(contract, dict):
        raise ValueError("Contract is not a JSON object.")

    actual_fields = set(contract.keys())

    if actual_fields != REQUIRED_FIELDS:
        missing = REQUIRED_FIELDS - actual_fields
        unexpected = actual_fields - REQUIRED_FIELDS

        raise ValueError(
            f"Invalid fields. Missing={sorted(missing)}, "
            f"Unexpected={sorted(unexpected)}"
        )

    if contract["article_id"] != expected_article_id:
        raise ValueError(
            f"Wrong article_id. "
            f"Expected={expected_article_id}, "
            f"Got={contract['article_id']}"
        )

    if contract["target_format"] != expected_format:
        raise ValueError(
            f"Wrong target_format. "
            f"Expected={expected_format}, "
            f"Got={contract['target_format']}"
        )

    if contract["target_format"] not in TARGET_FORMATS:
        raise ValueError("Invalid target format.")

    for field in [
        "required_sections",
        "information_priorities",
        "allowed_transformations",
        "forbidden_transformations",
        "protected_information",
    ]:
        if not isinstance(contract[field], list):
            raise ValueError(f"{field} must be a list.")

        if not contract[field]:
            raise ValueError(f"{field} cannot be empty.")

        if not all(isinstance(item, str) for item in contract[field]):
            raise ValueError(f"{field} must contain only strings.")

    if contract["required_sections"] != REQUIRED_SECTIONS:
        raise ValueError(
            "required_sections does not match the required schema."
        )

    if not isinstance(contract["validation_priorities"], dict):
        raise ValueError("validation_priorities must be an object.")

    required_validation_keys = {
        "numbers",
        "dates",
        "named_entities",
        "attributed_claims",
        "general_paraphrases",
    }

    actual_validation_keys = set(contract["validation_priorities"].keys())

    if actual_validation_keys != required_validation_keys:
        raise ValueError(
            "validation_priorities contains incorrect keys."
        )


# ============================================================
# Input loading
# ============================================================

def load_pilot_articles() -> list[dict]:
    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Input file not found: {INPUT_FILE}"
        )

    with INPUT_FILE.open(
        "r",
        encoding="utf-8-sig",
        newline=""
    ) as file:

        reader = csv.DictReader(file)

        required_columns = {"article", "id"}

        if not required_columns.issubset(reader.fieldnames or []):
            raise ValueError(
                f"train.csv must contain {sorted(required_columns)}. "
                f"Found: {reader.fieldnames}"
            )

        rows = list(reader)

    if len(rows) < PILOT_ARTICLES:
        raise ValueError(
            f"Need at least {PILOT_ARTICLES} articles, "
            f"but only {len(rows)} were found."
        )

    selected = rows[:PILOT_ARTICLES]

    for row in selected:
        if not row["id"]:
            raise ValueError("Pilot article has an empty ID.")

        if not row["article"]:
            raise ValueError(
                f"Article {row['id']} has empty article text."
            )

    return selected


# ============================================================
# Existing output / resume support
# ============================================================

def load_existing_contract_keys() -> set[tuple[str, str]]:
    keys = set()

    if not OUTPUT_FILE.exists():
        return keys

    with OUTPUT_FILE.open(
        "r",
        encoding="utf-8"
    ) as file:

        for line_number, line in enumerate(file, start=1):
            line = line.strip()

            if not line:
                continue

            try:
                contract = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON in output at line {line_number}."
                ) from exc

            key = (
                contract.get("article_id"),
                contract.get("target_format"),
            )

            keys.add(key)

    return keys


# ============================================================
# Main
# ============================================================

def main() -> None:
    print("STAGE 3 - TRANSFORMATION CONTRACT GENERATION")
    print("=" * 55)

    print(f"Input : {INPUT_FILE}")
    print(f"Output: {OUTPUT_FILE}")
    print(f"Model : {MODEL}")
    print()

    articles = load_pilot_articles()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    existing_keys = load_existing_contract_keys()

    print(f"Pilot articles: {len(articles)}")
    print(f"Target formats: {len(TARGET_FORMATS)}")
    print(f"Expected contracts: {len(articles) * len(TARGET_FORMATS)}")
    print(f"Existing contracts: {len(existing_keys)}")
    print()

    generated = 0
    skipped = 0

    with OUTPUT_FILE.open(
        "a",
        encoding="utf-8"
    ) as output:

        for article in articles:
            article_id = article["id"]
            source_text = article["article"]

            for target_format in TARGET_FORMATS:

                key = (article_id, target_format)

                if key in existing_keys:
                    print(
                        f"SKIP  {article_id} | {target_format} "
                        f"(already generated)"
                    )
                    skipped += 1
                    continue

                print(
                    f"GENERATING  {article_id} | {target_format}"
                )

                prompt = build_prompt(
                    article_id,
                    source_text,
                    target_format,
                )

                max_attempts = 3
                last_error = None

                for attempt in range(1, max_attempts + 1):
                    try:
                        request_prompt = (
                            prompt
                            if attempt == 1
                            else build_validation_retry_prompt(
                                prompt,
                                str(last_error),
                            )
                        )

                        if attempt > 1:
                            print(
                                f"RETRY  {article_id} | {target_format} "
                                f"(attempt {attempt}/{max_attempts})"
                            )

                        contract = call_ollama(request_prompt)

                        validate_contract(
                            contract,
                            article_id,
                            target_format,
                        )

                        output.write(
                            json.dumps(
                                contract,
                                ensure_ascii=False
                            )
                            + "\n"
                        )

                        output.flush()

                        existing_keys.add(key)
                        generated += 1

                        print(
                            f"PASS  {article_id} | {target_format}"
                        )
                        break

                    except Exception as exc:
                        last_error = exc

                        if attempt == max_attempts:
                            print(
                                f"FAIL  {article_id} | {target_format}"
                            )
                            print(f"      {exc}")
                            raise

    expected = len(articles) * len(TARGET_FORMATS)
    total = len(existing_keys)

    print()
    print("=" * 55)
    print("STAGE 3 RESULT")
    print("=" * 55)
    print(f"Expected contracts : {expected}")
    print(f"Generated this run : {generated}")
    print(f"Skipped existing   : {skipped}")
    print(f"Total valid records: {total}")

    if total != expected:
        raise RuntimeError(
            f"Expected {expected} contracts but found {total}."
        )

    print()
    print("STAGE 3 PASS")


if __name__ == "__main__":
    main()