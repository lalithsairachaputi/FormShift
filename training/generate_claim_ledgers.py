import csv
import json
import re
import urllib.request
from pathlib import Path


# ============================================================
# Configuration
# ============================================================
def build_ledger_retry_prompt(original_prompt: str, validation_error: str) -> str:
    return (
        "The previous JSON response failed local validation.\n\n"
        "Validation error:\n"
        + validation_error
        + "\n\n"
        "Return a corrected JSON object for the SAME article and SAME task.\n\n"
        "The JSON object MUST contain EXACTLY these top-level fields:\n"
        "- article_id\n"
        "- claims\n\n"
        "The claims field MUST be a JSON array.\n"
        "EVERY claim object MUST contain EXACTLY these fields:\n"
        "- id\n"
        "- text\n"
        "- type\n"
        "- importance\n"
        "- source_sentence_id\n\n"
        "Allowed claim types are:\n"
        "- event\n"
        "- fact\n"
        "- entity\n"
        "- date\n"
        "- number\n"
        "- quote\n"
        "- relationship\n\n"
        "importance MUST be a numeric value between 0 and 1.\n"
        "source_sentence_id MUST refer to a valid source sentence from the article.\n"
        "Do not add, remove, rename, or omit required fields.\n"
        "Do not invent claims.\n"
        "Keep article_id exactly as specified.\n"
        "Return ONLY the corrected JSON object. No Markdown and no explanation.\n\n"
        "Original task:\n"
        + original_prompt
    ).strip()
PROJECT_ROOT = Path(__file__).resolve().parent.parent

INPUT_FILE = PROJECT_ROOT / "data" / "raw" / "cnn_dailymail_100.csv"
OUTPUT_DIR = PROJECT_ROOT / "training_datasets"
OUTPUT_FILE = OUTPUT_DIR / "claim_ledgers_full.jsonl"

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL = "qwen2.5:7b-instruct"

PILOT_ARTICLES = 100

ALLOWED_CLAIM_TYPES = {
    "event",
    "fact",
    "entity",
    "date",
    "number",
    "quote",
    "relationship",
}


# ============================================================
# Sentence splitting
# ============================================================

def split_into_sentences(text: str) -> list[str]:
    """
    Simple deterministic sentence segmentation.

    This is intentionally lightweight. We only need stable sentence
    identifiers for provenance at this stage.
    """

    text = re.sub(r"\s+", " ", text).strip()

    if not text:
        return []

    sentences = re.split(r"(?<=[.!?])\s+", text)

    return [
        sentence.strip()
        for sentence in sentences
        if sentence.strip()
    ]


def build_sentence_map(article: str) -> list[dict]:
    sentences = split_into_sentences(article)

    return [
        {
            "id": f"S{i:03d}",
            "text": sentence,
        }
        for i, sentence in enumerate(sentences, start=1)
    ]


# ============================================================
# Prompt
# ============================================================

def build_prompt(
    article_id: str,
    sentence_map: list[dict],
) -> str:

    source_sentences = "\n".join(
        f"{item['id']}: {item['text']}"
        for item in sentence_map
    )

    return f"""
You are extracting a Claim Ledger from a source article.

ARTICLE ID:
{article_id}

Your job is to identify important factual claims explicitly supported
by the source sentences below.

IMPORTANT RULES:

1. Use ONLY information contained in the source sentences.
2. Do not use outside knowledge.
3. Do not invent facts.
4. Do not invent names.
5. Do not invent numbers.
6. Do not invent dates.
7. Do not invent relationships.
8. Do not invent quotations.
9. Every claim MUST be traceable to one source sentence.
10. source_sentence_id MUST be one of the sentence IDs provided below.
11. Claim IDs must be sequential: C001, C002, C003, ...
12. importance must be a number between 0 and 1.
13. Extract important claims rather than every trivial sentence.
14. Preserve the meaning of the source.
15. A claim may paraphrase its source sentence, but it must not add
    information that is absent from that sentence.

Allowed claim types:

- event
- fact
- entity
- date
- number
- quote
- relationship

Return ONLY a JSON object in this exact structure:

{{
  "article_id": "{article_id}",
  "claims": [
    {{
      "id": "C001",
      "text": "...",
      "type": "fact",
      "importance": 0.90,
      "source_sentence_id": "S001"
    }}
  ]
}}

SOURCE SENTENCES:

{source_sentences}
""".strip()


# ============================================================
# Ollama
# ============================================================

def call_ollama(prompt: str) -> dict:

    payload = {
        "model": MODEL,
        "prompt": prompt,
        "stream": False,
        "temperature": 0,
        "format": "json",
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

    with urllib.request.urlopen(
        request,
        timeout=600
    ) as response:

        response_data = response.read().decode("utf-8")

    result = json.loads(response_data)

    if "response" not in result:
        raise RuntimeError(
            "Ollama response did not contain 'response'."
        )

    return json.loads(result["response"])


# ============================================================
# Validation
# ============================================================

def validate_ledger(
    ledger: dict,
    article_id: str,
    sentence_map: list[dict],
) -> None:

    if not isinstance(ledger, dict):
        raise ValueError("Ledger is not a JSON object.")

    if set(ledger.keys()) != {"article_id", "claims"}:
        raise ValueError(
            "Ledger must contain exactly article_id and claims."
        )

    if ledger["article_id"] != article_id:
        raise ValueError(
            f"Wrong article_id. "
            f"Expected {article_id}, got {ledger['article_id']}"
        )

    if not isinstance(ledger["claims"], list):
        raise ValueError("claims must be a list.")

    source_ids = {
        sentence["id"]
        for sentence in sentence_map
    }

    previous_id = 0

    for claim in ledger["claims"]:

        if not isinstance(claim, dict):
            raise ValueError("Each claim must be an object.")

        required_fields = {
            "id",
            "text",
            "type",
            "importance",
            "source_sentence_id",
        }

        if set(claim.keys()) != required_fields:
            raise ValueError(
                f"Invalid claim fields for {claim.get('id')}. "
                f"Expected {sorted(required_fields)}, "
                f"got {sorted(claim.keys())}"
            )

        claim_id = claim["id"]

        match = re.fullmatch(r"C(\d{3,})", claim_id)

        if not match:
            raise ValueError(
                f"Invalid claim ID: {claim_id}"
            )

        numeric_id = int(match.group(1))

        if numeric_id != previous_id + 1:
            raise ValueError(
                f"Claim IDs are not sequential at {claim_id}."
            )

        previous_id = numeric_id

        if not isinstance(claim["text"], str):
            raise ValueError(
                f"{claim_id}: text must be a string."
            )

        if not claim["text"].strip():
            raise ValueError(
                f"{claim_id}: text is empty."
            )

        if claim["type"] not in ALLOWED_CLAIM_TYPES:
            raise ValueError(
                f"{claim_id}: invalid claim type "
                f"{claim['type']}"
            )

        if not isinstance(
            claim["importance"],
            (int, float)
        ):
            raise ValueError(
                f"{claim_id}: importance must be numeric."
            )

        if not 0 <= claim["importance"] <= 1:
            raise ValueError(
                f"{claim_id}: importance must be between 0 and 1."
            )

        if claim["source_sentence_id"] not in source_ids:
            raise ValueError(
                f"{claim_id}: source sentence "
                f"{claim['source_sentence_id']} "
                f"does not exist."
            )


# ============================================================
# Input
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

        if not reader.fieldnames:
            raise ValueError("train.csv has no header.")

        required_columns = {"article", "id"}

        if not required_columns.issubset(
            reader.fieldnames
        ):
            raise ValueError(
                f"Missing required columns. "
                f"Found: {reader.fieldnames}"
            )

        rows = list(reader)

    if len(rows) < PILOT_ARTICLES:
        raise ValueError(
            f"Need {PILOT_ARTICLES} articles."
        )

    rows = rows[:PILOT_ARTICLES]

    for row in rows:

        if not row["id"]:
            raise ValueError(
                "Article has an empty ID."
            )

        if not row["article"]:
            raise ValueError(
                f"Article {row['id']} has empty text."
            )

    return rows


# ============================================================
# Resume support
# ============================================================

def load_existing_ids() -> set[str]:

    existing = set()

    if not OUTPUT_FILE.exists():
        return existing

    with OUTPUT_FILE.open(
        "r",
        encoding="utf-8"
    ) as file:

        for line_number, line in enumerate(
            file,
            start=1
        ):

            line = line.strip()

            if not line:
                continue

            try:
                ledger = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON at output line "
                    f"{line_number}."
                ) from exc

            if "article_id" in ledger:
                existing.add(
                    ledger["article_id"]
                )

    return existing


# ============================================================
# Main
# ============================================================

def main() -> None:

    print("STAGE 4 - CLAIM LEDGER EXTRACTION")
    print("=" * 55)

    print(f"Input : {INPUT_FILE}")
    print(f"Output: {OUTPUT_FILE}")
    print(f"Model : {MODEL}")
    print()

    articles = load_pilot_articles()

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    existing_ids = load_existing_ids()

    print(f"Pilot articles: {len(articles)}")
    print(f"Expected ledgers: {len(articles)}")
    print(f"Existing ledgers: {len(existing_ids)}")
    print()

    generated = 0
    skipped = 0

    with OUTPUT_FILE.open(
        "a",
        encoding="utf-8"
    ) as output:

        for article in articles:

            article_id = article["id"]

            if article_id in existing_ids:

                print(
                    f"SKIP  {article_id} "
                    f"(already generated)"
                )

                skipped += 1
                continue

            print(
                f"GENERATING  {article_id}"
            )

            sentence_map = build_sentence_map(
                article["article"]
            )

            if not sentence_map:
                raise ValueError(
                    f"No sentences found for "
                    f"{article_id}"
                )

            print(
                f"  Source sentences: "
                f"{len(sentence_map)}"
            )

            prompt = build_prompt(
                article_id,
                sentence_map
            )

            max_attempts = 3
            last_error = None

            for attempt in range(1, max_attempts + 1):

                try:

                    request_prompt = (
                        prompt
                        if attempt == 1
                        else build_ledger_retry_prompt(
                            prompt,
                            str(last_error),
                        )
                    )

                    if attempt > 1:
                        print(
                            f"RETRY  {article_id} "
                            f"(attempt {attempt}/{max_attempts})"
                        )

                    ledger = call_ollama(
                        request_prompt
                    )

                    validate_ledger(
                        ledger,
                        article_id,
                        sentence_map,
                    )

                    output.write(
                        json.dumps(
                            ledger,
                            ensure_ascii=False
                        )
                        + "\n"
                    )

                    output.flush()

                    existing_ids.add(article_id)
                    generated += 1

                    print(
                        f"PASS  {article_id} | "
                        f"claims={len(ledger['claims'])}"
                    )

                    break

                except Exception as exc:

                    last_error = exc

                    if attempt == max_attempts:

                        print(
                            f"FAIL  {article_id}"
                        )

                        print(
                            f"      {exc}"
                        )

                        raise

    print()
    print("=" * 55)
    print("STAGE 4 RESULT")
    print("=" * 55)

    print(
        f"Expected ledgers : {len(articles)}"
    )

    print(
        f"Generated this run : {generated}"
    )

    print(
        f"Skipped existing : {skipped}"
    )

    print(
        f"Total valid ledgers: {len(existing_ids)}"
    )

    if len(existing_ids) != len(articles):

        raise RuntimeError(
            f"Expected {len(articles)} valid ledgers, "
            f"found {len(existing_ids)}."
        )

    print()
    print("STATUS: PASS")


if __name__ == "__main__":
    main()