from pathlib import Path
import ast

ROOT = Path(__file__).resolve().parents[1]
TARGET = ROOT / 'training' / 'generate_transformation_contracts.py'

HELPER = '''
def build_validation_retry_prompt(original_prompt: str, validation_error: str) -> str:
    return (
        'The previous JSON response failed local validation.\n\n'
        'Validation error:\n' + validation_error + '\n\n'
        'Return a corrected JSON object for the SAME task.\n\n'
        'The JSON object MUST contain EXACTLY these top-level fields:\n'
        '- article_id\n'
        '- target_format\n'
        '- required_sections\n'
        '- information_priorities\n'
        '- allowed_transformations\n'
        '- forbidden_transformations\n'
        '- protected_information\n'
        '- validation_priorities\n\n'
        'The validation_priorities object MUST contain EXACTLY these keys:\n'
        '- numbers\n'
        '- dates\n'
        '- named_entities\n'
        '- attributed_claims\n'
        '- general_paraphrases\n\n'
        'Do not add, remove, rename, or nest these keys differently.\n'
        'Keep article_id and target_format exactly as specified in the original task.\n'
        'Return ONLY the corrected JSON object. No Markdown and no explanation.\n\n'
        'Original task:\n' + original_prompt
    ).strip()
'''

TARGET_BLOCK = '''                try:
                    contract = call_ollama(prompt)

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
                        + '\\n'
                    )

                    output.flush()

                    existing_keys.add(key)
                    generated += 1

                    print(
                        f'PASS  {article_id} | {target_format}'
                    )

                except Exception as exc:
                    print(
                        f'FAIL  {article_id} | {target_format}'
                    )
                    print(f'      {exc}')
                    raise
'''

REPLACEMENT_BLOCK = '''                max_attempts = 3
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
                                f'RETRY {article_id} | {target_format} '
                                f'(attempt {attempt}/{max_attempts})'
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
                            + '\\n'
                        )

                        output.flush()

                        existing_keys.add(key)
                        generated += 1

                        print(
                            f'PASS  {article_id} | {target_format}'
                        )
                        break

                    except Exception as exc:
                        last_error = exc

                        if attempt == max_attempts:
                            print(
                                f'FAIL  {article_id} | {target_format}'
                            )
                            print(f'      {exc}')
                            raise
'''

if not TARGET.exists():
    raise FileNotFoundError(TARGET)

text = TARGET.read_text(encoding='utf-8')

if 'def build_validation_retry_prompt(' not in text:
    marker = '\n\n# ============================================================\n# Local validation'
    if text.count(marker) != 1:
        raise RuntimeError('Could not find the local-validation marker exactly once.')
    text = text.replace(marker, '\n\n' + HELPER + marker, 1)
    print('Added validation retry helper.')
else:
    print('Retry helper already exists.')

if REPLACEMENT_BLOCK in text:
    print('Retry loop already installed.')
else:
    if text.count(TARGET_BLOCK) != 1:
        raise RuntimeError('Could not find the current generation try-block exactly once.')
    text = text.replace(TARGET_BLOCK, REPLACEMENT_BLOCK, 1)
    print('Installed 3-attempt validation retry loop.')

ast.parse(text, filename=str(TARGET))
TARGET.write_text(text, encoding='utf-8')
print('Syntax check: PASS')
print('Existing valid contracts are preserved.')
