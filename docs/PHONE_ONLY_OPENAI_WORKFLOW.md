# Phone-only workflow: current OpenAI setup

This document captures the current phone-only operating mode for this repo and the main project changes that support it.

## Current way we run phone extraction

The current primary workflow is:
- use the workbook wrapper `scripts/run_phone_extract_workbook.py`
- run the phone-only pipeline, not the full sales pipeline
- use OpenAI as the phone LLM provider
- write the results back into a new Excel workbook, preserving original sheets

Recommended command:

```bash
python scripts/run_phone_extract_workbook.py --input-xlsx "data\Assemblean Manuav DACH Data enrichment.xlsx" --run-phone-extract --workers 50 --suffix assemblean_dach_openai54mini_phone50_fresh --write-back-workbook
```

What this command does:
- reads every sheet from the source workbook
- combines all rows into one prepared CSV with provenance columns
- launches `phone_extract.py` as the phone-only engine
- runs in parallel with 50 workers
- writes a new workbook with phone results merged back into the original sheet layout

This is one of the main supported ways to run phone extraction only, and it is the main bulk-run method currently in use.

## Current `.env` setup for phone-only runs

Recommended settings:

```dotenv
PHONE_LLM_PROVIDER="openai"
OPENAI_API_KEY="..."
OPENAI_MODEL_NAME="gpt-5.4-mini-2026-03-17"
OPENAI_SERVICE_TIER="flex"
```

Relevant notes:
- `PHONE_LLM_PROVIDER` only switches the phone-only LLM path. It does not change the full pipeline's Gemini configuration.
- `OPENAI_SERVICE_TIER="flex"` is the current preferred mode for the phone-only workflow.
- prompt caching remains supported for OpenAI, but the client now uses a short digest-based `prompt_cache_key` so longer model names do not break Responses API requests.

## When to use which entry point

Use `scripts/run_phone_extract_workbook.py` when:
- the source input is an `.xlsx` workbook
- the workbook has multiple sheets
- you want rows written back into their original sheets
- you want a single operator-friendly command for bulk phone extraction

Use `phone_extract.py` directly when:
- the input is already a single CSV
- workbook write-back is not needed
- you are running a smaller local test or a narrow row slice

Example direct CSV run:

```bash
python phone_extract.py -i "data\your_input.csv" --workers 10 -s phone_only_test
```

## Output expectations for the current workflow

The workbook-wrapper run produces:
- a prepared combined CSV under `data/prepared/`
- a standard phone-only run folder under `output_data/<run_id>/`
- a new augmented workbook with phone output columns written back into each original sheet

Typical outputs:
- `data/prepared/<workbook_stem>__all_sheets.csv`
- `data/prepared/<workbook_stem>__phone_augmented.xlsx`
- `output_data/<run_id>/phone_extraction_results_<run_id>_merged.csv`
- `output_data/<run_id>/phone_extract_<run_id>_live_status.json`

Common output columns used operationally:
- `Top_Number_1`
- `Top_Number_2`
- `Top_Number_3`
- `MainOffice_Number`
- `PhoneNumber_Found`
- `PhoneType_Found`
- `PhoneSources_Found`
- `PhoneExtract_Outcome`

## Main project changes made for this workflow

### 1. Added OpenAI as a phone-only LLM backend

The phone-only flow can now run on either:
- Gemini
- OpenAI Responses API

Key changes:
- added OpenAI configuration in `.env`, `env.example`, and `src/core/config.py`
- added `PHONE_LLM_PROVIDER` to choose the phone-only backend
- added `src/phone_retrieval/llm_clients/openai_client.py`
- updated phone extraction code to choose the correct client at runtime

Why this was added:
- Gemini throttling caused repeated `429 Resource exhausted` failures on larger phone-only runs
- OpenAI gave a practical alternative for bulk execution

### 2. Split Gemini and OpenAI ranking prompts

The phone ranking prompt is no longer a single shared file.

Current prompt split:
- `prompts/phone_ranking_prompt_gemini.txt`
- `prompts/phone_ranking_prompt_openai.txt`

Why this was added:
- Gemini and OpenAI needed to stay independently tunable
- OpenAI ranking was tuned more aggressively for the actual outreach use case

### 3. Tightened OpenAI ranking rules for DACH outreach

The OpenAI ranking prompt was strengthened so that top-ranked numbers better match the intended call strategy.

Current OpenAI ranking behavior emphasizes:
- DACH numbers over non-DACH numbers
- entity-specific numbers over parent-organization switchboards
- explicit sales / business development / founder / CEO / managing director lines over assistants, admin, support, or order-processing lines
- leaving `ranked_numbers` shorter than 3 when the remaining options are poor outreach targets

Important enforced rule:
- if any DE / AT / CH number exists, non-DACH numbers should not appear in `ranked_numbers`

### 4. Added workbook orchestration and write-back

The workbook wrapper `scripts/run_phone_extract_workbook.py` was extended so that it can:
- read all sheets from a workbook
- preserve provenance for each row
- run phone extraction once on the combined dataset
- write the results back into a new workbook with the original sheet structure preserved

Why this matters:
- this is the operationally easiest way to process large Excel inputs
- it avoids manual post-run row stitching

### 5. Refactored chunk processing for multi-provider support

The phone extraction chunk processor was updated to work with both:
- raw-text Gemini style responses
- structured OpenAI Responses API outputs

Related improvements:
- provider-aware LLM calling in `llm_chunk_processor.py`
- schema adaptation for OpenAI strict JSON output
- normalized phone-number matching so format differences such as spaces or hyphens do not break candidate reconciliation

### 6. Fixed OpenAI structured-output compatibility issues

The project was updated to handle OpenAI structured outputs correctly.

This included:
- adapting schemas so required fields are explicit enough for OpenAI validation
- handling optional fields in a way the Responses API accepts

Why this matters:
- without this, OpenAI rejected some requests with invalid schema errors

### 7. Fixed the `prompt_cache_key` length problem for newer models

When the phone-only workflow moved to `gpt-5.4-mini-2026-03-17`, the previous cache key format could exceed OpenAI's 64-character limit.

The fix:
- generate a shorter digest-based cache key
- if OpenAI still rejects prompt caching, retry without prompt caching instead of failing the whole request

Why this matters:
- it keeps newer model names usable without breaking the phone-only workflow

### 8. Tested the OpenAI path under higher concurrency

The OpenAI phone-only integration was tested with:
- small smoke tests
- stricter prompt validation samples
- higher-concurrency stress tests, including 50-worker attempts

What this established:
- the OpenAI path is functionally viable for the phone-only workflow
- the machine can still be constrained by Playwright and memory pressure, but the LLM integration and ranking logic are working

## Current operational guidance

For large phone-only jobs:
- prefer the workbook wrapper command above
- keep `PHONE_LLM_PROVIDER="openai"`
- keep the OpenAI-specific ranking prompt in place
- use the workbook write-back mode so results return directly to the source sheet structure

For legacy compatibility:
- Gemini remains available
- the Gemini ranking prompt is preserved separately
- direct `phone_extract.py` CSV runs remain supported

## Related files

- `scripts/run_phone_extract_workbook.py`
- `phone_extract.py`
- `src/core/config.py`
- `src/phone_retrieval/extractors/llm_extractor.py`
- `src/phone_retrieval/extractors/llm_chunk_processor.py`
- `src/phone_retrieval/llm_clients/openai_client.py`
- `src/phone_retrieval/llm_clients/gemini_client.py`
- `src/phone_retrieval/utils/llm_processing_helpers.py`
- `prompts/phone_ranking_prompt_openai.txt`
- `prompts/phone_ranking_prompt_gemini.txt`
