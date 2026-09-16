# LankaListings Media Service

FastAPI service for newspaper image ingestion, Sinhala/English OCR, and the review candidates a
moderator approves before anything is published.

Two paths live here side by side. The **batch ingestion pipeline** is the real one: upload many scans,
watch per-image progress, and get zero-to-many independent advertisement candidates per page. The
**single-image endpoints** are the original prototype, kept working unchanged until the management portal
moves off them.

## Responsibilities

- Accept a batch of images, validate the whole set, and store each image once by content hash.
- Run the staged pipeline -- preprocess, OCR, extraction -- in a background worker, resuming from whatever
  survived a crash rather than starting over.
- Produce candidates that are never publicly readable until a person approves them.
- Report per-item progress, and let an operator retry or reprocess what failed.
- Serve the source scans and their derivatives back to operators as evidence.

## Run Locally

PostgreSQL is required -- it is the same engine in development, test, and production.

```bash
docker compose up -d postgres     # from the repository root
cd media-service
python -m venv .venv
.venv\Scripts\activate
pip install -e ".[dev]"
alembic upgrade head
uvicorn media_service.main:app --reload --port 8001
```

Health check:

```bash
curl http://localhost:8001/health
```

### Batch ingestion

Upload a batch. The answer is `202 Accepted` with a `Location` header: the images are durable, and nothing
has been extracted yet.

```bash
curl -X POST http://localhost:8001/api/v1/ingestion-batches ^
  -H "Idempotency-Key: monday-classifieds-1" ^
  -F "images=@page-1.png" -F "images=@page-2.png" -F "images=@page-3.png"
```

Watch it converge. `counts` always carries all seven keys, including the zeroes.

```bash
curl http://localhost:8001/api/v1/ingestion-batches/{batch_id}
```

Retry what failed, or reprocess something that finished:

```bash
curl -X POST http://localhost:8001/api/v1/ingestion-items/{item_id}/retry
curl -X POST "http://localhost:8001/api/v1/ingestion-items/{item_id}/retry?mode=reprocess"
curl -X POST http://localhost:8001/api/v1/ingestion-batches/{batch_id}/retry
```

Fetch the evidence behind a candidate -- the original scan, or the preprocessed image OCR actually read:

```bash
curl http://localhost:8001/api/v1/media/assets/{asset_id}/original
curl http://localhost:8001/api/v1/media/assets/{asset_id}/ocr_input
```

### Single-image endpoints (prototype)

Extract text from an image:

```bash
curl -X POST http://localhost:8001/api/v1/media/extract ^
  -H "X-Correlation-Id: local-demo" ^
  -F "file=@property-flyer.png"
```

Create a draft advertisement from a newspaper article image:

```bash
curl -X POST http://localhost:8001/api/v1/newspaper-articles/extract ^
  -F "image=@newspaper-article.png"
```

List pending OCR drafts for human review:

```bash
curl http://localhost:8001/api/v1/advertisements/review
```

Approve a reviewed advertisement so it appears publicly:

```bash
curl -X POST http://localhost:8001/api/v1/advertisements/{advertisement_id}/approve
```

List approved advertisements for public web and mobile feeds:

```bash
curl http://localhost:8001/api/v1/advertisements
```

Retrieve an extraction result:

```bash
curl http://localhost:8001/api/v1/media/extractions/{extraction_id}
```

Responses use the shared API envelope:

```json
{
  "data": {
    "extraction_id": "uuid",
    "detected_text": "Rs. 45,000,000\nColombo 07",
    "asset": {
      "id": "uuid",
      "content_type": "image/png",
      "byte_size": 12345,
      "checksum": "sha256",
      "width": 1200,
      "height": 800,
      "created_at": "2026-09-04T00:00:00Z"
    },
    "metadata": {
      "engine": "tesseract",
      "model_version": "tesseract-local",
      "language": "eng",
      "confidence": "medium",
      "processing_ms": 120
    }
  },
  "error": null
}
```

## OCR Runtime

Python dependencies install the `pytesseract` wrapper, but the Tesseract binary must also exist on the host.

Install Tesseract separately:

- Windows: install Tesseract OCR. The service auto-detects `C:\Program Files\Tesseract-OCR\tesseract.exe`; if it is installed somewhere else, set `TESSERACT_CMD`.
- macOS: `brew install tesseract`
- Ubuntu/Debian: `sudo apt-get install tesseract-ocr`

### Language data (required for Sinhala)

Trained data files are **not committed** — they are large third-party model artifacts. Download them into
`.tessdata/`, which the service auto-detects:

```bash
mkdir -p .tessdata
curl -L -o .tessdata/eng.traineddata https://github.com/tesseract-ocr/tessdata/raw/main/eng.traineddata
curl -L -o .tessdata/sin.traineddata https://github.com/tesseract-ocr/tessdata/raw/main/sin.traineddata
```

Alternatively point `TESSERACT_TESSDATA_DIR` at an existing tessdata directory.

If `sin` is missing, the service does not fail silently — `GET /health` reports
`ocr_available: false` with `Missing Tesseract language data: sin.`, and extraction requests return a
structured `OCR_UNAVAILABLE` error.

Verify the runtime:

```bash
python -c "import pytesseract; print(pytesseract.get_tesseract_version(), pytesseract.get_languages(config=''))"
# expected: 5.4.0.20240606 ['eng', 'sin']
```

## Configuration

Database, storage, and the worker:

- `DATABASE_URL`: default `postgresql+psycopg://lankalistings:lankalistings@localhost:5434/lankalistings`.
- `TEST_DATABASE_URL`: the suite's database. Never the development one -- it is truncated between tests.
- `MEDIA_STORAGE_ROOT`: where blobs are written. Default `.data/media`.
- `MAX_IMAGES_PER_BATCH` (25), `MAX_IMAGE_BYTES` (10 MiB), `MAX_BATCH_BYTES` (100 MiB). The total is
  binding and is checked first: 25 images at the per-image limit would exceed it.
- `JOB_DISPATCH_MODE`: `local_pool` (default), `inline`, `manual`, or `none`.
- `MAX_WORKER_CONCURRENCY` (2), `LEASE_SECONDS` (300), `POLL_INTERVAL_MS` (2000),
  `REAPER_INTERVAL_MS` (15000), `MAX_ITEM_ATTEMPTS` (3).
- `WORKER_SINGLE_INSTANCE`: default `true`. Requeues everything in flight at startup, which is correct
  only when no other worker is running. Set it to `false` before starting a second instance.
- `OPERATOR_AUTH_MODE`: `none` (local default) or `static_token` with `OPERATOR_API_TOKEN`. `none` is
  refused outside `local` and `test`.
- `MEDIA_REPOSITORY`: `json` (default) or `sql` -- where the prototype endpoints keep their data.

Optional:

- `MEDIA_SERVICE_MAX_UPLOAD_BYTES`: default `10485760`.
- `MEDIA_SERVICE_METADATA_PATH`: default `.data/media_metadata.json`.
- `MEDIA_SERVICE_CORS_ORIGINS`: comma-separated browser origins allowed to call the API. Defaults to local Next.js and Vite dev origins.
- `TESSERACT_CMD`: explicit path to the Tesseract executable.
- `TESSERACT_LANG`: OCR languages. Defaults to `sin+eng`.
- `TESSERACT_TESSDATA_DIR`: explicit traineddata directory. Defaults to the project-local `.tessdata` folder when it exists.

## Operational commands

```bash
python -m media_service.cli import-legacy-json --dry-run [--include-extractions]
python -m media_service.cli storage-gc --dry-run [--min-age-seconds 3600]
```

The import moves the prototype's JSON store into PostgreSQL and is safe to run twice; a dry run executes
against a real transaction and rolls it back, so it reports what would actually happen. The JSON file is
read, never written, never deleted -- rolling back the cutover is `MEDIA_REPOSITORY=json`.

`storage-gc` removes blobs no row references. Files are written before the rows that reference them, so a
crash in between leaves an orphan; that ordering is deliberate, and this is the other half of it.

## Tests

```bash
docker compose up -d postgres     # required
cd media-service
pytest
```

Tests cover the prototype's endpoints, the batch pipeline end to end, both crash-resume points, the queue
under concurrent claims, lease expiry and restart recovery, idempotent uploads, operator auth, the
category catalog, prompt templates, and the regression corpus.

Tests that need the database skip with instructions rather than erroring when PostgreSQL is not running.

No test requires a live OCR or LLM provider. Tests that would are marked `live` and excluded by default; run them with `pytest -m live` once credentials are configured.

## Regression Corpus

`tests/fixtures/corpus/` holds the versioned fixtures the extraction pipeline is measured against. Each case directory contains:

| File | Meaning |
|---|---|
| `image.png` | The source page, rendered deterministically |
| `expected_ocr.json` | What Tesseract **actually** produces, including its mistakes |
| `expected.json` | What extraction **should** recover — the hand-authored ground truth |

Images are rendered rather than photographed so the corpus is reproducible and contains no real personal data. Real newspaper classifieds carry real phone numbers and addresses, which must not be committed.

The ten cases cover Sinhala-only, English-only, mixed-language, multi-column with three independent ads, no-ad editorial content, prompt injection, duplicate phone numbers, an ambiguous `O`-for-zero price, a rotated scan, and a low-resolution clipping.

`sinhala_only` is deliberately the hardest case. Tesseract corrupts its Sinhala conjuncts badly (mean confidence around 0.58) while the price and phone number survive intact — so it tests that extraction reports a damaged title with low confidence instead of inventing a clean one.

Regenerating fixtures (development only, requires Tesseract and Windows fonts):

```bash
python -m tests.fixtures.generate_corpus      # render image.png for every case
python -m tests.fixtures.capture_ocr          # re-capture expected_ocr.json
python -m tests.fixtures.author_expectations  # rewrite expected.json
```

A diff in a fixture always means a case changed on purpose. Re-capturing after an OCR configuration change shows exactly which cases it affected.

## Prompts

Extraction prompts are versioned under `src/media_service/llm/prompts/<family>/<version>/`. Every run records the prompt version *and* a checksum of the template files.

`tests/test_prompts.py` pins that checksum. Editing a prompt without bumping its version fails the suite, because a silent prompt edit invalidates the regression corpus and every accuracy measurement taken against it with nothing in the provenance record showing that anything changed.

## How the pipeline holds together

An item moves `uploaded -> preprocessing -> ocr_processing -> llm_processing -> awaiting_review`, and
`ingestion_items` *is* the queue -- there is no second jobs table to drift out of step with it.

Claims use `SELECT ... FOR UPDATE SKIP LOCKED`, so a second worker takes the next row instead of
contending on the hot one. Every worker write matches on the status it expects to replace and the claim
token it still holds; a write that affects no rows means the item moved on, and the worker stops rather
than overwriting.

Crash recovery is a pure function over what survived. A surviving `ocr_input` derivative skips
preprocessing, a completed OCR row skips Tesseract, and a validated extraction run skips the provider
entirely -- so the crash between a paid call and its candidate rows costs nothing. Dispatch is
at-least-once and artifact uniqueness is enforced per generation, which together give effectively-once
artifacts.

Candidates, the item's status, its candidate count, and the batch projection are written in one commit, so
a worker cannot die between creating candidates and recording that it did.

### What is still to come

- **Phase 2** replaces the OCR stage with a structured provider: blocks, bounding boxes, and per-word
  confidence, switchable by `OCR_PROVIDER`.
- **Phase 3** replaces the rule-based extractor -- which produces at most one candidate per page -- with
  an LLM behind `LLM_PROVIDER`, which is what makes zero-to-many candidates real.
- **Phase 4** rebuilds the management portal around bulk intake, batch progress, and multi-candidate
  review.
