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
- `MAX_IMAGE_PIXELS` (40M): declared dimensions, checked before anything decodes a full buffer. A
  highly compressible image passes the byte cap and still costs hundreds of megabytes to decode.
- `JOB_DISPATCH_MODE`: `local_pool` (default), `inline`, `manual`, or `none`.
- `MAX_WORKER_CONCURRENCY` (2), `LEASE_SECONDS` (300), `POLL_INTERVAL_MS` (2000),
  `REAPER_INTERVAL_MS` (15000), `MAX_ITEM_ATTEMPTS` (3).
- `WORKER_SINGLE_INSTANCE`: default `true`. Requeues everything in flight at startup, which is correct
  only when no other worker is running. Set it to `false` before starting a second instance.
- `OPERATOR_AUTH_MODE`: `none` (local default) or `static_token` with `OPERATOR_API_TOKEN`. `none` is
  refused outside `local` and `test`.
- `MEDIA_REPOSITORY`: `json` (default) or `sql` -- where the prototype endpoints keep their data.

OCR:

- `OCR_PROVIDER`: `tesseract` (default), `paddle_tesseract`, or `vision_llm`. An unknown name stops
  the service starting and lists the valid ones -- a typo that silently selected a different engine
  would change every extraction with nothing in the output saying so.
- `OCR_LANGUAGES` (`sin+eng`), `OCR_TIMEOUT_SECONDS` (60), `OCR_CONCURRENCY` (2). The last gates how
  many pages may be inside the engine at once, which is a different question from how many items may
  be in flight.
- `OCR_LOW_CONFIDENCE_THRESHOLD` (0.60) and `OCR_EMPTY_TEXT_MIN_CHARS` (8) decide the
  `OCR_LOW_CONFIDENCE` and `OCR_EMPTY_TEXT` warnings a reviewer sees.
- `OCR_PREPROCESS_*`: orientation is always applied; greyscale, autocontrast, denoise, threshold and
  deskew are switches. Only orientation and greyscale are on by default -- see below for why.
- `OCR_TESSERACT_{PSM,OEM,REGION_PSM}` and `OCR_PADDLE_*` tune the two engines.

Extraction:

- `LLM_PROVIDER`: `openai_compatible`, `gemini`, `anthropic`, `fake` (default), or `rule_based`.
  The last two invent advertisements without a model, so the service **refuses to construct them
  outside local and test** -- at startup, not at extraction time.
  `MEDIA_SERVICE_ALLOW_FAKE_PROVIDERS` is the deliberate override.
- `LLM_MAX_ATTEMPTS` (3) and `LLM_MAX_REPAIRS` (1) are **independent budgets**: a rate limit must
  not consume the item's one repair, and a schema failure must not eat the retries it may need.
- `LLM_STRUCTURED_MODE` (`auto`) picks strict `json_schema` or `json_object` by base URL.
- `LLM_MAX_CANDIDATES_PER_IMAGE` (20), `LLM_MAX_OCR_CHARS` (24000), `LLM_TOTAL_DEADLINE_SECONDS`.
- `OPENAI_*`, `GEMINI_*`, `ANTHROPIC_*` per provider. Keys are read as secrets and never printed;
  `/health` says `GEMINI_API_KEY is not set`, never a value.

Optional:

- `MEDIA_SERVICE_MAX_UPLOAD_BYTES`: default `10485760`.
- `MEDIA_SERVICE_METADATA_PATH`: default `.data/media_metadata.json`.
- `MEDIA_SERVICE_CORS_ORIGINS`: comma-separated browser origins allowed to call the API. Defaults
  to local Next.js and Vite dev origins.
- `TESSERACT_CMD`: explicit path to the Tesseract executable. Auto-detected on Windows.
- `TESSERACT_TESSDATA_DIR`: traineddata directory. Defaults to the project-local `.tessdata` when it
  exists, which is what makes Sinhala work -- the system install ships `eng` and not `sin`.
  (`TESSERACT_LANG` is still read as a fallback for `OCR_LANGUAGES`.)

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

### How a page is read

OCR returns **blocks**, not a wall of text: each carries its text, its bounding box, its confidence,
and where the box came from. That last field is why a candidate can cite the regions it was built
from, and why a reviewer can be shown the exact part of the scan an answer was read out of.

Block ids are assigned here, not taken from the engine. Tesseract's own numbering restarts per page,
can be zero, and follows its segmentation rather than the columns a person reads; ids are computed
1-based in column-then-down order instead, and the engine's value is kept in `source_ref` for
tracing. **The persisted ids are authoritative** -- evidence validation compares a candidate's
citations against the stored record, never a recomputed one.

Preprocessing defaults were **measured**, not assumed. Across the ten corpus cases, greyscale and
autocontrast left every block and every confidence figure identical, while denoising dropped mean
confidence from 0.8932 to 0.8713 and thresholding to 0.8662. So orientation and greyscale are on,
the rest are switches, and `preprocess_version` is recorded on every extraction so two results read
under different settings are never silently compared.

`tests/test_ocr_corpus.py` asserts the provider still reproduces every captured fixture block for
block. `python -m tests.fixtures.capture_ocr` regenerates them through the same production code, so
the corpus and the provider cannot drift apart.

### How a page becomes candidates

The extraction stage renders the OCR blocks into a versioned prompt, calls a provider, and validates
what comes back **in two tiers with deliberately different consequences**.

*Tier 1 is structural*: unknown fields, wrong types, out-of-range or non-finite confidence. A failure
there means the model misunderstood the schema, which another turn can fix, so it is **repairable**.

*Tier 2 is semantic*: the candidate cap, category mapping, phone normalisation, and -- the important
one -- whether every cited block actually exists in the recorded OCR result. A failure there is the
model being wrong about the world, which asking again does not fix, so Tier 2 **never retries**. It
warns, strips, or drops one candidate.

Getting that backwards is expensive in a specific way, which is why `category` is typed as a plain
string rather than an enum: as an enum, an unfamiliar trade would be a structural failure and would
burn the item's one paid repair on something policy says to accept with a warning.

A candidate whose evidence is entirely unknown is **discarded**, not warned about. It is the concrete
detector for an advertisement the model invented: a fabrication has nowhere real to point, and a
reviewer shown one has no way to tell it from a real advertisement.

Retry and repair are two independent budgets. Truncation raises the output limit instead of
repairing, because repair cannot lengthen a response that was cut off mid-object. A provider's own
`Retry-After` is honoured over our backoff. One run row is written **before** each call, so a hung
provider is visible in the store rather than invisible.

The repair message carries field paths and error messages and **never field values** -- an
OCR-derived phone number must not be echoed into a second prompt.

### What is still to come

- **Phase 4** rebuilds the management portal around bulk intake, batch progress, and multi-candidate
  review.
- `OCR_PROVIDER=vision_llm` now has the adapters it needs; wiring it is a small follow-on.
