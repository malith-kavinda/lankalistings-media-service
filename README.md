# LankaListings Media Service

FastAPI service for image upload validation and OCR/text extraction.

This is the first Python-native backend slice for the media-service described in the discovery pack. It does not persist to PostgreSQL yet; it stores extraction metadata in a local JSON file so the service boundary and response contract can be exercised before the storage layer is added.

## Responsibilities

- Validate uploaded images.
- Capture basic asset metadata: generated asset id, content type, byte size, checksum, width, and height.
- Run local OCR with Tesseract when available.
- Return a structured `OCR_UNAVAILABLE` error when OCR runtime dependencies are not installed or configured.
- Retrieve previous extraction results by extraction id.

## Run Locally

```bash
cd media-service
python -m venv .venv
.venv\Scripts\activate
pip install -e ".[dev]"
uvicorn media_service.main:app --reload --port 8001
```

Health check:

```bash
curl http://localhost:8001/health
```

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

Optional environment variables:

- `MEDIA_SERVICE_MAX_UPLOAD_BYTES`: default `10485760`.
- `MEDIA_SERVICE_METADATA_PATH`: default `.data/media_metadata.json`.
- `MEDIA_SERVICE_CORS_ORIGINS`: comma-separated browser origins allowed to call the API. Defaults to local Next.js and Vite dev origins.
- `TESSERACT_CMD`: explicit path to the Tesseract executable.
- `TESSERACT_LANG`: OCR languages. Defaults to `sin+eng`.
- `TESSERACT_TESSDATA_DIR`: explicit traineddata directory. Defaults to the project-local `.tessdata` folder when it exists.

## Tests

```bash
cd media-service
pytest
```

Tests cover health, image validation, OCR unavailable behavior, mocked successful extraction, retrieval, not found behavior, and JSON-backed local persistence.

## Persistence Roadmap

The JSON metadata store should migrate to persistent tables when the backend data layer is introduced:

- `media_assets`: `id`, `owner_account_id`, `storage_key`, `content_type`, `byte_size`, `width`, `height`, `checksum`, `created_at`.
- `media_derivatives`: derivative rows for thumbnail/card/gallery/main variants.
- `extraction_jobs`: `source_asset_id`, `status`, `raw_ocr_text`, `model_version`, `created_at`.
- `extracted_fields`: `job_id`, `field_key`, `extracted_value`, `confidence`, `warning_note`, `accepted_value`, `accepted_by`.
