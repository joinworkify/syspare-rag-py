PORT ?= 8000
WORKERS ?= 4

dev:
	RAG_BUILD_WORKERS=$(WORKERS) uv run uvicorn rag_server:app --reload --port $(PORT)

serve:
	uv run gunicorn rag_server:app --workers 1 --worker-class uvicorn.workers.UvicornWorker --bind 0.0.0.0:$(PORT)

install:
	uv sync

build-cache:
	curl -s -X POST "http://localhost:$(PORT)/api/build-cache" | python3 -m json.tool

build-cache-operation:
	curl -s -X POST "http://localhost:$(PORT)/api/build-cache?manual_id=YM358_operation" | python3 -m json.tool

build-cache-skip-images:
	curl -s -X POST "http://localhost:$(PORT)/api/build-cache?skip_existing_images=true" | python3 -m json.tool

build-cache-skip-images-operation:
	curl -s -X POST "http://localhost:$(PORT)/api/build-cache?manual_id=YM358_operation&skip_existing_images=true" | python3 -m json.tool

sync-s3:
	curl -s -X POST "http://localhost:$(PORT)/api/sync-to-s3" | python3 -m json.tool

pull-s3:
	curl -s -X POST "http://localhost:$(PORT)/api/pull-from-s3" | python3 -m json.tool

pull-s3-operation:
	curl -s -X POST "http://localhost:$(PORT)/api/pull-from-s3?manual_id=YM358_operation" | python3 -m json.tool

test:
	uv run pytest

.PHONY: dev serve install build-cache build-cache-operation build-cache-skip-images build-cache-skip-images-operation sync-s3 pull-s3 pull-s3-operation test
