# s3_storage.py — Sync RAG cache and PDFs with AWS S3.
# Uses env: AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, AWS_DEFAULT_REGION,
#           S3_BUCKET_NAME, S3_RAG_PREFIX (optional, default "rag-data")

import os
from pathlib import Path
from typing import Optional

# S3 layout:
#   {S3_RAG_PREFIX}/pdfs/     -> local PDF_FOLDER (uploaded PDFs)
#   {S3_RAG_PREFIX}/cache/    -> local CACHE_DIR (metadata + images)


def _env(key: str, default: str = "") -> str:
    return (os.environ.get(key) or default).strip()


def is_s3_configured() -> bool:
    bucket = _env("S3_BUCKET_NAME")
    key = _env("AWS_ACCESS_KEY_ID")
    secret = _env("AWS_SECRET_ACCESS_KEY")
    return bool(bucket and key and secret)


def get_s3_prefix() -> str:
    return _env("S3_RAG_PREFIX", "rag-data").rstrip("/")


def get_bucket_name() -> str:
    return _env("S3_BUCKET_NAME")


def _get_client():
    import boto3
    return boto3.client(
        "s3",
        region_name=_env("AWS_DEFAULT_REGION") or "us-east-2",
        aws_access_key_id=_env("AWS_ACCESS_KEY_ID"),
        aws_secret_access_key=_env("AWS_SECRET_ACCESS_KEY"),
    )


def sync_s3_to_local(bucket: str, s3_prefix: str, local_dir: str) -> int:
    """Download all objects under s3_prefix into local_dir. Returns number of files downloaded."""
    client = _get_client()
    local_path = Path(local_dir)
    local_path.mkdir(parents=True, exist_ok=True)
    prefix = f"{s3_prefix}/" if not s3_prefix.endswith("/") else s3_prefix
    count = 0
    paginator = client.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents") or []:
            key = obj["Key"]
            if key.endswith("/"):
                continue
            rel = key[len(prefix) :].lstrip("/")
            if not rel:
                continue
            dest = local_path / rel
            dest.parent.mkdir(parents=True, exist_ok=True)
            client.download_file(bucket, key, str(dest))
            count += 1
    return count


def sync_local_to_s3(local_dir: str, bucket: str, s3_prefix: str) -> int:
    """Upload local_dir tree to s3_prefix. Returns number of files uploaded."""
    client = _get_client()
    local_path = Path(local_dir)
    if not local_path.exists():
        return 0
    prefix = f"{s3_prefix}/" if not s3_prefix.endswith("/") else s3_prefix
    count = 0
    for f in local_path.rglob("*"):
        if f.is_file():
            rel = f.relative_to(local_path)
            key = prefix + str(rel).replace("\\", "/")
            client.upload_file(str(f), bucket, key)
            count += 1
    return count


def download_pdfs_from_s3(pdf_folder: str) -> int:
    """Download S3 rag-data/pdfs/ into pdf_folder. Returns file count."""
    if not is_s3_configured():
        return 0
    bucket = get_bucket_name()
    prefix = f"{get_s3_prefix()}/pdfs"
    return sync_s3_to_local(bucket, prefix, pdf_folder)


def upload_pdfs_to_s3(pdf_folder: str) -> int:
    """Upload pdf_folder to S3 rag-data/pdfs/. Returns file count."""
    if not is_s3_configured():
        return 0
    bucket = get_bucket_name()
    prefix = f"{get_s3_prefix()}/pdfs"
    return sync_local_to_s3(pdf_folder, bucket, prefix)


def download_cache_from_s3(cache_dir: str) -> int:
    """Download S3 rag-data/cache/ into cache_dir. Returns file count."""
    if not is_s3_configured():
        return 0
    bucket = get_bucket_name()
    prefix = f"{get_s3_prefix()}/cache"
    return sync_s3_to_local(bucket, prefix, cache_dir)


def upload_cache_to_s3(cache_dir: str) -> int:
    """Upload cache_dir to S3 rag-data/cache/. Returns file count."""
    if not is_s3_configured():
        return 0
    bucket = get_bucket_name()
    prefix = f"{get_s3_prefix()}/cache"
    return sync_local_to_s3(cache_dir, bucket, prefix)


def upload_pdf_file_to_s3(local_path: str, object_name: Optional[str] = None) -> bool:
    """Upload a single PDF file to S3 rag-data/pdfs/. object_name = filename if not set. Returns True on success."""
    if not is_s3_configured():
        return False
    path = Path(local_path)
    if not path.is_file():
        return False
    name = object_name or path.name
    key = f"{get_s3_prefix()}/pdfs/{name}"
    client = _get_client()
    client.upload_file(str(path), get_bucket_name(), key)
    return True
