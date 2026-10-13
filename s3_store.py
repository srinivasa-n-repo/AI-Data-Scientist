"""
S3 wiring for the AgentCore Data Scientist
==========================================
AgentCore Runtime only exchanges a JSON payload with the caller — there are no
upload/download HTTP endpoints like the old FastAPI app had. So datasets come
IN from S3 and every generated artifact (plots, report.md, pipeline.ipynb,
parquet checkpoints, project memory) goes OUT to S3, handed back to the caller
as short-lived presigned GET URLs.

Only boto3 is imported here (light); it is safe to import at module load time.
The execution role attached to the AgentCore Runtime needs s3:GetObject on the
dataset location and s3:PutObject on the artifact bucket/prefix.
"""

import os
from urllib.parse import urlparse

import boto3
from botocore.config import Config

_REGION = os.environ.get("AWS_REGION", "ap-south-1")

# Workspace files that are internal plumbing, not user-facing artifacts.
_SKIP_PREFIXES = ("_run_",)          # scratch scripts execute_python writes
_SKIP_NAMES = {"_notebook_log.jsonl"}  # raw log; the .ipynb export is uploaded


def _client():
    # Pin the REGIONAL endpoint + virtual addressing so presigned URLs look like
    # https://<bucket>.s3.<region>.amazonaws.com/... . The global endpoint
    # (s3.amazonaws.com) answers a regional bucket with a 307 redirect whose
    # response carries no CORS headers, which breaks browser PUT/GET even when
    # the bucket's CORS is configured. s3v4 signing is required for that host.
    return boto3.client(
        "s3",
        region_name=_REGION,
        endpoint_url=f"https://s3.{_REGION}.amazonaws.com",
        config=Config(signature_version="s3v4",
                      s3={"addressing_style": "virtual"}),
    )


def parse_s3_uri(uri: str) -> tuple[str, str]:
    """'s3://bucket/some/key.csv' -> ('bucket', 'some/key.csv')."""
    parts = urlparse(uri)
    if parts.scheme != "s3" or not parts.netloc:
        raise ValueError(f"not an s3:// uri: {uri!r}")
    return parts.netloc, parts.path.lstrip("/")


def download(uri: str, dest_dir: str) -> str:
    """Download an s3:// object into dest_dir. Returns the local filename."""
    bucket, key = parse_s3_uri(uri)
    name = os.path.basename(key) or "dataset"
    _client().download_file(bucket, key, os.path.join(dest_dir, name))
    return name


def presign_put(bucket: str, key: str, expires: int = 3600) -> str:
    """Presigned PUT URL so the browser can upload a dataset straight to S3
    without holding any AWS credentials (only its Cognito token).

    ContentType is intentionally NOT bound into the signature so the browser
    can PUT the raw bytes without having to match a Content-Type header.
    The bucket must allow PUT via CORS from the site origin.
    """
    return _client().generate_presigned_url(
        "put_object", Params={"Bucket": bucket, "Key": key}, ExpiresIn=expires)


def upload_dir_presigned(local_dir: str, bucket: str, prefix: str,
                         expires: int = 3600) -> dict[str, dict[str, str]]:
    """Upload every real artifact in local_dir under prefix/.

    Returns {filename: {"s3": "s3://…", "url": "<presigned GET>"}} so the
    caller (e.g. a browser UI) can render plots/reports directly.
    """
    client = _client()
    out: dict[str, dict[str, str]] = {}
    prefix = prefix.rstrip("/")
    for fn in sorted(os.listdir(local_dir)):
        path = os.path.join(local_dir, fn)
        if not os.path.isfile(path):
            continue
        if fn in _SKIP_NAMES or fn.startswith(_SKIP_PREFIXES):
            continue
        key = f"{prefix}/{fn}"
        client.upload_file(path, bucket, key)
        url = client.generate_presigned_url(
            "get_object", Params={"Bucket": bucket, "Key": key},
            ExpiresIn=expires)
        out[fn] = {"s3": f"s3://{bucket}/{key}", "url": url}
    return out
