from __future__ import annotations

import io
from pathlib import Path

import boto3
from botocore.config import Config as BotoConfig
from botocore.exceptions import ClientError

from app.core.config import settings


def _get_client():
    return boto3.client(
        "s3",
        endpoint_url=settings.S3_ENDPOINT,
        aws_access_key_id=settings.S3_ACCESS_KEY,
        aws_secret_access_key=settings.S3_SECRET_KEY,
        region_name=settings.S3_REGION,
        config=BotoConfig(signature_version="s3v4"),
    )


def ensure_bucket(bucket: str | None = None) -> None:
    bucket = bucket or settings.S3_BUCKET_TRACKS
    client = _get_client()
    try:
        client.head_bucket(Bucket=bucket)
    except ClientError:
        client.create_bucket(Bucket=bucket)


def upload_file(data: bytes, key: str, content_type: str = "audio/aiff", bucket: str | None = None) -> str:
    bucket = bucket or settings.S3_BUCKET_TRACKS
    ensure_bucket(bucket)
    client = _get_client()
    client.put_object(Bucket=bucket, Key=key, Body=data, ContentType=content_type)
    return key


def download_file(key: str, bucket: str | None = None) -> bytes:
    bucket = bucket or settings.S3_BUCKET_TRACKS
    client = _get_client()
    resp = client.get_object(Bucket=bucket, Key=key)
    return resp["Body"].read()


def download_to_path(key: str, dest: Path, bucket: str | None = None) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(download_file(key, bucket))
    return dest


def file_exists(key: str, bucket: str | None = None) -> bool:
    bucket = bucket or settings.S3_BUCKET_TRACKS
    client = _get_client()
    try:
        client.head_object(Bucket=bucket, Key=key)
        return True
    except Exception:
        return False


def delete_file(key: str, bucket: str | None = None) -> None:
    bucket = bucket or settings.S3_BUCKET_TRACKS
    client = _get_client()
    client.delete_object(Bucket=bucket, Key=key)


def _get_public_client():
    """Client using the browser-accessible endpoint for presigned URLs."""
    return boto3.client(
        "s3",
        endpoint_url=settings.S3_PUBLIC_ENDPOINT,
        aws_access_key_id=settings.S3_ACCESS_KEY,
        aws_secret_access_key=settings.S3_SECRET_KEY,
        region_name=settings.S3_REGION,
        config=BotoConfig(signature_version="s3v4"),
    )


def generate_presigned_url(key: str, expires_in: int = 3600, bucket: str | None = None) -> str:
    bucket = bucket or settings.S3_BUCKET_TRACKS
    client = _get_public_client()
    return client.generate_presigned_url(
        "get_object",
        Params={"Bucket": bucket, "Key": key},
        ExpiresIn=expires_in,
    )
