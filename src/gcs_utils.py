"""
gcs_utils.py
------------
Google Cloud Storage utilities for SupplAI.

Provides a single helper — `ensure_local` — that:
  1. Returns the local path immediately if the file already exists locally.
  2. Downloads the file from GCS when GCS_BUCKET_NAME is set and the local
     file is missing (e.g., first start of a Cloud Run instance).
  3. Falls through to the local path if GCS is not configured, so the app
     still works in a plain local-dev environment.

Environment variables used:
  GCS_BUCKET_NAME   – GCS bucket that holds runtime assets (models, data).
                      If unset, GCS download is skipped.
  GCP_PROJECT_ID    – GCP project (used for ADC authentication on Cloud Run).

GCS object layout expected:
  gs://<GCS_BUCKET_NAME>/models/delay_model.pkl
  gs://<GCS_BUCKET_NAME>/models/anomaly_model.pkl
  gs://<GCS_BUCKET_NAME>/datasets/order_large.csv
  gs://<GCS_BUCKET_NAME>/datasets/distance.csv
  gs://<GCS_BUCKET_NAME>/data/supply_chain.csv
  gs://<GCS_BUCKET_NAME>/data/wits_tariffs.csv
  gs://<GCS_BUCKET_NAME>/data/ofac_sanctions.json

On Cloud Run the service account has roles/storage.objectViewer on the bucket,
so no explicit credentials are needed (Application Default Credentials work).
"""

from __future__ import annotations

import os
from pathlib import Path

# Project root is two levels up from this file (src/gcs_utils.py → root)
PROJECT_ROOT = Path(__file__).resolve().parent.parent

_GCS_BUCKET = os.getenv("GCS_BUCKET_NAME", "").strip()
_GCS_CLIENT = None   # lazy-initialised


def _get_client():
    """Lazily initialise the GCS client (avoids import cost when GCS is unused)."""
    global _GCS_CLIENT
    if _GCS_CLIENT is None:
        from google.cloud import storage  # type: ignore
        _GCS_CLIENT = storage.Client()
    return _GCS_CLIENT


def _gcs_object_key(local_path: Path) -> str:
    """
    Derive the GCS object key from the local path relative to PROJECT_ROOT.
    E.g.  /app/models/delay_model.pkl  →  models/delay_model.pkl
    """
    return str(local_path.relative_to(PROJECT_ROOT)).replace("\\", "/")


def ensure_local(local_path: Path) -> Path:
    """
    Ensure *local_path* exists locally, downloading from GCS if needed.

    Parameters
    ----------
    local_path : Path
        Absolute path where the file should live locally.

    Returns
    -------
    Path
        The same *local_path* (guaranteed to exist after this call if GCS
        download succeeded, or if the file was already present).
    """
    if local_path.exists():
        return local_path

    if not _GCS_BUCKET:
        # GCS not configured — return path as-is (caller handles missing file)
        return local_path

    blob_name = _gcs_object_key(local_path)
    print(f"  [gcs_utils] {local_path.name} not found locally — "
          f"downloading from gs://{_GCS_BUCKET}/{blob_name} …")

    try:
        client = _get_client()
        bucket = client.bucket(_GCS_BUCKET)
        blob   = bucket.blob(blob_name)

        local_path.parent.mkdir(parents=True, exist_ok=True)
        blob.download_to_filename(str(local_path))
        print(f"  [gcs_utils] Downloaded {local_path.name} "
              f"({local_path.stat().st_size / 1024:.0f} KB)")
    except Exception as exc:
        print(f"  [gcs_utils] WARNING: Could not download {blob_name} "
              f"from GCS: {exc}")

    return local_path


def upload_to_gcs(local_path: Path, blob_name: str | None = None) -> bool:
    """
    Upload a local file to GCS.  Useful for persisting retrained models.

    Parameters
    ----------
    local_path : Path  Absolute path of the file to upload.
    blob_name  : str   GCS object key.  Defaults to path relative to PROJECT_ROOT.

    Returns True on success, False on failure.
    """
    if not _GCS_BUCKET:
        print("  [gcs_utils] GCS_BUCKET_NAME not set — skipping upload.")
        return False

    if blob_name is None:
        blob_name = _gcs_object_key(local_path)

    try:
        client = _get_client()
        bucket = client.bucket(_GCS_BUCKET)
        blob   = bucket.blob(blob_name)
        blob.upload_from_filename(str(local_path))
        print(f"  [gcs_utils] Uploaded {local_path.name} → "
              f"gs://{_GCS_BUCKET}/{blob_name}")
        return True
    except Exception as exc:
        print(f"  [gcs_utils] WARNING: Upload failed: {exc}")
        return False
