"""Phase 3 — Google Cloud mirror of the local warehouse.

`sync_to_gcp` exports every DuckDB table to Parquet, uploads to GCS (data
lake) and loads into BigQuery (warehouse). Google client libraries are an
optional extra (`uv sync --extra cloud`) and credentials come from ADC /
`PITCH_EDGE_GCP_PROJECT` / `PITCH_EDGE_GCS_BUCKET`; without them the
function reports what it *would* do and exits cleanly — the code path is
real, the deployment is whatever the reader provisions. `deploy/` holds the
Dockerfile, Cloud Run job and Cloud Scheduler definitions.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

from pitch_edge.config import get_settings
from pitch_edge.data.storage import Warehouse

logger = logging.getLogger(__name__)


@dataclass
class SyncReport:
    configured: bool
    uploaded: list[str] = field(default_factory=list)
    loaded: list[str] = field(default_factory=list)
    skipped_reason: str | None = None


def gcp_available() -> tuple[bool, str | None]:
    s = get_settings()
    if not (s.gcp_project and s.gcs_bucket):
        return False, "PITCH_EDGE_GCP_PROJECT / PITCH_EDGE_GCS_BUCKET not set"
    try:
        import google.cloud.bigquery  # noqa: F401
        import google.cloud.storage  # noqa: F401
    except ImportError:
        return False, "google-cloud libraries missing (uv sync --extra cloud)"
    return True, None


def sync_to_gcp(wh: Warehouse, lake_dir: str | Path | None = None, dry_run: bool = False) -> SyncReport:
    ok, reason = gcp_available()
    paths = wh.export_parquet(lake_dir)
    if not ok or dry_run:
        return SyncReport(
            configured=ok, skipped_reason=reason or "dry_run", uploaded=[p.name for p in paths] if dry_run else []
        )
    from google.cloud import bigquery, storage

    s = get_settings()
    report = SyncReport(configured=True)
    gcs = storage.Client(project=s.gcp_project)
    bucket = gcs.bucket(s.gcs_bucket)
    bq = bigquery.Client(project=s.gcp_project)
    dataset_ref = bigquery.Dataset(f"{s.gcp_project}.{s.bigquery_dataset}")
    bq.create_dataset(dataset_ref, exists_ok=True)
    for p in paths:
        blob_name = f"lake/{p.name}"
        bucket.blob(blob_name).upload_from_filename(str(p))
        report.uploaded.append(blob_name)
        table_id = f"{s.gcp_project}.{s.bigquery_dataset}.{p.stem}"
        job = bq.load_table_from_uri(
            f"gs://{s.gcs_bucket}/{blob_name}",
            table_id,
            job_config=bigquery.LoadJobConfig(
                source_format=bigquery.SourceFormat.PARQUET, write_disposition="WRITE_TRUNCATE"
            ),
        )
        job.result()
        report.loaded.append(table_id)
        logger.info("loaded %s", table_id)
    return report
