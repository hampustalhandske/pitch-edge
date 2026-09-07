# Phase 3 — Google Cloud deployment

Everything below is real, runnable configuration; it has **not** been exercised
against a live GCP project in this repo (no credentials are assumed). The
storage abstraction (`Warehouse.export_parquet` → `pitch_edge.cloud.sync_to_gcp`)
is the hand-off: local DuckDB tables → Parquet → GCS → BigQuery.

```bash
export PROJECT=your-project REGION=europe-north1 BUCKET=pitch-edge-lake
gcloud services enable run.googleapis.com cloudscheduler.googleapis.com bigquery.googleapis.com \
  secretmanager.googleapis.com artifactregistry.googleapis.com cloudbuild.googleapis.com

# image
gcloud builds submit --tag $REGION-docker.pkg.dev/$PROJECT/pitch-edge/app:latest -f deploy/Dockerfile .

# secrets (only an LLM key exists in this system — there is no bookmaker credential anywhere)
echo -n "$ANTHROPIC_API_KEY" | gcloud secrets create anthropic-api-key --data-file=-

# nightly refresh job (ingest -> features -> backtest -> RAG) + GCS/BigQuery sync
gcloud run jobs create pitch-edge-refresh --image $REGION-docker.pkg.dev/$PROJECT/pitch-edge/app:latest \
  --region $REGION --memory 4Gi --cpu 2 --task-timeout 3600 \
  --set-env-vars PITCH_EDGE_GCP_PROJECT=$PROJECT,PITCH_EDGE_GCS_BUCKET=$BUCKET \
  --command pitch-edge --args refresh,--fast
gcloud scheduler jobs create http pitch-edge-nightly --schedule "30 3 * * *" --location $REGION \
  --uri "https://$REGION-run.googleapis.com/apis/run.googleapis.com/v1/namespaces/$PROJECT/jobs/pitch-edge-refresh:run" \
  --http-method POST --oauth-service-account-email $PROJECT@appspot.gserviceaccount.com

# hourly news / prediction-market snapshots
gcloud run jobs create pitch-edge-news --image $REGION-docker.pkg.dev/$PROJECT/pitch-edge/app:latest --region $REGION \
  --command python --args -c,"from pitch_edge.scheduler import job_news_and_markets; job_news_and_markets()"
gcloud scheduler jobs create http pitch-edge-hourly --schedule "0 * * * *" --location $REGION \
  --uri "https://$REGION-run.googleapis.com/apis/run.googleapis.com/v1/namespaces/$PROJECT/jobs/pitch-edge-news:run" \
  --http-method POST --oauth-service-account-email $PROJECT@appspot.gserviceaccount.com

# dashboard service
gcloud run deploy pitch-edge-dashboard --image $REGION-docker.pkg.dev/$PROJECT/pitch-edge/app:latest --region $REGION \
  --allow-unauthenticated --memory 2Gi --set-secrets ANTHROPIC_API_KEY=anthropic-api-key:latest
```

Vector search: `pitch_edge.rag.index.VectorIndex` wraps Chroma behind a
two-method interface (`add`, `query`); a Vertex AI Vector Search backend is a
drop-in third `_Backend` class. Pub/Sub is not needed for the current job
shapes (Scheduler → Run Jobs is sufficient and cheaper).
