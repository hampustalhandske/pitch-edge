# Cloud Run image: same code path as local, storage pointed at GCS/BigQuery via env.
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 UV_SYSTEM_PYTHON=1 PITCH_EDGE_DATA_DIR=/data
RUN pip install --no-cache-dir uv
WORKDIR /app
COPY pyproject.toml uv.lock README.md ./
COPY src ./src
RUN uv sync --frozen --no-dev --extra cloud
ENV PATH="/app/.venv/bin:$PATH"

# Jobs: `pitch-edge refresh --fast` (nightly), `pitch-edge signals`, `python -m pitch_edge.cloud.sync`
# Service: streamlit dashboard
EXPOSE 8080
CMD ["streamlit", "run", "src/pitch_edge/dashboard/app.py", "--server.port", "8080", "--server.address", "0.0.0.0", "--server.headless", "true"]
