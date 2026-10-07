# ai-service: FastAPI + SigLIP (CPU) untuk embedding gambar produk.
# Weights DI-BAKE ke image supaya container tidak mengunduh model saat start
# (cold start jadi detik, bukan menit, dan tidak bergantung jaringan runtime).
FROM python:3.11-slim AS base

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/opt/hf \
    TRANSFORMERS_OFFLINE=1 \
    HF_HUB_OFFLINE=1

WORKDIR /app

# libgomp1 dibutuhkan torch CPU; sisanya untuk dekode JPEG/PNG oleh Pillow.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./
# --extra-index-url CPU wheel: image ~2.5GB, bukan ~7GB seperti build CUDA.
RUN pip install --no-cache-dir \
      --extra-index-url https://download.pytorch.org/whl/cpu \
      -r requirements.txt

ARG AI_MODEL_NAME=google/siglip-base-patch16-224
ENV AI_MODEL_NAME=${AI_MODEL_NAME}

# Pre-download weights ke HF_HOME pada build time.
# local_files_only=False SAAT BUILD = download dari HF. Nanti runtime,
# TRANSFORMERS_OFFLINE=1 + HF_HUB_OFFLINE=1 di atas = force pakai cache saja.
RUN TRANSFORMERS_OFFLINE=0 HF_HUB_OFFLINE=0 python -c "\
from transformers import AutoModel, AutoProcessor; \
import os; \
m = os.environ['AI_MODEL_NAME']; \
AutoProcessor.from_pretrained(m); \
AutoModel.from_pretrained(m); \
print('weights cached for', m); \
import glob; \
snap = glob.glob(os.path.join(os.environ['HF_HOME'], 'hub', 'models--*', 'snapshots', '*')); \
print('snapshot dirs:', len(snap)); \
[print(' ', os.path.basename(os.path.dirname(os.path.dirname(s)))) for s in snap]"

COPY main.py ./

EXPOSE 8000

# Model besar → 1 worker saja. Beberapa worker berarti beberapa kopi model di RAM.
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
