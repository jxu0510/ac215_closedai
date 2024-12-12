#!/bin/bash

echo "Container is running!!!"

export BASE_DIR=$(pwd)/..
export SECRETS_DIR=$(pwd)/../secrets/
export GCP_PROJECT="xenon-depth-434717-n0"
export GCS_BUCKET_NAME="closed-ai"
export GCS_RAG_BUCKET_NAME="closed-ai-rag"
export GOOGLE_APPLICATION_CREDENTIALS="/secrets/llm-service-account.json"
export GCP_SERVICE_ACCOUNT="llm-service-account@xenon-depth-434717-n0.iam.gserviceaccount.com"
export LOCATION="us-central1"

if [ "${DEV}" = 1 ]; then
  pipenv shell
else
  pipenv run python dataloader.py
  pipenv run python preprocess_rag.py --chunk --embed --load
fi
