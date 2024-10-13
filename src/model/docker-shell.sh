#!/bin/bash

# exit immediately if a command exits with a non-zero status
set -e

export GCP_PROJECT="ac215-438023" # CHANGE TO YOUR PROJECT ID
export GOOGLE_APPLICATION_CREDENTIALS="/secrets/llm-service-account.json"

# Create the network if we don't have it yet
docker network inspect llm-rag-network >/dev/null 2>&1 || docker network create llm-rag-network

docker-compose build model

docker-compose run --rm --service-ports model
