#!/bin/bash

# exit immediately if a command exits with a non-zero status
set -e

source ../env.dev

export GCP_PROJECT="xenon-depth-434717-n0" # CHANGE TO YOUR PROJECT ID
export GOOGLE_APPLICATION_CREDENTIALS="/secrets/llm-service-account.json"

# Create the network if we don't have it yet
docker network inspect llm-rag-network >/dev/null 2>&1 || docker network create llm-rag-network

docker-compose build empathos
docker-compose up -d chromadb
docker-compose run --rm --service-ports empathos
