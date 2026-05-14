#!/usr/bin/env bash
set -euo pipefail

MONGO_URI=$(grep -E '^MONGO_URI=' "$(dirname "$0")/../.env" | cut -d= -f2-)
MONGO_USER=$(echo "$MONGO_URI" | sed -E 's|mongodb://([^:]+):.*|\1|')
MONGO_PASS=$(echo "$MONGO_URI" | sed -E 's|mongodb://[^:]+:([^@]+)@.*|\1|')
MONGO_PORT=$(echo "$MONGO_URI" | sed -E 's|.*:([0-9]+)$|\1|')

docker run -d \
  --name mongo-rag \
  -p "${MONGO_PORT}:27017" \
  -e MONGO_INITDB_ROOT_USERNAME="${MONGO_USER}" \
  -e MONGO_INITDB_ROOT_PASSWORD="${MONGO_PASS}" \
  -v mongo-rag-data:/data/db \
  mongo:latest

echo "MongoDB running on port ${MONGO_PORT} (user: ${MONGO_USER})"
