#!/usr/bin/env bash
set -euo pipefail

PROJECT_ID="${GCP_PROJECT_ID:-$(gcloud config get-value project 2>/dev/null)}"
REGION="${GCP_REGION:-asia-southeast2}"
REPO="plastitrace"
API_SERVICE="plastitrace-api"
WEB_SERVICE="plastitrace-web"
REGISTRY="${REGION}-docker.pkg.dev/${PROJECT_ID}/${REPO}"

if [[ -z "${PROJECT_ID}" || "${PROJECT_ID}" == "(unset)" ]]; then
  echo "Error: set GCP_PROJECT_ID or run: gcloud config set project YOUR_PROJECT_ID"
  exit 1
fi

if [[ ! -f "models/plastitrace.pth" ]]; then
  echo "Error: models/plastitrace.pth not found"
  exit 1
fi

echo "==> Project: ${PROJECT_ID}"
echo "==> Region:  ${REGION}"

echo "==> Enabling required APIs..."
gcloud services enable run.googleapis.com artifactregistry.googleapis.com cloudbuild.googleapis.com \
  --project="${PROJECT_ID}"

if ! gcloud artifacts repositories describe "${REPO}" \
  --location="${REGION}" --project="${PROJECT_ID}" &>/dev/null; then
  echo "==> Creating Artifact Registry repo..."
  gcloud artifacts repositories create "${REPO}" \
    --repository-format=docker \
    --location="${REGION}" \
    --project="${PROJECT_ID}"
fi

echo "==> Building API image..."
gcloud builds submit . \
  --tag "${REGISTRY}/api:latest" \
  --project="${PROJECT_ID}" \
  --timeout=1200

echo "==> Deploying API to Cloud Run..."
gcloud run deploy "${API_SERVICE}" \
  --image "${REGISTRY}/api:latest" \
  --region "${REGION}" \
  --project "${PROJECT_ID}" \
  --allow-unauthenticated \
  --memory 4Gi \
  --cpu 2 \
  --cpu-boost \
  --timeout 120 \
  --port 8080 \
  --set-env-vars "ALLOWED_ORIGINS=*"

API_URL="$(gcloud run services describe "${API_SERVICE}" \
  --region "${REGION}" --project "${PROJECT_ID}" \
  --format='value(status.url)')"
echo "==> API URL: ${API_URL}"

echo "==> Building web image..."
TMP_CB=$(mktemp /tmp/cloudbuild-XXXX.yaml)
cat > "${TMP_CB}" <<YAML
steps:
  - name: gcr.io/cloud-builders/docker
    args:
      - build
      - --build-arg=NEXT_PUBLIC_API_URL=${API_URL}
      - -t
      - ${REGISTRY}/web:latest
      - .
images:
  - ${REGISTRY}/web:latest
YAML
gcloud builds submit web/ \
  --config="${TMP_CB}" \
  --project="${PROJECT_ID}" \
  --timeout=1200
rm -f "${TMP_CB}"

echo "==> Deploying web to Cloud Run..."
gcloud run deploy "${WEB_SERVICE}" \
  --image "${REGISTRY}/web:latest" \
  --region "${REGION}" \
  --project "${PROJECT_ID}" \
  --allow-unauthenticated \
  --memory 512Mi

WEB_URL="$(gcloud run services describe "${WEB_SERVICE}" \
  --region "${REGION}" --project "${PROJECT_ID}" \
  --format='value(status.url)')"
echo "==> Web URL: ${WEB_URL}"

echo "==> Locking down API CORS to web origin..."
gcloud run services update "${API_SERVICE}" \
  --region "${REGION}" \
  --project "${PROJECT_ID}" \
  --set-env-vars "^|^ALLOWED_ORIGINS=${WEB_URL},http://localhost:3000"

echo ""
echo "Deployment complete."
echo "  Web: ${WEB_URL}"
echo "  API: ${API_URL}"
