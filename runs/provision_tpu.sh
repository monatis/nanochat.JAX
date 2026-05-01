#!/bin/bash
# =============================================================================
# Provision a TPU VM via flex-start and run the nanochat smoke test
#
# This script runs from YOUR LOCAL MACHINE (not the TPU VM).
# It provisions a TPU, SSHes in, clones the repo, and kicks off the smoke test.
#
# Prerequisites:
#   1. gcloud CLI installed and authenticated: gcloud auth login
#   2. Project set: gcloud config set project YOUR_PROJECT_ID
#   3. TPU API enabled: gcloud services enable tpu.googleapis.com
#   4. Alpha components: gcloud components install alpha --quiet
#
# Usage:
#   # Quick test on smallest TPU (v6e-1, 1 chip):
#   bash runs/provision_tpu.sh
#
#   # Custom configuration:
#   TPU_TYPE=v5e-4 ZONE=us-west4-a RUNTIME=v2-alpha-tpuv5-lite DURATION=2h \
#     bash runs/provision_tpu.sh
#
# Environment variables (all optional, with defaults):
#   PROJECT_ID      - GCP project ID (default: current gcloud project)
#   ZONE            - TPU zone (default: us-central1-a)
#   TPU_TYPE        - Accelerator type (default: v6e-1)
#   RUNTIME         - Runtime version (default: v2-alpha-tpuv6e)
#   DURATION        - Max run duration (default: 2h)
#   NODE_NAME       - TPU node name (default: nanochat-smoke-test)
#   QUEUE_NAME      - Queued resource name (default: nanochat-queue)
#   REPO_URL        - Git repo URL (default: https://github.com/monatis/nanochat.JAX)
#   SKIP_PROVISION  - Set to 1 to skip provisioning (reuse existing TPU)
#   SKIP_CLEANUP    - Set to 1 to keep TPU alive after test
# =============================================================================

set -euo pipefail

# ---------------------------------------------------------------------------
# Configuration (override via environment variables)
# ---------------------------------------------------------------------------

PROJECT_ID="${PROJECT_ID:-$(gcloud config get-value project 2>/dev/null)}"
ZONE="${ZONE:-us-central1-a}"
TPU_TYPE="${TPU_TYPE:-v6e-1}"
RUNTIME="${RUNTIME:-v2-alpha-tpuv6e}"
DURATION="${DURATION:-2h}"
NODE_NAME="${NODE_NAME:-nanochat-smoke-test}"
QUEUE_NAME="${QUEUE_NAME:-nanochat-queue}"
REPO_URL="${REPO_URL:-https://github.com/monatis/nanochat.JAX}"
SKIP_PROVISION="${SKIP_PROVISION:-0}"
SKIP_CLEANUP="${SKIP_CLEANUP:-0}"

# Runtime version mapping (auto-detect from TPU_TYPE if not explicitly set)
if [ "$RUNTIME" = "auto" ]; then
    case "$TPU_TYPE" in
        v6e*) RUNTIME="v2-alpha-tpuv6e" ;;
        v5p*) RUNTIME="v2-alpha-tpuv5" ;;
        v5e*) RUNTIME="v2-alpha-tpuv5-lite" ;;
        *)    echo "ERROR: Cannot auto-detect runtime for $TPU_TYPE. Set RUNTIME explicitly."; exit 1 ;;
    esac
fi

echo "============================================================"
echo " nanochat TPU Provisioning & Smoke Test"
echo "============================================================"
echo ""
echo "  Project     : $PROJECT_ID"
echo "  Zone        : $ZONE"
echo "  TPU type    : $TPU_TYPE"
echo "  Runtime     : $RUNTIME"
echo "  Max duration: $DURATION"
echo "  Node        : $NODE_NAME"
echo "  Queue       : $QUEUE_NAME"
echo "  Repo        : $REPO_URL"
echo ""

# ---------------------------------------------------------------------------
# Step 1: Provision TPU via flex-start queued resource
# ---------------------------------------------------------------------------

if [ "$SKIP_PROVISION" != "1" ]; then
    echo ">>> Step 1: Creating TPU queued resource (flex-start)..."
    echo ""

    # Delete any existing queued resource with the same name (ignore errors)
    gcloud alpha compute tpus queued-resources delete "$QUEUE_NAME" \
        --project="$PROJECT_ID" \
        --zone="$ZONE" \
        --quiet 2>/dev/null || true

    gcloud alpha compute tpus queued-resources create "$QUEUE_NAME" \
        --project="$PROJECT_ID" \
        --zone="$ZONE" \
        --accelerator-type="$TPU_TYPE" \
        --runtime-version="$RUNTIME" \
        --node-id="$NODE_NAME" \
        --provisioning-model=flex-start \
        --max-run-duration="$DURATION" \
        --valid-until-duration="$DURATION"

    echo ""
    echo ">>> Waiting for TPU to be provisioned..."
    echo "    (This may take a few minutes depending on capacity)"
    echo ""

    # Poll until the TPU is ACTIVE
    MAX_WAIT=600  # 10 minutes
    ELAPSED=0
    POLL_INTERVAL=15

    while [ $ELAPSED -lt $MAX_WAIT ]; do
        STATUS=$(gcloud alpha compute tpus queued-resources describe "$QUEUE_NAME" \
            --project="$PROJECT_ID" \
            --zone="$ZONE" \
            --format="value(state.state)" 2>/dev/null || echo "UNKNOWN")

        echo "    Status: $STATUS (${ELAPSED}s elapsed)"

        if [ "$STATUS" = "ACTIVE" ]; then
            echo ""
            echo "    ✓ TPU is ACTIVE!"
            break
        elif [ "$STATUS" = "FAILED" ] || [ "$STATUS" = "SUSPENDED" ]; then
            echo ""
            echo "    ✗ TPU provisioning failed with status: $STATUS"
            echo "    Check: gcloud alpha compute tpus queued-resources describe $QUEUE_NAME --zone=$ZONE"
            exit 1
        fi

        sleep $POLL_INTERVAL
        ELAPSED=$((ELAPSED + POLL_INTERVAL))
    done

    if [ $ELAPSED -ge $MAX_WAIT ]; then
        echo ""
        echo "    ✗ Timed out waiting for TPU (${MAX_WAIT}s)."
        echo "    The request may still be pending. Check manually:"
        echo "    gcloud alpha compute tpus queued-resources describe $QUEUE_NAME --zone=$ZONE"
        exit 1
    fi
else
    echo ">>> Step 1: SKIPPED (SKIP_PROVISION=1)"
fi

echo ""

# ---------------------------------------------------------------------------
# Step 2: SSH into TPU VM and run the smoke test
# ---------------------------------------------------------------------------

echo ">>> Step 2: Running smoke test on TPU VM..."
echo ""

# The remote script: clone repo, install deps, run smoke test
REMOTE_SCRIPT=$(cat <<'REMOTE_EOF'
#!/bin/bash
set -euo pipefail

echo "=== TPU VM: Starting nanochat smoke test ==="
echo "Hostname: $(hostname)"
echo "Date: $(date)"
echo ""

# Clone the repo (or update if already present)
REPO_DIR="$HOME/nanochat"
REPO_URL_VAR="__REPO_URL__"

if [ -d "$REPO_DIR/.git" ]; then
    echo "Repo already exists, pulling latest..."
    cd "$REPO_DIR"
    git pull origin main || git pull origin master || true
else
    echo "Cloning $REPO_URL_VAR ..."
    git clone "$REPO_URL_VAR" "$REPO_DIR"
    cd "$REPO_DIR"
fi

echo ""
echo "=== Running smoke test ==="
bash runs/tpu_smoke_test.sh

echo ""
echo "=== Smoke test finished ==="
REMOTE_EOF
)

# Inject the actual REPO_URL into the script
REMOTE_SCRIPT="${REMOTE_SCRIPT/__REPO_URL__/$REPO_URL}"

gcloud compute tpus tpu-vm ssh "$NODE_NAME" \
    --project="$PROJECT_ID" \
    --zone="$ZONE" \
    --command="$REMOTE_SCRIPT"

SSH_EXIT=$?

echo ""

# ---------------------------------------------------------------------------
# Step 3: Cleanup (optional)
# ---------------------------------------------------------------------------

if [ "$SKIP_CLEANUP" != "1" ]; then
    echo ">>> Step 3: Cleaning up TPU resources..."
    gcloud alpha compute tpus queued-resources delete "$QUEUE_NAME" \
        --project="$PROJECT_ID" \
        --zone="$ZONE" \
        --quiet 2>/dev/null || true
    echo "    ✓ Queued resource deleted"
else
    echo ">>> Step 3: SKIPPED (SKIP_CLEANUP=1)"
    echo "    TPU is still running. Don't forget to clean up!"
    echo "    gcloud alpha compute tpus queued-resources delete $QUEUE_NAME --zone=$ZONE --quiet"
fi

echo ""
echo "============================================================"
if [ $SSH_EXIT -eq 0 ]; then
    echo " ✓ TPU smoke test completed successfully!"
else
    echo " ✗ TPU smoke test FAILED (exit code $SSH_EXIT)"
fi
echo "============================================================"
exit $SSH_EXIT
