#!/usr/bin/bash

################################################################################
# Batch DAPI Swap for Existing MCMICRO Outputs
#
# Retroactively runs swap_dapi_channel.py on all *_staged directories that
# have registration/, backsub/, and markers.csv outputs from MCMICRO.
################################################################################

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
DRY_RUN=false

#==============================================================================
# USAGE
#==============================================================================

usage() {
    cat <<EOF
Batch DAPI Swap — replace Cycle1 DAPI with registered Scan DAPI in backsub outputs.

Usage: $0 <staging-base-dir> [--dry-run]

Arguments:
  staging-base-dir    Directory containing *_staged folders (MCMICRO outputs)
  --dry-run, -d       Preview which folders would be processed without running

Example:
  $0 /data/staged --dry-run
  $0 /data/staged
EOF
    exit "${1:-0}"
}

#==============================================================================
# HELPER FUNCTIONS
#==============================================================================

# Return the highest-exposure directory under base_dir, or base_dir itself.
# Same logic as get_highest_exposure_dir in run_MCMICRO_general.sh.
get_highest_exposure_dir() {
    local base_dir="$1"
    local highest_raw

    highest_raw=$(find "$base_dir" -type d -name raw 2>/dev/null | sort | tail -n 1)

    if [ -z "$highest_raw" ]; then
        echo "$base_dir"
        return 0
    fi

    highest_raw="${highest_raw%/}"
    local parent_dir="${highest_raw%/*}"

    echo "$parent_dir"
}

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"
}

#==============================================================================
# ARGUMENT PARSING
#==============================================================================

STAGING_BASE_DIR=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --dry-run|-d)
            DRY_RUN=true
            shift
            ;;
        --help|-h)
            usage 0
            ;;
        -*)
            echo "Unknown option: $1" >&2
            usage 1
            ;;
        *)
            if [ -z "$STAGING_BASE_DIR" ]; then
                STAGING_BASE_DIR="$1"
            else
                echo "Unexpected argument: $1" >&2
                usage 1
            fi
            shift
            ;;
    esac
done

if [ -z "$STAGING_BASE_DIR" ]; then
    echo "ERROR: staging-base-dir is required" >&2
    echo ""
    usage 1
fi

if [ ! -d "$STAGING_BASE_DIR" ]; then
    echo "ERROR: staging base directory does not exist: $STAGING_BASE_DIR" >&2
    exit 1
fi

#==============================================================================
# MAIN
#==============================================================================

total=0
processed=0
skipped=0
failed=0

if [ "$DRY_RUN" = true ]; then
    log "[DRY-RUN] Scanning $STAGING_BASE_DIR for *_staged directories..."
else
    log "Scanning $STAGING_BASE_DIR for *_staged directories..."
fi

for staged_dir in "$STAGING_BASE_DIR"/*_staged/; do
    [ -d "$staged_dir" ] || continue

    total=$((total + 1))
    roi_name=$(basename "$staged_dir" | sed 's/_staged$//')

    # Resolve highest-exposure subdirectory
    mcmicro_dir=$(get_highest_exposure_dir "$staged_dir")

    reg_dir="$mcmicro_dir/registration"
    backsub_dir="$mcmicro_dir/backsub"
    markers_csv="$mcmicro_dir/markers.csv"

    # Check prerequisites
    missing=""
    [ ! -d "$reg_dir" ] && missing="${missing} registration/"
    [ ! -d "$backsub_dir" ] && missing="${missing} backsub/"
    [ ! -f "$markers_csv" ] && missing="${missing} markers.csv"

    if [ -n "$missing" ]; then
        log "SKIP $roi_name — missing:${missing} (in $mcmicro_dir)"
        skipped=$((skipped + 1))
        continue
    fi

    if [ "$DRY_RUN" = true ]; then
        log "[DRY-RUN] Would process: $roi_name"
        log "[DRY-RUN]   MCMICRO dir: $mcmicro_dir"
        processed=$((processed + 1))
        continue
    fi

    log "Processing: $roi_name"
    log "  MCMICRO dir: $mcmicro_dir"

    if python3 "$SCRIPT_DIR/swap_dapi_channel.py" \
        --registration-dir "$reg_dir" \
        --backsub-dir "$backsub_dir" \
        --markers-csv "$markers_csv"; then
        log "SUCCESS: $roi_name"
        processed=$((processed + 1))
    else
        log "FAILED: $roi_name"
        failed=$((failed + 1))
    fi
done

echo ""
log "=========================================="
if [ "$DRY_RUN" = true ]; then
    log "[DRY-RUN] Summary"
else
    log "Summary"
fi
log "=========================================="
log "Total *_staged dirs found: $total"
log "Would process / processed: $processed"
log "Skipped (missing files):   $skipped"
if [ "$DRY_RUN" = false ]; then
    log "Failed:                    $failed"
    if [ $failed -gt 0 ]; then
        exit 1
    fi
fi
