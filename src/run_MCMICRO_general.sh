#!/usr/bin/bash

################################################################################
# MACSima Pipeline - Staging and MCMICRO Processing
# Processes ROIs from MACSima device through staging and MCMICRO analysis
################################################################################

set -euo pipefail

# Prevent zsh from erroring on non-matching globs (match bash default behavior)
if [ -n "${ZSH_VERSION:-}" ]; then
    setopt NO_NOMATCH
fi

#==============================================================================
# DEFAULTS
#==============================================================================

DRY_RUN=false
SKIP_EXPERIMENTS=""
USE_SCAN_DAPI=false
USE_HIGHEST_EXPOSURE=true
REFERENCE_MARKER="DAPI"
EXPERIMENT_FILTER=""
CLEANUP_STAGED=false
RECOMPUTE=false
KEEP_DAPI_CYCLES=""   # Comma-separated cycle numbers whose DAPI to keep (cycle 1 always kept)

# Config paths (set via CLI flags or environment variables)
ROOT_DIR="${MCMICRO_ROOT_DIR:-}"
STAGING_CONTAINER="${MCMICRO_STAGING_CONTAINER:-}"
STAGING_BASE_DIR="${MCMICRO_STAGING_DIR:-}"
SINGULARITY_CONFIG="${MCMICRO_SINGULARITY_CONFIG:-}"
PARAMS_FILE="${MCMICRO_PARAMS_FILE:-}"
MCMICRO_OUTPUT_BASE="${MCMICRO_OUTPUT_DIR:-}"
MCMICRO_WORK_DIR=""  # Derived from MCMICRO_OUTPUT_BASE in validate_config

LOG_FILE=""  # Set in validate_config once MCMICRO_OUTPUT_BASE is known

# Counters (global, updated by processing functions)
TOTAL_ROIS=0
PROCESSED_ROIS=0
FAILED_ROIS=0
SKIPPED_EXPERIMENTS=0

#==============================================================================
# LOGGING FUNCTIONS
#==============================================================================

log_info() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] [INFO] $*" | tee -a "$LOG_FILE"
}

log_success() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] [SUCCESS] $*" | tee -a "$LOG_FILE"
}

log_error() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] [ERROR] $*" | tee -a "$LOG_FILE"
}

log_warning() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] [WARNING] $*" | tee -a "$LOG_FILE"
}

# Unified logging: prepends [DRY-RUN] in dry-run mode, writes to log file otherwise
log_msg() {
    if [ "$DRY_RUN" = true ]; then
        echo "[DRY-RUN] $*"
    else
        log_info "$*"
    fi
}

#==============================================================================
# CONFIGURATION VALIDATION
#==============================================================================

validate_config() {
    local errors=0

    if [ -z "$ROOT_DIR" ]; then
        echo "ERROR: Root directory not set. Use --root-dir or \$MCMICRO_ROOT_DIR" >&2
        errors=$((errors + 1))
    elif [ ! -d "$ROOT_DIR" ]; then
        echo "ERROR: Root directory does not exist: $ROOT_DIR" >&2
        errors=$((errors + 1))
    fi

    if [ -z "$STAGING_CONTAINER" ]; then
        echo "ERROR: Staging container not set. Use --container or \$MCMICRO_STAGING_CONTAINER" >&2
        errors=$((errors + 1))
    elif [ ! -f "$STAGING_CONTAINER" ]; then
        echo "ERROR: Staging container not found: $STAGING_CONTAINER" >&2
        errors=$((errors + 1))
    fi

    if [ -z "$STAGING_BASE_DIR" ]; then
        echo "ERROR: Staging directory not set. Use --staging-dir or \$MCMICRO_STAGING_DIR" >&2
        errors=$((errors + 1))
    fi

    if [ -z "$SINGULARITY_CONFIG" ]; then
        echo "ERROR: Singularity config not set. Use --singularity-config or \$MCMICRO_SINGULARITY_CONFIG" >&2
        errors=$((errors + 1))
    elif [ ! -f "$SINGULARITY_CONFIG" ]; then
        echo "ERROR: Singularity config not found: $SINGULARITY_CONFIG" >&2
        errors=$((errors + 1))
    fi

    if [ -z "$PARAMS_FILE" ]; then
        echo "ERROR: Params file not set. Use --params or \$MCMICRO_PARAMS_FILE" >&2
        errors=$((errors + 1))
    elif [ ! -f "$PARAMS_FILE" ]; then
        echo "ERROR: Params file not found: $PARAMS_FILE" >&2
        errors=$((errors + 1))
    fi

    if [ -z "$MCMICRO_OUTPUT_BASE" ]; then
        echo "ERROR: Output directory not set. Use --output-dir or \$MCMICRO_OUTPUT_DIR" >&2
        errors=$((errors + 1))
    fi

    if [ $errors -gt 0 ]; then
        echo "" >&2
        echo "Use --help for usage information" >&2
        exit 1
    fi

    # Derive work directory
    MCMICRO_WORK_DIR="${MCMICRO_OUTPUT_BASE}/work"

    # Set log file path in the output directory (now that MCMICRO_OUTPUT_BASE is known)
    LOG_FILE="${MCMICRO_OUTPUT_BASE}/macsima_pipeline_$(date +%Y%m%d_%H%M%S).log"
}

#==============================================================================
# HELPER FUNCTIONS
#==============================================================================

should_skip_experiment() {
    local exp_name="$1"

    if [ -z "$SKIP_EXPERIMENTS" ]; then
        return 1
    fi

    local SKIP_LIST=()
    local _item
    while IFS= read -r _item; do
        _item=$(echo "$_item" | xargs)  # trim whitespace
        [ -n "$_item" ] && SKIP_LIST+=("$_item")
    done < <(echo "$SKIP_EXPERIMENTS" | tr ',' '\n')

    for skip_exp in "${SKIP_LIST[@]}"; do
        if [ "$exp_name" = "$skip_exp" ]; then
            return 0
        fi
    done

    return 1
}

matches_experiment_filter() {
    local exp_name="$1"
    if [ -z "$EXPERIMENT_FILTER" ]; then
        return 0  # No filter = match all
    fi
    if [[ "$exp_name" =~ $EXPERIMENT_FILTER ]]; then
        return 0
    fi
    return 1
}

# Staging is complete only when the resolved MCMICRO input dir has markers.csv.
# A non-empty staged dir WITHOUT markers.csv is a partial/interrupted stage.
is_staging_complete() {
    local staged_dir="$1"
    [ -d "$staged_dir" ] || return 1
    local input_dir
    input_dir=$(get_highest_exposure_dir "$staged_dir")
    [ -f "$input_dir/markers.csv" ]
}

# MCMICRO is complete for an ROI once it has written its registration output
# into the resolved input dir (coarse per-ROI resume marker).
is_mcmicro_complete() {
    local input_dir="$1"
    [ -d "$input_dir/registration" ] && [ -n "$(ls -A "$input_dir/registration" 2>/dev/null)" ]
}

swap_scan_dapi_into_cycle1() {
    local roi_path="$1"
    local scan_dir="${roi_path}/3_Scan2"
    local backup_dir="${roi_path}/.dapi_backup"

    # Find the Cycle1 directory (e.g. 6_Cycle1)
    local cycle1_dir
    cycle1_dir=$(find "$roi_path" -maxdepth 1 -type d -name '*_Cycle1' | head -1)

    if [ ! -d "$scan_dir" ]; then
        log_warning "3_Scan2 directory not found in $roi_path — skipping DAPI swap"
        return 1
    fi

    if [ -z "$cycle1_dir" ]; then
        log_warning "No *_Cycle1 directory found in $roi_path — skipping DAPI swap"
        return 1
    fi

    # If backup already exists (interrupted previous run), restore first
    if [ -d "$backup_dir" ]; then
        log_warning "Found stale .dapi_backup in $roi_path — restoring before re-swapping"
        restore_cycle1_dapi "$roi_path"
    fi

    mkdir -p "$backup_dir"

    # Collect unique ROI identifiers from scan DAPI files
    local rois=()
    local roi_id
    for f in "$scan_dir"/*_D-DAPI_*.tif; do
        [ -f "$f" ] || continue
        roi_id=$(basename "$f" | grep -oE 'ROI-[0-9]+')
        if [ -n "$roi_id" ]; then
            local found=false
            for r in "${rois[@]+"${rois[@]}"}"; do
                if [ "$r" = "$roi_id" ]; then
                    found=true
                    break
                fi
            done
            if [ "$found" = false ]; then
                rois+=("$roi_id")
            fi
        fi
    done

    if [ ${#rois[@]} -eq 0 ]; then
        log_warning "No DAPI files found in $scan_dir — skipping DAPI swap"
        rmdir "$backup_dir" 2>/dev/null || true
        return 1
    fi

    local total_swapped=0
    for roi_id in "${rois[@]}"; do
        # Collect scan DAPI files for this ROI, sorted by F-number
        local scan_dapis=()
        while IFS= read -r f; do
            scan_dapis+=("$f")
        done < <(find "$scan_dir" -name "*_${roi_id}_*_D-DAPI_*.tif" -type f | sort -V)

        # Collect Cycle1 Stain DAPI files for this ROI, sorted by F-number
        local cycle1_dapis=()
        while IFS= read -r f; do
            cycle1_dapis+=("$f")
        done < <(find "$cycle1_dir" -name "*_ST-S_*_${roi_id}_*_D-DAPI_*.tif" -type f | sort -V)

        if [ ${#scan_dapis[@]} -ne ${#cycle1_dapis[@]} ]; then
            log_error "DAPI count mismatch for $roi_id: ${#scan_dapis[@]} scan vs ${#cycle1_dapis[@]} cycle1 ST-S"
            # Restore anything already backed up
            restore_cycle1_dapi "$roi_path"
            return 1
        fi

        while IFS=$'\t' read -r scan_file cycle1_file; do
            # Backup the original cycle1 DAPI
            cp "$cycle1_file" "$backup_dir/"
            # Overwrite cycle1 DAPI with scan DAPI (keep the cycle1 filename)
            cp "$scan_file" "$cycle1_file"
            total_swapped=$((total_swapped + 1))
        done < <(paste \
            <(printf '%s\n' "${scan_dapis[@]}") \
            <(printf '%s\n' "${cycle1_dapis[@]}"))
    done

    log_info "Swapped $total_swapped scan DAPI tiles into Cycle1 in $roi_path (${#rois[@]} ROIs)"

    # Also move bleach (ST-B) files out of cycle1 so macsima2mc only produces the stain OME-TIFF.
    # This prevents cross-round registration issues between bleach DAPI and scan DAPI.
    local bleach_moved=0
    for bleach_file in "$cycle1_dir"/*_ST-B_*.tif; do
        [ -f "$bleach_file" ] || continue
        mv "$bleach_file" "$backup_dir/"
        bleach_moved=$((bleach_moved + 1))
    done

    if [ $bleach_moved -gt 0 ]; then
        log_info "Moved $bleach_moved bleach (ST-B) files out of $cycle1_dir for staging"
    fi

    return 0
}

duplicate_cycle1_as_cycle999() {
    local roi_path="$1"

    # Find the Cycle1 directory (e.g. 6_Cycle1)
    local cycle1_dir
    cycle1_dir=$(find "$roi_path" -maxdepth 1 -type d -name '*_Cycle1' | head -1)

    if [ -z "$cycle1_dir" ]; then
        log_warning "No *_Cycle1 directory found in $roi_path — skipping Cycle999 duplication"
        return 1
    fi

    # Determine the directory prefix number (e.g. "6" from "6_Cycle1") and create 999_Cycle999
    local cycle999_dir="${roi_path}/999_Cycle999"
    mkdir -p "$cycle999_dir"

    # Copy only stain files (ST-S: DAPI + markers) from Cycle1
    local copied=0
    for stain_file in "$cycle1_dir"/*_ST-S_*.tif; do
        [ -f "$stain_file" ] || continue
        local basename_f
        basename_f=$(basename "$stain_file")
        # Rename CYC-001 → CYC-999 in filename
        local new_name="${basename_f//CYC-001/CYC-999}"
        cp "$stain_file" "$cycle999_dir/$new_name"
        copied=$((copied + 1))
    done

    if [ $copied -eq 0 ]; then
        log_warning "No ST-S files found in $cycle1_dir — removing empty Cycle999 dir"
        rmdir "$cycle999_dir" 2>/dev/null || true
        return 1
    fi

    log_info "Duplicated $copied Cycle1 stain files into $cycle999_dir (CYC-001 → CYC-999)"
    return 0
}

restore_cycle1_dapi() {
    local roi_path="$1"
    local backup_dir="${roi_path}/.dapi_backup"

    if [ ! -d "$backup_dir" ]; then
        return 0
    fi

    local cycle1_dir
    cycle1_dir=$(find "$roi_path" -maxdepth 1 -type d -name '*_Cycle1' | head -1)

    if [ -z "$cycle1_dir" ]; then
        log_error "Cannot restore DAPI: no *_Cycle1 directory found in $roi_path"
        return 1
    fi

    local restored=0
    for backup_file in "$backup_dir"/*.tif; do
        [ -f "$backup_file" ] || continue
        mv "$backup_file" "$cycle1_dir/"
        restored=$((restored + 1))
    done

    rm -rf "$backup_dir"
    log_info "Restored $restored files to Cycle1 ($roi_path)"

    # Remove the fake Cycle999 directory if it exists (created by duplicate_cycle1_as_cycle999)
    local cycle999_dir="${roi_path}/999_Cycle999"
    if [ -d "$cycle999_dir" ]; then
        rm -rf "$cycle999_dir"
        log_info "Removed Cycle999 duplicate directory ($roi_path)"
    fi

    return 0
}

clean_markers_background() {
    local staged_dir="$1"

    local markers_csv
    while IFS= read -r markers_csv; do
        [ -f "$markers_csv" ] || continue
        python3 -c "
import csv, sys

path = sys.argv[1]
with open(path) as f:
    reader = csv.DictReader(f)
    fieldnames = reader.fieldnames
    rows = list(reader)

existing = {r['marker_name'] for r in rows}
cleaned = 0
for r in rows:
    bg = r.get('background', '')
    if bg and bg not in existing:
        r['background'] = ''
        cleaned += 1

if cleaned > 0:
    with open(path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f'Cleared {cleaned} invalid background references in {path}')
" "$markers_csv" 2>&1 | while read -r line; do log_info "$line"; done
    done < <(find "$staged_dir" -name "markers.csv" -type f)
}

mark_cycle1_markers_removed() {
    local staged_dir="$1"

    local markers_csv
    while IFS= read -r markers_csv; do
        [ -f "$markers_csv" ] || continue
        python3 -c "
import csv, sys

path = sys.argv[1]
with open(path) as f:
    reader = csv.DictReader(f)
    fieldnames = reader.fieldnames
    rows = list(reader)

marked = 0
for r in rows:
    cycle = int(r.get('cycle_number', 0))
    marker = r.get('marker_name', '')
    # Cycle 1 non-DAPI markers are misaligned (acquired at original positions, not Scan positions)
    if cycle == 1 and marker != 'DAPI':
        r['remove'] = 'TRUE'
        marked += 1
    # Cycle 999 DAPI is the original bad-quality DAPI — only needed for alignment
    elif cycle == 999 and marker == 'DAPI':
        r['remove'] = 'TRUE'
        marked += 1

if marked > 0:
    with open(path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f'Marked {marked} channels as remove=TRUE in {path} (Cycle1 markers + Cycle999 DAPI)')
" "$markers_csv" 2>&1 | while read -r line; do log_info "$line"; done
    done < <(find "$staged_dir" -name "markers.csv" -type f)
}

# Additionally retain the requested cycles' DAPI in the FINAL stitched stack.
# By default only the reference DAPI (cycle 1) survives to the final image; every
# other cycle's DAPI is marked remove=TRUE. Simply un-removing one would leave TWO
# channels named "DAPI", desyncing the OME-TIFF plane count from its channel metadata
# (QuPath "Index N out of bounds for length N"). So we RENAME each kept cycle's DAPI to
# a unique name (DAPI_cycle<N>) and mark it kept. Cycle 1 (the reference) is untouched.
# Alignment is unaffected — ASHLAR aligns by channel index, not marker name.
keep_dapi_cycles() {
    local staged_dir="$1"
    local keep_list="$2"

    local markers_csv
    while IFS= read -r markers_csv; do
        [ -f "$markers_csv" ] || continue
        python3 -c "
import csv, sys

path = sys.argv[1]
ref_marker = sys.argv[2]
# Parse comma-separated list of extra cycles to keep (cycle 1 is always the reference)
keep = set()
for tok in sys.argv[3].split(','):
    tok = tok.strip()
    if tok:
        keep.add(int(tok))

with open(path) as f:
    reader = csv.DictReader(f)
    fieldnames = reader.fieldnames
    rows = list(reader)

# Per-cycle lookup of DAPI background rows (macsima2mc emits these as
# 'bg_*_DAPI-DAPI'). backsub subtracts the channel named in 'background', so a
# kept DAPI must point at its own cycle's DAPI background to be subtracted.
dapi_bg_by_cycle = {}
for r in rows:
    name = r.get('marker_name', '')
    if name == ref_marker:
        continue  # real DAPI rows, not backgrounds
    # A DAPI background row: a non-reference row that still references DAPI and
    # is a bg_* channel. Discriminates against the real DAPI + other markers.
    if 'DAPI' in name.upper() and name.lower().startswith('bg'):
        try:
            c = int(r.get('cycle_number', 0))
        except (ValueError, TypeError):
            continue
        dapi_bg_by_cycle[c] = name

kept = []
for r in rows:
    # Only the real per-cycle DAPI rows (bg_*_DAPI-DAPI rows have a different name)
    if r.get('marker_name', '') != ref_marker:
        continue
    cycle = int(r.get('cycle_number', 0))
    if cycle == 1:
        continue  # reference DAPI — must stay named '<ref>' and is already kept
    if cycle in keep:
        r['marker_name'] = f'{ref_marker}_cycle{cycle}'  # unique name -> retained as distinct channel
        r['remove'] = ''                                 # blank = keep in final stack
        bg = dapi_bg_by_cycle.get(cycle, '')
        if bg:
            r['background'] = bg   # backsub will subtract this cycle's DAPI background
        else:
            print(f'WARNING: no DAPI background row found for cycle {cycle}; '
                  f'{ref_marker}_cycle{cycle} will NOT be background-subtracted')
        kept.append(cycle)

if kept:
    with open(path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows)
    print(f'Kept {len(kept)} extra {ref_marker} channel(s) in final stack, cycles {sorted(kept)} '
          f'(renamed {ref_marker}_cycle<N>) in {path}')
" "$markers_csv" "$REFERENCE_MARKER" "$keep_list" 2>&1 | while read -r line; do log_info "$line"; done
    done < <(find "$staged_dir" -name "markers.csv" -type f)
}

# Return the highest-exposure directory under base_dir, or base_dir itself.
get_highest_exposure_dir() {
    local base_dir="$1"
    local highest_raw

    # Find last (highest) raw dir alphabetically — works in both bash and zsh
    highest_raw=$(find "$base_dir" -type d -name raw 2>/dev/null | sort | tail -n 1)

    if [ -z "$highest_raw" ]; then
        echo "$base_dir"
        return 0
    fi

    highest_raw="${highest_raw%/}"
    local parent_dir="${highest_raw%/*}"

    echo "$parent_dir"
}

#==============================================================================
# STAGING FUNCTION
#==============================================================================

stage_roi() {
    local roi_path="$1"
    local output_dir="$2"
    local roi_name="$3"

    if [ "$DRY_RUN" = true ]; then
        log_msg "  Would stage ROI: $roi_name"
        log_msg "    Source: $roi_path"
        log_msg "    Destination: $output_dir"
        if [ "$USE_HIGHEST_EXPOSURE" = true ]; then
            log_msg "    Mode: highest exposure only (-he)"
        fi

        for cycle in "${roi_path}"/*_Cycle*/; do
            [ -d "$cycle" ] || continue
            local cycle_folder
            cycle_folder=$(basename "$cycle")
            log_msg "      Cycle: $cycle_folder"
        done
        return 0
    fi

    log_info "Staging ROI: $roi_name from $roi_path"
    mkdir -p "$output_dir"

    if [ ! -d "$roi_path" ]; then
        log_error "ROI path does not exist: $roi_path"
        return 1
    fi

    local cycle_count=0
    for cycle in "${roi_path}"/*_Cycle*/; do
        [ -d "$cycle" ] || continue

        local cycle_folder
        cycle_folder=$(basename "$cycle")

        cycle_count=$((cycle_count + 1))
        log_info "  Processing cycle: $cycle_folder"

        local he_flag=""
        if [ "$USE_HIGHEST_EXPOSURE" = true ]; then
            he_flag="-he"
        fi

        # Run staging with singularity
        if singularity exec \
            --pwd /tmp \
            --bind "$roi_path:/mnt,$output_dir:/media" \
            --no-home \
            "$STAGING_CONTAINER" \
            python /staging/macsima2mc/macsima2mc.py \
            -i "/mnt/$cycle_folder" \
            -rm "$REFERENCE_MARKER" \
            -rr \
            -o "/media/1" \
            -ic \
            $he_flag >> "$LOG_FILE" 2>&1; then
            log_info "  Cycle $cycle_folder staged successfully"
        else
            log_error "  Failed to stage cycle $cycle_folder"
            return 1
        fi
    done

    if [ $cycle_count -eq 0 ]; then
        log_warning "No cycle folders found in $roi_path (or all were skipped)"
        return 1
    fi

    log_success "Staging completed for $roi_name ($cycle_count cycles processed)"
    return 0
}

#==============================================================================
# MCMICRO FUNCTION
#==============================================================================

run_mcmicro() {
    local staged_dir="$1"
    local roi_name="$2"
    local timestamp
    timestamp=$(date +%Y%m%d_%H%M%S)
    local output_report="${MCMICRO_OUTPUT_BASE}/${roi_name}_report_${timestamp}.html"

    if [ "$DRY_RUN" = true ]; then
        log_msg "  Would run MCMICRO for: $roi_name"
        log_msg "    Input: $staged_dir"
        log_msg "    Report: $output_report"
        return 0
    fi

    log_info "Running MCMICRO for: $roi_name"
    log_info "  Input: $staged_dir"

    # Run nextflow from the output directory so .nextflow/ and .nextflow.log
    # are created there instead of wherever the script was launched from.
    if (cd "$MCMICRO_OUTPUT_BASE" && nextflow run \
        -c "$SINGULARITY_CONFIG" \
        labsyspharm/mcmicro \
        --in "$staged_dir" \
        -profile singularity \
        --params "$PARAMS_FILE" \
        -work-dir "$MCMICRO_WORK_DIR" \
        -with-report "$output_report") >> "$LOG_FILE" 2>&1; then
        log_success "MCMICRO completed for $roi_name"
        log_info "  Report saved to: $output_report"
        return 0
    else
        log_error "MCMICRO failed for $roi_name"
        return 1
    fi
}

#==============================================================================
# CLEANUP FUNCTIONS
#==============================================================================

cleanup_staged() {
    local staged_dir="$1"
    local roi_name="$2"

    log_info "Cleaning up staged data for: $roi_name"

    if [ -d "$staged_dir/raw" ]; then
        if rm -rf "$staged_dir/raw"; then
            log_success "Staged data deleted: $staged_dir/raw"
            return 0
        else
            log_error "Failed to delete staged data: $staged_dir/raw"
            return 1
        fi
    else
        log_warning "Staged directory not found: $staged_dir/raw"
        return 1
    fi
}

cleanup_mcmicro_work() {
    local roi_name="$1"

    log_info "Cleaning up MCMICRO work directory for: $roi_name"

    if [ -d "$MCMICRO_WORK_DIR" ]; then
        local work_size
        work_size=$(du -sh "$MCMICRO_WORK_DIR" 2>/dev/null | cut -f1)
        log_info "  Work directory size: $work_size"

        if rm -rf "$MCMICRO_WORK_DIR"/*; then
            log_success "MCMICRO work directory cleaned: $MCMICRO_WORK_DIR"
            return 0
        else
            log_error "Failed to clean MCMICRO work directory: $MCMICRO_WORK_DIR"
            return 1
        fi
    else
        log_warning "MCMICRO work directory not found: $MCMICRO_WORK_DIR"
        return 0
    fi
}

#==============================================================================
# ROI PROCESSING
#==============================================================================

process_roi() {
    local roi_path="$1"
    local roi_name="$2"
    local staged_dir="${STAGING_BASE_DIR}/${roi_name}_staged"

    log_msg "Processing ROI: $roi_name"

    # Stage the ROI. "Complete" = markers.csv present in the resolved input dir;
    # a partial stage (no markers.csv) is redone. --recompute always re-stages.
    if is_staging_complete "$staged_dir" && [ "$RECOMPUTE" = false ]; then
        log_msg "  Staging complete (markers.csv present), skipping: $staged_dir"
    else
        # About to (re)stage — clear any partial/previous output so macsima2mc
        # starts clean instead of appending to a half-staged dir.
        if [ -d "$staged_dir" ]; then
            if [ "$DRY_RUN" = true ]; then
                log_msg "  Would remove existing staged dir before (re)staging: $staged_dir"
            elif [ "$RECOMPUTE" = true ]; then
                log_info "  Removing existing staged dir for recompute: $staged_dir"
                rm -rf "$staged_dir"
            else
                log_warning "  Staging incomplete (no markers.csv) — removing partial staged dir before re-staging: $staged_dir"
                rm -rf "$staged_dir"
            fi
        fi
        if [ "$USE_SCAN_DAPI" = true ]; then
            if [ "$DRY_RUN" = true ]; then
                log_msg "  Would swap scan DAPI into Cycle1"
                log_msg "  Would duplicate Cycle1 as Cycle999 for alignment"
            else
                swap_scan_dapi_into_cycle1 "$roi_path"
                duplicate_cycle1_as_cycle999 "$roi_path"
            fi
        fi
        local staging_failed=false
        if ! stage_roi "$roi_path" "$staged_dir" "$roi_name"; then
            staging_failed=true
        fi
        if [ "$USE_SCAN_DAPI" = true ] && [ "$DRY_RUN" = false ]; then
            restore_cycle1_dapi "$roi_path"
            clean_markers_background "$staged_dir"
            mark_cycle1_markers_removed "$staged_dir"
        fi
        if [ "$staging_failed" = true ]; then
            if [ "$DRY_RUN" = false ]; then
                log_error "FAILED - Staging failed for $roi_name"
                echo "$roi_name,STAGING_FAILED,$(date '+%Y-%m-%d %H:%M:%S')" >> "${LOG_FILE%.log}_summary.csv"
            fi
            return 1
        fi
    fi

    # Keep only the requested cycles' DAPI (cycle 1 always kept). Runs whether the ROI
    # was freshly staged or staging was skipped; idempotent on re-run.
    if [ -n "$KEEP_DAPI_CYCLES" ]; then
        if [ "$DRY_RUN" = true ]; then
            log_msg "  Would additionally keep DAPI of cycles $KEEP_DAPI_CYCLES in final stack (as DAPI_cycle<N>)"
        else
            keep_dapi_cycles "$staged_dir" "$KEEP_DAPI_CYCLES"
        fi
    fi

    # Resolve the actual directory to feed into MCMICRO
    local mcmicro_input_dir
    mcmicro_input_dir=$(get_highest_exposure_dir "$staged_dir")
    log_msg "  Resolved MCMICRO input: $mcmicro_input_dir"

    # Skip MCMICRO if it already produced registration output (unless --recompute).
    if [ "$RECOMPUTE" = false ] && is_mcmicro_complete "$mcmicro_input_dir"; then
        log_success "  MCMICRO already complete (registration/ present), skipping: $roi_name"
        if [ "$DRY_RUN" = false ]; then
            echo "$roi_name,ALREADY_COMPLETE,$(date '+%Y-%m-%d %H:%M:%S')" >> "${LOG_FILE%.log}_summary.csv"
        fi
        return 0
    fi

    # Run MCMICRO
    if ! run_mcmicro "$mcmicro_input_dir" "$roi_name"; then
        if [ "$DRY_RUN" = false ]; then
            log_error "FAILED - MCMICRO failed for $roi_name"
            echo "$roi_name,MCMICRO_FAILED,$(date '+%Y-%m-%d %H:%M:%S')" >> "${LOG_FILE%.log}_summary.csv"
            cleanup_staged "$staged_dir" "$roi_name"
            cleanup_mcmicro_work "$roi_name"
        fi
        return 1
    fi

    if [ "$DRY_RUN" = false ]; then
        log_success "COMPLETED - Successfully processed $roi_name"
        echo "$roi_name,SUCCESS,$(date '+%Y-%m-%d %H:%M:%S')" >> "${LOG_FILE%.log}_summary.csv"
        if [ "$CLEANUP_STAGED" = true ]; then
            cleanup_staged "$staged_dir" "$roi_name"
        fi
    else
        if [ "$CLEANUP_STAGED" = true ]; then
            log_msg "  Would clean up staged data: $staged_dir/raw"
        fi
    fi
    return 0
}

#==============================================================================
# EXPERIMENT PROCESSING
#==============================================================================

process_experiment() {
    local exp_dir="$1"
    local exp_name="$2"

    for data_dir in "$exp_dir"/*/; do
        [ -d "$data_dir" ] || continue

        # Search for RawData directory (1-2 levels deep)
        local raw_data_dir
        raw_data_dir=$(find "$data_dir" -maxdepth 2 -type d -name "RawData" -print -quit)

        if [ -z "$raw_data_dir" ]; then
            continue
        fi

        log_msg "  RawData: $raw_data_dir"

        for r_folder in "$raw_data_dir"/R*/; do
            [ -d "$r_folder" ] || continue

            local r_name
            r_name=$(basename "$r_folder")

            if [ "$r_name" = "R0" ]; then
                log_msg "    Skipping: $r_name"
                continue
            fi

            log_msg "    Rack: $r_name"

            for a_folder in "$r_folder"/*/; do
                [ -d "$a_folder" ] || continue

                local a_name
                a_name=$(basename "$a_folder")
                log_msg "      Position: $a_name"

                for roi_folder in "$a_folder"/ROI*/; do
                    [ -d "$roi_folder" ] || continue

                    local roi_name
                    roi_name=$(basename "$roi_folder")

                    if [ "$roi_name" = "ROI0" ]; then
                        log_msg "        Skipping: $roi_name"
                        continue
                    fi

                    TOTAL_ROIS=$((TOTAL_ROIS + 1))
                    local roi_identifier="${exp_name}_${r_name}_${a_name}_${roi_name}"

                    if process_roi "$roi_folder" "$roi_identifier"; then
                        PROCESSED_ROIS=$((PROCESSED_ROIS + 1))
                    else
                        FAILED_ROIS=$((FAILED_ROIS + 1))
                    fi
                done
            done
        done
    done
}

#==============================================================================
# SUMMARY FUNCTIONS
#==============================================================================

print_config_summary() {
    log_msg "=========================================="
    if [ "$DRY_RUN" = true ]; then
        echo "[DRY-RUN] DRY RUN MODE - No operations will be performed"
    else
        log_info "MACSima Pipeline Started"
        log_info "Log file: $LOG_FILE"
        log_info "Summary file: ${LOG_FILE%.log}_summary.csv"
    fi
    log_msg "=========================================="
    log_msg "Root directory:     $ROOT_DIR"
    log_msg "Staging container:  $STAGING_CONTAINER"
    log_msg "Staging directory:  $STAGING_BASE_DIR"
    log_msg "Singularity config: $SINGULARITY_CONFIG"
    log_msg "Params file:        $PARAMS_FILE"
    log_msg "Output directory:   $MCMICRO_OUTPUT_BASE"
    log_msg "Reference marker:   $REFERENCE_MARKER"
    if [ -n "$SKIP_EXPERIMENTS" ]; then
        log_msg "Skipping experiments: $SKIP_EXPERIMENTS"
    fi
    if [ -n "$EXPERIMENT_FILTER" ]; then
        log_msg "Experiment filter:  $EXPERIMENT_FILTER"
    fi
    if [ "$USE_SCAN_DAPI" = true ]; then
        log_msg "Swap scan DAPI into Cycle1: YES"
    fi
    if [ "$USE_HIGHEST_EXPOSURE" = true ]; then
        log_msg "Using highest exposure only (-he for staging, highest folder for MCMICRO)"
    else
        log_msg "Using all exposures (MCMICRO still uses highest exposure folder if multiple exist)"
    fi
    if [ "$CLEANUP_STAGED" = true ]; then
        log_msg "Cleanup staged data: YES (after successful processing)"
    fi
    if [ "$RECOMPUTE" = true ]; then
        log_msg "Recompute mode:     YES (force re-stage and re-process)"
    fi
    if [ -n "$KEEP_DAPI_CYCLES" ]; then
        log_msg "Keep extra DAPI in final stack for cycles: $KEEP_DAPI_CYCLES (as DAPI_cycle<N>; reference cycle 1 always kept)"
    fi
    echo ""

    if [ "$DRY_RUN" = false ]; then
        echo "ROI_Name,Status,Timestamp" > "${LOG_FILE%.log}_summary.csv"
    fi
}

print_final_summary() {
    echo ""
    log_msg "=========================================="
    if [ "$DRY_RUN" = true ]; then
        echo "[DRY-RUN] Dry Run Complete"
    else
        log_info "Pipeline Complete"
    fi
    log_msg "=========================================="
    log_msg "Total ROIs found: $TOTAL_ROIS"
    if [ "$DRY_RUN" = false ]; then
        log_info "Successfully processed: $PROCESSED_ROIS"
        log_info "Failed: $FAILED_ROIS"
    fi
    if [ $SKIPPED_EXPERIMENTS -gt 0 ]; then
        log_msg "Experiments skipped: $SKIPPED_EXPERIMENTS"
    fi
    if [ "$DRY_RUN" = true ]; then
        echo ""
        echo "To run the actual pipeline, execute without --dry-run flag"
    else
        log_info "Log file: $LOG_FILE"
        log_info "Summary file: ${LOG_FILE%.log}_summary.csv"
        if [ $FAILED_ROIS -gt 0 ]; then
            exit 1
        fi
    fi
}

#==============================================================================
# MAIN
#==============================================================================

main() {
    validate_config

    if [ "$DRY_RUN" = false ]; then
        mkdir -p "$STAGING_BASE_DIR"
        mkdir -p "$MCMICRO_OUTPUT_BASE"
    fi

    print_config_summary

    for exp_dir in "$ROOT_DIR"/*/; do
        [ -d "$exp_dir" ] || continue

        local exp_name
        exp_name=$(basename "$exp_dir")

        if ! matches_experiment_filter "$exp_name"; then
            log_msg "Skipping experiment: $exp_name (does not match filter)"
            SKIPPED_EXPERIMENTS=$((SKIPPED_EXPERIMENTS + 1))
            continue
        fi

        if should_skip_experiment "$exp_name"; then
            log_msg "Skipping experiment: $exp_name (in skip list)"
            SKIPPED_EXPERIMENTS=$((SKIPPED_EXPERIMENTS + 1))
            continue
        fi

        log_msg "Experiment: $exp_name"
        process_experiment "$exp_dir" "$exp_name"
    done

    print_final_summary
    exit 0
}

#==============================================================================
# COMMAND LINE ARGUMENT PARSING
#==============================================================================

while [[ $# -gt 0 ]]; do
    case $1 in
        --root-dir)
            ROOT_DIR="$2"
            shift 2
            ;;
        --container)
            STAGING_CONTAINER="$2"
            shift 2
            ;;
        --staging-dir)
            STAGING_BASE_DIR="$2"
            shift 2
            ;;
        --singularity-config)
            SINGULARITY_CONFIG="$2"
            shift 2
            ;;
        --params)
            PARAMS_FILE="$2"
            shift 2
            ;;
        --output-dir)
            MCMICRO_OUTPUT_BASE="$2"
            shift 2
            ;;
        --reference-marker)
            REFERENCE_MARKER="$2"
            shift 2
            ;;
        --dry-run|-d)
            DRY_RUN=true
            shift
            ;;
        --skip-exp|--skip-experiments)
            SKIP_EXPERIMENTS="$2"
            shift 2
            ;;
        --experiment-filter)
            EXPERIMENT_FILTER="$2"
            shift 2
            ;;
        --use-scan-dapi)
            USE_SCAN_DAPI=true
            shift
            ;;
        --highest-exposure-only|-he)
            USE_HIGHEST_EXPOSURE=true
            shift
            ;;
        --cleanup-staged)
            CLEANUP_STAGED=true
            shift
            ;;
        --recompute)
            RECOMPUTE=true
            shift
            ;;
        --keep-dapi)
            KEEP_DAPI_CYCLES="$2"
            shift 2
            ;;
        --help|-h)
            cat <<EOF
MACSima Pipeline - Staging and MCMICRO Processing

Usage: $0 [OPTIONS]

Required arguments (or set via environment variables):
  --root-dir DIR              Root directory containing experiment folders
                              (env: MCMICRO_ROOT_DIR)
  --container FILE            Path to macsima2mc Singularity container
                              (env: MCMICRO_STAGING_CONTAINER)
  --staging-dir DIR           Base directory for staged output
                              (env: MCMICRO_STAGING_DIR)
  --singularity-config FILE   Nextflow Singularity config file
                              (env: MCMICRO_SINGULARITY_CONFIG)
  --params FILE               MCMICRO parameters YAML file
                              (env: MCMICRO_PARAMS_FILE)
  --output-dir DIR            Base directory for MCMICRO output and logs
                              (env: MCMICRO_OUTPUT_DIR)

Optional arguments:
  --reference-marker NAME     Reference marker for staging (default: DAPI)
  --dry-run, -d               Run in dry-run mode (preview without executing)
  --skip-exp <list>           Skip specific experiments (comma-separated)
  --skip-experiments <list>   Same as --skip-exp
  --experiment-filter REGEX   Only process experiments matching REGEX (bash regex)
  --use-scan-dapi             Swap scan DAPI tiles into Cycle1 before staging (backs up originals, restores after)
  --highest-exposure-only     Use only highest exposure in staging (-he flag)
  -he                         Same as --highest-exposure-only
  --cleanup-staged            Delete staged raw data after successful processing
  --recompute                 Force re-staging and re-processing (ignore existing data)
  --keep-dapi <cycles>        Additionally retain these cycles' DAPI in the final stitched
                              stack (comma-separated), each renamed DAPI_cycle<N>. The
                              reference DAPI (cycle 1) is always kept. Does not affect alignment.
  --help, -h                  Show this help message

Examples:
  # Full pipeline run with all required arguments
  $0 --root-dir /data/CRC_study \\
     --container /opt/macsima2mc.sif \\
     --staging-dir /data/staged \\
     --singularity-config /etc/singularity.config \\
     --params /etc/mcmicro_params.yml \\
     --output-dir /results/mcmicro

  # Preview what would be processed
  $0 --dry-run --root-dir /data/CRC_study ...

  # Skip specific experiments
  $0 --skip-exp EXP_001,EXP_003 ...

  # Only process folders starting with EXP_
  $0 --experiment-filter "^EXP_" ...

  # Use a different nuclear marker
  $0 --reference-marker "Hoechst" ...

  # Clean up staged data after each successful ROI
  $0 --cleanup-staged ...

  # Force full re-processing of all ROIs
  $0 --recompute ...

  # Using environment variables
  export MCMICRO_ROOT_DIR=/data/CRC_study
  export MCMICRO_STAGING_CONTAINER=/opt/macsima2mc.sif
  export MCMICRO_STAGING_DIR=/data/staged
  export MCMICRO_SINGULARITY_CONFIG=/etc/singularity.config
  export MCMICRO_PARAMS_FILE=/etc/mcmicro_params.yml
  export MCMICRO_OUTPUT_DIR=/results/mcmicro
  $0 --dry-run
EOF
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Run main function
main
