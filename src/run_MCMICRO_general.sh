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
USE_BACKGROUND_ALIGN=false
CLEANUP_BACKGROUND=false
USE_HIGHEST_EXPOSURE=true
REFERENCE_MARKER="DAPI"
EXPERIMENT_FILTER=""
CLEANUP_STAGED=false
RECOMPUTE=false

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

    # Cleanup-only mode needs only staging and output dirs
    if [ "$CLEANUP_BACKGROUND" = true ]; then
        if [ -z "$STAGING_BASE_DIR" ] || [ -z "$MCMICRO_OUTPUT_BASE" ]; then
            echo "ERROR: --staging-dir and --output-dir required for --cleanup-background" >&2
            exit 1
        fi
        MCMICRO_WORK_DIR="${MCMICRO_OUTPUT_BASE}/work"
        LOG_FILE="${MCMICRO_OUTPUT_BASE}/macsima_pipeline_$(date +%Y%m%d_%H%M%S).log"
        return 0
    fi

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

    if [ "$USE_SCAN_DAPI" = true ] && [ "$USE_BACKGROUND_ALIGN" = true ]; then
        echo "ERROR: --use-scan-dapi and --use-background-align are mutually exclusive" >&2
        errors=$((errors + 1))
    fi

    if [ $errors -gt 0 ]; then
        echo "" >&2
        echo "Use --help for usage information" >&2
        exit 1
    fi

    # Override reference marker for background alignment
    if [ "$USE_BACKGROUND_ALIGN" = true ]; then
        REFERENCE_MARKER="BGREF"
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

is_already_staged() {
    local staged_dir="$1"
    if [ -d "$staged_dir" ] && [ -n "$(ls -A "$staged_dir" 2>/dev/null)" ]; then
        return 0
    fi
    return 1
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

inject_background_ref() {
    local roi_path="$1"
    local scan_dir="${roi_path}/3_Scan2"
    local backup_dir="${roi_path}/.bgref_backup"

    # Find the Cycle1 directory
    local cycle1_dir
    cycle1_dir=$(find "$roi_path" -maxdepth 1 -type d -name '*_Cycle1' | head -1)

    if [ ! -d "$scan_dir" ]; then
        log_error "3_Scan2 directory not found in $roi_path — cannot inject background ref"
        return 1
    fi

    if [ -z "$cycle1_dir" ]; then
        log_error "No *_Cycle1 directory found in $roi_path — cannot inject background ref"
        return 1
    fi

    # If stale backup exists (interrupted previous run), restore first
    if [ -d "$backup_dir" ]; then
        log_warning "Found stale .bgref_backup in $roi_path — restoring before re-injecting"
        restore_background_ref "$roi_path"
    fi

    mkdir -p "$backup_dir"

    #--- Cycle1: Inject highest-exposure PE from 3_Scan2 as BGREF ---

    # Find all PE files in 3_Scan2
    local pe_files=()
    while IFS= read -r f; do
        pe_files+=("$f")
    done < <(find "$scan_dir" -maxdepth 1 -name '*_D-PE_*.tif' -type f 2>/dev/null)

    if [ ${#pe_files[@]} -eq 0 ]; then
        log_error "No PE files found in $scan_dir — cannot inject background ref"
        rmdir "$backup_dir" 2>/dev/null || true
        return 1
    fi

    # Determine highest exposure among PE files
    local highest_exp="0"
    for f in "${pe_files[@]}"; do
        local exp_val
        exp_val=$(basename "$f" | grep -oP 'EXP-\K[0-9]+(\.[0-9]+)?')
        if [ -n "$exp_val" ]; then
            if awk "BEGIN {exit !($exp_val > $highest_exp)}"; then
                highest_exp="$exp_val"
            fi
        fi
    done

    # Copy highest-exposure PE tiles into Cycle1 as BGREF
    local injected_cycle1=0
    for f in "${pe_files[@]}"; do
        local bname
        bname=$(basename "$f")
        # Only use the highest exposure tiles
        local exp_val
        exp_val=$(echo "$bname" | grep -oP 'EXP-\K[0-9]+(\.[0-9]+)?')
        if [ "$exp_val" != "$highest_exp" ]; then
            continue
        fi
        # Extract ROI, field, R, W identifiers from the scan filename
        local r_id w_id roi_id f_id
        r_id=$(echo "$bname" | grep -oP 'R-\K[0-9]+')
        w_id=$(echo "$bname" | grep -oP 'W-\K[A-Z][0-9]+')
        roi_id=$(echo "$bname" | grep -oP 'ROI-\K[0-9]+')
        f_id=$(echo "$bname" | grep -oP 'F-\K[0-9]+')
        # Build the BGREF filename for Cycle1
        local new_name="CYC-001_SCN-002_ST-S_R-${r_id}_W-${w_id}_ROI-${roi_id}_F-${f_id}_A-BGREF_C-Reference_D-PE_EXP-${highest_exp}.tif"
        cp "$f" "$cycle1_dir/$new_name"
        # Also create ST-B counterpart (SCN-001) so macsima2mc can pair stain+bleach
        local new_name_b="CYC-001_SCN-001_ST-B_R-${r_id}_W-${w_id}_ROI-${roi_id}_F-${f_id}_A-BGREF_C-Reference_D-PE_EXP-${highest_exp}.tif"
        cp "$f" "$cycle1_dir/$new_name_b"
        injected_cycle1=$((injected_cycle1 + 1))
    done

    log_info "Injected $injected_cycle1 BGREF tiles into Cycle1 from 3_Scan2 PE (EXP-${highest_exp})"

    #--- Cycle999 + other cycles: Inject highest-exposure ST-B as BGREF ---

    local total_injected_other=0
    for cycle_dir in "$roi_path"/*_Cycle*/; do
        [ -d "$cycle_dir" ] || continue
        local cycle_basename
        cycle_basename=$(basename "$cycle_dir")
        # Skip Cycle1 (already handled above)
        if [[ "$cycle_basename" == *_Cycle1 ]]; then
            continue
        fi
        # Extract cycle number from directory name (e.g. "999" from "999_Cycle999")
        local cycle_num
        cycle_num=$(echo "$cycle_basename" | grep -oP 'Cycle\K[0-9]+')
        local cyc_padded
        cyc_padded=$(printf '%03d' "$cycle_num")

        # Determine where to search for ST-B files
        # Cycle999 is a duplicate of Cycle1 stain files and has no ST-B; use Cycle1's
        local stb_search_dir="$cycle_dir"
        if [[ "$cycle_basename" == *_Cycle999 ]]; then
            stb_search_dir="$cycle1_dir"
        fi

        # Find ST-B files with filter priority: PE > FITC > APC
        local chosen_filter=""
        local stb_files=()
        for filter_name in PE FITC APC; do
            local candidates=()
            while IFS= read -r f; do
                candidates+=("$f")
            done < <(find "$stb_search_dir" -maxdepth 1 -name "*_ST-B_*_D-${filter_name}_*.tif" -type f 2>/dev/null)
            if [ ${#candidates[@]} -gt 0 ]; then
                chosen_filter="$filter_name"
                stb_files=("${candidates[@]}")
                break
            fi
        done

        if [ -z "$chosen_filter" ]; then
            log_warning "No suitable ST-B files found in $cycle_dir — skipping BGREF injection for this cycle"
            continue
        fi

        # Determine highest exposure among chosen filter's ST-B files
        local he_stb="0"
        for f in "${stb_files[@]}"; do
            local exp_val
            exp_val=$(basename "$f" | grep -oP 'EXP-\K[0-9]+(\.[0-9]+)?')
            if [ -n "$exp_val" ]; then
                if awk "BEGIN {exit !($exp_val > $he_stb)}"; then
                    he_stb="$exp_val"
                fi
            fi
        done

        # Copy highest-exposure ST-B tiles as BGREF
        for f in "${stb_files[@]}"; do
            local bname
            bname=$(basename "$f")
            local exp_val
            exp_val=$(echo "$bname" | grep -oP 'EXP-\K[0-9]+(\.[0-9]+)?')
            if [ "$exp_val" != "$he_stb" ]; then
                continue
            fi
            local r_id w_id roi_id f_id
            r_id=$(echo "$bname" | grep -oP 'R-\K[0-9]+')
            w_id=$(echo "$bname" | grep -oP 'W-\K[A-Z][0-9]+')
            roi_id=$(echo "$bname" | grep -oP 'ROI-\K[0-9]+')
            f_id=$(echo "$bname" | grep -oP 'F-\K[0-9]+')
            local new_name="CYC-${cyc_padded}_SCN-002_ST-S_R-${r_id}_W-${w_id}_ROI-${roi_id}_F-${f_id}_A-BGREF_C-Reference_D-${chosen_filter}_EXP-${he_stb}.tif"
            cp "$f" "$cycle_dir/$new_name"
            # Also create ST-B counterpart (SCN-001) so macsima2mc can pair stain+bleach
            local new_name_b="CYC-${cyc_padded}_SCN-001_ST-B_R-${r_id}_W-${w_id}_ROI-${roi_id}_F-${f_id}_A-BGREF_C-Reference_D-${chosen_filter}_EXP-${he_stb}.tif"
            cp "$f" "$cycle_dir/$new_name_b"
            total_injected_other=$((total_injected_other + 1))
        done
    done

    log_info "Injected $total_injected_other BGREF tiles into other cycles (Cycle999 + remaining)"

    # Move Cycle1 ST-B (bleach) files to backup to prevent staging issues
    # Done AFTER the other-cycles loop so Cycle999 can still find Cycle1's ST-B files
    local bleach_moved=0
    for bleach_file in "$cycle1_dir"/*_ST-B_*.tif; do
        [ -f "$bleach_file" ] || continue
        # Don't move injected BGREF ST-B files — they need to stay for macsima2mc pairing
        [[ "$(basename "$bleach_file")" == *_A-BGREF_* ]] && continue
        mv "$bleach_file" "$backup_dir/"
        bleach_moved=$((bleach_moved + 1))
    done
    if [ $bleach_moved -gt 0 ]; then
        log_info "Moved $bleach_moved Cycle1 bleach (ST-B) files to .bgref_backup"
    fi

    return 0
}

restore_background_ref() {
    local roi_path="$1"
    local backup_dir="${roi_path}/.bgref_backup"

    # Remove all injected BGREF files from all cycle directories
    local removed=0
    for cycle_dir in "$roi_path"/*_Cycle*/; do
        [ -d "$cycle_dir" ] || continue
        for bgref_file in "$cycle_dir"/*_A-BGREF_C-Reference_*.tif; do
            [ -f "$bgref_file" ] || continue
            rm "$bgref_file"
            removed=$((removed + 1))
        done
    done
    if [ $removed -gt 0 ]; then
        log_info "Removed $removed injected BGREF files from cycle directories"
    fi

    # Restore Cycle1 ST-B files from backup
    if [ -d "$backup_dir" ]; then
        local cycle1_dir
        cycle1_dir=$(find "$roi_path" -maxdepth 1 -type d -name '*_Cycle1' | head -1)
        if [ -n "$cycle1_dir" ]; then
            local restored=0
            for backup_file in "$backup_dir"/*.tif; do
                [ -f "$backup_file" ] || continue
                mv "$backup_file" "$cycle1_dir/"
                restored=$((restored + 1))
            done
            log_info "Restored $restored Cycle1 bleach files from .bgref_backup"
        else
            log_error "Cannot restore bleach files: no *_Cycle1 directory found in $roi_path"
        fi
        rm -rf "$backup_dir"
    fi

    # Remove Cycle999 directory
    local cycle999_dir="${roi_path}/999_Cycle999"
    if [ -d "$cycle999_dir" ]; then
        rm -rf "$cycle999_dir"
        log_info "Removed Cycle999 duplicate directory ($roi_path)"
    fi

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

mark_background_markers_removed() {
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
    # All BGREF channels are alignment helpers — remove them
    if marker == 'BGREF':
        r['remove'] = 'TRUE'
        marked += 1
    # Cycle 1 non-DAPI markers are misaligned (acquired at original positions, not Scan positions)
    elif cycle == 1 and marker != 'DAPI':
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
    print(f'Marked {marked} channels as remove=TRUE in {path} (BGREF + Cycle1 markers + Cycle999 DAPI)')
" "$markers_csv" 2>&1 | while read -r line; do log_info "$line"; done
    done < <(find "$staged_dir" -name "markers.csv" -type f)
}

cleanup_background_markers() {
    local count=0
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
    marker = r.get('marker_name', '')
    if marker == 'BGREF' and r.get('remove', '').upper() != 'TRUE':
        r['remove'] = 'TRUE'
        marked += 1

if marked > 0:
    with open(path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f'Marked {marked} BGREF channels as remove=TRUE in {path}')
else:
    print(f'No unmarked BGREF channels found in {path}')
" "$markers_csv" 2>&1 | while read -r line; do log_info "$line"; done
        count=$((count + 1))
    done < <(find "$STAGING_BASE_DIR" -name "markers.csv" -type f)
    log_info "Processed $count markers.csv files for BGREF cleanup"
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

    # Stage the ROI (skip if already staged, unless --recompute)
    if is_already_staged "$staged_dir" && [ "$RECOMPUTE" = false ]; then
        log_msg "  Staging already exists, skipping: $staged_dir"
    else
        if [ "$RECOMPUTE" = true ] && [ -d "$staged_dir" ]; then
            if [ "$DRY_RUN" = true ]; then
                log_msg "  Would remove existing staged dir for recompute: $staged_dir"
            else
                log_info "  Removing existing staged dir for recompute: $staged_dir"
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
        if [ "$USE_BACKGROUND_ALIGN" = true ]; then
            if [ "$DRY_RUN" = true ]; then
                log_msg "  Would duplicate Cycle1 as Cycle999 for alignment"
                log_msg "  Would inject background reference (BGREF) into all cycles"
            else
                duplicate_cycle1_as_cycle999 "$roi_path"
                inject_background_ref "$roi_path"
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
        if [ "$USE_BACKGROUND_ALIGN" = true ] && [ "$DRY_RUN" = false ]; then
            restore_background_ref "$roi_path"
            clean_markers_background "$staged_dir"
            mark_background_markers_removed "$staged_dir"
        fi
        if [ "$staging_failed" = true ]; then
            if [ "$DRY_RUN" = false ]; then
                log_error "FAILED - Staging failed for $roi_name"
                echo "$roi_name,STAGING_FAILED,$(date '+%Y-%m-%d %H:%M:%S')" >> "${LOG_FILE%.log}_summary.csv"
            fi
            return 1
        fi
    fi

    # Resolve the actual directory to feed into MCMICRO
    local mcmicro_input_dir
    mcmicro_input_dir=$(get_highest_exposure_dir "$staged_dir")
    log_msg "  Resolved MCMICRO input: $mcmicro_input_dir"

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
    if [ "$USE_BACKGROUND_ALIGN" = true ]; then
        log_msg "Background-based alignment: YES (using BGREF from ST-B/PE autofluorescence)"
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

    # Standalone BGREF cleanup mode — mark and exit
    if [ "$CLEANUP_BACKGROUND" = true ]; then
        mkdir -p "$MCMICRO_OUTPUT_BASE"
        log_info "Running standalone BGREF cleanup on $STAGING_BASE_DIR"
        cleanup_background_markers
        exit 0
    fi

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
        --use-background-align)
            USE_BACKGROUND_ALIGN=true
            shift
            ;;
        --cleanup-background)
            CLEANUP_BACKGROUND=true
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
  --use-background-align      Align cycles using ST-B autofluorescence instead of DAPI (mutually exclusive with --use-scan-dapi)
  --cleanup-background        Standalone: mark BGREF as remove=TRUE in all staged markers.csv (only needs --staging-dir and --output-dir)
  --highest-exposure-only     Use only highest exposure in staging (-he flag)
  -he                         Same as --highest-exposure-only
  --cleanup-staged            Delete staged raw data after successful processing
  --recompute                 Force re-staging and re-processing (ignore existing data)
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
