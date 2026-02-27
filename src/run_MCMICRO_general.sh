#!/usr/bin/bash

################################################################################
# MACSima Pipeline - Staging and MCMICRO Processing
# Processes ROIs from MACSima device through staging and MCMICRO analysis
################################################################################

set -euo pipefail

#==============================================================================
# DEFAULTS
#==============================================================================

DRY_RUN=false
SKIP_EXPERIMENTS=""
USE_OVERVIEW_SCAN_DAPI=false
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

LOG_FILE="$(pwd)/macsima_pipeline_$(date +%Y%m%d_%H%M%S).log"

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
}

#==============================================================================
# HELPER FUNCTIONS
#==============================================================================

should_skip_experiment() {
    local exp_name="$1"

    if [ -z "$SKIP_EXPERIMENTS" ]; then
        return 1
    fi

    IFS=',' read -ra SKIP_LIST <<< "$SKIP_EXPERIMENTS"

    for skip_exp in "${SKIP_LIST[@]}"; do
        skip_exp=$(echo "$skip_exp" | xargs)
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

inject_scan2_as_cycle0() {
    local roi_path="$1"
    local scan2_dir="${roi_path}/3_Scan2"
    local cycle0_dir="${roi_path}/5_Cycle0"

    if [ ! -d "$scan2_dir" ]; then
        log_warning "3_Scan2 directory not found in $roi_path — skipping Cycle0 injection"
        return 1
    fi

    if [ -d "$cycle0_dir" ]; then
        log_info "5_Cycle0 already exists in $roi_path — skipping injection (idempotent)"
        return 0
    fi

    mkdir -p "$cycle0_dir"

    local file_count=0
    for tif in "$scan2_dir"/*.tif; do
        [ -f "$tif" ] || continue
        cp "$tif" "$cycle0_dir/CYC-000_$(basename "$tif")"
        file_count=$((file_count + 1))
    done

    log_info "Injected $file_count .tif files from 3_Scan2 as Cycle0 in $roi_path"
    return 0
}

cleanup_cycle0() {
    local roi_path="$1"
    local cycle0_dir="${roi_path}/5_Cycle0"

    if [ -d "$cycle0_dir" ]; then
        rm -rf "$cycle0_dir"
        log_info "Cleaned up injected Cycle0 directory: $cycle0_dir"
    fi
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
        if [ "$USE_OVERVIEW_SCAN_DAPI" = true ]; then
            if [ "$DRY_RUN" = true ]; then
                log_msg "  Would inject 3_Scan2 as Cycle0"
            else
                inject_scan2_as_cycle0 "$roi_path"
            fi
        fi
        if ! stage_roi "$roi_path" "$staged_dir" "$roi_name"; then
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
        if [ "$USE_OVERVIEW_SCAN_DAPI" = true ]; then
            cleanup_cycle0 "$roi_path"
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
    if [ "$USE_OVERVIEW_SCAN_DAPI" = true ]; then
        log_msg "Using 3_Scan2 DAPI as Cycle 0: YES"
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
        --use-overview-scan-dapi)
            USE_OVERVIEW_SCAN_DAPI=true
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
  --use-overview-scan-dapi    Use DAPI from 3_Scan2 as Cycle 0 (copies folder, renames files, cleans up on success)
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
