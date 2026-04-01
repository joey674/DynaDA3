#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
INPUTS_DIR="${REPO_ROOT}/inputs"
C3VD_ROOT="${INPUTS_DIR}/c3vd"
DOWNLOAD_DIR="${C3VD_ROOT}/_downloads"
TMP_DIR="${C3VD_ROOT}/_tmp"

# Set to 1 if you want to keep the downloaded archives in inputs/c3vd/_downloads.
KEEP_ARCHIVES=0

# Set to 1 to re-extract / overwrite already prepared items.
FORCE=0

# Default selection when the script is run without positional arguments.
# Fill this array with the item names you want, for example:
# SELECTED_DATASETS=("cecum_t1_a" "screening_t1" "full_model")
SELECTED_DATASETS=(
)

CATALOG=$(cat <<'EOF'
# key|filename|file_id|kind|size
# Registered videos
cecum_t1_a|cecum_t1_a.zip|14o6_4GQLZWx5dQq2L_drzmN_rlCT7Yhr|registered|2.86 GB
cecum_t1_b|cecum_t1_b.zip|1z3AHdnBH_YoCMnnTfDa8SNPQYIsvaBO3|registered|8.36 GB
cecum_t2_a|cecum_t2_a.zip|13XhJIev9memFtwUf_dnjJ7o8z6O_c-xW|registered|3.71 GB
cecum_t2_b|cecum_t2_b.zip|1ykYtQGiFesev5QLfz_avYuQ5a7Zs8kgF|registered|11.06 GB
cecum_t2_c|cecum_t2_c.zip|1tNoBLpPbrQexKlnOKMK2peERn9Rj_9Dp|registered|6.13 GB
cecum_t3_a|cecum_t3_a.zip|1Uw8uCRRDm_RrgkccGbiBXZHf9P-THM2Q|registered|6.80 GB
cecum_t4_a|cecum_t4_a.zip|1FC-dR__0LVb7WH02KpUx9TZVvvvN-Gyx|registered|5.04 GB
cecum_t4_b|cecum_t4_b.zip|11SbH2AZsuciTu3iGxdXCdQZky6uDTyS5|registered|4.41 GB
desc_t4_a|desc_t4_a.zip|1d9HDNg4-Og1cTWM-eIU5SWM2BMrMWhwQ|registered|1.24 GB
sigmoid_t1_a|sigmoid_t1_a.zip|19VGDuZ73OWNwM8eIgDkkZYBPJPQ5BD91|registered|5.20 GB
sigmoid_t2_a|sigmoid_t2_a.zip|16epAys428g9vBQgm611TElMyAXORo7rH|registered|4.22 GB
sigmoid_t3_a|sigmoid_t3_a.zip|1ZRU2KuHoc2XCbKSY_A1S-7BxEfKf9xPr|registered|4.58 GB
sigmoid_t3_b|sigmoid_t3_b.zip|1XfZFAQ5_Wxle8d5wSlOCumKSg4IP8wTv|registered|4.21 GB
trans_t1_a|trans_t1_a.zip|1urFuVo8ZalwPmsXEZg3xzhuqhpgWV8hw|registered|0.59 GB
trans_t1_b|trans_t1_b.zip|1hyjmd7vn86McE1nUnYCzvOm8LlyHLYwt|registered|5.07 GB
trans_t2_a|trans_t2_a.zip|1ylZWWtVlXfDx9dhPIeWJ1HqDHJZ3QKdH|registered|1.58 GB
trans_t2_b|trans_t2_b.zip|1vru228_TEgxT3aS90CmvOWsMB0RLnAxn|registered|0.97 GB
trans_t2_c|trans_t2_c.zip|12YpowbP6zhoO_Qx9UBwhfRLXNJN1EAu4|registered|1.83 GB
trans_t3_a|trans_t3_a.zip|1B4aeZfAqmUJgWr8e-2YAibUe4er30ncr|registered|1.83 GB
trans_t3_b|trans_t3_b.zip|1ZpbYcDVP-sCTjjQrDc303olFgsr2nA5J|registered|1.66 GB
trans_t4_a|trans_t4_a.zip|18qzXMifS54jAx29yROKXXxxZg0qo-iTz|registered|3.10 GB
trans_t4_b|trans_t4_b.zip|1C-nw6MR7sxssw3LS-GpiPmwBzEYhUCHN|registered|4.61 GB
# Screening videos
screening_t1|screening_t1.zip|18m3Z5zJtljor_AGmPW8OgO9fRuactuNk|screening|8.13 GB
screening_t2|screening_t2.zip|1kn_qevX7lLh3gWkiKAt3hgWFpy0P68s6|screening|7.09 GB
screening_t3|screening_t3.zip|1RmOnnjJBzCMwO5gPY4h3e4MpcDDOvOb6|screening|7.07 GB
screening_t4|screening_t4.zip|1sYps79WjJ0ETRtuWtHd_1zeuyPLtoMM7|screening|7.36 GB
# 3D models and molds
ascend_model|ascend_model.obj|1EsWdG3r9WYUpONzNKFbNVl_u9zpTQPCd|model|25.4 MB
ascend_mold|ascend_mold.zip|1coep9FX1AV0rWqFNpvxsKbDBRDUC7hQE|mold|18.7 MB
cecum_model|cecum_model.obj|1-1jVQn6lFHP22qyVO-fHA3f9Vvqiztpb|model|54.8 MB
cecum_mold|cecum_mold.zip|1SlDiYm6B7t0Ce2ztE9ThA34AE7dwu7UN|mold|24.9 MB
desc_model|desc_model.obj|1B8CNXxAM5t8C2W14YAnzWJYUu5Lmtit7|model|38.0 MB
desc_mold|desc_mold.zip|1kRPcB0IK_WdZtgp3b8SpD99vURd_NxAb|mold|26.6 MB
sigmoid_model|sigmoid_model.obj|1ySPh9w9Ix-fxh4r6cn3LYdu_A_1QWGAp|model|20.8 MB
sigmoid_mold|sigmoid_mold.zip|11k7AuwHVEae6EhQQ98R0tAwbbbtv8gdU|mold|42.2 MB
trans_model|trans_model.obj|1--_lUA1fRlIB6Ed1rf1FfqsSWxe9vKYQ|model|18.3 MB
trans_mold|trans_mold.zip|1Icvgs1DGQQ1fGS3bgAjbsA9NJYNQhGrB|mold|24.1 MB
full_model|full_model.obj|12gKUf4HvPYm3DiRxFmoum70--kkSewhq|model|194.8 MB
# Calibration files
cfhq190l_10x10mm_checkerboard_images|cfhq190l_10x10mm_checkerboard_images.zip|1ZTNyLx0p19U2Q3vl8dUe2YxxxA-9WisI|calibration|46 MB
cfhq190l_omnidirectional_params|cfhq190l_omnidirectional_params.zip|1gUA7mAM7DSD9oCvPH1hgQ0s2hhg1kOaC|calibration|unknown
EOF
)

declare -A ITEM_FILENAME
declare -A ITEM_ID
declare -A ITEM_KIND
declare -A ITEM_SIZE
ALL_ITEMS=()
REGISTERED_ITEMS=()
SCREENING_ITEMS=()
ASSET_ITEMS=()

parse_catalog() {
    local key filename file_id kind size
    while IFS='|' read -r key filename file_id kind size; do
        [[ -z "${key}" ]] && continue
        [[ "${key}" == \#* ]] && continue

        ITEM_FILENAME["${key}"]="${filename}"
        ITEM_ID["${key}"]="${file_id}"
        ITEM_KIND["${key}"]="${kind}"
        ITEM_SIZE["${key}"]="${size}"
        ALL_ITEMS+=("${key}")

        case "${kind}" in
            registered)
                REGISTERED_ITEMS+=("${key}")
                ;;
            screening)
                SCREENING_ITEMS+=("${key}")
                ;;
            model|mold|calibration)
                ASSET_ITEMS+=("${key}")
                ;;
            *)
                printf 'Unknown catalog kind: %s\n' "${kind}" >&2
                exit 1
                ;;
        esac
    done <<< "${CATALOG}"
}

usage() {
    cat <<EOF
Usage:
  bash scripts/download_c3vd.sh --list
  bash scripts/download_c3vd.sh cecum_t1_a screening_t1
  bash scripts/download_c3vd.sh all_registered

Options:
  --list           Show all available item names and exit.
  --force          Re-download / re-extract items that are already prepared.
  --keep-archives  Keep downloaded archives under inputs/c3vd/_downloads.
  -h, --help       Show this help.

Selection aliases:
  all              All videos + assets.
  all_registered   All 22 registered videos.
  all_screening    All 4 screening videos.
  all_assets       All models, molds, and calibration files.

Prepared layout:
  inputs/c3vd/<sequence>/raw       Original extracted content
  inputs/c3vd/<sequence>/images    Symlink to RGB frames for this repo's model
  inputs/c3vd/assets/...           Models / molds / calibration files
EOF
}

print_group() {
    local title="$1"
    shift
    local key

    printf '%s\n' "${title}"
    for key in "$@"; do
        printf '  %-38s %-12s %s\n' "${key}" "${ITEM_SIZE[${key}]}" "${ITEM_FILENAME[${key}]}"
    done
    printf '\n'
}

print_catalog() {
    print_group "Registered Videos" "${REGISTERED_ITEMS[@]}"
    print_group "Screening Videos" "${SCREENING_ITEMS[@]}"
    print_group "Assets" "${ASSET_ITEMS[@]}"
}

require_command() {
    local cmd="$1"
    if ! command -v "${cmd}" >/dev/null 2>&1; then
        printf 'Missing required command: %s\n' "${cmd}" >&2
        exit 1
    fi
}

check_dependencies() {
    require_command curl
    require_command unzip
    require_command find
    require_command mktemp
    require_command sort
    require_command head
    require_command cp
    require_command mv
    require_command rm
}

download_url_for_id() {
    local file_id="$1"
    printf 'https://drive.usercontent.google.com/download?id=%s&export=view&confirm=t' "${file_id}"
}

dedupe_lines() {
    declare -A seen=()
    local line
    while IFS= read -r line; do
        [[ -z "${line}" ]] && continue
        if [[ -z "${seen[${line}]:-}" ]]; then
            seen["${line}"]=1
            printf '%s\n' "${line}"
        fi
    done
}

expand_selection() {
    local token
    for token in "$@"; do
        case "${token}" in
            all)
                printf '%s\n' "${ALL_ITEMS[@]}"
                ;;
            all_registered)
                printf '%s\n' "${REGISTERED_ITEMS[@]}"
                ;;
            all_screening)
                printf '%s\n' "${SCREENING_ITEMS[@]}"
                ;;
            all_assets)
                printf '%s\n' "${ASSET_ITEMS[@]}"
                ;;
            *)
                printf '%s\n' "${token}"
                ;;
        esac
    done | dedupe_lines
}

find_image_root() {
    local base_dir="$1"
    local candidate=""

    if [[ -d "${base_dir}/rgb" ]]; then
        printf '%s\n' "${base_dir}/rgb"
        return 0
    fi

    if [[ -d "${base_dir}/images" ]]; then
        printf '%s\n' "${base_dir}/images"
        return 0
    fi

    if find "${base_dir}" -maxdepth 1 -type f \
        \( -iname '*.png' -o -iname '*.jpg' -o -iname '*.jpeg' -o -iname '*.bmp' -o \
        -iname '*.tif' -o -iname '*.tiff' -o -iname '*.webp' \) | grep -q .; then
        printf '%s\n' "${base_dir}"
        return 0
    fi

    candidate="$(find "${base_dir}" -mindepth 1 -maxdepth 2 -type d \( -iname 'rgb' -o -iname 'images' \) | sort | head -n 1 || true)"
    if [[ -n "${candidate}" ]]; then
        printf '%s\n' "${candidate}"
        return 0
    fi

    candidate="$(find "${base_dir}" -mindepth 1 -maxdepth 2 -type f \
        \( -iname '*.png' -o -iname '*.jpg' -o -iname '*.jpeg' -o -iname '*.bmp' -o \
        -iname '*.tif' -o -iname '*.tiff' -o -iname '*.webp' \) \
        -printf '%h\n' | sort -u | head -n 1 || true)"
    if [[ -n "${candidate}" ]]; then
        printf '%s\n' "${candidate}"
        return 0
    fi

    return 1
}

link_path() {
    local link_path="$1"
    local target_path="$2"

    rm -rf "${link_path}"
    ln -s "${target_path}" "${link_path}"
}

normalize_extracted_tree() {
    local staging_dir="$1"
    local final_dir="$2"
    local source_root="${staging_dir}"
    local path
    local entries=()

    rm -rf "${final_dir}"
    mkdir -p "${final_dir}"

    shopt -s dotglob nullglob
    for path in "${staging_dir}"/*; do
        [[ "$(basename "${path}")" == "__MACOSX" ]] && continue
        entries+=("${path}")
    done

    if (( ${#entries[@]} == 1 )) && [[ -d "${entries[0]}" ]]; then
        source_root="${entries[0]}"
        entries=()
        for path in "${source_root}"/*; do
            [[ "$(basename "${path}")" == "__MACOSX" ]] && continue
            entries+=("${path}")
        done
    fi

    if (( ${#entries[@]} == 0 )); then
        shopt -u dotglob nullglob
        printf 'Archive extraction produced no usable files.\n' >&2
        exit 1
    fi

    mv "${entries[@]}" "${final_dir}/"
    shopt -u dotglob nullglob
}

extract_zip_to_dir() {
    local archive_path="$1"
    local final_dir="$2"
    local stage_dir="${TMP_DIR}/$(basename "${archive_path}" .zip).$$"

    rm -rf "${stage_dir}"
    mkdir -p "${stage_dir}"
    unzip -q "${archive_path}" -d "${stage_dir}"
    normalize_extracted_tree "${stage_dir}" "${final_dir}"
    rm -rf "${stage_dir}"
}

download_file() {
    local key="$1"
    local file_id="${ITEM_ID[${key}]}"
    local filename="${ITEM_FILENAME[${key}]}"
    local archive_path="${DOWNLOAD_DIR}/${filename}"
    local partial_path="${archive_path}.part"
    local url

    mkdir -p "${DOWNLOAD_DIR}"

    if [[ -s "${archive_path}" ]]; then
        printf '[skip] archive exists: %s\n' "${archive_path}" >&2
        printf '%s\n' "${archive_path}"
        return 0
    fi

    url="$(download_url_for_id "${file_id}")"
    printf '[download] %s (%s)\n' "${key}" "${ITEM_SIZE[${key}]}" >&2
    curl -L --fail --retry 5 --retry-all-errors --retry-delay 5 --continue-at - \
        -o "${partial_path}" "${url}"

    if [[ ! -s "${partial_path}" ]]; then
        printf 'Download failed or produced an empty file for %s\n' "${key}" >&2
        exit 1
    fi

    mv "${partial_path}" "${archive_path}"
    printf '%s\n' "${archive_path}"
}

find_single_file() {
    local search_root="$1"
    local filename="$2"
    find "${search_root}" -maxdepth 2 -type f -name "${filename}" | sort | head -n 1 || true
}

sequence_output_dir() {
    local key="$1"
    printf '%s/%s\n' "${C3VD_ROOT}" "${key}"
}

sequence_ready() {
    local key="$1"
    [[ -f "$(sequence_output_dir "${key}")/.ready" ]]
}

prepare_sequence() {
    local key="$1"
    local archive_path="$2"
    local item_dir
    local raw_dir
    local ready_file
    local image_root
    local pose_path
    local coverage_path

    item_dir="$(sequence_output_dir "${key}")"
    raw_dir="${item_dir}/raw"
    ready_file="${item_dir}/.ready"

    rm -rf "${item_dir}"
    mkdir -p "${item_dir}"
    extract_zip_to_dir "${archive_path}" "${raw_dir}"

    image_root="$(find_image_root "${raw_dir}" || true)"
    if [[ -n "${image_root}" ]]; then
        link_path "${item_dir}/images" "${image_root}"
    else
        printf '[warn] no RGB image directory detected for %s\n' "${key}" >&2
    fi

    pose_path="$(find_single_file "${raw_dir}" 'pose.txt')"
    if [[ -n "${pose_path}" ]]; then
        link_path "${item_dir}/pose.txt" "${pose_path}"
    fi

    coverage_path="$(find_single_file "${raw_dir}" 'coverage_mesh.obj')"
    if [[ -n "${coverage_path}" ]]; then
        link_path "${item_dir}/coverage_mesh.obj" "${coverage_path}"
    fi

    {
        printf 'key=%s\n' "${key}"
        printf 'kind=%s\n' "${ITEM_KIND[${key}]}"
        printf 'filename=%s\n' "${ITEM_FILENAME[${key}]}"
    } > "${ready_file}"

    printf '[ready] %s -> %s\n' "${key}" "${item_dir}" >&2
}

asset_output_dir() {
    local key="$1"
    case "${ITEM_KIND[${key}]}" in
        mold)
            printf '%s/assets/molds/%s\n' "${C3VD_ROOT}" "${key}"
            ;;
        calibration)
            printf '%s/assets/calibration/%s\n' "${C3VD_ROOT}" "${key}"
            ;;
        *)
            printf '%s/assets/%s\n' "${C3VD_ROOT}" "${key}"
            ;;
    esac
}

model_output_path() {
    local key="$1"
    printf '%s/assets/models/%s\n' "${C3VD_ROOT}" "${ITEM_FILENAME[${key}]}"
}

asset_ready() {
    local key="$1"
    case "${ITEM_KIND[${key}]}" in
        model)
            [[ -s "$(model_output_path "${key}")" ]]
            ;;
        mold|calibration)
            [[ -f "$(asset_output_dir "${key}")/.ready" ]]
            ;;
        *)
            return 1
            ;;
    esac
}

prepare_zip_asset() {
    local key="$1"
    local archive_path="$2"
    local item_dir
    local raw_dir

    item_dir="$(asset_output_dir "${key}")"
    raw_dir="${item_dir}/raw"

    rm -rf "${item_dir}"
    mkdir -p "${item_dir}"
    extract_zip_to_dir "${archive_path}" "${raw_dir}"
    {
        printf 'key=%s\n' "${key}"
        printf 'kind=%s\n' "${ITEM_KIND[${key}]}"
        printf 'filename=%s\n' "${ITEM_FILENAME[${key}]}"
    } > "${item_dir}/.ready"

    printf '[ready] %s -> %s\n' "${key}" "${item_dir}" >&2
}

prepare_model_asset() {
    local key="$1"
    local archive_path="$2"
    local output_path

    output_path="$(model_output_path "${key}")"
    mkdir -p "$(dirname "${output_path}")"
    cp -f "${archive_path}" "${output_path}"
    printf '[ready] %s -> %s\n' "${key}" "${output_path}" >&2
}

cleanup_archive_if_needed() {
    local archive_path="$1"
    if [[ "${KEEP_ARCHIVES}" -eq 0 ]]; then
        rm -f "${archive_path}"
    fi
}

process_item() {
    local key="$1"
    local archive_path

    if [[ -z "${ITEM_ID[${key}]:-}" ]]; then
        printf 'Unknown C3VD item: %s\n' "${key}" >&2
        printf 'Run `bash scripts/download_c3vd.sh --list` to see valid names.\n' >&2
        exit 1
    fi

    case "${ITEM_KIND[${key}]}" in
        registered|screening)
            if [[ "${FORCE}" -eq 0 ]] && sequence_ready "${key}"; then
                printf '[skip] already prepared: %s\n' "$(sequence_output_dir "${key}")" >&2
                return 0
            fi
            ;;
        model|mold|calibration)
            if [[ "${FORCE}" -eq 0 ]] && asset_ready "${key}"; then
                case "${ITEM_KIND[${key}]}" in
                    model)
                        printf '[skip] already prepared: %s\n' "$(model_output_path "${key}")" >&2
                        ;;
                    *)
                        printf '[skip] already prepared: %s\n' "$(asset_output_dir "${key}")" >&2
                        ;;
                esac
                return 0
            fi
            ;;
        *)
            printf 'Unsupported item kind: %s\n' "${ITEM_KIND[${key}]}" >&2
            exit 1
            ;;
    esac

    archive_path="$(download_file "${key}")"

    case "${ITEM_KIND[${key}]}" in
        registered|screening)
            prepare_sequence "${key}" "${archive_path}"
            ;;
        model)
            prepare_model_asset "${key}" "${archive_path}"
            ;;
        mold|calibration)
            prepare_zip_asset "${key}" "${archive_path}"
            ;;
    esac

    cleanup_archive_if_needed "${archive_path}"
}

main() {
    local -a cli_selection=()
    local -a requested_items=()
    local item

    parse_catalog

    while (( "$#" )); do
        case "$1" in
            --list)
                print_catalog
                return 0
                ;;
            --force)
                FORCE=1
                ;;
            --keep-archives)
                KEEP_ARCHIVES=1
                ;;
            -h|--help)
                usage
                return 0
                ;;
            *)
                cli_selection+=("$1")
                ;;
        esac
        shift
    done

    if (( ${#cli_selection[@]} > 0 )); then
        mapfile -t requested_items < <(expand_selection "${cli_selection[@]}")
    else
        mapfile -t requested_items < <(expand_selection "${SELECTED_DATASETS[@]}")
    fi

    if (( ${#requested_items[@]} == 0 )); then
        printf 'No C3VD items selected.\n\n' >&2
        usage >&2
        printf '\n' >&2
        print_catalog >&2
        return 1
    fi

    check_dependencies
    mkdir -p "${INPUTS_DIR}" "${C3VD_ROOT}" "${TMP_DIR}"

    printf 'Preparing %d C3VD item(s) under %s\n' "${#requested_items[@]}" "${C3VD_ROOT}" >&2
    for item in "${requested_items[@]}"; do
        process_item "${item}"
    done

    printf '\nUsable sequence inputs:\n' >&2
    for item in "${requested_items[@]}"; do
        case "${ITEM_KIND[${item}]}" in
            registered|screening)
                if [[ -e "$(sequence_output_dir "${item}")/images" ]]; then
                    printf '  %s -> %s/images\n' "${item}" "$(sequence_output_dir "${item}")" >&2
                fi
                ;;
        esac
    done
}

main "$@"
