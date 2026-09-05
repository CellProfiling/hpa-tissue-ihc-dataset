#!/usr/bin/env bash
# End-to-end smoke test of the HPA tissue dataset pipeline on a small XML subset.
#
#   crawl -> (IDR table) -> split -> download -> verify -> tissue masks -> crops
#
# Usage:
#   smoke_test_pipeline.sh <proteinatlas_subset.xml> <work_dir> [idr_table.csv]
#
# Environment (all optional):
#   PYTHON       python interpreter (default: python)
#   IDR_DIR      directory with idr0043-experimentA-annotation-run*.csv; together with
#                IDR_LISTING this also exercises build_idr_ftp_paths.py and the resulting
#                table replaces the [idr_table.csv] argument
#   IDR_LISTING  recursive listing of ftp.ebi.ac.uk/pub/databases/IDR/idr0043-uhlen-humanproteinatlas/
#   SOURCE       idr (default, needs an IDR table) or all
#   N_PER_SET    images downloaded per split (default 2)
#   WORKERS      parallel workers (default 2)
#
# Make a small XML from the full release (first 60 genes, ~50 MB):
#   head -c 400000000 proteinatlas.xml | awk '{print} /^\t<\/entry>$/{n++; if(n==60){print "</proteinAtlas>"; exit}}' > subset.xml
set -euo pipefail

XML=${1:?proteinatlas_subset.xml}
WORK=${2:?work_dir}
IDR_TABLE=${3:-}
PYTHON=${PYTHON:-python}
SOURCE=${SOURCE:-idr}
N_PER_SET=${N_PER_SET:-2}
WORKERS=${WORKERS:-2}

HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
S=$HERE/scripts
META=$WORK/metadata
IMAGES=$WORK/images
mkdir -p "$META" "$IMAGES"

step() { echo; echo "=== [$(date +%H:%M:%S)] $*"; }

if [[ -n "${IDR_DIR:-}" ]]; then
  step "0. build_idr_ftp_paths.py"
  IDR_TABLE=$META/idr_ftp_paths.csv
  "$PYTHON" "$S/build_idr_ftp_paths.py" --idr-dir "$IDR_DIR" --listing "${IDR_LISTING:?IDR_LISTING}" --output "$IDR_TABLE"
fi

step "1. hpa_xml_parser.py  ($XML)"
FTP_ARGS=()
[[ -n "$IDR_TABLE" ]] && FTP_ARGS=(--ftp-paths "$IDR_TABLE")
"$PYTHON" "$S/hpa_xml_parser.py" --xml-file "$XML" --output-dir "$WORK" --format csv \
  --log-file "$META/hpa_xml_parser.log" ${FTP_ARGS[@]+"${FTP_ARGS[@]}"}
echo "images.csv rows: $(($(wc -l < "$META/images.csv") - 1))"

step "2. prepare_pilot_dataset.py  (--source $SOURCE)"
IDR_ARGS=()
[[ -n "$IDR_TABLE" ]] && IDR_ARGS=(--idr-table "$IDR_TABLE")
"$PYTHON" "$S/prepare_pilot_dataset.py" --images-csv "$META/images.csv" --antibodies-csv "$META/antibodies.csv" \
  ${IDR_ARGS[@]+"${IDR_ARGS[@]}"} --source "$SOURCE" --stage both --out-dir "$META" --save-plots --plot-dir "$META" \
  --log-file "$META/prepare_pilot_dataset.log"
PILOT=$META/HPA_pilot_dataset_$SOURCE.csv
echo "--- step counts"; cat "$PILOT.steps.csv"

step "2b. build_annotations.py"
"$PYTHON" "$S/build_annotations.py" --dataset-csv "$PILOT" --metadata-dir "$META" --out-dir "$META"
head -3 "$META/HPA_pilot_dataset_${SOURCE}_cell_annotations.csv" | cut -c1-200

step "3. pick $N_PER_SET images per split -> smoke_subset.csv"
SUBSET=$META/smoke_subset.csv
awk -F, -v n="$N_PER_SET" 'NR==1{print; next} {c[$NF]++; if (c[$NF] <= n) print}' "$PILOT" > "$SUBSET"
cut -d, -f1,5,14 "$SUBSET"

step "4. download_dataset_images.py"
"$PYTHON" "$S/download_dataset_images.py" --csv "$SUBSET" --root "$IMAGES" --workers "$WORKERS"

step "5. download_dataset_images.py --verify"
"$PYTHON" "$S/download_dataset_images.py" --csv "$SUBSET" --root "$IMAGES" --verify --workers "$WORKERS"

step "6. generate_tissue_masks.py"
"$PYTHON" "$S/generate_tissue_masks.py" --csv-file "$SUBSET" --img-root "$IMAGES" --num-workers "$WORKERS" --report-dir "$META"

step "7. crop_images_to_masks.py"
"$PYTHON" "$S/crop_images_to_masks.py" --csv-file "$SUBSET" --img-root "$IMAGES" --mask-format npy --padding 20 --num-workers "$WORKERS"

step "8. outputs under $IMAGES"
find "$IMAGES" -type f \( -name '*.tif' -o -name '*_mask.npy' -o -name '*_crop.npy' \) | sort
echo "tifs: $(find "$IMAGES" -name '*.tif' | wc -l)  masks: $(find "$IMAGES" -name '*_mask.npy' | wc -l)  crops: $(find "$IMAGES" -name '*_crop.npy' | wc -l)"
echo; echo "SMOKE TEST OK"
