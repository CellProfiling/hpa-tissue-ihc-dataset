# hpa-tissue-ihc-dataset

Build, download and preprocess a dataset of **Human Protein Atlas (HPA) tissue immunohistochemistry (IHC)
images** for training vision models. Bright-field tissue microarray cores, 3000x3000 RGB, one protein
stained brown (DAB) per image, 45 normal tissue categories.

Everything starts from public sources (the HPA XML release and, optionally, the Image Data Resource) and
ends with a dataset CSV plus the downloaded, segmented and cropped images:

```
crawl HPA XML  ->  IDR table (optional)  ->  split into full + pilot CSVs  ->  download  ->  tissue masks  ->  crops
hpa_xml_parser.py  build_idr_ftp_paths.py    prepare_pilot_dataset.py        download_dataset_images.py  generate_tissue_masks.py  crop_images_to_masks.py
```

Annotations (cell type, staining intensity and quantity, subcellular location, patient sex and age) are
joined onto any dataset CSV by `build_annotations.py` (section 4).

The pipeline produces two CSVs: the **full dataset** (all 45 tissue categories, ~700k images from IDR or ~1.2M
from HPA) and a **pilot dataset** (20 tissues of different complexity, ~190k images), both with download
URLs and an antibody-disjoint train/val/test split. The CSVs we built from the December 2025 release
(`HPA_pilot_dataset_idr.csv` and the IDR table) are shared separately; they are outputs of this repository,
not part of it.

* Have a dataset CSV? Start at [section 2](#2-download-and-preprocess-a-dataset-csv) (download, masks, crops).
* Want to build one from the current HPA release? [Section 3](#3-build-a-dataset-from-an-hpa-release), then section 2.

## Repository layout

```
README.md
LICENSE                             MIT
requirements.txt                    dependencies (incl. pytest)
smoke_test_pipeline.sh              runs the whole chain on a small XML (section 5)
scripts/
  hpa_xml_parser.py                 HPA XML -> images.csv, antibodies.csv, ...
  build_idr_ftp_paths.py            IDR table: which HPA images are mirrored on IDR, and where
  prepare_pilot_dataset.py          filters the crawl, builds full + pilot CSVs with train/val/test splits
  build_annotations.py              joins the HPA cell-type and patient annotations onto a dataset CSV (section 4)
  download_dataset_images.py        downloads (and optionally verifies) the images of a dataset CSV
  generate_tissue_masks.py          tissue segmentation masks
  crop_images_to_masks.py           crops images to their mask
tests/                              pytest unit tests for every script (no network)
```

## 1. Setup

```bash
git clone https://github.com/CellProfiling/hpa-tissue-ihc-dataset.git
cd hpa-tissue-ihc-dataset
python -m venv .venv && source .venv/bin/activate      # Python >= 3.10
pip install -r requirements.txt
```

* All scripts print `--help` and read `.csv` and `.csv.gz` alike.
* Disk. One image is a 3000x3000 RGB TIFF, **27 MB** on disk (uncompressed); its mask is 9 MB and its
  crop 13-19 MB in the default `.npy` format (see [section 6](#6-output-formats) for smaller formats).

  | dataset (December 2025 release) | images | TIFFs on disk |
  |---|---|---|
  | pilot, `--source idr` (20 tissues) | 187 685 | 5.1 TB |
  | full, `--source idr` (45 tissues, IDR-mirrored images) | 697 989 | 19 TB |
  | full, `--source all` (45 tissues, every HPA image) | 1 190 508 | 32 TB |

* Run the long steps in the background (`nohup`, `tmux`, or your job scheduler). Download and mask
  generation are resumable and skip files that already exist.

## 2. Download and preprocess a dataset CSV

A dataset CSV (`HPA_pilot_dataset_idr.csv`, or any output of section 3) has one row per image:

```
image_id, antibody_id, ensembl_id, gene_name, tissue, organ, patient_id, image_type,
hpa_url, idr_available, idr_ftp_path, idr_url, local_path, set
```

* `set` is `train` / `val` / `test` (pilot: 133 203 / 18 526 / 35 956). Splits are by antibody (no antibody
  appears in two splits) and stratified by tissue.
* `idr_url` is the EBI/IDR mirror (`https://ftp.ebi.ac.uk/pub/databases/IDR/idr0043-uhlen-humanproteinatlas/...`,
  answers `200 image/tiff`). `hpa_url` is the HPA link; it answers `302` and redirects to the EBI
  BioStudies mirror. Both give the same TIFF.
* `local_path` is the sub-directory an image is stored in. All scripts use the layout
  `<root>/<local_path>/<image_id>.tif`; keep it, do not rename files.

### Step 1 - download

```bash
python scripts/download_dataset_images.py --csv HPA_pilot_dataset_idr.csv --root ./hpa_tissue --workers 8
# one split only, or a first look:
python scripts/download_dataset_images.py --csv HPA_pilot_dataset_idr.csv --root ./hpa_tissue --set test --max-images 100
```

Tries `idr_url` first, then `hpa_url` (`--source-order hpa,idr` to swap). Each file is streamed to a temp
name, checked (HTTP 200, image content type, TIFF header, complete transfer) and then moved into place.
Re-run the same command to resume; it skips existing valid files. Failures go to
`<root>/download_failed_<timestamp>.csv` and the exit code is 1. A few failures per run are normal
(server hiccups); just re-run.

Throughput is limited by the EBI servers, not by your CPU. Measure it first with `--max-images 100` and
plan the full run from that. Keep `--workers` at 8-16; both mirrors throttle aggressive clients.

Optional: `--verify` opens and fully decodes every listed file without downloading anything and reports
`missing` / `corrupt` files to `<root>/verify_failed_<timestamp>.csv` (`--delete-corrupt` removes the
corrupt ones so that the next download run fetches them again). Useful after a disk or transfer problem.

### Step 2 - tissue masks (segmentation)

```bash
python scripts/generate_tissue_masks.py --csv-file HPA_pilot_dataset_idr.csv --img-root ./hpa_tissue \
    --num-workers 8 --output-format npy
```

Writes `<root>/<local_path>/masks/<image_id>_mask.npy` (uint8, 0/1, full resolution). The mask is a
rough tissue-vs-background segmentation: grayscale threshold at 0.2x downsampling, small components
(`--min-component-area`, in downsampled pixels; default 1000 = ~25 000 full-resolution pixels) removed,
holes filled, convex hull per component. Reports: `<img-root>/metadata/generate_tissue_masks_failed_<ts>.csv`
(exceptions; exit code 1) and `..._no_tissue_<ts>.csv` (empty masks, e.g. blank cores). Existing
non-empty masks are skipped, so it is resumable.

### Step 3 - crops

```bash
python scripts/crop_images_to_masks.py --csv-file HPA_pilot_dataset_idr.csv --img-root ./hpa_tissue \
    --mask-format npy --padding 20 --num-workers 8
```

Crops each image to the bounding box of its mask plus `--padding` pixels and writes
`<root>/<local_path>/crops/<image_id>_crop.npy` (RGB uint8). Images without a mask are reported and skipped.

Resulting layout:

```
hpa_tissue/
  20190109-ftp/51818/117278_A_8_2.tif
  20190109-ftp/51818/masks/117278_A_8_2_mask.npy
  20190109-ftp/51818/crops/117278_A_8_2_crop.npy
  download_failed_<ts>.csv, verify_failed_<ts>.csv           (only when something failed)
  metadata/generate_tissue_masks_{failed,no_tissue}_<ts>.csv
```

## 3. Build a dataset from an HPA release

Two sources to choose from:

1. **IDR** (`--source idr`): only images mirrored on the Image Data Resource (idr0043). Fast, stable
   downloads, TIFF guaranteed. Needs the IDR table. ~700k images.
2. **HPA website** (`--source all`): every image in the HPA release, downloaded via the HPA links (which
   redirect to the EBI BioStudies mirror). ~1.2M images.

**IDR is optional.** With source 2 nothing IDR-related is needed: skip Step 1, leave out `--ftp-paths` in
Step 2 and `--idr-table` in Step 3. The downloader then uses `hpa_url` for every image.

### Step 0 - get the XML

```bash
wget https://www.proteinatlas.org/download/proteinatlas.xml.gz && gunzip proteinatlas.xml.gz   # ~14 GB unpacked
```

### Step 1 - IDR table (source 1 only; skip for source 2)

The IDR table lists every HPA image mirrored on IDR with its path there. The one we built
(`idr0043-experimentA-annotation-M-00100_ftp_paths.csv`) is shared together with the pilot CSV. To build it
from public sources you need:

* **IDR annotation CSVs**: one per HPA batch, from the IDR metadata repository
  https://github.com/IDR/idr0043-uhlen-humanproteinatlas, file
  `experimentA/hpa_run_XX/idr0043-experimentA-annotation.csv.gz` for `XX` = 01 ... 12 (more if IDR added
  batches). Unpack them into one directory as `idr0043-experimentA-annotation-runXX.csv`. Their columns start
  with `Dataset Name, Image Name, ..., Characteristics [Pathology] Accession, ...`.
* **FTP listing**: a recursive listing of the IDR FTP folder, one path per line (directories end with `/`):
  ```bash
  lftp -c 'open ftp.ebi.ac.uk; find /pub/databases/IDR/idr0043-uhlen-humanproteinatlas/' > IDR_all_paths_raw.txt
  ```
  (~6 M lines; any tool that prints full paths works, e.g. `rclone lsf -R` against an FTP remote.)

```bash
python scripts/build_idr_ftp_paths.py --idr-dir ./IDR --listing ./IDR_all_paths_raw.txt \
    --output ./idr_ftp_paths.csv          # ~3 min, ~8 GB RAM
```

Keeps normal-tissue rows (SNOMED `M-00100`) and adds `ftp_path` (`<batch>/<antibody>/<image>.tif`);
when an image exists in several batches the newest wins.

### Step 2 - crawl

```bash
mkdir -p ./hpa_tissue/metadata
python scripts/hpa_xml_parser.py --xml-file proteinatlas.xml --output-dir ./hpa_tissue --format csv \
    --ftp-paths ./idr_ftp_paths.csv       # --ftp-paths optional
```

Streams the XML (a few hours, low memory) and writes `./hpa_tissue/metadata/images.csv`,
`antibodies.csv`, `genes.csv`, `tissues.csv`, `cells.csv`, `patients.csv`. Only normal-tissue IHC
(`assayType="tissue"`, SNOMED `M-00100`) is kept; TIFF links are preferred over JPG. `--ftp-paths` only
makes `local_path` follow the IDR batch folders; without it images are grouped into synthetic batches.

### Step 3 - split into full + pilot CSVs

```bash
# source 1: IDR
python scripts/prepare_pilot_dataset.py \
    --images-csv ./hpa_tissue/metadata/images.csv --antibodies-csv ./hpa_tissue/metadata/antibodies.csv \
    --idr-table ./idr_ftp_paths.csv --source idr \
    --stage both --out-dir ./hpa_tissue/metadata --save-plots --plot-dir ./hpa_tissue/metadata
# source 2: HPA website
python scripts/prepare_pilot_dataset.py \
    --images-csv ./hpa_tissue/metadata/images.csv --antibodies-csv ./hpa_tissue/metadata/antibodies.csv \
    --source all --stage both --out-dir ./hpa_tissue/metadata --save-plots --plot-dir ./hpa_tissue/metadata
```

#### How the dataset is built

The crawl (`images.csv`) has one row per image and tissue: 2.68 M rows for the December 2025 release. Every
step below is a function in `prepare_pilot_dataset.py`; the script writes the row counts after each step to
`<pilot>.steps.csv`. Counts in brackets are from the December 2025 release with `--source idr`.

**A. Full dataset** [2 680 905 image rows, 21 195 antibodies, 14 798 genes, 49 tissue categories]

1. **Source.** `--source idr` keeps images that exist on the IDR mirror (matched by `image_id` against the
   IDR table) [1 462 106]; `--source all` keeps everything.
2. **TIFF available.** Keep an image if HPA offers it as TIFF or if it is on IDR (IDR holds TIFFs even where
   the HPA XML only lists a JPG). `--allow-jpg` disables this [1 462 106].
3. **Secondary tissue categories.** HPA has a second tissue-microarray series for four organs, labelled
   "Endometrium 2", "Stomach 2", "Soft tissue 2" and "Skin 2". They are dropped so each organ is one
   category [1 344 921, 45 tissue categories].
4. **Antibody filters.** Every HPA image is stained with one antibody, and HPA scores each antibody's
   reliability (`enhanced` > `supported` > `approved` > `uncertain`). Three steps:
   * drop antibodies scored `uncertain` [981 066];
   * drop antibodies that target more than one gene, according to `antibodies.csv` [930 038];
   * keep **one antibody per gene**: among the antibodies that still have images, the one with the highest
     reliability score, ties broken by a seeded shuffle [709 860 images, 6 126 antibodies = 6 126 genes].
     The dataset is therefore one antibody, i.e. one protein, per gene, which keeps the gene set as large
     as possible without near-duplicate stainings of the same protein.
5. **Unique image id.** HPA cross-lists every "Soft tissue 1" image under "Adipose tissue" as well (same
   image, same patient). The default `--duplicate-policy prefer` keeps the "Soft tissue 1" row; `drop`
   removes both; `fail` stops and lets you inspect `<output>.csv.duplicates.csv` [697 989].
6. **Split** (see C) [train 493 852 / val 69 721 / test 134 416 for the IDR full set].

**B. Pilot dataset**

1. **Tissues.** 20 of the 45 tissue categories, chosen to cover tissues of different complexity. The
   tissues were binned by how hard their morphology is to tell apart (easy: distinctive architecture;
   moderate; difficult: cellular, homogeneous tissues that look alike). The pilot takes all difficult ones
   plus a random sample of the others (seed 42; the resulting list is frozen in the script) [308 220]:

   | complexity | pilot tissues |
   |---|---|
   | difficult (all 5) | Bone marrow, Caudate, Lung, Lymph node, Tonsil |
   | moderate (8 of 17) | Adrenal gland, Breast, Cerebellum, Cerebral cortex, Hippocampus, Parathyroid gland, Seminal vesicle, Testis |
   | easy (7 of 22) | Appendix, Colon, Duodenum, Gallbladder, Heart muscle, Small intestine, Urinary bladder |

2. **Thinning.** HPA usually has 1-3 cores (images) per antibody and tissue, from different patients. To
   spend the image budget on more antibodies rather than on near-replicates of the same staining, per
   (antibody, tissue) group with exactly 3 images one is dropped [231 229], and in 40 % of the groups with
   exactly 2 images one is dropped [187 685]. Which image goes is decided by a seeded shuffle.
3. **Split** as in C, redone on the pilot rows [train 133 203 / val 18 526 / test 35 956].

**C. Train / val / test split**

* **Unit is the antibody, not the image.** All images of one antibody, across all tissues and patients, go
  to the same split. Because the full set has one antibody per gene, no gene appears in two splits either.
  This prevents leakage: a model cannot recognise a protein's staining pattern in the test set because it
  saw the same antibody in training.
* **Stratified by tissue.** Each antibody is described by the set of tissues it has images in (a
  multi-label vector). The antibodies are then split with iterative stratification (Sechidis et al., 2011;
  `scikit-multilearn`, second-order label combinations), which balances the tissue combinations across the
  splits. As a result each tissue has close to 70/10/20 % of its images in train/val/test.
* **Two stages.** First 20 % of the antibodies are held out as test, then the remaining 80 % are split
  70:10 into train and val with the same procedure. Fractions are `--train-frac/--val-frac/--test-frac`.
* **Deterministic.** All randomness (antibody tie-breaks, tissue sampling, thinning, stratification
  tie-breaks) is driven by `--seed` (default 42). The same crawl and seed give the same CSVs.
* **Inheriting an existing split.** `--split-from <csv>` copies the `set` label from an existing dataset
  CSV by `image_id`. If an antibody has some labelled images, its unlabelled images get the same label
  (majority vote), so the antibody rule still holds; only antibodies with no labelled image go through the
  stratified split. Use this to keep test images fixed across HPA releases.
* The script aborts if any antibody ends up in more than one split.

Outputs in `--out-dir`: `HPA_full_dataset_<source>.csv`, `HPA_pilot_dataset_<source>.csv`,
`<pilot>.steps.csv` (row count after each filter), `<full|pilot>.csv.duplicates.csv` (the cross-listed ids),
optional tissue/organ distribution PNGs. Runs in a few minutes on a whole crawl (~2.7 M rows).

Then continue with [section 2](#2-download-and-preprocess-a-dataset-csv) on the CSV you want; the
downloader works for the full set too.

## 4. Metadata and annotations

The crawl (section 3, step 2) writes six tables to `./hpa_tissue/metadata/`. HPA annotates staining per
**antibody x tissue x cell type**, not per image: all images (patients) of one antibody in one tissue share
the same annotation.

| table | one row per | columns |
|---|---|---|
| `images.csv` | image x tissue | `antibody_id, ensembl_id, gene_name, tissue, organ, patient_id, image_type, image_url, image_id, local_path, ...` |
| `antibodies.csv` | antibody x gene | `antibody_id, ensembl_id, gene_name, Reliability score` (enhanced / supported / approved / uncertain) |
| `genes.csv` | gene | `ensembl_id, name, synonyms` |
| `tissues.csv` | antibody x tissue | `antibody_id, ensembl_id, tissue, organ, ontology_terms` (UBERON) |
| `cells.csv` | antibody x tissue x cell type | `antibody_id, ensembl_id, tissue, cell_type, staining, intensity, quantity, location` |
| `patients.csv` | antibody x tissue x patient | `antibody_id, ensembl_id, tissue, patient_id, sex, age` |

The join key from a dataset CSV to `cells.csv` and `tissues.csv` is `(antibody_id, ensembl_id, tissue)`; to
`patients.csv` add `patient_id`. `build_annotations.py` does these joins for you and writes, next to the
dataset CSV:

```bash
python scripts/build_annotations.py --dataset-csv HPA_pilot_dataset_idr.csv --metadata-dir ./hpa_tissue/metadata
```

* **`<dataset>_cell_annotations.csv`**, one row per image x annotated cell type (the cell-type-level labels):
  the dataset columns plus `cell_type, staining, intensity, quantity, location` as HPA strings and ordinal
  codes `staining_code, intensity_code, quantity_code, location_code`:

  | code | staining | intensity | quantity | location |
  |---|---|---|---|---|
  | 0 | not detected | negative | none | none |
  | 1 | low | weak | <25% | cytoplasmic/membranous |
  | 2 | medium | moderate | 25%-75% | nuclear |
  | 3 | high | strong | >75% | cytoplasmic/membranous and nuclear |
  | -1 | missing or "not representative" | | | |

* **`<dataset>_image_annotations.csv`**, one row per image (the image-level labels): the dataset columns plus
  `sex, age, ontology_terms`, the annotated `cell_types` (`;`-separated), `n_cell_types`, and the maximum over
  the cell types `max_staining_code, max_intensity_code, max_quantity_code`; `protein_detected` is true if any
  cell type is stained.
* **`<dataset>_cell_types.csv`**, the cell-type vocabulary of the dataset with image, antibody and tissue counts.

Cell types are tissue-specific (e.g. "Glandular cells" in Colon, "Pneumocytes" in Lung); the December 2025
crawl has 108 distinct cell-type names, the full dataset 107 (either source), the pilot dataset 59 (395 132
image x cell-type rows for 187 685 images; every image has at least one annotated cell type). Should a metadata table contain repeated
keys, the script drops exact duplicates, keeps the first value of a conflict and logs the counts.

## 5. Smoke test (~3 min, ~200 MB)

`smoke_test_pipeline.sh` runs the whole chain on a small XML and downloads two images per split:

```bash
head -c 400000000 proteinatlas.xml | awk '{print} /^\t<\/entry>$/{n++; if(n==60){print "</proteinAtlas>"; exit}}' > subset.xml
./smoke_test_pipeline.sh subset.xml ./smoke_run ./idr_ftp_paths.csv      # source 1 (needs the IDR table)
SOURCE=all ./smoke_test_pipeline.sh subset.xml ./smoke_run_all             # source 2
```

It ends with `SMOKE TEST OK` and lists the TIFFs, masks and crops under `./smoke_run/images/`. Every
intermediate file (crawl CSVs, split CSVs, step counts, reports) is in `./smoke_run/metadata/`. Reference
run (60 genes): crawl 12 s (11 701 image rows), split 19 s (pilot 670 rows), 6 downloads ~1 min, masks 6 s,
crops 5 s. Run it as a single process (on Slurm: `srun -n1 ...`).

## 6. Output formats

Masks and crops are written as `.npy` by default: a raw `uint8` array with a 128-byte header. Loading one
is a single read (or a memory map) with no decoding, which is what makes training data loaders fast; PNG
or compressed TIFF must be inflated on every access, which is CPU-bound and several times slower per image.

The price is size: `.npy` is uncompressed (mask 9 MB, crop 13-19 MB). If storage is the bottleneck:

```bash
python scripts/generate_tissue_masks.py ... --output-format png       # masks: ~1000x smaller
python scripts/crop_images_to_masks.py  ... --save-format png         # lossless PNG, roughly 2-4x smaller
python scripts/crop_images_to_masks.py  ... --save-format tif         # zlib (deflate) compressed TIFF, similar size
```

Both are lossless; expect slower training-time loading. The crop script accepts every mask format the mask
script can write (`--mask-format png|npy|pt`).

## 7. Tests

```bash
pytest -q
```

Unit tests cover every script with synthetic data and a fake HTTP session; nothing touches the network.
They also run on GitHub Actions for every push.

## 8. Data licences

* **Code**: MIT (see `LICENSE`).
* **Images and metadata** come from the Human Protein Atlas, licensed
  [CC BY-SA 4.0](https://www.proteinatlas.org/about/licence), and from the Image Data Resource study
  idr0043, licensed [CC BY 4.0](https://idr.openmicroscopy.org/about/). Dataset CSVs produced by this
  pipeline are derived from that metadata and inherit these terms: attribute HPA and IDR in anything you publish.
* Please cite the Human Protein Atlas (Uhlén et al., *Science* 2015, doi:10.1126/science.1260419) and, for
  IDR, Williams et al., *Nature Methods* 2017 (doi:10.1038/nmeth.4326).

## References

* HPA: https://www.proteinatlas.org (XML: https://www.proteinatlas.org/download/proteinatlas.xml.gz)
* IDR study idr0043: https://idr.openmicroscopy.org/webclient/?show=project-1201 ,
  FTP `ftp.ebi.ac.uk/pub/databases/IDR/idr0043-uhlen-humanproteinatlas/`,
  metadata https://github.com/IDR/idr0043-uhlen-humanproteinatlas
* Sizes and counts above were measured on the December 2025 HPA release (v25).
