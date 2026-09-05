#!/usr/bin/env python3
"""
Crawl the Human Protein Atlas XML release into CSV tables.

Streams ``proteinatlas.xml`` (https://www.proteinatlas.org/download/proteinatlas.xml.gz)
and keeps normal-tissue IHC only (``assayType="tissue"``, SNOMED ``M-00100``).
Writes ``<output-dir>/metadata/images.csv`` (one row per image x tissue, TIFF URL
preferred over JPG), ``antibodies.csv``, ``genes.csv``, ``tissues.csv``, ``cells.csv``
and ``patients.csv``. ``--ftp-paths`` (optional IDR table) makes ``local_path`` follow
the IDR batch folders.

Example:
    python hpa_xml_parser.py --xml-file proteinatlas.xml --output-dir ./hpa_tissue --format csv
"""
import xml.etree.ElementTree as ET
import pandas as pd
import json
from collections import defaultdict
import os
import argparse
import sys
import re
import csv
import time
import gzip


# Helper function for logging
def log_message(message, log_file=None):
    print(message, flush=True)
    if log_file:
        with open(log_file, "a") as f:
            f.write(message + "\n")


def extract_hpa_data(
    xml_file,
    output_dir=None,
    output_format="img_json",
    ftp_paths_file=None,
    log_file=None,
):
    """
    Extract gene, antibody and tissue expression data from the HPA XML file.
    Only includes normal tissue data (assayType="tissue") with healthy tissue samples (M-00100).

    Args:
        xml_file: Path to the XML file
        output_dir: Directory to save output files (defaults to same directory as input file)
        output_format: 'csv', 'json', 'img_json', or 'all'
        ftp_paths_file: CSV file containing FTP paths for downloaded images
        log_file: Path to log file (optional)

    Returns:
        Dictionary of extracted data
    """
    if output_dir is None:
        output_dir = os.path.dirname(os.path.abspath(xml_file))
        if not output_dir:
            output_dir = "."
    if log_file is None:
        log_file = os.path.join(output_dir, "hpa_xml_parser.log")

    log_message(f"Parsing XML file: {xml_file}", log_file)
    log_message(f"Output will be saved to: {output_dir}", log_file)

    # Load FTP paths if provided
    ftp_path_mapping = {}
    if ftp_paths_file and os.path.exists(ftp_paths_file):
        log_message(f"Loading FTP paths from: {ftp_paths_file}", log_file)
        opener = gzip.open if ftp_paths_file.endswith(".gz") else open
        with opener(ftp_paths_file, "rt") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if "ftp_path" in row:
                    path = row["ftp_path"]
                    image_filename = os.path.basename(path)
                    image_id = os.path.splitext(image_filename)[0]
                    ftp_path_mapping[image_id] = path
        log_message(f"Loaded {len(ftp_path_mapping)} FTP paths", log_file)

    context = ET.iterparse(xml_file, events=("start", "end"))
    all_data = []
    current_gene = {"ensembl_id": None, "name": None, "synonyms": []}
    current_antibody = None
    current_data = None
    current_tissue = None
    current_tissue_cell = None
    current_patient = None
    current_image = None
    in_data_section = False
    in_tissue_expression = False
    current_assay_type = None
    current_verification_score = None
    skipped_cancer_sections = 0
    total_tissues_processed = 0
    healthy_tissues_processed = 0
    tif_count = 0
    jpg_count = 0
    verification_count = 0  # Add counter for verification elements
    for event, elem in context:
        # Gene/entry information
        if event == "start" and elem.tag == "entry":
            # Reset current gene for new entry
            current_gene = {"ensembl_id": None, "name": None, "synonyms": []}

            # Try to get URL which often contains ensembl ID
            entry_url = elem.get("url", "")
            if entry_url and "/" in entry_url:
                current_gene["ensembl_id"] = entry_url.split("/")[-1]

        elif event == "end" and elem.tag == "name" and current_gene["name"] is None:
            current_gene["name"] = elem.text

        elif event == "end" and elem.tag == "synonym":
            if elem.text:
                current_gene["synonyms"].append(elem.text)

        elif (
            event == "start"
            and elem.tag == "identifier"
            and elem.get("db") == "Ensembl"
        ):
            current_gene["ensembl_id"] = elem.get("id")

        # Start of antibody - capture ID
        elif event == "start" and elem.tag == "antibody":
            current_antibody = {
                "id": elem.get("id"),
                "gene": current_gene.copy(),  # Associate gene info with antibody
                "tissues": [],
                "reliability_score": None,  # Add reliability score field
            }

        # Tissue expression section - check assay type
        elif event == "start" and elem.tag == "tissueExpression":
            in_tissue_expression = True
            current_assay_type = elem.get("assayType")
            current_verification_score = (
                None  # Track verification for this tissueExpression
            )

            # Skip cancer sections entirely
            if current_assay_type == "cancer":
                skipped_cancer_sections += 1

        # Start of data section within tissueExpression (only for tissue, not cancer)
        elif (
            event == "start"
            and elem.tag == "data"
            and current_antibody is not None
            and in_tissue_expression
            and current_assay_type == "tissue"
        ):
            in_data_section = True
            current_data = {
                "tissue": None,
                "organ": None,
                "ontology_terms": None,
                "tissue_cells": [],
                "patients": [],
            }
            total_tissues_processed += 1

        # Tissue information within data section
        elif event == "end" and elem.tag == "tissue" and in_data_section:
            current_tissue = elem.text
            current_data["tissue"] = current_tissue
            current_data["organ"] = elem.get("organ")
            current_data["ontology_terms"] = elem.get("ontologyTerms")

        # Start of tissueCell within data section
        elif event == "start" and elem.tag == "tissueCell" and in_data_section:
            current_tissue_cell = {
                "cell_type": None,
                "staining": None,
                "intensity": None,
                "quantity": None,
                "location": None,
            }

        # Cell type within tissueCell
        elif (
            event == "end"
            and elem.tag == "cellType"
            and current_tissue_cell is not None
        ):
            current_tissue_cell["cell_type"] = elem.text

        # Level (staining or intensity) within tissueCell
        elif event == "end" and elem.tag == "level" and current_tissue_cell is not None:
            level_type = elem.get("type")
            if level_type == "staining":
                current_tissue_cell["staining"] = elem.text
            elif level_type == "intensity":
                current_tissue_cell["intensity"] = elem.text

        # Quantity within tissueCell
        elif (
            event == "end"
            and elem.tag == "quantity"
            and current_tissue_cell is not None
        ):
            current_tissue_cell["quantity"] = elem.text

        # Location within tissueCell
        elif (
            event == "end"
            and elem.tag == "location"
            and current_tissue_cell is not None
        ):
            current_tissue_cell["location"] = elem.text

        # End of tissueCell - add to current data
        elif (
            event == "end"
            and elem.tag == "tissueCell"
            and current_tissue_cell is not None
        ):
            if not current_tissue_cell.get("location"):
                log_message(
                    f"WARNING: Missing location! Antibody={current_antibody['id'] if current_antibody else None} | Tissue={current_data['tissue'] if current_data else None} | CellType={current_tissue_cell.get('cell_type')}",
                    log_file,
                )
            current_data["tissue_cells"].append(current_tissue_cell)
            current_tissue_cell = None

        # Start of patient within data section
        elif event == "start" and elem.tag == "patient" and in_data_section:
            current_patient = {
                "sex": None,
                "age": None,
                "patient_id": None,
                "images": [],
                "has_normal_tissue": False,  # Track if this patient has M-00100 code
            }

        # NOTE: elem.text is only guaranteed populated on the "end" event.
        # Reading it on "start" returns None whenever the tag straddles an
        # iterparse read-block boundary. Only attribute reads may use "start".
        # Sex within patient
        elif event == "end" and elem.tag == "sex" and current_patient is not None:
            current_patient["sex"] = elem.text

        # Age within patient
        elif event == "end" and elem.tag == "age" and current_patient is not None:
            current_patient["age"] = elem.text

        # PatientId within patient
        elif (
            event == "end" and elem.tag == "patientId" and current_patient is not None
        ):
            current_patient["patient_id"] = elem.text

        # SNOMED code - check for normal tissue (M-00100)
        elif event == "start" and elem.tag == "snomed" and current_patient is not None:
            snomed_code = elem.get("snomedCode")
            if snomed_code == "M-00100":
                current_patient["has_normal_tissue"] = True

        # Start of image element - initialize image data collection
        elif event == "start" and elem.tag == "image" and current_patient is not None:
            current_image = {
                "type": elem.get("imageType"),
                "tif_url": None,
                "jpg_url": None,
            }

        # TIF URL within image element
        elif (
            event == "end" and elem.tag == "imageUrlTif" and current_image is not None
        ):
            current_image["tif_url"] = elem.text

        # JPG URL within image element
        elif event == "end" and elem.tag == "imageUrl" and current_image is not None:
            current_image["jpg_url"] = elem.text

        # End of image element - add collected image data to patient
        elif event == "end" and elem.tag == "image" and current_image is not None:
            if current_image["tif_url"]:
                current_patient["images"].append(
                    {"type": "tif", "url": current_image["tif_url"]}
                )
                tif_count += 1
            elif current_image["jpg_url"]:
                current_patient["images"].append(
                    {"type": "jpg", "url": current_image["jpg_url"]}
                )
                jpg_count += 1
            current_image = None

        # End of patient - add to current data ONLY if it has normal tissue
        elif event == "end" and elem.tag == "patient" and current_patient is not None:
            if current_patient["has_normal_tissue"]:
                # Remove the tracking flag before adding to data
                del current_patient["has_normal_tissue"]
                current_data["patients"].append(current_patient)
                healthy_tissues_processed += 1
            current_patient = None

        # End of data section - add to tissues list in current antibody
        elif event == "end" and elem.tag == "data" and in_data_section:
            if current_antibody and current_data and current_data["patients"]:
                # Only add if there are patients (which means they have normal tissue)
                current_antibody["tissues"].append(current_data.copy())
            current_data = None
            in_data_section = False

        # Extract <verification type="validation"> value for reliability score
        elif (
            event == "end"
            and elem.tag == "verification"
            and in_tissue_expression
            and current_assay_type == "tissue"
        ):
            verification_count += 1  # Count all verification elements

            if elem.get("type") == "validation":
                if current_antibody is not None:
                    # Overwrite with the latest non-empty value
                    if elem.text and elem.text.strip():
                        # Check if we're overwriting an existing value
                        if current_antibody.get("reliability_score") is not None:
                            log_message(
                                f"Overwriting reliability score for antibody {current_antibody['id']} from {current_antibody['reliability_score']} to {elem.text.strip()}",
                                log_file,
                            )
                        current_antibody["reliability_score"] = elem.text.strip()

        # End of tissueExpression section
        elif event == "end" and elem.tag == "tissueExpression":
            in_tissue_expression = False
            current_assay_type = None

        # End of antibody - add to all_data
        elif event == "end" and elem.tag == "antibody":
            if current_antibody and current_antibody["tissues"]:
                # Only add antibodies that have tissue data
                all_data.append(current_antibody.copy())

                # Log if antibody is missing reliability score
                if current_antibody.get("reliability_score") is None:
                    log_message(
                        f"WARNING: Antibody {current_antibody['id']} has no reliability score",
                        log_file,
                    )
            current_antibody = None

        # Clear element to save memory
        if event == "end":
            elem.clear()

    # Summarize extraction
    gene_count = len(
        set(
            data["gene"]["ensembl_id"]
            for data in all_data
            if data["gene"]["ensembl_id"]
        )
    )
    antibody_count = len(all_data)
    tissue_count = sum(len(data["tissues"]) for data in all_data)
    total_images = tif_count + jpg_count

    log_message(
        f"Extracted data for {gene_count} genes, {antibody_count} antibodies", log_file
    )
    log_message(f"Skipped {skipped_cancer_sections} cancer sections", log_file)
    log_message(
        f"Processed {total_tissues_processed} tissues, found {healthy_tissues_processed} with healthy samples (M-00100)",
        log_file,
    )
    log_message(f"Found {verification_count} verification elements", log_file)

    # Print URL type statistics
    if total_images > 0:
        tif_percent = (tif_count / total_images) * 100
        jpg_percent = (jpg_count / total_images) * 100
        log_message(f"Image URL statistics:", log_file)
        log_message(f"- TIF URLs: {tif_count} ({tif_percent:.1f}%)", log_file)
        log_message(f"- JPG URLs: {jpg_count} ({jpg_percent:.1f}%)", log_file)
        log_message(f"- Total images: {total_images}", log_file)

    # Convert to the desired output format(s)
    if output_format in ["csv", "all"]:
        save_as_csv(all_data, output_dir, ftp_path_mapping, log_file)

    if output_format in ["json", "all"]:
        save_as_json(all_data, output_dir, log_file)

    if output_format in ["img_json", "all"]:
        save_image_metadata(all_data, output_dir, ftp_path_mapping, log_file)

    return all_data


class BatchFolderAssigner:
    """Assigns web-downloaded images to synthetic grandparent 'batch' folders.

    Every new *parent* folder (HPA antibody number from the URL) is placed in the
    current batch; after `parents_per_batch` new parents the batch id advances.
    Mirrors the historical layout: 20250612-xml, 20250613-xml, ...
    """

    def __init__(self, start_batch_id=12, parents_per_batch=500):
        self.next_batch_id = start_batch_id
        self.parents_per_batch = parents_per_batch
        self._count = 0
        self.parent_folders = {}  # parent folder -> grandparent batch folder

    @property
    def current_batch_id(self):
        return f"202506{self.next_batch_id}-xml"

    def grandparent_for(self, parent_folder):
        if parent_folder not in self.parent_folders:
            self.parent_folders[parent_folder] = self.current_batch_id
            self._count += 1
            if self._count >= self.parents_per_batch:
                self.next_batch_id += 1
                self._count = 0
        return self.parent_folders[parent_folder]


def resolve_image_folders(image_url, antibody_id, image_id, ftp_path_mapping, assigner):
    """Return (grandparent_folder, parent_folder, source) for one image.

    source is "ftp" when a valid FTP path (>= 3 segments) exists for image_id,
    otherwise "web" with a synthetic batch grandparent from `assigner`.
    """
    url_parts = image_url.split("/")
    if len(url_parts) >= 2:
        parent_folder = url_parts[-2]
    else:
        match = re.search(r"(HPA|CAB)0*(\d+)", antibody_id)
        if match:
            parent_folder = match.group(2)
        else:
            parent_folder = antibody_id.replace("HPA", "").replace("CAB", "")

    if ftp_path_mapping and image_id in ftp_path_mapping:
        path_parts = ftp_path_mapping[image_id].split("/")
        if len(path_parts) >= 3:
            return path_parts[-3], path_parts[-2], "ftp"

    return assigner.grandparent_for(parent_folder), parent_folder, "web"


def save_as_csv(all_data, output_dir, ftp_path_mapping=None, log_file=None):
    """Save the extracted data as CSV files in the metadata directory"""
    metadata_dir = os.path.join(output_dir, "metadata")
    os.makedirs(metadata_dir, exist_ok=True)

    # Create flattened dataframes
    gene_data = []
    antibody_data = []
    tissue_data = []
    cell_data = []
    patient_data = []
    image_data = []

    # For folder structure tracking (same logic as save_image_metadata)
    assigner = BatchFolderAssigner()
    seen_genes = set()

    for antibody_entry in all_data:
        antibody_id = antibody_entry["id"]
        gene_info = antibody_entry["gene"]
        ensembl_id = gene_info["ensembl_id"]
        gene_name = gene_info["name"]

        # Add gene info
        if ensembl_id and ensembl_id not in seen_genes:
            seen_genes.add(ensembl_id)
            gene_data.append(
                {
                    "ensembl_id": ensembl_id,
                    "name": gene_name,
                    "synonyms": ", ".join(gene_info["synonyms"]),
                }
            )

        # Add antibody info
        antibody_data.append(
            {
                "antibody_id": antibody_id,
                "ensembl_id": ensembl_id,
                "gene_name": gene_name,
                "Reliability score": antibody_entry.get(
                    "reliability_score"
                ),  # Add reliability score to CSV
            }
        )

        # Process tissues
        for tissue_entry in antibody_entry["tissues"]:
            tissue = tissue_entry["tissue"]
            organ = tissue_entry["organ"]
            ontology_terms = tissue_entry["ontology_terms"]

            # Add tissue data
            tissue_data.append(
                {
                    "antibody_id": antibody_id,
                    "ensembl_id": ensembl_id,
                    "tissue": tissue,
                    "organ": organ,
                    "ontology_terms": ontology_terms,
                }
            )

            # Add cell data
            for cell in tissue_entry["tissue_cells"]:
                cell_data.append(
                    {
                        "antibody_id": antibody_id,
                        "ensembl_id": ensembl_id,
                        "tissue": tissue,
                        "cell_type": cell["cell_type"],
                        "staining": cell["staining"],
                        "intensity": cell["intensity"],
                        "quantity": cell.get("quantity"),
                        "location": cell["location"],
                    }
                )

            # Add patient and image data
            for patient in tissue_entry["patients"]:
                patient_id = patient["patient_id"]

                patient_data.append(
                    {
                        "antibody_id": antibody_id,
                        "ensembl_id": ensembl_id,
                        "tissue": tissue,
                        "patient_id": patient_id,
                        "sex": patient["sex"],
                        "age": patient["age"],
                    }
                )

                for img in patient["images"]:
                    if "url" in img and img["url"]:  # Ensure URL exists
                        image_filename = os.path.basename(img["url"])
                        image_id = os.path.splitext(image_filename)[0]
                        grandparent_folder, parent_folder, _ = resolve_image_folders(
                            img["url"], antibody_id, image_id, ftp_path_mapping, assigner
                        )

                        # Determine download status
                        download_status = "pending"
                        if ftp_path_mapping and image_id in ftp_path_mapping:
                            download_status = "ftp_available"

                        # Create local path
                        local_path = f"{grandparent_folder}/{parent_folder}"

                        image_data.append(
                            {
                                "antibody_id": antibody_id,
                                "ensembl_id": ensembl_id,
                                "gene_name": gene_name,
                                "tissue": tissue,
                                "organ": organ,
                                "patient_id": patient_id,
                                "image_type": img["type"],
                                "image_url": img["url"],
                                "image_id": image_id,
                                "local_path": local_path,
                                "download_status": download_status,
                                "download_date": "",
                                "file_size": "",
                                "url_used": "",
                            }
                        )

    # Deduplicate by (image_id, patient_id, tissue), keeping TIF over JPG
    log_message(f"Total image rows before deduplication: {len(image_data)}", log_file)

    image_data_dict = {}  # {(image_id, patient_id, tissue): row_dict}
    jpg_removed = 0

    for row in image_data:
        key = (row["image_id"], row["patient_id"], row["tissue"])

        if key in image_data_dict:
            # Duplicate found - prefer TIF over JPG
            existing_type = image_data_dict[key]["image_type"]
            current_type = row["image_type"]

            if current_type == "tif" and existing_type == "jpg":
                # Replace JPG with TIF
                image_data_dict[key] = row
                jpg_removed += 1
            elif current_type == "jpg" and existing_type == "tif":
                # Keep existing TIF, skip current JPG
                jpg_removed += 1
            # Otherwise keep existing (both same format)
        else:
            # First occurrence of this key
            image_data_dict[key] = row

    # Convert dict back to list
    image_data = list(image_data_dict.values())

    log_message(f"Total image rows after deduplication: {len(image_data)}", log_file)
    log_message(f"Removed {jpg_removed} JPG rows where TIF version exists", log_file)

    # Save to CSV files in metadata directory
    pd.DataFrame(gene_data).to_csv(os.path.join(metadata_dir, "genes.csv"), index=False)
    pd.DataFrame(antibody_data).to_csv(
        os.path.join(metadata_dir, "antibodies.csv"), index=False
    )
    pd.DataFrame(tissue_data).to_csv(
        os.path.join(metadata_dir, "tissues.csv"), index=False
    )
    pd.DataFrame(cell_data).to_csv(os.path.join(metadata_dir, "cells.csv"), index=False)
    pd.DataFrame(patient_data).to_csv(
        os.path.join(metadata_dir, "patients.csv"), index=False
    )
    pd.DataFrame(image_data).to_csv(
        os.path.join(metadata_dir, "images.csv"), index=False
    )

    log_message(f"Data saved as CSV files in '{metadata_dir}'", log_file)
    log_message(f"- {len(gene_data)} genes", log_file)
    log_message(f"- {len(antibody_data)} antibodies", log_file)
    log_message(f"- {len(image_data)} images", log_file)

    # Log download status statistics
    ftp_count = sum(
        1 for img in image_data if img["download_status"] == "ftp_available"
    )
    pending_count = sum(1 for img in image_data if img["download_status"] == "pending")
    log_message(f"- {ftp_count} images available via FTP", log_file)
    log_message(f"- {pending_count} images need web download", log_file)


def save_as_json(all_data, output_dir, log_file=None):
    """Save the extracted data as JSON files in the metadata directory"""
    metadata_dir = os.path.join(output_dir, "metadata")
    os.makedirs(metadata_dir, exist_ok=True)

    # Group data by gene
    gene_data = {}

    for antibody_entry in all_data:
        ensembl_id = antibody_entry["gene"]["ensembl_id"]

        if not ensembl_id:
            continue

        if ensembl_id not in gene_data:
            gene_data[ensembl_id] = {
                "ensembl_id": ensembl_id,
                "name": antibody_entry["gene"]["name"],
                "synonyms": antibody_entry["gene"]["synonyms"],
                "antibodies": [],
            }

        # Add this antibody to the gene's data
        gene_data[ensembl_id]["antibodies"].append(antibody_entry)

    # Save full data to a file in metadata directory
    with open(os.path.join(metadata_dir, "hpa_full_data.json"), "w") as f:
        json.dump(gene_data, f)

    # Also save individual files per gene in metadata/genes directory
    gene_dir = os.path.join(metadata_dir, "genes")
    os.makedirs(gene_dir, exist_ok=True)

    for ensembl_id, data in gene_data.items():
        with open(os.path.join(gene_dir, f"{ensembl_id}.json"), "w") as f:
            json.dump(data, f, indent=2)

    log_message(f"Gene data saved as JSON files in '{metadata_dir}'", log_file)
    log_message(f"Created {len(gene_data)} gene-specific JSON files", log_file)


def save_image_metadata(all_data, output_dir, ftp_path_mapping=None, log_file=None):
    """
    Save metadata organized by image for fast retrieval.
    Files are saved in output_dir/grandparent/parent/metadata/image_id.json
    """
    log_message(f"Starting image metadata extraction...", log_file)
    start_time = time.time()
    last_update_time = start_time

    # Stats tracking
    image_count = 0
    errors_count = 0
    ftp_images_count = 0
    web_images_count = 0
    assigner = BatchFolderAssigner()

    # image_id -> metadata; all (tissue) annotations for an image are aggregated
    image_metadata_dict = {}

    # Process all data
    for antibody_entry in all_data:
        antibody_id = antibody_entry["id"]
        gene_info = antibody_entry["gene"]
        ensembl_id = gene_info["ensembl_id"]
        gene_name = gene_info["name"]

        # Process all tissues
        for tissue_entry in antibody_entry["tissues"]:
            tissue = tissue_entry["tissue"]
            organ = tissue_entry["organ"]
            ontology_terms = tissue_entry["ontology_terms"]

            # Process patients and their images
            for patient in tissue_entry["patients"]:
                patient_id = patient["patient_id"]
                sex = patient["sex"]
                age = patient["age"]

                for img in patient["images"]:
                    image_url = img.get("url")
                    image_type = img.get("type")

                    # Skip if URL is None or empty
                    if not image_url:
                        errors_count += 1
                        continue

                    try:
                        # Extract image ID from URL
                        image_filename = os.path.basename(image_url)
                        image_id = os.path.splitext(image_filename)[0]

                        # Skip if we couldn't extract a valid image ID
                        if not image_id:
                            errors_count += 1
                            continue

                        grandparent_folder, parent_folder, source = resolve_image_folders(
                            image_url, antibody_id, image_id, ftp_path_mapping, assigner
                        )
                        if source == "ftp":
                            ftp_images_count += 1
                        else:
                            web_images_count += 1

                        tissue_entry_out = {
                            "tissue": tissue,
                            "organ": organ,
                            "ontology_terms": ontology_terms,
                            "tissue_cells": [dict(c) for c in tissue_entry["tissue_cells"]],
                        }

                        if image_id in image_metadata_dict:
                            meta = image_metadata_dict[image_id]
                            # Prefer the TIF URL if this occurrence has one
                            if image_type == "tif" and meta["image_type"] != "tif":
                                meta["image_url"] = image_url
                                meta["image_type"] = "tif"
                            if (tissue, organ, ontology_terms) not in {
                                (t["tissue"], t["organ"], t["ontology_terms"])
                                for t in meta["tissues"]
                            }:
                                meta["tissues"].append(tissue_entry_out)
                        else:
                            image_metadata_dict[image_id] = {
                                "image_id": image_id,
                                "image_url": image_url,
                                "image_type": image_type,
                                "antibody_id": antibody_id,
                                "ensembl_id": ensembl_id,
                                "gene_name": gene_name,
                                "patient_id": patient_id,
                                "sex": sex,
                                "age": age,
                                "path": {
                                    "grandparent": grandparent_folder,
                                    "parent": parent_folder,
                                },
                                "tissues": [tissue_entry_out],
                            }

                        image_count += 1

                        # Print status update every minute
                        current_time = time.time()
                        if current_time - last_update_time > 60:
                            elapsed = current_time - start_time
                            images_per_sec = image_count / elapsed if elapsed > 0 else 0
                            log_message(
                                f"Processed {image_count} images ({images_per_sec:.1f} images/sec)",
                                log_file,
                            )
                            log_message(
                                f"- FTP: {ftp_images_count}, Web: {web_images_count}",
                                log_file,
                            )
                            last_update_time = current_time

                    except Exception as e:
                        errors_count += 1
                        log_message(f"Error: {str(e)}", log_file)
                        continue

    # Save deduplicated metadata JSON files
    log_message(
        f"Saving {len(image_metadata_dict)} deduplicated metadata JSON files...",
        log_file,
    )
    saved_count = 0
    for metadata in image_metadata_dict.values():
        try:
            image_id = metadata["image_id"]
            grandparent_folder = metadata["path"]["grandparent"]
            parent_folder = metadata["path"]["parent"]

            # Save in output_dir/grandparent/parent/metadata/
            metadata_path = os.path.join(
                output_dir, grandparent_folder, parent_folder, "metadata"
            )
            os.makedirs(metadata_path, exist_ok=True)

            # Save the metadata file
            metadata_file = os.path.join(metadata_path, f"{image_id}.json")
            with open(metadata_file, "w") as f:
                json.dump(metadata, f)
            saved_count += 1
        except Exception as e:
            log_message(f"Error saving metadata for {image_id}: {str(e)}", log_file)
            continue

    log_message(f"Saved {saved_count} metadata JSON files", log_file)

    # Create global metadata directory for other files
    global_metadata_dir = os.path.join(output_dir, "metadata")
    os.makedirs(global_metadata_dir, exist_ok=True)

    # Save a parent folder mapping file for reference
    parent_mapping_file = os.path.join(
        global_metadata_dir, "parent_folder_mapping.json"
    )
    with open(parent_mapping_file, "w") as f:
        json.dump(assigner.parent_folders, f, indent=2)

    elapsed = time.time() - start_time
    log_message(
        f"Metadata saved for {image_count} images in {elapsed:.1f} seconds", log_file
    )
    log_message(f"- FTP structure: {ftp_images_count}", log_file)
    log_message(f"- Web structure: {web_images_count}", log_file)
    log_message(
        f"Created {len(assigner.parent_folders)} parent folders across {assigner.next_batch_id - 11} batches",
        log_file,
    )

    if errors_count > 0:
        log_message(
            f"WARNING: Encountered {errors_count} errors during image metadata processing",
            log_file,
        )

    return image_count, errors_count


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="Extract gene, antibody and tissue expression data from HPA XML file"
    )
    parser.add_argument(
        "--xml-file",
        required=True,
        help="Path to proteinatlas.xml (https://www.proteinatlas.org/download/proteinatlas.xml.gz)",
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        help="Output directory; CSVs go to <output-dir>/metadata/ (default: directory of the XML file)",
        default=None,
    )
    parser.add_argument(
        "--format",
        "-f",
        choices=["csv", "json", "img_json", "all"],
        default="csv",
        help="Output format: csv, json, img_json, or all (default: csv)",
    )
    parser.add_argument(
        "--ftp-paths",
        help="Optional IDR table (build_idr_ftp_paths.py output) so local_path follows the IDR batch folders",
        default=None,
    )
    parser.add_argument(
        "--log-file",
        help="Log file (default: <output-dir>/hpa_xml_parser.log)",
        default=None,
    )

    args = parser.parse_args()

    # Process the XML file
    extract_hpa_data(
        args.xml_file, args.output_dir, args.format, args.ftp_paths, args.log_file
    )


if __name__ == "__main__":
    main()
