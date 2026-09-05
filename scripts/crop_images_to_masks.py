#!/usr/bin/env python3
"""
Crop HPA tissue images to the bounding box of their tissue mask.

Input is a dataset CSV (``--csv-file`` + ``--img-root``; paths are
``<img-root>/<local_path>/<image_id>.tif``) or a directory tree (``--root``). Masks are
read from ``<image_dir>/masks/<image_id>_mask.<npy|png|pt>`` (``generate_tissue_masks.py``)
and crops are written to ``<image_dir>/crops/<image_id>_crop.<npy|png|tif>``.

Example:
    python crop_images_to_masks.py --csv-file HPA_pilot_dataset_idr.csv.gz --img-root ./hpa_tissue \\
        --mask-format npy --padding 20 --num-workers 8
"""
import argparse
from pathlib import Path
from PIL import Image
import numpy as np
import cv2
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
import pandas as pd
import os


def load_mask(mask_path):
    """
    Load a mask from .pt, .png, or .npy format.

    Args:
        mask_path: Path to mask file

    Returns:
        numpy array of the mask (binary, 0 or 1)
    """
    mask_path = Path(mask_path)

    if mask_path.suffix == '.pt':
        import torch  # only needed for .pt masks
        mask = torch.load(mask_path, map_location='cpu', weights_only=True)
        if isinstance(mask, torch.Tensor):
            mask = mask.numpy()
    elif mask_path.suffix == '.png':
        # Load PNG image
        mask = np.array(Image.open(mask_path))
        # Convert to binary (assume non-zero values are tissue)
        mask = (mask > 0).astype(np.uint8)
    elif mask_path.suffix == '.npy':
        # Load NPY file
        mask = np.load(mask_path)
        # Ensure binary format
        if mask.max() > 1:
            mask = (mask > 0).astype(np.uint8)
    else:
        raise ValueError(f"Unsupported mask format: {mask_path.suffix}")

    return mask


def get_bounding_box(mask):
    """
    Get the bounding box of non-zero pixels in the mask.
    
    Args:
        mask: Binary mask array
        
    Returns:
        tuple: (min_y, min_x, max_y, max_x) or None if no tissue found
    """
    # Find coordinates of non-zero pixels
    coords = np.where(mask > 0)
    
    if len(coords[0]) == 0:
        return None
    
    min_y, max_y = coords[0].min(), coords[0].max()
    min_x, max_x = coords[1].min(), coords[1].max()
    
    return min_y, min_x, max_y, max_x


def crop_image_to_mask(img_path, mask_path, crop_path, padding=0, save_format="npy"):
    """
    Crop an image to the bounding box of its tissue mask.
    
    Args:
        img_path: Path to input image
        mask_path: Path to tissue mask
        crop_path: Path to save cropped image
        padding: Additional padding around bounding box
        save_format: "npy" (raw uint8 array, fastest to load), "png" (lossless, smaller,
            slower to decode) or "tif" (zlib/deflate-compressed TIFF, smaller, slower to decode)
    """
    try:
        # Load image and mask
        img = Image.open(img_path)
        mask = load_mask(mask_path)
        
        # Get bounding box
        bbox = get_bounding_box(mask)
        if bbox is None:
            print(f"Warning: No tissue found in mask for {img_path}")
            return
        
        min_y, min_x, max_y, max_x = bbox
        
        # Add padding
        if padding > 0:
            h, w = mask.shape
            min_y = max(0, min_y - padding)
            min_x = max(0, min_x - padding)
            max_y = min(h - 1, max_y + padding)
            max_x = min(w - 1, max_x + padding)
        
        # Crop image (PIL uses (left, top, right, bottom) format)
        cropped = img.crop((min_x, min_y, max_x + 1, max_y + 1))
        
        # Save in the specified format
        if save_format == "npy":
            np.save(crop_path, np.array(cropped, dtype=np.uint8))
        elif save_format == "tif":
            cropped.save(crop_path, "TIFF", compression="tiff_adobe_deflate")
        else:
            cropped.save(crop_path, "PNG", optimize=True)
        
    except Exception as e:
        print(f"Error processing {img_path}: {e}")


def get_mask_path(img_path, mask_format="pt"):
    """
    Get the expected mask path for an image.

    Args:
        img_path: Path to image file
        mask_format: Format of mask file ("pt", "png", or "npy")

    Returns:
        Path to mask file
    """
    img_path = Path(img_path)
    mask_dir = img_path.parent / "masks"
    if mask_format == "pt":
        suffix = "_mask.pt"
    elif mask_format == "npy":
        suffix = "_mask.npy"
    else:  # png
        suffix = "_mask.png"
    return mask_dir / f"{img_path.stem}{suffix}"


def get_crop_path(img_path, save_format="npy"):
    """
    Get the output path for cropped image.
    
    Args:
        img_path: Path to original image
        save_format: Format to save ("npy" or "png")
        
    Returns:
        Path to save cropped image
    """
    img_path = Path(img_path)
    crop_dir = img_path.parent / "crops"
    crop_dir.mkdir(exist_ok=True)
    extension = {"npy": ".npy", "png": ".png", "tif": ".tif"}[save_format]
    return crop_dir / f"{img_path.stem}_crop{extension}"


def process_image(args):
    """
    Process a single image: load, crop to mask bounding box, save as NPY.
    
    Args:
        args: tuple of (img_path, mask_format, padding, save_format)
    """
    img_path, mask_format, padding, save_format = args
    
    # Get paths
    mask_path = get_mask_path(img_path, mask_format)
    crop_path = get_crop_path(img_path, save_format)
    
    # Skip if crop already exists
    if crop_path.exists():
        return
    
    # Skip if mask doesn't exist
    if not mask_path.exists():
        print(f"Warning: Mask not found for {img_path}")
        return
    
    # Crop image
    crop_image_to_mask(img_path, mask_path, crop_path, padding, save_format)


def get_image_paths_from_csv(csv_file, img_root=""):
    """
    Extract image paths from CSV file.
    
    Args:
        csv_file: Path to CSV file with columns 'local_path' and 'image_id'
        img_root: Root directory for images (optional)
    
    Returns:
        List of full image paths
    """
    df = pd.read_csv(csv_file)
    
    # Check required columns
    required_cols = ['local_path', 'image_id']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"CSV file missing required columns: {missing_cols}")
    
    image_paths = []
    for _, row in df.iterrows():
        local_path = row['local_path']
        image_id = row['image_id']
        
        # Construct full path: img_root/local_path/image_id.tif
        if os.path.isabs(local_path):
            img_path = os.path.join(local_path, f"{image_id}.tif")
        else:
            img_path = os.path.join(img_root, local_path, f"{image_id}.tif")
        
        # Only add if file exists
        if os.path.exists(img_path):
            image_paths.append(img_path)
        else:
            print(f"Warning: Image file not found: {img_path}")
    
    return image_paths


def main():
    parser = argparse.ArgumentParser(
        description="Crop images to tissue mask bounding boxes and save as PNG files."
    )
    
    # Make root and csv-file mutually exclusive
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--root", type=str, 
                            help="Root directory to recursively search for .tif files")
    input_group.add_argument("--csv-file", type=str, 
                            help="CSV file with image paths (requires 'local_path' and 'image_id' columns)")
    
    parser.add_argument("--img-root", type=str, default=None,
                       help="Root directory for images when using --csv-file (required with --csv-file)")
    parser.add_argument("--mask-format", choices=["png", "pt", "npy"], default="pt",
                       help="Format of mask files")
    parser.add_argument("--save-format", choices=["npy", "png", "tif"], default="npy",
                       help="npy: raw uint8 array, fastest to load; png or tif (deflate): 2-4x smaller, slower to decode")
    parser.add_argument("--padding", type=int, default=5,
                       help="Additional padding around bounding box")
    parser.add_argument("--max-images", type=int, default=None,
                       help="Maximum number of images to process")
    parser.add_argument("--num-workers", type=int, default=1,
                       help="Number of parallel workers")
    
    args = parser.parse_args()

    if args.csv_file and not args.img_root:
        parser.error("--img-root is required with --csv-file")

    # Get list of image files
    if args.csv_file:
        print(f"Reading image paths from CSV: {args.csv_file}")
        files = get_image_paths_from_csv(args.csv_file, args.img_root)
        files = [Path(f) for f in files]  # Convert to Path objects
    else:
        print(f"Searching for .tif files in: {args.root}")
        files = list(Path(args.root).rglob("*.tif"))
    
    if args.max_images:
        files = files[:args.max_images]

    print(f"Processing {len(files)} images")
    print(f"Save format: {args.save_format}")
    if len(files) == 0:
        print("No images found to process!")
        return
    
    # Process images in parallel
    if len(files) <= 10000:  # For smaller datasets, submit all at once
        with ProcessPoolExecutor(max_workers=args.num_workers) as executor:
            # Submit tasks one by one - this starts processing immediately
            futures = []
            for p in files:
                future = executor.submit(process_image, (str(p), args.mask_format, args.padding, args.save_format))
                futures.append(future)
            
            # Process completed tasks with progress bar
            for future in tqdm(as_completed(futures), total=len(futures), desc="Processing images"):
                try:
                    future.result()  # This will raise any exceptions that occurred
                except Exception as e:
                    print(f"Error in worker: {e}")
    else:
        # Option 2: Process in chunks for large datasets
        chunk_size = 1000  # Process 1000 images at a time
        total_processed = 0
        
        with tqdm(total=len(files), desc="Processing images") as pbar:
            for i in range(0, len(files), chunk_size):
                chunk = files[i:i + chunk_size]
                
                with ProcessPoolExecutor(max_workers=args.num_workers) as executor:
                    futures = []
                    for p in chunk:
                        future = executor.submit(process_image, (str(p), args.mask_format, args.padding, args.save_format))
                        futures.append(future)
                    
                    # Process this chunk
                    for future in as_completed(futures):
                        try:
                            future.result()
                            total_processed += 1
                            pbar.update(1)
                        except Exception as e:
                            print(f"Error in worker: {e}")
                            pbar.update(1)

    print("Cropping completed!")


if __name__ == "__main__":
    main() 