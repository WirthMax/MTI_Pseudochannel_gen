import argparse
import numpy as np
import tifffile
from PIL import Image
import pathlib
from numba import njit, prange
import os

@njit(parallel=True, fastmath=True)
def separation_border_inplace(image):
    x, y = image.shape

    for i in prange(1, x - 1):
        for j in range(1, y - 1):
            cur_val = image[i, j]
            if cur_val == 0:
                continue

            # Define neighbors
            neighbor_top = image[i - 1, j]
            neighbor_bottom = image[i + 1, j]
            neighbor_left = image[i, j - 1]
            neighbor_right = image[i, j + 1]
            neighbor_top_left = image[i - 1, j - 1]
            neighbor_top_right = image[i - 1, j + 1]
            neighbor_bottom_left = image[i + 1, j - 1]
            neighbor_bottom_right = image[i + 1, j + 1]

            # Check all 8 neighbors
            if (
                (neighbor_top != 0 and neighbor_top != cur_val)
                or (neighbor_bottom != 0 and neighbor_bottom != cur_val)
                or (neighbor_left != 0 and neighbor_left != cur_val)
                or (neighbor_right != 0 and neighbor_right != cur_val)
                or (neighbor_top_left != 0 and neighbor_top_left != cur_val)
                or (neighbor_top_right != 0 and neighbor_top_right != cur_val)
                or (neighbor_bottom_left != 0 and neighbor_bottom_left != cur_val)
                or (neighbor_bottom_right != 0 and neighbor_bottom_right != cur_val)
            ):
                image[i, j] = 0
    return image

def load_image(image_path):
    """Automatically detect image format and load with appropriate reader."""
    ext = pathlib.Path(image_path).suffix.lower()
    
    if ext in ['.tif', '.tiff']:
        return tifffile.imread(image_path)
    elif ext in ['.png', '.jpg', '.jpeg', '.bmp', '.gif']:
        Image.MAX_IMAGE_PIXELS = None  # Disable decompression bomb check
        return np.array(Image.open(image_path))
    else:
        raise ValueError(f"Unsupported image format: {ext}. Supported: .tif, .tiff, .png, .jpg, .jpeg, .bmp, .gif")

def save_image(output_path, image_array):
    """Save image array as TIFF using tifffile."""
    tifffile.imwrite(output_path, image_array.astype(np.uint8), compression = True)

def main():
    parser = argparse.ArgumentParser(
        description="Apply separation border processing to images or directory of images."
    )
    parser.add_argument(
        'input',
        help="Input image file or directory path"
    )
    parser.add_argument(
        '-o', '--output',
        help="Output directory (default: same as input directory)",
        default=None
    )
    parser.add_argument(
        '--single-file',
        action='store_true',
        help="Process single input file (default: process all images in directory)"
    )
    
    args = parser.parse_args()
    
    input_path = pathlib.Path(args.input)
    output_dir = pathlib.Path(args.output) if args.output else input_path.parent
    
    # Create output directory if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if args.single_file or input_path.is_file():
        # Process single file
        if not input_path.is_file():
            raise FileNotFoundError(f"Input file not found: {input_path}")
        
        print(f"Processing: {input_path}")
        img = load_image(input_path)
        print(f"Input shape: {img.shape}, unique values: {np.unique(img)}")
        
        mask = separation_border_inplace(img.copy())
        mask[mask != 0] = 1
        print(f"Output unique values: {np.unique(mask)}")
        
        output_path = output_dir / f"{input_path.stem}_MacsIQView.tif"
        save_image(output_path, mask)
        print(f"Saved: {output_path}")
        
    else:
        # Process all images in directory
        if not input_path.is_dir():
            raise NotADirectoryError(f"Input is not a directory: {input_path}")
        
        image_extensions = ['.tif', '.tiff', '.png', '.jpg', '.jpeg', '.bmp', '.gif']
        image_files = [f for f in input_path.iterdir() 
                      if f.suffix.lower() in image_extensions]
        
        if not image_files:
            print(f"No supported image files found in {input_path}")
            return
        
        print(f"Found {len(image_files)} images to process")
        
        for img_path in image_files:
            print(f"\nProcessing: {img_path}")
            img = load_image(img_path)
            print(f"Input shape: {img.shape}, unique values: {np.unique(img)}")
            
            mask = separation_border_inplace(img.copy())
            mask[mask != 0] = 1
            print(f"Output unique values: {np.unique(mask)}")
            
            output_path = output_dir / f"{img_path.stem}_MacsIQView.tif"
            save_image(output_path, mask)
            print(f"Saved: {output_path}")

if __name__ == "__main__":
    main()