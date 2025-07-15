from PIL import Image
import numpy as np
import os
import pandas as pd
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
import multiprocessing
import zipfile
import io
import tempfile


def image_to_bitmap_bytes(image_path, target_size=None, preserve_aspect_ratio=True):
    """
    Convert image to uncompressed bitmap bytes for compression analysis.
    
    Args:
        image_path: Path to the image file
        target_size: Tuple (width, height) to resize to, or None to keep original size
        preserve_aspect_ratio: If True, pad with black to preserve aspect ratio when resizing
    
    Returns:
        bytes: Raw bitmap data (RGB pixels)
    """
    with Image.open(image_path) as img:
        # Convert to RGB to ensure consistent format
        img = img.convert("RGB")
        
        if target_size is not None:
            target_width, target_height = target_size
            
            if preserve_aspect_ratio:
                # Calculate scaling to fit within target size while preserving aspect ratio
                img_width, img_height = img.size
                scale = min(target_width / img_width, target_height / img_height)
                
                # Resize maintaining aspect ratio
                new_width = int(img_width * scale)
                new_height = int(img_height * scale)
                img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
                
                # Create black canvas of target size and paste resized image
                canvas = Image.new("RGB", target_size, (0, 0, 0))
                offset_x = (target_width - new_width) // 2
                offset_y = (target_height - new_height) // 2
                canvas.paste(img, (offset_x, offset_y))
                img = canvas
            else:
                # Simple resize without preserving aspect ratio
                img = img.resize(target_size, Image.Resampling.LANCZOS)
        
        # Convert to raw bytes (RGB format)
        return img.tobytes()


def calculate_compression_ratio(data_bytes):
    """
    Calculate ZIP compression ratio for raw data bytes.
    Returns compression ratio (0-1, where higher values indicate higher complexity).
    """
    original_size = len(data_bytes)
    if original_size == 0:
        return np.nan
    
    # Create a temporary in-memory zip file
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, 'w', zipfile.ZIP_DEFLATED, compresslevel=9) as zip_file:
        zip_file.writestr('data', data_bytes)
    
    compressed_size = buffer.tell()
    compression_ratio = compressed_size / original_size
    
    return compression_ratio


def calculate_compression_complexity(image_path, target_size=None, preserve_aspect_ratio=True):
    """
    Calculate complexity metrics based on ZIP compression of bitmap data.
    
    Args:
        image_path: Path to the image file
        target_size: Tuple (width, height) to resize to, or None to keep original size
        preserve_aspect_ratio: If True, pad with black to preserve aspect ratio when resizing
    
    Returns:
        tuple: (compression_ratio, complexity_score, final_width, final_height, data_size)
    """
    try:
        # Convert image to uncompressed bitmap bytes
        bitmap_bytes = image_to_bitmap_bytes(image_path, target_size, preserve_aspect_ratio)
        
        # Calculate compression ratio
        compression_ratio = calculate_compression_ratio(bitmap_bytes)
        
        # Use compression ratio directly as complexity score
        # Higher compression ratio = harder to compress = higher complexity
        complexity_score = compression_ratio
        
        # Calculate final dimensions
        if target_size is not None:
            final_width, final_height = target_size
        else:
            with Image.open(image_path) as img:
                final_width, final_height = img.size
        
        data_size = len(bitmap_bytes)
        
        return compression_ratio
        
    except Exception as e:
        return np.nan


def process_single_image(image_path, target_size=None, preserve_aspect_ratio=True):
    """
    Process a single image and return its compression-based complexity metrics.
    This function is designed to be used with parallel processing.
    
    Args:
        image_path: Path to the image file
        target_size: Tuple (width, height) to resize to, or None to keep original size
        preserve_aspect_ratio: If True, pad with black to preserve aspect ratio when resizing
    """
    try:
        filename = os.path.basename(image_path)
        # Extract MBID from filename
        mbid = filename.replace('.jpg', '').replace('.png', '').replace('.jpeg', '')
        
        # Calculate compression metrics
        compression_ratio = calculate_compression_complexity(
            image_path, target_size, preserve_aspect_ratio
        )
        
        # Get original image info
        with Image.open(image_path) as img:
            orig_width, orig_height = img.size
            file_size = os.path.getsize(image_path)
        
        return {
            'album_group_mbid': mbid,
            'compression_ratio': compression_ratio,
            'status': 'success'
        }
        
    except Exception as e:
        filename = os.path.basename(image_path)
        mbid = filename.replace('.jpg', '').replace('.png', '').replace('.jpeg', '')
        return {
            'album_group_mbid': mbid,
            'compression_ratio': np.nan,
            'status': 'error',
            'error': str(e)
        }


def analyze_dataset(image_files_full_paths, target_size=None, preserve_aspect_ratio=True, max_workers=None):
    """
    Analyze a dataset of images using parallel processing with progress tracking.
    
    Args:
        image_files_full_paths: List of full paths to image files
        target_size: Tuple (width, height) to resize to, or None to keep original size
        preserve_aspect_ratio: If True, pad with black to preserve aspect ratio when resizing
        max_workers: Maximum number of worker processes (None for automatic)
    """
    if max_workers is None:
        max_workers = min(multiprocessing.cpu_count(), len(image_files_full_paths))
    
    size_info = f" (resized to {target_size})" if target_size else " (original size)"
    print(f"Analyzing {len(image_files_full_paths)} images using {max_workers} workers{size_info}...")
    
    results = []
    errors = []
    
    # Use ProcessPoolExecutor for CPU-bound image processing
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        future_to_path = {executor.submit(process_single_image, path, target_size, preserve_aspect_ratio): path 
                         for path in image_files_full_paths}
        
        # Process completed tasks with progress bar
        with tqdm(total=len(image_files_full_paths), 
                 desc="Processing", 
                 unit="img") as pbar:
            
            for future in as_completed(future_to_path):
                image_path = future_to_path[future]
                try:
                    result = future.result()
                    results.append(result)
                    
                    if result['status'] == 'error':
                        errors.append(f"{os.path.basename(image_path)}: {result['error']}")
                    
                except Exception as exc:
                    filename = os.path.basename(image_path)
                    mbid = filename.replace('.jpg', '').replace('.png', '').replace('.jpeg', '')
                    results.append({
                        'album_group_mbid': mbid,
                        'compression_ratio': np.nan,
                        'status': 'error'
                    })
                    errors.append(f"{filename}: {str(exc)}")
                
                pbar.update(1)
    
    # Print error summary if any
    if errors:
        print(f"Warning: {len(errors)} images failed to process")
        if len(errors) <= 3:
            for error in errors:
                print(f"  {error}")
        else:
            for error in errors[:3]:
                print(f"  {error}")
            print(f"  ... and {len(errors) - 3} more errors")
    
    successful_results = [r for r in results if r['status'] == 'success']
    print(f"Successfully processed {len(successful_results)}/{len(image_files_full_paths)} images")
    
    return results


def process_album_images_and_merge(csv_file, image_directory, output_file, 
                                 target_size=None, preserve_aspect_ratio=True, max_workers=None):
    """
    Process all album images, calculate compression-based complexity metrics, and merge with CSV data.
    
    Args:
        csv_file: Path to CSV file with album data
        image_directory: Directory containing album cover images
        output_file: Path for output CSV file with complexity metrics
        target_size: Tuple (width, height) to resize to, or None to keep original size
        preserve_aspect_ratio: If True, pad with black to preserve aspect ratio when resizing
        max_workers: Maximum number of worker processes for parallel processing (None for automatic)
    """
    print(f"Loading CSV data from {os.path.basename(csv_file)}...")
    df = pd.read_csv(csv_file)
    print(f"Loaded {len(df)} albums")
    
    # Get set of album MBIDs from CSV for filtering
    album_mbids = set(df['album_group_mbid'].astype(str))
    
    # Get list of image files
    print(f"Scanning image directory: {image_directory}")
    all_image_files = []
    if os.path.exists(image_directory):
        all_files = os.listdir(image_directory)
        for filename in tqdm(all_files, desc="Scanning", unit="file"):
            if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                all_image_files.append(os.path.join(image_directory, filename))
    else:
        print(f"Error: Image directory not found: {image_directory}")
        return
    
    print(f"Found {len(all_image_files)} total image files")
    
    # Filter images to only include those that match album MBIDs in the CSV
    relevant_image_files = []
    for image_path in all_image_files:
        filename = os.path.basename(image_path)
        mbid = filename.replace('.jpg', '').replace('.png', '').replace('.jpeg', '')
        if mbid in album_mbids:
            relevant_image_files.append(image_path)
    
    print(f"Found {len(relevant_image_files)} images matching albums in CSV")
    
    if len(relevant_image_files) == 0:
        print("Warning: No matching image files found for albums in CSV!")
        # Create empty complexity results for merging
        complexity_results = []
    else:
        # Analyze only relevant images with parallel processing
        complexity_results = analyze_dataset(relevant_image_files, target_size, preserve_aspect_ratio, max_workers)
    
    # Convert results to DataFrame and clean up
    complexity_df = pd.DataFrame(complexity_results)
    # Remove the status column as it's not needed in the final output
    complexity_df = complexity_df.drop(['status'], axis=1, errors='ignore')
    if 'error' in complexity_df.columns:
        complexity_df = complexity_df.drop(['error'], axis=1)
    
    successful_count = complexity_df['compression_ratio'].notna().sum() if len(complexity_df) > 0 else 0
    print(f"Successfully analyzed {successful_count} images")
    
    # Merge with original CSV data
    print("Merging with original dataset...")
    merged_df = df.merge(complexity_df, on='album_group_mbid', how='left')
    
    # Fill NaN values for albums without images
    columns_to_fill = [
        'compression_ratio'
    ]
    for col in columns_to_fill:
        if col in merged_df.columns:
            merged_df[col] = merged_df[col].fillna(np.nan)
    
    # Save to new file
    print(f"Saving results to: {output_file}")
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    merged_df.to_csv(output_file, index=False)
    
    # Print summary statistics
    ratio_count = merged_df['compression_ratio'].notna().sum()
    
    print(f"\nSummary:")
    print(f"  Total albums: {len(merged_df):,}")
    print(f"  Albums with analysis: {ratio_count:,} ({ratio_count/len(merged_df)*100:.1f}%)")
    print(f"  Albums without analysis: {len(merged_df) - ratio_count:,}")
    
    if ratio_count > 0:
        print(f"\nCompression ratio stats: mean={merged_df['compression_ratio'].mean():.3f}, "
              f"std={merged_df['compression_ratio'].std():.3f}")
    
    print(f"Analysis complete. Results saved to: {output_file}")


if __name__ == "__main__":
    import argparse
    
    # Configuration
    max_workers = None  # None for auto-detect, or set to specific number (e.g., 4, 8)
    
    # Standard sizes for comparison
    STANDARD_SIZE = (224, 224)  # Common album cover size    # Smaller for faster processing
    
    billboard_csv = '../data/billboard_album_final_super_genres.csv'
    mumu_msdi_csv = '../data/merged_dataset_mumu_msdi_final_cleaned.csv'

    # Process with different configurations
    # process_album_images_and_merge(
    #     mumu_msdi_csv, 
    #     '../data/img_all', 
    #     '../results/zip/mumu_msdi_album_compression_512x512.csv',
    #     target_size=STANDARD_SIZE,
    #     preserve_aspect_ratio=True,
    #     max_workers=max_workers
    # )

    print("\n=== Processing with original sizes ===")
    process_album_images_and_merge(
        mumu_msdi_csv, 
        '../data/img_all', 
        '../results/zip/mumu_msdi_album_compression.csv',
        target_size=None,
        max_workers=max_workers
    )

    process_album_images_and_merge(
        billboard_csv, 
        '../data/img_all', 
        '../results/zip/billboard_album_compression.csv',
        target_size=None,
        max_workers=max_workers
    )