from PIL import Image
import numpy as np
import os
from itertools import permutations
import pandas as pd
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
import multiprocessing

import matplotlib.pyplot as plt


def image_to_grayscale_matrix(image_path):
    img = Image.open(image_path).convert("L")
    return np.array(img)


def ordinal_pattern_distribution(matrix, dx=2, dy=2):
    patterns = list(permutations(range(dx * dy)))
    pattern_counts = {p: 0 for p in patterns}
    num_total_patterns = len(patterns)

    if matrix.shape[0] < dx or matrix.shape[1] < dy:
        # Not enough data to form any patches
        return np.zeros(num_total_patterns)

    for i in range(matrix.shape[0] - dx + 1):
        for j in range(matrix.shape[1] - dy + 1):
            patch = matrix[i:i+dx, j:j+dy].flatten()
            ranks = tuple(np.argsort(patch))
            pattern_counts[ranks] += 1

    total = sum(pattern_counts.values())
    if total == 0:
        return np.zeros(num_total_patterns)
        
    probs = np.array([count / total for count in pattern_counts.values()])
    return probs


def permutation_entropy(probs):
    n_total_patterns = len(probs) # n = (dx*dy)!

    if n_total_patterns <= 1:
        return 0.0

    probs_filtered = probs[probs > 0] # use only non-zero probabilities

    if len(probs_filtered) == 0: # All patterns have zero probability (empty/too small image)
        return 0.0
    
    normalized_entropy = shannon_entropy(probs_filtered) / np.log(n_total_patterns)
    return normalized_entropy

def shannon_entropy(p):
    p = p[p > 0]
    return -np.sum(p * np.log(p))


def statistical_complexity(probs):
    uniform = np.ones_like(probs) / len(probs)
    M = 0.5 * (probs + uniform)
    D_JS = shannon_entropy(M) - 0.5 * (shannon_entropy(probs) + shannon_entropy(uniform))
    D_max = -0.5 * ((len(probs)+1)/len(probs)*np.log(len(probs)+1) + np.log(len(probs)) - 2*np.log(2*len(probs)))
    H = permutation_entropy(probs)
    C = D_JS * H / D_max
    return C


def process_single_image(image_path):
    """
    Process a single image and return its complexity metrics.
    This function is designed to be used with parallel processing.
    """
    try:
        filename = os.path.basename(image_path)
        # Extract MBID from filename
        mbid = filename.replace('.jpg', '').replace('.png', '').replace('.jpeg', '')
        
        # Convert image to grayscale matrix
        matrix = image_to_grayscale_matrix(image_path)
        
        # Calculate ordinal pattern distribution
        probs = ordinal_pattern_distribution(matrix, dx=2, dy=2)
        
        # Calculate permutation entropy and statistical complexity
        entropy = permutation_entropy(probs)
        complexity = statistical_complexity(probs)
        
        return {
            'album_group_mbid': mbid,
            'permutation_entropy': entropy,
            'statistical_complexity': complexity,
            'status': 'success'
        }
        
    except Exception as e:
        filename = os.path.basename(image_path)
        mbid = filename.replace('.jpg', '').replace('.png', '').replace('.jpeg', '')
        return {
            'album_group_mbid': mbid,
            'permutation_entropy': np.nan,
            'statistical_complexity': np.nan,
            'status': 'error',
            'error': str(e)
        }


def analyze_dataset(image_files_full_paths, max_workers=None):
    """
    Analyze a dataset of images using parallel processing with progress tracking.
    
    Args:
        image_files_full_paths: List of full paths to image files
        max_workers: Maximum number of worker processes (None for automatic)
    """
    if max_workers is None:
        max_workers = min(multiprocessing.cpu_count(), len(image_files_full_paths))
    
    print(f"Analyzing {len(image_files_full_paths)} images using {max_workers} workers...")
    
    results = []
    errors = []
    
    # Use ProcessPoolExecutor for CPU-bound image processing
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        future_to_path = {executor.submit(process_single_image, path): path 
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
                        'permutation_entropy': np.nan,
                        'statistical_complexity': np.nan,
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

def process_album_images_and_merge(csv_file, image_directory, output_file, max_workers=None):
    """
    Process all album images, calculate complexity metrics, and merge with CSV data.
    
    Args:
        csv_file: Path to CSV file with album data
        image_directory: Directory containing album cover images
        output_file: Path for output CSV file with complexity metrics
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
        complexity_results = analyze_dataset(relevant_image_files, max_workers=max_workers)
    
    # Convert results to DataFrame and clean up
    complexity_df = pd.DataFrame(complexity_results)
    # Remove the status column as it's not needed in the final output
    complexity_df = complexity_df.drop(['status'], axis=1, errors='ignore')
    if 'error' in complexity_df.columns:
        complexity_df = complexity_df.drop(['error'], axis=1)
    
    successful_count = complexity_df['permutation_entropy'].notna().sum() if len(complexity_df) > 0 else 0
    print(f"Successfully analyzed {successful_count} images")
    
    # Merge with original CSV data
    print("Merging with original dataset...")
    merged_df = df.merge(complexity_df, on='album_group_mbid', how='left')
    
    # Fill NaN values for albums without images
    merged_df['permutation_entropy'] = merged_df['permutation_entropy'].fillna(np.nan)
    merged_df['statistical_complexity'] = merged_df['statistical_complexity'].fillna(np.nan)
    
    # Save to new file
    print(f"Saving results to: {output_file}")
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    merged_df.to_csv(output_file, index=False)
    
    # Print summary statistics
    entropy_count = merged_df['permutation_entropy'].notna().sum()
    complexity_count = merged_df['statistical_complexity'].notna().sum()
    
    print(f"\nSummary:")
    print(f"  Total albums: {len(merged_df):,}")
    print(f"  Albums with analysis: {entropy_count:,} ({entropy_count/len(merged_df)*100:.1f}%)")
    print(f"  Albums without analysis: {len(merged_df) - entropy_count:,}")
    
    if entropy_count > 0:
        print(f"\nEntropy stats: mean={merged_df['permutation_entropy'].mean():.3f}, "
              f"std={merged_df['permutation_entropy'].std():.3f}")
        print(f"Complexity stats: mean={merged_df['statistical_complexity'].mean():.3f}, "
              f"std={merged_df['statistical_complexity'].std():.3f}")
    
    print(f"Analysis complete. Results saved to: {output_file}")

if __name__ == "__main__":
    import argparse
    
    # parser = argparse.ArgumentParser(description='Analyze album cover image complexity')
    # parser.add_argument('-i', '--input-csv', default='../data/billboard/billboard_album_final_super_genres.csv',
    #                    help='Input CSV file with album data')
    # parser.add_argument('-d', '--image-dir', default='../data/img_all',
    #                    help='Directory containing album cover images')
    # parser.add_argument('-o', '--output', default='../results/plane/billboard_album_entr_compl.csv',
    #                    help='Output CSV file with complexity metrics')
    # parser.add_argument('-w', '--workers', type=int, default=None,
    #                    help='Number of parallel workers (default: auto-detect)')
    
    # args = parser.parse_args()
    
    # Configuration
    max_workers = None  # None for auto-detect, or set to specific number (e.g., 4, 8)
    
    billboard_csv = '../data/billboard_album_final_super_genres.csv'
    mumu_msdi_csv = '../data/merged_dataset_mumu_msdi_final_cleaned.csv'


    # process_album_images_and_merge(
    #     billboard_csv, 
    #     '../data/img_all', 
    #     '../results/plane/billboard_album_entr_compl.csv',
    #     max_workers=max_workers
    # )

    process_album_images_and_merge(
        mumu_msdi_csv, 
        '../data/img_all', 
        '../results/plane/mumu_msdi_album_entr_compl.csv',
        max_workers=max_workers
    )