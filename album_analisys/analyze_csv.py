import os
import csv
import time
from album_cover_analyzer import AlbumCoverAnalyzer
import pandas as pd
import sys
import multiprocessing as mp
from functools import partial
import threading
from queue import Queue


def load_processed_albums(processed_file):
    """Load list of already processed album_group_mbids from file"""
    if os.path.exists(processed_file):
        with open(processed_file, 'r') as f:
            processed_ids = set(line.strip() for line in f if line.strip())
        print(f"Found {len(processed_ids)} already processed albums")
        return processed_ids
    return set()


def save_processed_album(processed_file, album_id):
    """Append a processed album_group_mbid to the tracking file"""
    with open(processed_file, 'a') as f:
        f.write(f"{album_id}\n")


def save_single_result(result_data, output_csv, complexity_columns):
    """Save a single result to CSV immediately - thread safe"""
    # Check if file exists and has content each time (more reliable for multiprocessing)
    file_exists = os.path.exists(output_csv) and os.path.getsize(output_csv) > 0
    write_header = not file_exists
    
    # Create a single row dataframe
    row_data = result_data.copy()
    for j, score in enumerate(result_data['scores']):
        row_data[complexity_columns[j]] = round(score, 4)
    row_data['complexity_overall_score'] = round(result_data['overall'], 4)
    
    # Remove the 'scores' and 'overall' keys as they're now in separate columns
    del row_data['scores']
    del row_data['overall']
    del row_data['process_time']
    
    # Convert to dataframe
    df_row = pd.DataFrame([row_data])
    
    # Use a file lock for thread safety
    lock_file = f"{output_csv}.lock"
    max_retries = 10
    retry_delay = 0.1
    
    for attempt in range(max_retries):
        try:
            # Try to acquire lock
            if not os.path.exists(lock_file):
                # Create lock file
                with open(lock_file, 'w') as lock:
                    lock.write("locked")
                
                try:
                    # Re-check file existence inside lock (another process might have created it)
                    file_exists = os.path.exists(output_csv) and os.path.getsize(output_csv) > 0
                    write_header = not file_exists
                    
                    # Save to CSV
                    mode = 'w' if write_header else 'a'
                    df_row.to_csv(output_csv, mode=mode, index=False, header=write_header)
                    
                finally:
                    # Release lock
                    if os.path.exists(lock_file):
                        os.remove(lock_file)
                
                return  # Success, exit
                
        except Exception as e:
            if attempt == max_retries - 1:
                print(f"Failed to save result after {max_retries} attempts: {e}")
                raise
            time.sleep(retry_delay * (attempt + 1))
            continue
        
        # If lock file exists, wait and retry
        time.sleep(retry_delay)


def setup_analysis_environment(output_csv):
    """Setup directories and file paths for analysis"""
    # Ensure output directory exists
    output_dir = os.path.dirname(output_csv)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Setup progress tracking files
    base_name = os.path.splitext(os.path.basename(output_csv))[0]
    processed_file = os.path.join(output_dir, f"{base_name}_processed_albums.txt")
    progress_file = os.path.join(output_dir, f"{base_name}_progress.txt")
    
    return processed_file, progress_file


def prepare_dataframe(input_csv, processed_albums):
    """Load CSV and prepare dataframe with complexity columns"""
    print(f"Loading CSV file: {input_csv}")
    df = pd.read_csv(input_csv)
    
    if 'album_group_mbid' not in df.columns:
        raise ValueError("CSV must contain 'album_group_mbid' column")
    
    # Filter out already processed albums
    if processed_albums:
        original_count = len(df)
        df = df[~df['album_group_mbid'].isin(processed_albums)]
        print(f"Filtered out {original_count - len(df)} already processed albums")
    
    if len(df) == 0:
        print("All albums have already been processed!")
        return df, []
    
    # Initialize complexity score columns
    num_levels = 4  # Based on the ComplexityMeasurer initialization
    complexity_columns = []
    for i in range(num_levels):
        level_description = "local_detail" if i == 0 else "global_structure" if i == num_levels-1 else f"level_{i+1}"
        column_name = f"complexity_level_{i+1}_{level_description}"
        complexity_columns.append(column_name)
    
    return df, complexity_columns


def process_album_parallel(row_data, complexity_columns, output_csv, processed_file):
    """Process a single album and save result immediately
    
    Returns processing status (success/failure)
    """
    img_name = row_data['album_group_mbid']
    img_path = f"./data/img_all/{img_name}.jpg"
    
    if pd.isna(img_name) or img_name == '':
        print(f"Warning: Empty album_group_mbid. Skipping.")
        return False
        
    if not os.path.exists(img_path):
        print(f"Warning: File {img_path} does not exist. Skipping.")
        return False
    
    try:
        # Start timing
        process_start_time = time.time()
        
        # Create a new analyzer instance for thread safety
        analyzer = AlbumCoverAnalyzer()
        analyzer.image_path = img_path
        scores = analyzer.analyze_complexity()
        overall_score = analyzer.get_overall_score()
        
        # Calculate processing time
        process_time = time.time() - process_start_time
        
        print(f"Processed {img_name} in {process_time:.3f} seconds (Overall score: {overall_score:.4f})")
        
        # Prepare result data - include all original row data plus complexity scores
        result_data = row_data.copy()
        result_data['scores'] = scores
        result_data['overall'] = overall_score
        result_data['process_time'] = process_time
        
        # Save result immediately
        save_single_result(result_data, output_csv, complexity_columns)
        
        # Save processed album ID
        save_processed_album(processed_file, img_name)
        
        return True
        
    except Exception as e:
        print(f"Error analyzing {img_path}: {e}")
        return False


def analyze_images_from_csv(input_csv, output_csv="./results/album_complexity_results.csv", num_workers=None):
    """
    Analyzes images listed in a CSV file and saves each result immediately.
    
    Parameters:
    input_csv (str): Path to CSV file containing image info
    output_csv (str): Path to save results
    num_workers (int): Number of parallel workers (default: CPU count - 1)
    
    Features:
    - Parallel processing with multiprocessing
    - Immediate saving after each album is processed
    - Crash recovery using processed albums tracking
    - Real-time progress monitoring
    """
    start_time = time.time()
    
    if not os.path.exists(input_csv):
        print(f"Input CSV file {input_csv} not found.")
        return None
    
    # Setup environment and tracking files
    processed_file, progress_file = setup_analysis_environment(output_csv)
    
    # Set number of workers
    if num_workers is None:
        # Use CPU count - 1 by default to leave one core free
        num_workers = max(1, mp.cpu_count() - 1)
    
    print(f"Using {num_workers} parallel workers")
    
    # Load already processed albums and prepare dataframe
    processed_albums = load_processed_albums(processed_file)
    df, complexity_columns = prepare_dataframe(input_csv, processed_albums)
    
    if len(df) == 0:
        return df
    
    print(f"Total rows to process: {len(df)}")
    print(f"Results will be saved immediately after each album is processed")
    print(f"Progress tracking files:")
    print(f"  - Processed albums: {processed_file}")
    print(f"  - Progress log: {progress_file}")
    
    # Convert dataframe rows to list for multiprocessing
    row_data_list = []
    for index, row in df.iterrows():
        row_data_list.append(row.to_dict())
        
    print(f"Starting parallel processing...")
        
    # Process albums in parallel
    with mp.Pool(processes=num_workers) as pool:
        # Create partial function with fixed arguments
        process_func = partial(
                process_album_parallel, 
            complexity_columns=complexity_columns,
            output_csv=output_csv,
            processed_file=processed_file
        )
        
        # Start processing
        results = pool.map(process_func, row_data_list)
    
    # Calculate final statistics
    successful_count = sum(results)
    failed_count = len(results) - successful_count
    
    # Final progress update
    with open(progress_file, 'w') as f:
        f.write(f"COMPLETED - All albums processed\n")
        f.write(f"Total processed: {successful_count}\n")
        f.write(f"Total skipped: {failed_count}\n")
        f.write(f"Completion timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    # Calculate and display final statistics
    end_time = time.time()
    elapsed_time = end_time - start_time
    minutes, seconds = divmod(elapsed_time, 60)
    
    print(f"\nAnalysis complete!")
    print(f"Total images processed: {successful_count}")
    print(f"Total images skipped: {failed_count}")
    print(f"Results saved to: {output_csv}")
    print(f"Total analysis runtime: {int(minutes)} minutes and {seconds:.2f} seconds")
    print(f"Overall processing rate: {successful_count/elapsed_time:.2f} images/second")
    
    return df


def check_progress(output_csv):
    """Check progress of a running or completed analysis"""
    output_dir = os.path.dirname(output_csv)
    base_name = os.path.splitext(os.path.basename(output_csv))[0]
    processed_file = os.path.join(output_dir, f"{base_name}_processed_albums.txt")
    progress_file = os.path.join(output_dir, f"{base_name}_progress.txt")
    
    print(f"Progress check for: {output_csv}")
    print("-" * 50)
    
    # Check processed albums count
    if os.path.exists(processed_file):
        with open(processed_file, 'r') as f:
            processed_count = len([line for line in f if line.strip()])
        print(f"Albums processed so far: {processed_count}")
    else:
        print("No processed albums file found - analysis not started")
        return
    
    # Check progress file
    if os.path.exists(progress_file):
        print("\nLatest progress:")
        with open(progress_file, 'r') as f:
            print(f.read())
    
    # Check output file
    if os.path.exists(output_csv):
        try:
            df = pd.read_csv(output_csv)
            print(f"Output CSV contains {len(df)} rows")
        except Exception as e:
            print(f"Error reading output CSV: {e}")
    else:
        print("Output CSV not found")


if __name__ == "__main__":    
    if len(sys.argv) > 1 and sys.argv[1] == "check":
        # Check progress mode
        if len(sys.argv) > 2:
            check_progress(sys.argv[2])
        else:
            print("Usage for progress check: python analyze_csv.py check <output_csv_path>")
    elif len(sys.argv) > 1 and sys.argv[1] == "parallel":
        # Parallel processing mode with custom worker count
        workers = int(sys.argv[2]) if len(sys.argv) > 2 else None
        print(f"Running in parallel mode with {workers if workers else 'auto'} workers")
        
        # Process  dataset
        print("\nProcessing dataset...")
        analyze_images_from_csv(
            "data/chunks/mumu_msdi_chunk_04_rows_30001-35899.csv", 
            "./results/mumu_msdi_chunk_04_rows_30001-35899_with_complexity.csv",
            num_workers=workers
        )
    else:
        # Normal processing mode
        # Process billboard dataset
        print("\nProcessing dataset...")
        # analyze_images_from_csv(
        #     "data/billboard_album_final_super_genres.csv", 
        #     "./results/billboard_album_with_complexity.csv"
        # )
        analyze_images_from_csv("./data/chunks/mumu_msdi_chunk_03_rows_20001-30000.csv", 
                                "./results/mumu_msdi_chunk_03_rows_20001-30000_with_complexity.csv"
        )
