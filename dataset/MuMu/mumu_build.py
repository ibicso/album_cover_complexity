import pandas as pd
import logging
import os
from pathlib import Path

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def process_mumu_dataset():
    """
    Process the MuMu dataset:
    1. Read both single-label and multi-label datasets
    2. Clean and prepare for analysis
    3. Save clean datasets
    """
    # File paths
    mumu_single_file = "dataset/MuMu/MuMu_dataset/MuMu_dataset_single-label.csv"
    mumu_multi_file = "dataset/MuMu/MuMu_dataset/MuMu_dataset_multi-label.csv"
    output_single_file = "dataset/MuMu/mumu_clean_single.csv"
    output_multi_file = "dataset/MuMu/mumu_clean_multi.csv"
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_single_file), exist_ok=True)
    
    # Process single-label dataset
    logging.info("Reading MuMu single-label dataset...")
    single_df = pd.read_csv(mumu_single_file)
    initial_single_count = len(single_df)
    logging.info(f"Found {initial_single_count} tracks in MuMu single-label dataset")
    
    # Remove duplicates based on msd_track_id
    single_df = single_df.drop_duplicates(subset=['MSD_track_id'], keep='first')
    after_dedup_single = len(single_df)
    
    # Save clean single-label dataset
    single_df.to_csv(output_single_file, index=False)
    logging.info(f"Saved clean single-label dataset to {output_single_file}")
    
    # Process multi-label dataset
    logging.info("\nReading MuMu multi-label dataset...")
    multi_df = pd.read_csv(mumu_multi_file)
    initial_multi_count = len(multi_df)
    logging.info(f"Found {initial_multi_count} tracks in MuMu multi-label dataset")
    
    # Remove duplicates based on msd_track_id
    multi_df = multi_df.drop_duplicates(subset=['MSD_track_id'], keep='first')
    after_dedup_multi = len(multi_df)
    
    # Save clean multi-label dataset
    multi_df.to_csv(output_multi_file, index=False)
    logging.info(f"Saved clean multi-label dataset to {output_multi_file}")
    
    # Print statistics
    logging.info("\nDataset Statistics:")
    logging.info("Single-label dataset:")
    logging.info(f"Original tracks: {initial_single_count}")
    logging.info(f"After removing duplicates: {after_dedup_single}")
    logging.info(f"Removed duplicates: {initial_single_count - after_dedup_single}")
    
    logging.info("\nMulti-label dataset:")
    logging.info(f"Original tracks: {initial_multi_count}")
    logging.info(f"After removing duplicates: {after_dedup_multi}")
    logging.info(f"Removed duplicates: {initial_multi_count - after_dedup_multi}")
    
    # Print sample of tracks from both datasets
    logging.info("\nSample of tracks from single-label dataset (first 5):")
    single_sample = single_df.head()
    for _, row in single_sample.iterrows():
        logging.info(f"Track ID: {row['MSD_track_id']}, Genre: {row['genres']}")
    
    logging.info("\nSample of tracks from multi-label dataset (first 5):")
    multi_sample = multi_df.head()
    for _, row in multi_sample.iterrows():
        logging.info(f"Track ID: {row['MSD_track_id']}")
    
    return single_df, multi_df

if __name__ == "__main__":
    process_mumu_dataset() 