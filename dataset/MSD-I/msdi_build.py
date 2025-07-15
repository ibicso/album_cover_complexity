import pandas as pd
import logging
import os
from pathlib import Path

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def add_mbids(msdi_df, mbid_df):
    """
    Add MBIDs to the MSD-I dataset using the mapping file
    """

    # Clean up track IDs (remove any potential whitespace)
    mbid_df['msd_track_id'] = mbid_df['msd_track_id'].str.strip()
    msdi_df['msd_track_id'] = msdi_df['msd_track_id'].str.strip()
    
    # Check for any track IDs that match
    sample_msdi_ids = set(msdi_df['msd_track_id'].head(100))
    sample_mapping_ids = set(mbid_df['msd_track_id'].head(100))
    logging.info("\nSample track IDs from MSD-I:")
    logging.info(list(sample_msdi_ids)[:5])
    logging.info("\nSample track IDs from mapping:")
    logging.info(list(sample_mapping_ids)[:5])
    
    # Merge the datasets
    logging.info("\nMerging datasets...")
    merged_df = pd.merge(msdi_df, mbid_df[['msd_track_id', 'mbid']], on='msd_track_id', how='left')

    
    return merged_df

def clean_and_compare(msdi_df, mumu_single_df, mumu_multi_df):
    """
    Clean MSD-I dataset and compare with MuMu:
    1. Remove songs without MBIDs
    2. Convert to CSV
    3. Remove duplicates with MuMu dataset
    4. Save unique songs
    """
    
    
    # Remove songs without MBIDs
    logging.info("Removing songs without MBIDs...")
    msdi_df = msdi_df.dropna(subset=['mbid'])
    after_mbid_count = len(msdi_df)
    logging.info(f"Tracks with MBIDs: {after_mbid_count}")
    

    
    # Get all track IDs from both MuMu datasets
    mumu_tracks = set(mumu_single_df['MSD_track_id'].unique())
    mumu_tracks.update(mumu_multi_df['MSD_track_id'].unique())
    logging.info(f"Total unique MuMu track IDs: {len(mumu_tracks)}")
    
    # Print sample track IDs before comparison
    logging.info("\nSample track IDs before comparison:")
    logging.info("MSD-I track IDs:")
    logging.info(msdi_df['msd_track_id'].head().tolist())
    logging.info("MuMu track IDs:")
    logging.info(list(mumu_tracks)[:5])
    
    # Remove tracks that exist in MuMu dataset
    logging.info("\nRemoving tracks that exist in MuMu dataset...")
    msdi_unique_df = msdi_df[~msdi_df['msd_track_id'].isin(mumu_tracks)]
    

    return msdi_unique_df

def filter_unique_albums(df):
    """
    Filter the unique MSD-I tracks to keep only one track per album.
    """
    # Group by album_index and keep first track
    logging.info("Filtering to keep one track per album...")
    df_albums = df.drop_duplicates(subset=['album_index'], keep='first')
    
    return df_albums

def process_msdi_dataset():
    mbid_mapping_file =pd.read_csv("msd-mbid-2016-01-results-ab.csv", header=None, 
                         names=['msd_track_id', 'mbid', 'track_name', 'artist_name'])
    msdi_file = pd.read_csv("MSD-I_dataset.tsv", sep='\t')
    df_with_mbids = add_mbids(msdi_file, mbid_mapping_file)
    mumu_single_file = pd.read_csv("../MuMu/MuMu_dataset/MuMu_dataset_single-label.csv")
    mumu_multi_file = pd.read_csv("../MuMu/MuMu_dataset/MuMu_dataset_multi-label.csv")
    df_unique_tracks = clean_and_compare(df_with_mbids, mumu_single_file, mumu_multi_file)
    df_unique_albums = filter_unique_albums(df_unique_tracks)  
    output_file = "msdi_unique_albums.csv"
    df_unique_albums.to_csv(output_file, index=False)


if __name__ == "__main__":
    process_msdi_dataset()
