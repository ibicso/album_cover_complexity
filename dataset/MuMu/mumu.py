import pandas as pd
import json
import ast
from pathlib import Path

def extract_unique_albums(input_file, output_file):
    """
    Extract unique album information from the MuMu dataset.
    
    Args:
        input_file (str): Path to the input CSV file
        output_file (str): Path to save the output CSV file
    """
    print(f"Reading {input_file}...")
    
    # Read the CSV file
    df = pd.read_csv(input_file)
    
    # Group by album_mbid and get unique artist_mbid and genres
    album_info = df.groupby('album_mbid').agg({
        'artist_mbid': 'first',  # Take the first artist_mbid since it should be the same for all tracks in an album
        'genres': lambda x: list(set(x))  # Get unique genres for the album
    }).reset_index()
    
    # Save to CSV
    print(f"Saving {len(album_info)} unique albums to {output_file}...")
    album_info.to_csv(output_file, index=False)
    print("Done!")

def parse_genres(genres_str):
    """
    Parse genres string into a list of unique genres.
    Handles both list-like strings and comma-separated strings.
    
    Args:
        genres_str: String representation of genres
    
    Returns:
        list: Sorted list of unique genres
    """
    try:
        if not isinstance(genres_str, str):
            return []
            
        # Remove any whitespace and quotes
        genres_str = genres_str.strip()
        
        # If it looks like a Python list (starts with [ and ends with ])
        if genres_str.startswith('[') and genres_str.endswith(']'):
            try:
                # Try to parse as a Python list first
                genres = ast.literal_eval(genres_str)
                if isinstance(genres, list):
                    # Split any genres that might contain commas
                    expanded_genres = []
                    for genre in genres:
                        expanded_genres.extend([g.strip() for g in genre.split(',')])
                    return sorted(list(set(expanded_genres)))
            except:
                pass
        
        # If it's a comma-separated string (without list brackets)
        if ',' in genres_str:
            return sorted(list(set([g.strip() for g in genres_str.split(',')])))
        
        # If it's a single genre
        return [genres_str]
    except:
        return []

def merge_album_files(single_label_file, multi_label_file, output_file):
    """
    Merge single-label and multi-label album files, handling duplicates and formatting genres.
    
    Args:
        single_label_file (str): Path to the single-label CSV file
        multi_label_file (str): Path to the multi-label CSV file
        output_file (str): Path to save the merged output CSV file
    """
    print("Merging album files...")
    
    # Read both CSV files
    single_df = pd.read_csv(single_label_file)
    multi_df = pd.read_csv(multi_label_file)
    
    # Parse genres for both dataframes
    single_df['genres'] = single_df['genres'].apply(parse_genres)
    multi_df['genres'] = multi_df['genres'].apply(parse_genres)
    
    # Concatenate the dataframes
    merged_df = pd.concat([single_df, multi_df], ignore_index=True)
    
    # Remove duplicates based on album_mbid, keeping the entry with more genres
    merged_df['genre_count'] = merged_df['genres'].apply(len)
    merged_df = merged_df.sort_values('genre_count', ascending=False)
    merged_df = merged_df.drop_duplicates(subset='album_mbid', keep='first')
    
    # Drop the helper column
    merged_df = merged_df.drop('genre_count', axis=1)
    
    # Sort by album_mbid for consistency
    merged_df = merged_df.sort_values('album_mbid')
    
    # Save to CSV
    print(f"Saving {len(merged_df)} unique albums to {output_file}...")
    merged_df.to_csv(output_file, index=False)
    print("Done!")

def get_unique_albums():
    # Set up paths
    base_dir = Path("MuMu_dataset")
    single_label_file = base_dir / "MuMu_dataset_single-label.csv"
    multi_label_file = base_dir / "MuMu_dataset_multi-label.csv"
    
    # Process single-label dataset
    output_single = base_dir / "unique_albums_single_label.csv"
    extract_unique_albums(single_label_file, output_single)
    
    # Process multi-label dataset
    output_multi = base_dir / "unique_albums_multi_label.csv"
    extract_unique_albums(multi_label_file, output_multi)
    
    # Merge the files
    output_merged = base_dir / "MuMu_albums.csv"
    merge_album_files(output_single, output_multi, output_merged)


def dataset_analysis():
    # Read the merged CSV file
    merged_df = pd.read_csv('MuMu_dataset/MuMu_albums.csv')
    single_df = pd.read_csv('MuMu_dataset/unique_albums_single_label.csv')
    multi_df = pd.read_csv('MuMu_dataset/unique_albums_multi_label.csv')
    mumu_df = pd.read_csv('MuMu_dataset/MuMu_dataset_single-label.csv')
    mumu_df_multi = pd.read_csv('MuMu_dataset/MuMu_dataset_multi-label.csv')
    
    # Get the genre list of the first row
    first_row_genres = merged_df.iloc[0]['genres']
    print(first_row_genres)

    # Get the number of rows for each dataframe
    merged_df_rows = len(merged_df)
    single_df_rows = len(single_df)
    multi_df_rows = len(multi_df)
    mumu_df_rows = len(mumu_df)
    mumu_df_multi_rows = len(mumu_df_multi)
    
    # Print the number of rows for each dataframe
    print(f"Merged DataFrame Rows: {merged_df_rows}")
    print(f"Single Label DataFrame Rows: {single_df_rows}")
    print(f"Multi Label DataFrame Rows: {multi_df_rows}")
    print(f"MuMu Single Label DataFrame Rows: {mumu_df_rows}")
    print(f"MuMu Multi Label DataFrame Rows: {mumu_df_multi_rows}")
if __name__ == "__main__":
    get_unique_albums() 
    dataset_analysis()