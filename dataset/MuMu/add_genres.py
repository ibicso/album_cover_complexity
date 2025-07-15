import pandas as pd
import logging
import os
import ast

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def convert_genres_to_list(genre_data):
    """
    Convert genre data to a proper list of strings.
    
    Args:
        genre_data: Genre data in various formats (string, comma-separated, or list)
        
    Returns:
        List of genre strings
    """
    if pd.isna(genre_data):
        return []
    
    # If it's already a list, return it
    if isinstance(genre_data, list):
        return [str(genre).strip() for genre in genre_data]
    
    # If it's a string that looks like a list representation
    if isinstance(genre_data, str):
        # Try to parse as Python literal first
        try:
            parsed = ast.literal_eval(genre_data)
            if isinstance(parsed, list):
                return [str(genre).strip() for genre in parsed]
        except (ValueError, SyntaxError):
            pass
        
        # If it contains commas, split by comma
        if ',' in genre_data:
            return [genre.strip() for genre in genre_data.split(',') if genre.strip()]
        else:
            # Single genre
            return [genre_data.strip()]
    
    # Fallback: convert to string and return as single-item list
    return [str(genre_data).strip()]

def add_genres():
    # Define file paths
    base_dir = os.path.dirname(os.path.abspath(__file__))
    mb_details_path = os.path.join(base_dir, 'mb_albums', 'musicbrainz_album_details.csv')
    single_label_path = os.path.join(base_dir, 'mumu_clean_single.csv')
    multi_label_path = os.path.join(base_dir, 'mumu_clean_multi.csv')
    
    # Read the files
    logging.info("Reading input files...")
    mb_details_df = pd.read_csv(mb_details_path)
    single_label_df = pd.read_csv(single_label_path)
    multi_label_df = pd.read_csv(multi_label_path)
    
    # Process single label dataset - group by album_mbid and collect genres into lists
    logging.info("Processing single label dataset...")
    single_grouped = single_label_df.groupby('album_mbid')['genres'].apply(
        lambda x: list(set(x.dropna()))  # Remove duplicates and NaN values
    ).reset_index()
    
    # Process multi label dataset - convert comma-separated strings to proper lists
    logging.info("Processing multi label dataset...")
    multi_label_df['genres'] = multi_label_df['genres'].apply(convert_genres_to_list)
    
    # Group multi-label data and merge genre lists
    multi_grouped = multi_label_df.groupby('album_mbid')['genres'].apply(
        lambda x: list(set([genre for genre_list in x for genre in genre_list if genre]))  # Flatten and deduplicate
    ).reset_index()
    
    # Combine single and multi label datasets
    logging.info("Combining genre datasets...")
    all_genres_list = []
    
    # Add single label data
    for _, row in single_grouped.iterrows():
        all_genres_list.append({
            'album_mbid': row['album_mbid'],
            'genres': row['genres']
        })
    
    # Add multi label data (merge if album already exists)
    for _, row in multi_grouped.iterrows():
        album_mbid = row['album_mbid']
        multi_genres = row['genres']
        
        # Check if this album already exists in single label data
        existing_entry = None
        for entry in all_genres_list:
            if entry['album_mbid'] == album_mbid:
                existing_entry = entry
                break
        
        if existing_entry:
            # Merge genres (combine and deduplicate)
            combined_genres = list(set(existing_entry['genres'] + multi_genres))
            existing_entry['genres'] = combined_genres
        else:
            # Add as new entry
            all_genres_list.append({
                'album_mbid': album_mbid,
                'genres': multi_genres
            })
    
    # Convert to DataFrame
    all_genres_df = pd.DataFrame(all_genres_list)
    
    logging.info(f"Combined dataset has {len(all_genres_df)} unique albums with genres")
    
    # Add genres using merge
    logging.info("Adding genres to album details...")
    mb_details_df = mb_details_df.merge(
        all_genres_df,
        left_on='album_group_mbid',
        right_on='album_mbid',
        how='left'
    )
    
    # Clean up columns
    mb_details_df = mb_details_df.drop('album_mbid', axis=1)
    
    # Convert NaN values in genres to empty lists
    mb_details_df['genres'] = mb_details_df['genres'].apply(
        lambda x: x if isinstance(x, list) else []
    )
    
    # Log statistics
    total_rows = len(mb_details_df)
    matched_rows = mb_details_df['genres'].apply(lambda x: len(x) > 0).sum()
    logging.info(f"Matched genres for {matched_rows} out of {total_rows} albums")
    
    # Log some examples of genre formats
    logging.info("Sample genre formats:")
    for i, row in mb_details_df[mb_details_df['genres'].apply(lambda x: len(x) > 0)].head(3).iterrows():
        logging.info(f"  Album {row.get('title', 'N/A')}: {row['genres']}")
    
    # Save the updated file
    output_file = os.path.join(base_dir, 'mb_albums', 'musicbrainz_album_details_with_genres.csv')
    mb_details_df.to_csv(output_file, index=False)
    logging.info(f"Saved updated album details to {output_file}")

if __name__ == "__main__":
    add_genres()
