import pandas as pd
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def convert_single_genre_to_list(genre_data):
    """
    Convert a single genre to a list format.
    
    Args:
        genre_data: Single genre string or NaN
        
    Returns:
        List with one genre string, or empty list if NaN
    """
    if pd.isna(genre_data):
        return []
    
    return [str(genre_data).strip()]

def add_genres():
    # Read the files
    logging.info("Reading input files...")
    unique_albums_df = pd.read_csv('msdi_unique_albums.csv')
    album_details_df = pd.read_csv('msdi_album_details.csv')
    
    logging.info(f"Processing {len(album_details_df)} albums from album_details...")
    
    # Initialize genres column with empty lists
    album_details_df['genres'] = [[] for _ in range(len(album_details_df))]
    
    # For each album in album_details, find matching entry in unique_albums
    matches = 0
    for idx, row in album_details_df.iterrows():
        image_url = row['msdi_image_url']
        if pd.notna(image_url):
            # Find matching row in unique_albums
            matching_rows = unique_albums_df[unique_albums_df['image_url'] == image_url]
            if not matching_rows.empty:
                genre = matching_rows.iloc[0]['genre']
                if pd.notna(genre):
                    # Convert single genre to list format
                    album_details_df.at[idx, 'genres'] = convert_single_genre_to_list(genre)
                    matches += 1
        
        # Log progress every 100 albums
        if (idx + 1) % 100 == 0:
            logging.info(f"Processed {idx + 1} albums, found {matches} genre matches so far...")
    
    # Log final statistics
    total_rows = len(album_details_df)
    matched_rows = album_details_df['genres'].apply(lambda x: len(x) > 0).sum()
    logging.info(f"Matched genres for {matched_rows} out of {total_rows} albums")
    
    # Log some examples of genre formats
    logging.info("Sample genre formats:")
    for i, row in album_details_df[album_details_df['genres'].apply(lambda x: len(x) > 0)].head(3).iterrows():
        logging.info(f"  Album {row.get('title', 'N/A')}: {row['genres']}")
    
    # Save the updated file
    output_file = 'msdi_album_details_with_genres.csv'
    album_details_df.to_csv(output_file, index=False)
    logging.info(f"Saved updated album details to {output_file}")

if __name__ == "__main__":
    add_genres()
