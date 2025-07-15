import pandas as pd
import json
import logging
import os

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def add_images():
    # Define file paths
    base_dir = os.path.dirname(os.path.abspath(__file__))
    mb_details_path = os.path.join(base_dir, 'mb_albums', 'musicbrainz_album_details_with_genres.csv')
    amazon_json_path = os.path.join(base_dir, 'MuMu_dataset', 'amazon_metadata_MuMu.json')
    
    # Read musicbrainz album details
    logging.info("Reading musicbrainz album details...")
    mb_details_df = pd.read_csv(mb_details_path)
    
    # Read Amazon JSON file
    logging.info("Reading Amazon albums JSON...")
    with open(amazon_json_path, 'r') as f:
        amazon_data = json.load(f)

    # Print first album to check structure
    print("First album structure:")
    print(json.dumps(amazon_data[0], indent=2))
    
    # Create a mapping from mbid to image URL
    logging.info("Creating MBID to image URL mapping...")
    mbid_to_image = {}
    for album in amazon_data:
        if 'release-group_mbid' in album and 'imUrl' in album:
            mbid_to_image[album['release-group_mbid']] = album['imUrl']
    
    logging.info(f"Found {len(mbid_to_image)} albums with image URLs")
    
    # Add image URLs to musicbrainz details
    logging.info("Adding image URLs to album details...")
    mb_details_df['amazon_image_url'] = mb_details_df['album_group_mbid'].map(mbid_to_image)
    
    # Log statistics
    total_rows = len(mb_details_df)
    matched_rows = mb_details_df['amazon_image_url'].notna().sum()
    logging.info(f"Added image URLs for {matched_rows} out of {total_rows} albums")
    
    # Save the updated file
    output_file = os.path.join(base_dir, 'mb_albums', 'musicbrainz_album_details_with_images.csv')
    mb_details_df.to_csv(output_file, index=False)
    logging.info(f"Saved updated album details to {output_file}")

if __name__ == "__main__":
    add_images()

