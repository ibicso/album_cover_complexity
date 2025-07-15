import pandas as pd
import os
from urllib.parse import urlparse

def is_valid_url(url):
    """Check if URL is valid and from Amazon."""
    if not isinstance(url, str):
        return False
    try:
        result = urlparse(url)
        return all([result.scheme, result.netloc]) and 'amazon' in result.netloc.lower()
    except:
        return False

def main():
    # Load data
    df = pd.read_csv("merged_dataset_renamed.csv")
    
    # Load processed MBIDs
    processed_mbids = set()
    if os.path.exists("processed_mbids_images.txt"):
        with open("processed_mbids_images.txt", 'r') as f:
            processed_mbids = set(line.strip() for line in f)
    
    # Get actual downloaded images
    img_dir = "img"
    downloaded_files = set()
    if os.path.exists(img_dir):
        downloaded_files = {f.replace('.jpg', '') for f in os.listdir(img_dir) if f.endswith('.jpg')}
    
    print(f"Total albums in CSV: {len(df)}")
    print(f"Processed MBIDs in file: {len(processed_mbids)}")
    print(f"Actually downloaded images: {len(downloaded_files)}")
    
    # Analyze missing images
    valid_mbids = 0
    valid_urls = 0
    both_valid = 0
    missing_images = []
    
    for _, row in df.iterrows():
        mbid = row['album_group_mbid']
        url = row['amazon_image_url']
        
        has_valid_mbid = isinstance(mbid, str) and mbid.strip()
        has_valid_url = is_valid_url(url)
        
        if has_valid_mbid:
            valid_mbids += 1
        if has_valid_url:
            valid_urls += 1
        if has_valid_mbid and has_valid_url:
            both_valid += 1
            
            # Check if image is missing
            if mbid not in downloaded_files:
                missing_images.append({
                    'mbid': mbid,
                    'url': url,
                    'in_processed_file': mbid in processed_mbids
                })
    
    print(f"\nValid analysis:")
    print(f"Albums with valid MBID: {valid_mbids}")
    print(f"Albums with valid URL: {valid_urls}")
    print(f"Albums with both valid MBID and URL: {both_valid}")
    print(f"Missing images: {len(missing_images)}")
    
    # Analyze missing images
    missing_but_processed = sum(1 for img in missing_images if img['in_processed_file'])
    missing_not_processed = sum(1 for img in missing_images if not img['in_processed_file'])
    
    print(f"\nMissing image breakdown:")
    print(f"Missing but marked as processed: {missing_but_processed}")
    print(f"Missing and not processed: {missing_not_processed}")
    
    # Show first 10 missing images that should be downloadable
    print(f"\nFirst 10 missing images that should be downloadable:")
    for i, img in enumerate(missing_images[:10]):
        status = "PROCESSED" if img['in_processed_file'] else "NOT PROCESSED"
        print(f"{i+1}. MBID: {img['mbid']} - Status: {status}")
        print(f"   URL: {img['url']}")
    
    # Check for MBIDs in processed file but not in downloaded files
    processed_but_not_downloaded = processed_mbids - downloaded_files
    print(f"\nMBIDs marked as processed but no image file: {len(processed_but_not_downloaded)}")
    
    if len(processed_but_not_downloaded) > 0:
        print("First 10 examples:")
        for i, mbid in enumerate(list(processed_but_not_downloaded)[:10]):
            print(f"{i+1}. {mbid}")

if __name__ == "__main__":
    main() 