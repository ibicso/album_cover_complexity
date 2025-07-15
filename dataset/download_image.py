import pandas as pd
import requests
import os
from urllib.parse import urlparse
import time
from tqdm import tqdm
import argparse

def is_valid_url(url):
    """Check if URL is valid and from a supported image hosting service."""
    if not isinstance(url, str):
        return False
    try:
        result = urlparse(url)
        if not all([result.scheme, result.netloc]):
            return False
        
        # List of supported image hosting domains
        supported_domains = [
            'amazon.com',
            'amazonaws.com', 
            '7static.com',
            'discogs.com',
            'last.fm',
            'lastfm.freetls.fastly.net',
            'musicbrainz.org',
            'coverartarchive.org',
            'spotify.com',
            'scdn.co',
            'imgur.com',
            'wikimedia.org',
            'wikipedia.org'
        ]
        
        # Check if the domain matches any supported service
        domain = result.netloc.lower()
        return any(supported_domain in domain for supported_domain in supported_domains)
    except:
        return False

def download_image(url, filepath, max_retries=3):
    """Download image with retry mechanism."""
    for attempt in range(max_retries):
        try:
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            
            # Check if response is actually an image
            content_type = response.headers.get('content-type', '')
            if not content_type.startswith('image/'):
                print(f"Warning: URL {url} did not return an image (content-type: {content_type})")
                return False
            
            with open(filepath, 'wb') as f:
                f.write(response.content)
            return True
        except requests.RequestException as e:
            if attempt == max_retries - 1:  # Last attempt
                print(f"Failed to download {url}: {str(e)}")
                return False
            time.sleep(1)  # Wait before retrying

def load_processed_mbids():
    """Load the set of already processed MBIDs."""
    processed_file = "processed_mbids_images.txt"
    if os.path.exists(processed_file):
        with open(processed_file, 'r') as f:
            return set(line.strip() for line in f)
    return set()

def save_processed_mbid(mbid):
    """Append a processed MBID to the tracking file."""
    with open("processed_mbids_images.txt", 'a') as f:
        f.write(f"{mbid}\n")

def should_process_mbid(mbid, processed_mbids, image_dir):
    """
    Check if an MBID should be processed (downloaded).
    Returns True if the MBID needs processing, False otherwise.
    """
    # Skip if MBID is not valid
    if not isinstance(mbid, str) or not mbid.strip():
        return False
    
    # Skip if already in processed list
    if mbid in processed_mbids:
        return False
    
    # Skip if image file already exists
    filepath = os.path.join(image_dir, f"{mbid}.jpg")
    if os.path.exists(filepath):
        # Add to processed list if file exists but wasn't tracked
        processed_mbids.add(mbid)
        save_processed_mbid(mbid)
        return False
    
    return True

def check_mbid_status(mbid, csv_file="merged_dataset_renamed.csv"):
    """
    Check the status of a specific MBID - whether it can be processed, already processed, etc.
    """
    # Load processed MBIDs
    processed_mbids = load_processed_mbids()
    
    # Load dataset
    df = pd.read_csv(csv_file)
    album_row = df[df['album_group_mbid'] == mbid]
    
    if album_row.empty:
        return f"MBID {mbid} not found in dataset"
    
    row = album_row.iloc[0]
    amazon_url = row['amazon_image_url']
    msdi_url = row.get('msdi_image_url', '')
    
    # Check URLs
    has_amazon = is_valid_url(amazon_url)
    has_msdi = is_valid_url(msdi_url)
    
    # Check if already processed
    is_processed = mbid in processed_mbids
    
    # Check if file exists
    filepath = os.path.join("img", f"{mbid}.jpg")
    file_exists = os.path.exists(filepath)
    
    status = f"""
MBID Status for: {mbid}
Album: {row['album_group_title']} by {row['artist_name']}
=====================================
Amazon URL: {amazon_url} (Valid: {has_amazon})
MSDI URL: {msdi_url} (Valid: {has_msdi})
In processed list: {is_processed}
Image file exists: {file_exists}
Can be downloaded: {(has_amazon or has_msdi) and not is_processed and not file_exists}
"""
    return status

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Download album cover images from Amazon')
    parser.add_argument('-n', '--num-images', type=int, default=1000,
                       help='Number of images to download in this run (default: all remaining)')
    parser.add_argument('--check-mbid', type=str, 
                       help='Check the status of a specific MBID instead of downloading')
    args = parser.parse_args()

    # If checking a specific MBID, do that and exit
    if args.check_mbid:
        print(check_mbid_status(args.check_mbid))
        return

    # Create images directory if it doesn't exist
    image_dir = "img/billboard"
    os.makedirs(image_dir, exist_ok=True)
    
    # Load already processed MBIDs
    processed_mbids = load_processed_mbids()
    print(f"Found {len(processed_mbids)} already processed albums")
    
    # Read the merged dataset
    # df = pd.read_csv("merged_dataset_renamed_no_duplicates.csv")
    df = pd.read_csv("./billboard/cleaned_merged_dataset.csv")
    
    # Keep track of successful and failed downloads
    successful = 0
    failed = 0
    skipped = 0
    
    print("Starting image downloads...")
    
    # Process each row
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Downloading images"):
        mbid = row['album_group_mbid']
        amazon_url = row.get('amazon_image_url', '')
        if not amazon_url:
            amazon_url = ''
        # amazon_url = row['amazon_image_url']
        msdi_url = row.get('msdi_image_url', '')
        
        # Skip if no valid MBID
        if not isinstance(mbid, str) or not mbid.strip():
            skipped += 1
            continue
        
        # Try Amazon URL first, then MSDI URL
        url = None
        if is_valid_url(amazon_url):
            url = amazon_url
        elif is_valid_url(msdi_url):
            url = msdi_url
        
        # Skip if no valid URL found
        if not url:
            skipped += 1
            continue
        
        # Skip if already processed
        if not should_process_mbid(mbid, processed_mbids, image_dir):
            skipped += 1
            continue
        
        # Create filepath for download
        filepath = os.path.join(image_dir, f"{mbid}.jpg")
        
        # Download the image
        if download_image(url, filepath):
            successful += 1
            processed_mbids.add(mbid)
            save_processed_mbid(mbid)
        else:
            failed += 1
            # Still mark as processed to avoid retrying failed downloads
            processed_mbids.add(mbid)
            save_processed_mbid(mbid)
        
        # Add a small delay to be nice to servers
        time.sleep(0.1)
        
        # Check if we've reached the requested number of downloads
        if args.num_images and successful >= args.num_images:
            print(f"\nReached requested limit of {args.num_images} images")
            break
    
    # Print summary
    print("\nDownload Summary:")
    print(f"Successful downloads: {successful}")
    print(f"Failed downloads: {failed}")
    print(f"Skipped (invalid URL, already exists, or processed): {skipped}")
    print(f"Total processed: {successful + failed + skipped}")
    print(f"Total unique albums processed so far: {len(processed_mbids)}")

if __name__ == "__main__":
    main()
