import requests
import time
import pandas as pd
import json
import logging
import asyncio
import aiohttp
from pathlib import Path
import os
from datetime import datetime, timedelta
import statistics

# Setup basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class ProcessingStats:
    def __init__(self):
        self.start_time = None
        self.batch_times = []  # List of seconds per batch
        self.albums_per_batch = []  # List of albums processed per batch
        self.total_albums_processed = 0
        self.stats_file = "processing_stats.txt"
    
    def start(self):
        self.start_time = time.time()
        # Initialize stats file
        with open(self.stats_file, 'w') as f:
            f.write("MusicBrainz Processing Statistics\n")
            f.write("================================\n")
            f.write(f"Process started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
    
    def add_batch(self, batch_time, albums_in_batch):
        self.batch_times.append(batch_time)
        self.albums_per_batch.append(albums_in_batch)
        self.total_albums_processed += albums_in_batch
    
    def get_stats(self):
        if not self.batch_times:
            return "No batches processed yet."
        
        current_duration = time.time() - self.start_time
        avg_batch_time = statistics.mean(self.batch_times)
        avg_album_time = sum(self.batch_times) / self.total_albums_processed
        
        stats = {
            "total_runtime": str(timedelta(seconds=int(current_duration))),
            "batches_completed": len(self.batch_times),
            "total_albums_processed": self.total_albums_processed,
            "avg_time_per_batch": str(timedelta(seconds=int(avg_batch_time))),
            "avg_time_per_album": f"{avg_album_time:.2f} seconds",
            "estimated_total_time": str(timedelta(seconds=int(avg_album_time * 31471))),  # Total MuMu dataset size
        }
        return stats

    def save_stats(self, stats, is_batch=True):
        """Save statistics to a file"""
        stats_dir = "stats"
        os.makedirs(stats_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"batch_stats_{timestamp}.txt" if is_batch else f"final_stats_{timestamp}.txt"
        filepath = os.path.join(stats_dir, filename)
        
        with open(filepath, 'w') as f:
            f.write("Processing Statistics:\n")
            f.write(f"- Batch processing time: {stats['batch_time']}\n")
            f.write(f"- Average time per item in batch: {stats['avg_time_per_item_batch']}\n")
            f.write(f"- Total runtime so far: {stats['total_runtime']}\n")
            f.write(f"- Estimated total time: {stats['estimated_total_time']}\n")

class MusicBrainzAPI:
    BASE_URL = "http://musicbrainz.org/ws/2/"
    BATCH_SIZE = 500  # Number of albums to process before saving
    MAX_CONCURRENT_REQUESTS = 3  # Maximum number of concurrent requests

    def __init__(self, user_agent="MyMusicApp/1.0 ( myemail@example.com )"):
        self.user_agent = user_agent
        self.headers = {
            "User-Agent": self.user_agent,
            "Accept": "application/json"
        }
        self.request_semaphore = asyncio.Semaphore(1)  # Ensure 1 request per second
        self.session = None
        self.stats = ProcessingStats()
        self.processed_albums_file = None
        self.run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.batches_dir = None
        logging.info(f"MusicBrainzAPI initialized with User-Agent: {self.user_agent}")

    async def _ensure_rate_limit(self):
        async with self.request_semaphore:
            await asyncio.sleep(1.0)  # Ensure 1 second between requests

    async def _execute_mb_request(self, endpoint, params):
        await self._ensure_rate_limit()
        url = f"{self.BASE_URL}{endpoint}"
        logging.info(f"Requesting URL: {url} with params: {params}")
        response_text = None
        try:
            async with self.session.get(url, headers=self.headers, params=params) as response:
                response_text = await response.text()
                response.raise_for_status()
                return await response.json()
        except aiohttp.ClientError as e:
            logging.error(f"HTTP error for {url}: {e}. Response content: {response_text}")
            return None
        except json.JSONDecodeError as e:
            logging.error(f"Error decoding JSON for {url}: {e}. Response content: {response_text}")
            return None

    async def get_recording_info(self, recording_mbid):
        """Get information about a recording (track) from MusicBrainz"""
        endpoint = f"recording/{recording_mbid}"
        params = {
            'fmt': 'json',
            'inc': 'artist-credits+releases+release-groups+isrcs'
        }
        logging.debug(f"Preparing to fetch recording data for MBID: {recording_mbid}")
        return await self._execute_mb_request(endpoint, params)

    async def get_release_group_info(self, release_group_mbid):
        """Get information about a release group from MusicBrainz"""
        endpoint = f"release-group/{release_group_mbid}"
        params = {"fmt": "json", "inc": "artists+releases"}
        logging.debug(f"Preparing to fetch release group data for MBID: {release_group_mbid}")
        return await self._execute_mb_request(endpoint, params)

    async def get_specific_release_info(self, release_mbid):
        """Get detailed information about a specific release"""
        endpoint = f"release/{release_mbid}"
        params = {"fmt": "json", "inc": "labels+media+artist-credits"}
        logging.debug(f"Preparing to fetch specific release data for MBID: {release_mbid}")
        return await self._execute_mb_request(endpoint, params)

    async def get_label_details(self, label_mbid):
        """Get detailed information about a label, including parent relationships"""
        endpoint = f"label/{label_mbid}"
        params = {"fmt": "json", "inc": "label-rels"}
        logging.debug(f"Preparing to fetch label details for MBID: {label_mbid}")
        return await self._execute_mb_request(endpoint, params)

    def is_album_release(self, release):
        """Check if a release is an album (not an EP, single, or other type)"""
        if not release:
            return False
        
        release_group = release.get('release-group')
        if not release_group:
            return False
        
        primary_type = release_group.get('primary-type', '').lower() if release_group.get('primary-type') else ''
        
        if primary_type == 'album':
            secondary_types = release_group.get('secondary-types', [])
            forbidden_types = {'compilation', 'soundtrack', 'live', 'remix', 'dj-mix', 'mixtape/street'}
            return not any(stype.lower() in forbidden_types for stype in secondary_types)
        
        return False

    async def find_oldest_album_release(self, recording_info):
        """Find the oldest album release from recording information. If no album is found, returns the oldest release of any type."""
        if not recording_info:
            logging.warning("No recording information provided")
            return None, None
        
        releases = recording_info.get('releases', [])
        if not releases:
            logging.warning("No releases found in recording information")
            return None, None
        
        oldest_album = None
        oldest_album_date = None
        oldest_release = None
        oldest_release_date = None
        
        for release in releases:
            if not release:
                continue
            
            release_date = release.get('date', '')
            if not release_date:
                continue
            
            try:
                if len(release_date) == 4:  # Year only
                    current_date = datetime.strptime(release_date, '%Y')
                elif len(release_date) == 7:  # Year-month
                    current_date = datetime.strptime(release_date, '%Y-%m')
                else:  # Full date
                    current_date = datetime.strptime(release_date, '%Y-%m-%d')
            
                # Check if it's an album and update oldest album if applicable
                if self.is_album_release(release):
                    if oldest_album_date is None or current_date < oldest_album_date:
                        oldest_album_date = current_date
                        oldest_album = release
            
                # Update oldest release regardless of type
                if oldest_release_date is None or current_date < oldest_release_date:
                    oldest_release_date = current_date
                    oldest_release = release
                
            except ValueError as e:
                logging.warning(f"Could not parse date {release_date}: {e}")
                continue
        
        # If we found an album, return that
        if oldest_album:
            logging.info("Found oldest album release")
            return oldest_album, oldest_album_date
        
        # If no album was found but we have another type of release, return that
        if oldest_release:
            release_group = oldest_release.get('release-group', {})
            primary_type = release_group.get('primary-type', 'unknown').lower() if release_group else 'unknown'
            logging.warning(f"No album release found, using oldest release of type: {primary_type}")
            return oldest_release, oldest_release_date
        
        logging.warning("No valid releases found with dates")
        return None, None

    async def get_album_details(self, release, release_group):
        """Extract album details from a release and its release group"""
        details = {
            'album_group_mbid': release_group.get('id', ''),
            'album_group_title': release_group.get('title', ''),
            'artist_name': ' & '.join(credit['artist']['name'] for credit in release_group.get('artist-credit', [])),
            'first_release_date_group': release_group.get('first-release-date', ''),
            'primary_type': release_group.get('primary-type', ''),
            'release_mbid': release.get('id', ''),
            'release_title': release.get('title', ''),
            'release_date': release.get('date', ''),
            'release_country': release.get('country', ''),
            'track_count': 0,
            'label_name': '',
            'label_mbid': '',
            'parent_label_name': '',
            'parent_label_mbid': ''
        }

        # Get track count
        release_info = await self.get_specific_release_info(release['id'])
        if release_info and 'media' in release_info:
            details['track_count'] = sum(medium.get('track-count', 0) for medium in release_info['media'])

        # Get label information
        if release_info and 'label-info' in release_info:
            label_info_list = release_info.get('label-info', [])
            if label_info_list:
                label_info = label_info_list[0]
                label = label_info.get('label', {})
                if label:
                    details['label_name'] = label.get('name', '')
                    label_mbid = label.get('id', '')
                    if label_mbid:
                        details['label_mbid'] = label_mbid
                        
                        # Get detailed label information to find parent label
                        label_details = await self.get_label_details(label_mbid)
                        if label_details and 'relations' in label_details:
                            for relation in label_details['relations']:
                                if (relation.get('type') == 'imprint' and 
                                    relation.get('direction') == 'backward' and
                                    'label' in relation):
                                    parent_label = relation['label']
                                    details['parent_label_name'] = parent_label.get('name', '')
                                    details['parent_label_mbid'] = parent_label.get('id', '')
                                    break

        return details

    async def process_single_track(self, recording_mbid, msdi_image_url=None):
        """Process a single track and find its oldest album release"""
        try:
            recording_info = await self.get_recording_info(recording_mbid)
            if not recording_info:
                logging.warning(f"No recording information found for MBID: {recording_mbid}")
                return None

            oldest_release, oldest_date = await self.find_oldest_album_release(recording_info)
            if not oldest_release:
                logging.warning(f"No album release found for recording MBID: {recording_mbid}")
                return None

            release_group = oldest_release.get('release-group')
            if not release_group:
                logging.warning(f"No release group found for oldest release of recording MBID: {recording_mbid}")
                return None

            album_details = await self.get_album_details(oldest_release, release_group)
            if album_details and msdi_image_url:
                album_details['msdi_image_url'] = msdi_image_url
            return album_details
        except Exception as e:
            logging.error(f"Error processing track {recording_mbid}: {e}")
            return None

    async def process_single_album(self, rg_mbid):
        """Process a single album from its release group MBID"""
        release_group_data = await self.get_release_group_info(rg_mbid)
        if not release_group_data:
            logging.warning(f"No data retrieved for release group MBID: {rg_mbid}")
            return None

        releases = release_group_data.get('releases', [])
        if not releases:
            logging.warning(f"No releases found for release group MBID: {rg_mbid}")
            return None

        release = releases[0]  # Take the first release
        return await self.get_album_details(release, release_group_data)

    def setup_batch_directory(self, output_dir):
        """Setup the directory for this run's batch files"""
        self.batches_dir = os.path.join(output_dir, f"batches_{self.run_timestamp}")
        os.makedirs(self.batches_dir, exist_ok=True)
        logging.info(f"Batch files will be saved in: {self.batches_dir}")

    def load_processed_items(self, processed_file_path):
        """Load the set of already processed MBIDs"""
        self.processed_albums_file = processed_file_path
        if os.path.exists(processed_file_path):
            with open(processed_file_path, 'r') as f:
                return set(line.strip() for line in f)
        return set()

    def save_processed_item(self, mbid):
        """Save a single processed MBID to the processed list"""
        if self.processed_albums_file:
            with open(self.processed_albums_file, 'a') as f:
                f.write(f"{mbid}\n")

    def save_batch(self, batch_data, output_csv_path, batch_num):
        """Save a batch of data to a CSV file and append to the main output file"""
        # Save to batch directory
        os.makedirs(self.batches_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        batch_file = os.path.join(self.batches_dir, f"msdi_album_details_batch_{batch_num}_{timestamp}.csv")
        
        # Convert list of dictionaries to DataFrame
        df = pd.DataFrame(batch_data)
        
        # Ensure consistent column order
        columns = ['album_group_mbid', 'album_group_title', 'artist_name', 'first_release_date_group',
                   'primary_type', 'release_mbid', 'release_title', 'release_date', 'release_country',
                   'track_count', 'label_name', 'label_mbid', 'parent_label_name', 'parent_label_mbid',
                   'msdi_image_url']
        
        # Reorder columns
        df = df.reindex(columns=columns)
        
        # Save batch file
        df.to_csv(batch_file, index=False)
        logging.info(f"Saved batch {batch_num} with {len(batch_data)} items to {batch_file}")
        
        # Append to main output file
        if os.path.exists(output_csv_path):
            df.to_csv(output_csv_path, mode='a', header=False, index=False)
        else:
            df.to_csv(output_csv_path, index=False)
        logging.info(f"Appended batch {batch_num} to main output file: {output_csv_path}")

    async def process_items_list(self, mbids, image_urls, output_csv_path, process_func):
        """Process a list of MBIDs using the specified processing function"""
        self.setup_batch_directory(os.path.dirname(output_csv_path))
        self.session = aiohttp.ClientSession()
        self.stats.start()
        
        try:
            all_data = []
            batch_num = 1
            total_items = len(mbids)
            
            for i in range(0, total_items, self.BATCH_SIZE):
                batch_start_time = time.time()
                batch = mbids[i:i + self.BATCH_SIZE]
                batch_data = []
                
                tasks = [process_func(mbid, image_urls.get(mbid)) for mbid in batch]
                completed = 0
                
                for mbid, completed_tasks in zip(batch, asyncio.as_completed(tasks)):
                    try:
                        item_data = await completed_tasks
                        completed += 1
                        
                        if item_data:
                            batch_data.append(item_data)
                            # Save the original track MBID instead of the album group MBID
                            self.save_processed_item(mbid)
                        
                        processed = len(all_data) + len(batch_data)
                        logging.info(f"Processed {processed}/{total_items} items ({(processed/total_items)*100:.2f}%)")
                    except Exception as e:
                        logging.error(f"Error processing task: {e}")
                        completed += 1
                
                if batch_data:
                    self.save_batch(batch_data, output_csv_path, batch_num)
                    all_data.extend(batch_data)
                
                batch_time = time.time() - batch_start_time
                if len(batch_data) > 0:  # Only add stats if we have processed items
                    self.stats.add_batch(batch_time, len(batch_data))
                
                current_stats = self.stats.get_stats()
                batch_stats = {
                    "batch_time": str(timedelta(seconds=int(batch_time))),
                    "avg_time_per_item_batch": f"{(batch_time/len(batch)):.2f} seconds",
                    "total_runtime": current_stats['total_runtime'],
                    "estimated_total_time": current_stats['estimated_total_time']
                }
                self.stats.save_stats(batch_stats, is_batch=True)
                
                logging.info(f"""
Batch {batch_num} Statistics:
- Batch processing time: {batch_stats['batch_time']}
- Average time per item in this batch: {batch_stats['avg_time_per_item_batch']}
- Total runtime so far: {batch_stats['total_runtime']}
- Estimated total time for full dataset: {batch_stats['estimated_total_time']}
""")
                
                batch_num += 1
            
            final_stats = self.stats.get_stats()
            total_runtime = time.time() - self.stats.start_time
            final_stats_formatted = {
                "batch_time": str(timedelta(seconds=int(total_runtime))),
                "avg_time_per_item_batch": f"{(total_runtime/len(all_data)):.2f} seconds" if all_data else "N/A",
                "total_runtime": final_stats['total_runtime'],
                "estimated_total_time": final_stats['estimated_total_time']
            }
            self.stats.save_stats(final_stats_formatted, is_batch=False)
            
            logging.info(f"""
Final Processing Statistics:
- Total runtime: {final_stats['total_runtime']}
- Total items processed: {len(all_data)}
- Average time per item: {final_stats_formatted['avg_time_per_item_batch']}
- Estimated time for full dataset: {final_stats['estimated_total_time']}
""")
            
        finally:
            await self.session.close()

    async def process_mumu_albums(self, album_mbids, output_csv_path="musicbrainz_album_details.csv"):
        """Process a list of MuMu album MBIDs"""
        await self.process_items_list(album_mbids, {}, output_csv_path, self.process_single_album)

    async def process_msdi_tracks(self, track_mbids, image_urls, output_csv_path="msdi_album_details.csv"):
        """Process a list of MSDI track MBIDs"""
        await self.process_items_list(track_mbids, image_urls, output_csv_path, self.process_single_track)

    def get_mbids_from_csv(self, csv_path, mbid_column_name):
        """Get list of MBIDs from CSV file, excluding already processed ones"""
        logging.info(f"Attempting to read MBIDs from: {csv_path}, column: {mbid_column_name}")
        
        # Read the CSV file
        df = pd.read_csv(csv_path)
        mbids_with_images = df[[mbid_column_name, 'image_url']].dropna()
        
        # Get the set of already processed MBIDs
        if self.processed_albums_file and os.path.exists(self.processed_albums_file):
            with open(self.processed_albums_file, 'r') as f:
                processed_mbids = {line.strip() for line in f}
            logging.info(f"Found {len(processed_mbids)} already processed MBIDs")
        else:
            processed_mbids = set()
            logging.info("No processed MBIDs found")
        
        # Filter out already processed MBIDs
        unprocessed_mbids = mbids_with_images[~mbids_with_images[mbid_column_name].isin(processed_mbids)]
        
        # Create dictionary of unprocessed MBIDs and their image URLs
        mbids_dict = dict(zip(unprocessed_mbids[mbid_column_name], unprocessed_mbids['image_url']))
        unique_mbids = list(mbids_dict.keys())
        
        total_mbids = len(mbids_with_images)
        remaining_mbids = len(unique_mbids)
        logging.info(f"Found {remaining_mbids} unprocessed MBIDs out of {total_mbids} total MBIDs in {csv_path}.")
        
        return unique_mbids, mbids_dict

async def process_mumu_dataset():
    """Process the MuMu dataset"""
    output_dir = "dataset/MuMu/mb_albums"
    os.makedirs(output_dir, exist_ok=True)
    
    mb_api = MusicBrainzAPI(user_agent="MuMuAlbumFetcher/1.0 (nicolas.fracaro@gmail.com)")
    
    mumu_albums_csv = "dataset/MuMu/MuMu_dataset/MuMu_albums.csv"
    processed_albums_path = os.path.join(output_dir, "processed_albums.txt")
    output_csv_path = os.path.join(output_dir, "musicbrainz_album_details.csv")
    
    all_album_mbids = mb_api.get_mbids_from_csv(mumu_albums_csv, "album_mbid")
    processed_mbids = mb_api.load_processed_items(processed_albums_path)
    remaining_mbids = [mbid for mbid in all_album_mbids if mbid not in processed_mbids]
    
    if remaining_mbids:
        logging.info(f"Found {len(remaining_mbids)} albums to process out of {len(all_album_mbids)} total albums.")
        logging.info(f"{len(processed_mbids)} albums were already processed.")
        
        test_mbids = remaining_mbids[:5000]  # Process first 5000 unprocessed albums
        logging.info(f"Running with first {len(test_mbids)} unprocessed albums")
        await mb_api.process_mumu_albums(test_mbids, output_csv_path)
    else:
        logging.warning("No new albums to process.")

async def process_msdi_dataset():
    """Process the MSDI dataset"""
    logging.info("Starting MSDI dataset processing...")
    
    # Initialize MusicBrainz API
    mb_api = MusicBrainzAPI(user_agent="MSDITrackFetcher/1.0 (nicolas.fracaro@gmail.com)")
    
    # Input and output paths
    input_csv_path = "dataset/MSD-I/msdi_unique_albums.csv"
    output_dir = "dataset/MSD-I"
    output_csv_path = os.path.join(output_dir, "msdi_album_details.csv")
    processed_albums_path = os.path.join(output_dir, "processed_albums.txt")
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Set the processed albums file path before loading MBIDs
    mb_api.processed_albums_file = processed_albums_path
    
    # Get unprocessed MBIDs and image URLs from CSV
    all_mbids, image_urls = mb_api.get_mbids_from_csv(input_csv_path, "mbid")
    
    if all_mbids:
        test_mbids = all_mbids  # Process first 10 unprocessed tracks
        logging.info(f"Processing {len(test_mbids)} tracks from the dataset.")
        
        # Process the tracks
        await mb_api.process_msdi_tracks(test_mbids, image_urls, output_csv_path)
    else:
        logging.warning("No new tracks to process.")

async def main():
    """Main entry point - choose which dataset to process"""
    # Process MuMu dataset
    # await process_mumu_dataset()
    
    # Process MSDI dataset
    await process_msdi_dataset()

if __name__ == "__main__":
    asyncio.run(main())