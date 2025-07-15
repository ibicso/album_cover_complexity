#!/usr/bin/env python3
"""
MusicBrainz Billboard Album Search Script (Optimized with async/aiohttp)

This script searches for albums from Billboard charts in MusicBrainz database
and retrieves detailed information along with cover art.

Required fields from all_unique_albums.csv:
- artist
- image
- isNew
- lastPos
- peakPos
- rank
- title
- weeks
- image_url
- year
- genre

Additional MusicBrainz fields:
- album_group_mbid
- album_group_title
- artist_name
- first_release_date_group
- primary_type
- release_mbid
- release_title
- release_date
- release_country
- track_count
- label_name
- label_mbid
- parent_label_name
- parent_label_mbid
- msdi_image_url
- genres
"""

import csv
import json
import os
import time
import asyncio
import aiohttp
from typing import Dict, List, Optional, Any, Tuple

import logging
import hashlib
from datetime import datetime, timedelta
import statistics

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('musicbrainz_search.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class ProcessingStats:
    def __init__(self):
        self.start_time = None
        self.batch_times = []  # List of seconds per batch
        self.albums_per_batch = []  # List of albums processed per batch
        self.total_albums_processed = 0
        self.stats_file = "billboard_processing_stats.txt"
        self.stats_dir = "stats"
        os.makedirs(self.stats_dir, exist_ok=True)
    
    def start(self):
        self.start_time = time.time()
        # Initialize stats file
        with open(os.path.join(self.stats_dir, self.stats_file), 'w') as f:
            f.write("Billboard MusicBrainz Processing Statistics\n")
            f.write("=========================================\n")
            f.write(f"Process started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
    
    def add_batch(self, batch_time, albums_in_batch):
        self.batch_times.append(batch_time)
        self.albums_per_batch.append(albums_in_batch)
        self.total_albums_processed += albums_in_batch
    
    def get_stats(self, total_albums=None, remaining_albums=None):
        if not self.batch_times:
            return {
                "total_runtime": "0:00:00",
                "batches_completed": 0,
                "total_albums_processed": 0,
                "avg_time_per_batch": "0:00:00",
                "avg_time_per_album": "0.00 seconds",
                "estimated_remaining_time": "0:00:00"
            }
        
        current_duration = time.time() - self.start_time
        avg_batch_time = statistics.mean(self.batch_times)
        avg_album_time = sum(self.batch_times) / self.total_albums_processed
        
        # Calculate estimated remaining time
        if remaining_albums is not None:
            estimated_remaining = remaining_albums * avg_album_time
        else:
            estimated_remaining = self.total_albums_processed * 0.1 * avg_album_time  # Default estimate
        
        stats = {
            "total_runtime": str(timedelta(seconds=int(current_duration))),
            "batches_completed": len(self.batch_times),
            "total_albums_processed": self.total_albums_processed,
            "avg_time_per_batch": str(timedelta(seconds=int(avg_batch_time))),
            "avg_time_per_album": f"{avg_album_time:.2f} seconds",
            "estimated_remaining_time": str(timedelta(seconds=int(estimated_remaining)))
        }
        return stats

    def save_stats(self, stats, is_batch=True):
        """Save statistics to a file"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"billboard_batch_stats_{timestamp}.txt" if is_batch else f"billboard_final_stats_{timestamp}.txt"
        filepath = os.path.join(self.stats_dir, filename)
        
        with open(filepath, 'w') as f:
            f.write("Billboard Processing Statistics:\n")
            f.write("==============================\n")
            for key, value in stats.items():
                f.write(f"{key}: {value}\n")

class MusicBrainzBillboardSearch:
    def __init__(self, user_agent: str = "BillboardMusicBrainzSearch/1.0"):
        """Initialize MusicBrainz search with async/aiohttp and rate limiting"""
        self.user_agent = user_agent
        self.headers = {
            'User-Agent': user_agent,
            'Accept': 'application/json'
        }
        self.session = None
        self.request_semaphore = asyncio.Semaphore(1)  # Ensure 1 request per second
        self.cover_art_base_url = "https://coverartarchive.org"
        self.musicbrainz_base_url = "https://musicbrainz.org/ws/2"
        self.billboard_placeholder_url = "https://www.billboard.com/wp-content/themes/vip/pmc-billboard-2021/assets/public/lazyload-fallback.gif"
        
    async def _ensure_rate_limit(self):
        """Ensure we respect MusicBrainz's 1 request per second rate limit"""
        async with self.request_semaphore:
            await asyncio.sleep(1.0)  # Ensure 1 second between requests
    
    async def _execute_mb_request(self, url: str, params: Dict = None) -> Optional[Dict[str, Any]]:
        """Execute a MusicBrainz API request with rate limiting and detailed logging"""
        await self._ensure_rate_limit()
        
        request_start_time = time.time()
        logger.info(f"Making API request to: {url}")
        if params:
            logger.debug(f"Request parameters: {params}")
        
        response_text = None
        try:
            async with self.session.get(url, headers=self.headers, params=params) as response:
                request_duration = time.time() - request_start_time
                logger.info(f"API request completed in {request_duration:.2f}s - Status: {response.status}")
                
                response_text = await response.text()
                response.raise_for_status()
                return await response.json()
                
        except aiohttp.ClientError as e:
            request_duration = time.time() - request_start_time
            logger.error(f"HTTP error for {url} after {request_duration:.2f}s: {e}. Response content: {response_text[:500]}...")
            return None
        except json.JSONDecodeError as e:
            request_duration = time.time() - request_start_time
            logger.error(f"JSON decode error for {url} after {request_duration:.2f}s: {e}. Response content: {response_text[:500]}...")
            return None
        except Exception as e:
            request_duration = time.time() - request_start_time
            logger.error(f"Unexpected error for {url} after {request_duration:.2f}s: {e}")
            return None
        
    def is_placeholder_image(self, image_url: str) -> bool:
        """Check if the image URL is the Billboard placeholder"""
        return image_url == self.billboard_placeholder_url
    
    async def search_album_group(self, artist_name: str, album_title: str) -> Optional[Dict[str, Any]]:
        """Search for album group in MusicBrainz using async API calls"""
        try:
            # Clean up search terms
            artist_clean = self._clean_search_term(artist_name)
            album_clean = self._clean_search_term(album_title)
            
            search_start_time = time.time()
            logger.info(f"Starting search for: {artist_clean} - {album_clean}")
            
            # Search for release groups (album groups)
            query = f'artist:"{artist_clean}" AND releasegroup:"{album_clean}"'
            url = f"{self.musicbrainz_base_url}/release-group"
            params = {"query": query, "limit": 10, "fmt": "json"}
            
            result = await self._execute_mb_request(url, params)
            
            if not result or not result.get('release-groups'):
                # Try broader search
                logger.info("No results with exact search, trying broader search...")
                query = f'artist:{artist_clean} AND releasegroup:{album_clean}'
                params = {"query": query, "limit": 10, "fmt": "json"}
                
                result = await self._execute_mb_request(url, params)
            
            if result and result.get('release-groups'):
                # Find best match
                best_match = self._find_best_match(
                    result['release-groups'],
                    artist_clean,
                    album_clean
                )
                search_duration = time.time() - search_start_time
                logger.info(f"Search completed in {search_duration:.2f}s - Found match: {best_match is not None}")
                return best_match
            else:
                search_duration = time.time() - search_start_time
                logger.info(f"Search completed in {search_duration:.2f}s - No matches found")
                
        except Exception as e:
            search_duration = time.time() - search_start_time
            logger.error(f"Error searching for {artist_name} - {album_title} after {search_duration:.2f}s: {e}")
            
        return None
    
    async def get_album_details(self, release_group: Dict[str, Any], billboard_year: str = None) -> Dict[str, Any]:
        """Get detailed information about an album group using async API calls"""
        try:
            rg_id = release_group['id']
            details_start_time = time.time()
            logger.info(f"Getting album details for release group: {rg_id}")
            
            # Get release group details with releases, genres, and artist-credits in one request
            url = f"{self.musicbrainz_base_url}/release-group/{rg_id}"
            params = {"inc": "releases+artist-credits+tags+genres", "fmt": "json"}
            
            rg_info = await self._execute_mb_request(url, params)
            if not rg_info:
                return {}
            
            # Get artist name from artist credits
            artist_name = ""
            if rg_info.get('artist-credit'):
                artist_name = "".join([
                    ac.get('name', '') + ac.get('joinphrase', '') 
                    for ac in rg_info['artist-credit']
                ]).strip()
            
            # Get the first/primary release for additional details, considering Billboard year
            primary_release = self._get_primary_release(rg_info.get('releases', []), billboard_year)
            
            if not primary_release:
                logger.warning(f"No primary release found for release group {rg_id}")
                return {}
            
            # Create tasks for parallel requests
            release_url = f"{self.musicbrainz_base_url}/release/{primary_release['id']}"
            release_params = {"inc": "recordings+labels", "fmt": "json"}
            cover_url = f"{self.cover_art_base_url}/release/{primary_release['id']}"
            
            # Execute requests in parallel (but still respecting rate limit through semaphore)
            release_task = self._execute_mb_request(release_url, release_params)
            
            cover_task = self._execute_mb_request(cover_url)
            
            release_info, cover_response = await asyncio.gather(release_task, cover_task, return_exceptions=True)
            
            # Handle exceptions
            if isinstance(release_info, Exception):
                logger.error(f"Error getting release info: {release_info}")
                release_info = None
            if isinstance(cover_response, Exception):
                logger.warning(f"Error getting cover art (non-critical): {cover_response}")
                cover_response = None
            
            if not release_info:
                logger.warning(f"No release info found for release {primary_release['id']}")
                return {}
            
            # Process cover art if available
            cover_art_url = None
            if cover_response and cover_response.get('images'):
                front_images = [img for img in cover_response['images'] if img.get('front', False)]
                if front_images:
                    cover_art_url = front_images[0].get('thumbnails', {}).get('large', front_images[0]['image'])
            
            # Get label information
            label_info = release_info.get('label-info', [{}])[0] if release_info.get('label-info') else {}
            label_data = label_info.get('label', {})
            
            # Get genres from the response
            genres = self._extract_genres_from_response(rg_info)
            
            details_duration = time.time() - details_start_time
            logger.info(f"Album details retrieved in {details_duration:.2f}s")
            
            return {
                'album_group_mbid': rg_id,
                'album_group_title': rg_info.get('title', ''),
                'artist_name': artist_name,
                'first_release_date_group': rg_info.get('first-release-date', ''),
                'primary_type': rg_info.get('primary-type', ''),
                'release_mbid': primary_release['id'],
                'release_title': release_info.get('title', ''),
                'release_date': release_info.get('date', ''),
                'release_country': release_info.get('country', ''),
                'track_count': len(release_info.get('media', [{}])[0].get('tracks', [])) if release_info.get('media') else 0,
                'label_name': label_data.get('name', ''),
                'label_mbid': label_data.get('id', ''),
                'parent_label_name': '',  # Would need additional API call
                'parent_label_mbid': '',  # Would need additional API call
                'msdi_image_url': cover_art_url,
                'genres': genres,
                'found_in_musicbrainz': True
            }
            
        except Exception as e:
            details_duration = time.time() - details_start_time
            logger.error(f"Error getting album details after {details_duration:.2f}s: {e}")
            return {}
    
    def _extract_genres_from_response(self, rg_info: Dict[str, Any]) -> List[str]:
        """Extract genres from MusicBrainz response"""
        genres = []
        
        # Get genres from the official MusicBrainz genres list
        if 'genres' in rg_info:
            for genre in rg_info['genres']:
                if 'name' in genre:
                    genre_name = genre['name'].strip()
                    if genre_name:
                        genres.append(genre_name)
        
        # If no official genres found, fall back to tags
        if not genres and 'tags' in rg_info:
            for tag in rg_info['tags']:
                if 'name' in tag:
                    tag_name = tag['name'].lower().strip()
                    vote_count = int(tag.get('count', 0))
                    
                    # Only consider tags with votes
                    if vote_count > 0:
                        genres.append(tag_name)
        
        # Remove duplicates and limit to top 5
        unique_genres = list(dict.fromkeys(genres))[:5]  # Preserves order
        
        return unique_genres
    
    def _clean_search_term(self, term: str) -> str:
        """Clean search terms for better matching"""
        # Remove common prefixes/suffixes that might cause issues
        replacements = {
            ' Featuring ': ' ',
            ' featuring ': ' ',
            ' feat. ': ' ',
            ' feat ': ' ',
            ' ft. ': ' ',
            ' ft ': ' ',
            ' & ': ' ',
            ' and ': ' ',
            '(Soundtrack)': '',
            '(EP)': '',
            '(Deluxe)': '',
            '(Deluxe Edition)': '',
            '(Expanded Edition)': '',
            '"': '',
            "'": ""
        }
        
        cleaned = term
        for old, new in replacements.items():
            cleaned = cleaned.replace(old, new)
        
        return cleaned.strip()
    
    def _find_best_match(self, release_groups: List[Dict], target_artist: str, target_album: str) -> Optional[Dict]:
        """Find the best matching release group"""
        best_match = None
        best_score = 0
        
        for rg in release_groups:
            score = 0
            
            # Check artist match
            if 'artist-credit' in rg:
                rg_artists = [ac.get('name', '').lower() for ac in rg['artist-credit']]
                if any(target_artist.lower() in artist for artist in rg_artists):
                    score += 10
            
            # Check album title match
            rg_title = rg.get('title', '').lower()
            if target_album.lower() in rg_title or rg_title in target_album.lower():
                score += 10
            
            # Prefer albums over other types
            if rg.get('primary-type', '').lower() == 'album':
                score += 5
            
            if score > best_score:
                best_score = score
                best_match = rg
        
        return best_match if best_score > 10 else None
    
    def _get_primary_release(self, releases: List[Dict], billboard_year: str = None) -> Optional[Dict]:
        """Get the primary/first release from a list, considering Billboard year if provided"""
        if not releases:
            return None
        
        def parse_date(date_str):
            """Parse date string and return a comparable tuple (year, month, day)"""
            if not date_str:
                return (9999, 12, 31)  # Put releases with no date at the end
            
            # Handle different date formats
            parts = date_str.split('-')
            year = int(parts[0]) if parts[0] else 9999
            month = int(parts[1]) if len(parts) > 1 and parts[1] else 12
            day = int(parts[2]) if len(parts) > 2 and parts[2] else 31
            
            return (year, month, day)
        
        def get_sort_key(release):
            """Generate sort key for release selection"""
            date_tuple = parse_date(release.get('date', ''))
            
            # Prefer official releases
            status_score = 0 if release.get('status', '').lower() == 'official' else 1
            
            # If we have a Billboard year, prefer releases closer to that year
            year_score = 0
            if billboard_year:
                try:
                    bb_year = int(billboard_year)
                    release_year = date_tuple[0]
                    if release_year != 9999:
                        year_score = abs(release_year - bb_year)
                except (ValueError, TypeError):
                    pass
            
            return (date_tuple, status_score, year_score)
        
        # Sort releases by date first, then by status, then by proximity to Billboard year
        sorted_releases = sorted(releases, key=get_sort_key)
        
        selected_release = sorted_releases[0] if sorted_releases else None
        
        if selected_release:
            logger.debug(f"Selected release: {selected_release.get('title', 'Unknown')} "
                        f"({selected_release.get('date', 'No date')}) "
                        f"Status: {selected_release.get('status', 'Unknown')} "
                        f"from {len(releases)} available releases")
            
            # Log other releases for debugging
            if len(releases) > 1:
                logger.debug("Other available releases:")
                for i, release in enumerate(sorted_releases[1:6], 1):  # Show up to 5 other releases
                    logger.debug(f"  {i}. {release.get('title', 'Unknown')} "
                               f"({release.get('date', 'No date')}) "
                               f"Status: {release.get('status', 'Unknown')}")
        
        return selected_release

    async def _download_cover_art(self, image_url: str, mbid: str, artist: str, album: str):
        """Download cover art if Billboard image is placeholder"""
        try:
            # Create safe filename
            safe_artist = "".join(c for c in artist if c.isalnum() or c in (' ', '-', '_')).rstrip()
            safe_album = "".join(c for c in album if c.isalnum() or c in (' ', '-', '_')).rstrip()
            filename = f"{safe_artist}_{safe_album}_{mbid}.jpg"
            
            # Create covers directory if it doesn't exist
            covers_dir = "covers"
            os.makedirs(covers_dir, exist_ok=True)
            
            filepath = os.path.join(covers_dir, filename)
            
            # Download image
            async with self.session.get(image_url) as response:
                response.raise_for_status()
                content = await response.read()
                
                with open(filepath, 'wb') as f:
                    f.write(content)
                
                logger.info(f"Downloaded cover art: {filepath}")
            
        except Exception as e:
            logger.error(f"Error downloading cover art: {e}")
    
    async def process_single_album(self, artist: str, title: str, billboard_image_url: str = '', billboard_year: str = None) -> Tuple[Dict[str, Any], bool]:
        """Process a single album and return the result data and success status"""
        album_start_time = time.time()
        logger.info(f"Starting processing: {artist} - {title} ({billboard_year or 'No year'})")
        
        # Search in MusicBrainz
        release_group = await self.search_album_group(artist, title)
        
        if release_group:
            album_data = await self.get_album_details(release_group, billboard_year)
            if album_data:
                # Download cover art only if Billboard image is placeholder
                #if self.is_placeholder_image(billboard_image_url) and album_data.get('msdi_image_url'):
                 #   await self._download_cover_art(
                   #     album_data['msdi_image_url'],
                   #     album_data['album_group_mbid'],
                    #    album_data['artist_name'],
                    #    album_data['album_group_title']
                    #)
                
                album_duration = time.time() - album_start_time
                logger.info(f"Successfully processed {artist} - {title} in {album_duration:.2f}s")
                return album_data, True
            else:
                album_duration = time.time() - album_start_time
                logger.warning(f"Found release group but no album details for {artist} - {title} after {album_duration:.2f}s")
                return {}, False
        else:
            album_duration = time.time() - album_start_time
            logger.warning(f"No release group found for {artist} - {title} after {album_duration:.2f}s")
            return {}, False

async def process_billboard_csv(csv_file: str, output_file: str, batch_size: int = 500, max_albums: int = None):
    """Process Billboard CSV and search for albums in MusicBrainz using async processing"""
    searcher = MusicBrainzBillboardSearch()
    stats = ProcessingStats()
    stats.start()
    
    # Initialize aiohttp session
    searcher.session = aiohttp.ClientSession()
    
    try:
        # Define output fields
        fieldnames = [
            'artist', 'image', 'isNew', 'lastPos', 'peakPos', 'rank', 'title',
            'weeks', 'image_url', 'year', 'genre',  # Original Billboard fields
            'album_group_mbid', 'album_group_title', 'artist_name',
            'first_release_date_group', 'primary_type', 'release_mbid',
            'release_title', 'release_date', 'release_country',
            'track_count', 'label_name', 'label_mbid',
            'parent_label_name', 'parent_label_mbid', 'msdi_image_url',
            'genres', 'found_in_musicbrainz'
        ]
        
        # Create progress directory if it doesn't exist
        progress_dir = "progress_retry"
        os.makedirs(progress_dir, exist_ok=True)
        
        # Read all entries first
        all_entries = []
        with open(csv_file, 'r', encoding='utf-8') as infile:
            reader = csv.DictReader(infile)
            all_entries = list(reader)
        
        total_entries_in_file = len(all_entries)
        
        # Find the last processed batch
        processed_files = [f for f in os.listdir(progress_dir) if f.startswith('batch_') and f.endswith('.csv')]
        last_batch = -1
        total_rows = sum(sum(1 for _ in open(os.path.join(progress_dir, f), 'r')) - 1 for f in processed_files)
        start_index = total_rows + 1
        logger.info(f"Starting from index {start_index}")
        
        # Calculate end index based on max_albums for this run
        if max_albums:
            end_index = min(start_index + max_albums, total_entries_in_file)
            logger.info(f"Will process {max_albums} albums in this run (from index {start_index} to {end_index - 1})")
        else:
            end_index = total_entries_in_file
            logger.info(f"Will process all remaining albums (from index {start_index} to {end_index - 1})")
        
        # Check if there's anything to process
        if start_index >= total_entries_in_file:
            logger.info(f"All albums have already been processed (start_index: {start_index}, total in file: {total_entries_in_file})")
            return
        
        if start_index >= end_index:
            logger.info(f"Nothing to process in this run (start_index: {start_index}, end_index: {end_index})")
            return
        
        processed_count = start_index
        found_count = 0  # Count of albums found in MusicBrainz in this run only
        current_batch_num = last_batch + 1
        
        # Process entries in batches
        for i in range(start_index, end_index, batch_size):
            batch_start_time = time.time()
            batch_entries = all_entries[i:i + batch_size]
            batch_data = []
            
            logger.info(f"Processing batch {current_batch_num} with {len(batch_entries)} entries")
            
            # Process entries in the batch
            for row in batch_entries:
                processed_count += 1
                artist = row.get('artist', '').strip()
                title = row.get('title', '').strip()
                billboard_image_url = row.get('image_url', '')
                billboard_year = row.get('year', '')
                
                if not artist or not title:
                    output_row = {**row, 'found_in_musicbrainz': False}
                    batch_data.append(output_row)
                    continue
                
                logger.info(f"Processing {processed_count}/{end_index}: {artist} - {title} ({billboard_year or 'No year'})")
                
                # Process the album
                album_data, found = await searcher.process_single_album(artist, title, billboard_image_url, billboard_year)
                
                if found:
                    found_count += 1
                    # Preserve original Billboard data
                    output_row = {**row, **album_data, 'found_in_musicbrainz': True}
                else:
                    output_row = {**row, 'found_in_musicbrainz': False}
                
                batch_data.append(output_row)
            
            # Save batch
            batch_end_time = time.time()
            batch_duration = batch_end_time - batch_start_time
            stats.add_batch(batch_duration, len(batch_data))
            
            # Add timestamp to batch filename
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            batch_file = os.path.join(progress_dir, f'batch_{current_batch_num}_{timestamp}.csv')
            with open(batch_file, 'w', encoding='utf-8', newline='') as batch_out:
                writer = csv.DictWriter(batch_out, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(batch_data)
            logger.info(f"Saved batch {current_batch_num} to {batch_file}")
            
            # Get and save batch statistics
            remaining_albums = end_index - processed_count
            current_stats = stats.get_stats(total_albums=end_index - start_index, remaining_albums=remaining_albums)
            batch_stats = {
                "batch_number": current_batch_num,
                "batch_time": str(timedelta(seconds=int(batch_duration))),
                "avg_time_per_item_batch": f"{(batch_duration/len(batch_data)):.2f} seconds",
                "total_runtime": current_stats['total_runtime'],
                "total_processed": current_stats['total_albums_processed'],
                "remaining_albums": remaining_albums,
                "estimated_remaining_time": current_stats['estimated_remaining_time'],
                "found_in_batch": sum(1 for row in batch_data if row.get('found_in_musicbrainz')),
                "success_rate_batch": f"{(sum(1 for row in batch_data if row.get('found_in_musicbrainz'))/len(batch_data))*100:.2f}%"
            }
            stats.save_stats(batch_stats, is_batch=True)
            
            logger.info(f"""
Batch {current_batch_num} Statistics:
- Batch processing time: {batch_stats['batch_time']}
- Average time per item in batch: {batch_stats['avg_time_per_item_batch']}
- Found in this batch: {batch_stats['found_in_batch']}/{len(batch_data)} ({batch_stats['success_rate_batch']})
- Total runtime so far: {batch_stats['total_runtime']}
- Total processed: {batch_stats['total_processed']}/{end_index}
- Remaining albums: {batch_stats['remaining_albums']}
- Estimated remaining time: {batch_stats['estimated_remaining_time']}
""")
            
            current_batch_num += 1
        
        # Save final statistics
        final_stats = stats.get_stats()
        albums_processed_this_run = processed_count - start_index
        final_stats.update({
            "total_entries_in_file": total_entries_in_file,
            "start_index": start_index,
            "end_index": end_index,
            "albums_processed_this_run": albums_processed_this_run,
            "total_found": found_count,
            "success_rate": f"{(found_count/albums_processed_this_run)*100:.2f}%" if albums_processed_this_run > 0 else "0.00%"
        })
        stats.save_stats(final_stats, is_batch=False)
        
        # Combine all batches into final output file
        with open(output_file, 'w', encoding='utf-8', newline='') as outfile:
            writer = csv.DictWriter(outfile, fieldnames=fieldnames)
            writer.writeheader()
            
            # Update batch file pattern to match timestamped files
            for batch_num in range(current_batch_num):
                # Find the matching batch file with timestamp
                batch_files = [f for f in os.listdir(progress_dir) if f.startswith(f'batch_{batch_num}_') and f.endswith('.csv')]
                if batch_files:  # Use the first matching file if found
                    batch_file = os.path.join(progress_dir, batch_files[0])
                    with open(batch_file, 'r', encoding='utf-8') as batch_in:
                        reader = csv.DictReader(batch_in)
                        writer.writerows(reader)
        
        logger.info(f"""
Final Processing Statistics:
- Total runtime: {final_stats['total_runtime']}
- Total entries in file: {final_stats['total_entries_in_file']}
- Processed in this run: {final_stats['albums_processed_this_run']} (from index {final_stats['start_index']} to {final_stats['end_index'] - 1})
- Total found in MusicBrainz: {final_stats['total_found']}
- Success rate: {final_stats['success_rate']}
- Average time per album: {final_stats['avg_time_per_album']}
""")
        logger.info(f"Results saved to: {output_file}")
        
    finally:
        # Close aiohttp session
        await searcher.session.close()

def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Search Billboard albums in MusicBrainz (Optimized)')
    parser.add_argument('-i', '--input_csv',default='not_found_in_musicbrainz.csv', help='Input Billboard CSV file')
    parser.add_argument('-o', '--output', default='not_found_in_musicbrainz_retry.csv',
                       help='Output CSV file (default: not_found_in_musicbrainz_retry.csv)')
    parser.add_argument('-b', '--batch-size', type=int, default=500,
                       help='Number of entries to process in each batch (default: 10)')
    parser.add_argument('-n', '--max-albums', type=int, default=5000,
                       help='Maximum number of albums to process in this run starting from last processed (default: process all remaining)')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.input_csv):
        logger.error(f"Input file not found: {args.input_csv}")
        return
    
    if args.max_albums:
        logger.info(f"Starting optimized MusicBrainz search for: {args.input_csv} (processing next {args.max_albums} albums from last position)")
    else:
        logger.info(f"Starting optimized MusicBrainz search for: {args.input_csv} (processing all remaining albums)")
    
    asyncio.run(process_billboard_csv(args.input_csv, args.output, args.batch_size, args.max_albums))

if __name__ == "__main__":
    main()
