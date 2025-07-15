import pandas as pd
import ast
import random
import google.generativeai as genai
import os
import json
import io
import time
from datetime import datetime

GEMINI_API_KEY = "AIzaSyBMC8_LdcQy5fuQGgV8v1mm2rKpOVzC4Rk"

# Configure the API key
genai.configure(api_key=GEMINI_API_KEY)

# The 11 supergenres used in the dataset
SUPERGENRES = [
    'Classical', 'Country & Folk', 'Electronic', 'Hip Hop', 
    'Jazz & Blues', 'Metal', 'Pop', 'R&B', 'Rock', 'Speciality', 'World music'
]

# Configuration
BATCH_SIZE = 75
OUTPUT_FOLDER = './data/gemini/genre_prediction'
PROGRESS_FILE = os.path.join(OUTPUT_FOLDER, 'progress.json')
FINAL_OUTPUT_FILE = os.path.join(OUTPUT_FOLDER, 'billboard_albums_with_genres.csv')


def setup_output_folder():
    """Create output folder and initialize progress tracking"""
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    print(f"Output folder created/verified: {OUTPUT_FOLDER}")


def load_unlabeled_albums(csv_path):
    """
    Load the billboard dataset without genres for prediction
    
    Args:
        csv_path: Path to the unlabeled billboard dataset CSV
    
    Returns:
        DataFrame: Albums to be classified
    """
    try:
        df = pd.read_csv(csv_path)
        print(f"Loaded {len(df)} albums for genre prediction")
        print(f"Columns available: {list(df.columns)}")
        return df
        
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return None


def load_progress():
    """Load progress from previous runs"""
    try:
        if os.path.exists(PROGRESS_FILE):
            with open(PROGRESS_FILE, 'r') as f:
                progress = json.load(f)
            print(f"Resuming from batch {progress.get('last_completed_batch', 0) + 1}")
            return progress
        else:
            return {'last_completed_batch': -1, 'processed_albums': 0, 'start_time': None}
    except Exception as e:
        print(f"Error loading progress: {e}")
        return {'last_completed_batch': -1, 'processed_albums': 0, 'start_time': None}


def save_progress(progress):
    """Save current progress"""
    try:
        with open(PROGRESS_FILE, 'w') as f:
            json.dump(progress, f, indent=2)
    except Exception as e:
        print(f"Error saving progress: {e}")


def create_gemini_prompt(albums_batch):
    """
    Create a prompt for Gemini to classify album genres
    
    Args:
        albums_batch: DataFrame containing a batch of albums
    
    Returns:
        str: Formatted prompt for Gemini
    """
    supergenres_str = ", ".join(SUPERGENRES)
    
    prompt = f"""
You are a music expert tasked with classifying album genres. I will provide you with a list of albums, and you need to classify each one using ONLY the following 11 supergenres:

{supergenres_str}

IMPORTANT INSTRUCTIONS:
1. Return ONLY a CSV format with the following columns: album_group_mbid, title, release_group_date, genres, sure
2. The genres column should contain a Python list format: ['Genre'], only 1 single genre per album is allowed
3. Use ONLY the 11 supergenres listed above - no other genres are allowed
4. If you're unsure about a genre, pick the most likely one from the 11 options but set "sure" to false
5. Do not include any explanations, headers, or additional text - ONLY the CSV data
6. Do not include markdown formatting or code blocks
7. CRITICAL: Use proper CSV escaping - put double quotes around any field that contains commas, quotes, or special characters
8. Example format: "album_id","Title, with comma","2023","['Pop']","true"

Here are the albums to classify:

"""
    
    # Add album information
    for _, row in albums_batch.iterrows():
        # Handle different possible column names
        artist = row.get('artist_name', row.get('artist', 'Unknown Artist'))
        album = row.get('album_group_title', row.get('title', 'Unknown Album'))
        date = row.get('first_release_date_group', row.get('year', 'Unknown'))
        mbid = row.get('album_group_mbid', row.get('mbid', f"unknown_{row.name}"))
        
        prompt += f"Album: {album} by {artist} ({date}), MBID: {mbid}\n"
    
    prompt += "\nReturn the CSV data now:"
    
    return prompt


def get_genres_from_gemini(albums_batch, batch_num):
    """
    Get genre classifications from Gemini for a batch of albums
    
    Args:
        albums_batch: DataFrame containing albums to classify
        batch_num: Batch number for progress tracking
    
    Returns:
        str: Gemini's response in CSV format
    """
    prompt = create_gemini_prompt(albums_batch)
    
    try:
        # Create the model
        model = genai.GenerativeModel('gemini-2.0-flash')
        
        print(f"Processing batch {batch_num}... ({len(albums_batch)} albums)")
        
        # Generate content
        response = model.generate_content(prompt)
        
        return response.text
        
    except Exception as e:
        print(f"Error calling Gemini API for batch {batch_num}: {e}")
        return None


def parse_gemini_response(response_text):
    """
    Parse Gemini's CSV response into a DataFrame
    
    Args:
        response_text: Raw response from Gemini
    
    Returns:
        DataFrame: Parsed CSV data
    """
    try:
        # Clean the response - remove any markdown or extra text
        lines = response_text.strip().split('\n')
        csv_lines = []
        
        for line in lines:
            line = line.strip()
            # Skip empty lines, markdown code blocks, or explanatory text
            if line and not line.startswith('```') and ',' in line:
                csv_lines.append(line)
        
        if not csv_lines:
            print("No valid CSV data found in response")
            return None
        
        # Add header if not present
        if not csv_lines[0].lower().startswith('album_group_mbid'):
            csv_lines.insert(0, 'album_group_mbid,title,release_group_date,genres,sure')
        
        # Create DataFrame from CSV lines with proper CSV parsing
        csv_data = '\n'.join(csv_lines)
        df = pd.read_csv(io.StringIO(csv_data), quotechar='"', skipinitialspace=True)
        
        return df
        
    except Exception as e:
        print(f"Error parsing Gemini response: {e}")
        
        # Try alternative parsing method
        try:
            print("Attempting alternative parsing...")
            return parse_gemini_response_fallback(response_text)
        except Exception as e2:
            print(f"Alternative parsing also failed: {e2}")
            return None


def parse_gemini_response_fallback(response_text):
    """
    Fallback parsing method for problematic CSV responses
    
    Args:
        response_text: Raw response from Gemini
    
    Returns:
        DataFrame: Parsed CSV data
    """
    lines = response_text.strip().split('\n')
    parsed_data = []
    
    for line in lines:
        line = line.strip()
        if not line or line.startswith('```'):
            continue
            
        # Count commas to estimate if this is a valid CSV line
        comma_count = line.count(',')
        if comma_count >= 4:  # Should have at least 4 commas for 5 fields
            # Split but be careful about commas in brackets
            parts = []
            current_part = ""
            in_brackets = False
            in_quotes = False
            
            for char in line:
                if char == '"' and not in_brackets:
                    in_quotes = not in_quotes
                elif char == '[' and not in_quotes:
                    in_brackets = True
                elif char == ']' and not in_quotes:
                    in_brackets = False
                elif char == ',' and not in_brackets and not in_quotes:
                    parts.append(current_part.strip().strip('"'))
                    current_part = ""
                    continue
                current_part += char
            
            # Add the last part
            parts.append(current_part.strip().strip('"'))
            
            if len(parts) == 5:
                parsed_data.append({
                    'album_group_mbid': parts[0],
                    'title': parts[1],
                    'release_group_date': parts[2],
                    'genres': parts[3],
                    'sure': parts[4]
                })
    
    if parsed_data:
        return pd.DataFrame(parsed_data)
    else:
        raise ValueError("No valid data could be parsed")


def save_batch_results(predictions_df, batch_num):
    """Save batch results to individual files"""
    batch_file = os.path.join(OUTPUT_FOLDER, f'batch_{batch_num:04d}.csv')
    predictions_df.to_csv(batch_file, index=False)
    print(f"Batch {batch_num} saved to {batch_file}")


def combine_all_batches():
    """Combine all batch files into final output"""
    print("Combining all batch results...")
    
    batch_files = [f for f in os.listdir(OUTPUT_FOLDER) if f.startswith('batch_') and f.endswith('.csv')]
    batch_files.sort()
    
    if not batch_files:
        print("No batch files found to combine")
        return
    
    combined_data = []
    for batch_file in batch_files:
        batch_path = os.path.join(OUTPUT_FOLDER, batch_file)
        try:
            batch_df = pd.read_csv(batch_path)
            combined_data.append(batch_df)
            print(f"Added {len(batch_df)} records from {batch_file}")
        except Exception as e:
            print(f"Error reading {batch_file}: {e}")
    
    if combined_data:
        final_df = pd.concat(combined_data, ignore_index=True)
        final_df.to_csv(FINAL_OUTPUT_FILE, index=False)
        print(f"Final results saved to {FINAL_OUTPUT_FILE}")
        print(f"Total albums processed: {len(final_df)}")
        
        # Generate summary statistics
        generate_summary_report(final_df)
        
        return final_df
    else:
        print("No data to combine")
        return None


def generate_summary_report(final_df):
    """Generate a summary report of the genre predictions"""
    summary_file = os.path.join(OUTPUT_FOLDER, 'summary_report.txt')
    
    with open(summary_file, 'w') as f:
        f.write("GEMINI GENRE PREDICTION SUMMARY REPORT\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Total albums processed: {len(final_df)}\n")
        f.write(f"Processing completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        # Genre distribution
        f.write("GENRE DISTRIBUTION:\n")
        f.write("-" * 20 + "\n")
        
        # Parse genres and count (only those with confidence true)
        genre_counts = {}
        confidence_counts = {'true': 0, 'false': 0}
        total_confident_albums = 0
        
        for _, row in final_df.iterrows():
            # Count confidence
            sure_val = str(row.get('sure', 'unknown')).lower()
            if sure_val in confidence_counts:
                confidence_counts[sure_val] += 1
            
            # Only count genres if confidence is true
            if sure_val == 'true':
                total_confident_albums += 1
                try:
                    genres = ast.literal_eval(row['genres'])
                    for genre in genres:
                        genre_counts[genre] = genre_counts.get(genre, 0) + 1
                except:
                    pass
        
        f.write(f"Albums with high confidence (sure=true): {total_confident_albums}\n")
        f.write(f"Albums with low confidence (sure=false): {confidence_counts['false']}\n\n")
        
        if total_confident_albums > 0:
            for genre, count in sorted(genre_counts.items(), key=lambda x: x[1], reverse=True):
                percentage = (count / total_confident_albums) * 100
                f.write(f"{genre}: {count} ({percentage:.1f}%)\n")
        else:
            f.write("No albums with high confidence found.\n")
        
        f.write(f"\nCONFIDENCE LEVELS:\n")
        f.write("-" * 20 + "\n")
        for conf, count in confidence_counts.items():
            percentage = (count / len(final_df)) * 100
            f.write(f"Sure {conf}: {count} ({percentage:.1f}%)\n")
    
    print(f"Summary report saved to {summary_file}")


def predict_genres():
    """
    Main function to predict genres for the unlabeled dataset
    """
    # Setup
    setup_output_folder()
    
    # Load the unlabeled dataset
    dataset_path = './data/billboard_album__no_genres.csv'
    albums_df = load_unlabeled_albums(dataset_path)
    
    if albums_df is None:
        return
    
    # Load progress
    progress = load_progress()
    start_batch = progress['last_completed_batch'] + 1
    
    if progress['start_time'] is None:
        progress['start_time'] = datetime.now().isoformat()
    
    total_batches = (len(albums_df) + BATCH_SIZE - 1) // BATCH_SIZE
    print(f"Total batches to process: {total_batches}")
    print(f"Starting from batch: {start_batch + 1}")
    
    # Process albums in batches
    successful_batches = 0
    
    for i in range(start_batch * BATCH_SIZE, len(albums_df), BATCH_SIZE):
        batch = albums_df.iloc[i:i+BATCH_SIZE]
        batch_num = i // BATCH_SIZE + 1
        
        print(f"\n--- Processing batch {batch_num}/{total_batches} ---")
        
        # Get predictions from Gemini
        gemini_response = get_genres_from_gemini(batch, batch_num)
        
        if gemini_response:
            # Parse the response
            predictions_df = parse_gemini_response(gemini_response)
            
            if predictions_df is not None:
                # Save batch results
                save_batch_results(predictions_df, batch_num)
                successful_batches += 1
                
                # Update progress
                progress['last_completed_batch'] = batch_num - 1
                progress['processed_albums'] = i + len(batch)
                save_progress(progress)
                
                print(f"Successfully processed batch {batch_num} ({len(predictions_df)} albums)")
                
                # Add a small delay to be respectful to the API
                time.sleep(1)
                
            else:
                print(f"Failed to parse response for batch {batch_num}")
        else:
            print(f"Failed to get response for batch {batch_num}")
    
    print(f"\n--- Processing Complete ---")
    print(f"Successfully processed {successful_batches} batches")
    
    # Combine all results
    final_df = combine_all_batches()
    
    if final_df is not None:
        print(f"\nGenre prediction complete!")
        print(f"Results saved in: {OUTPUT_FOLDER}")
        print(f"Final file: {FINAL_OUTPUT_FILE}")


if __name__ == "__main__":
    # predict_genres()
    df = pd.read_csv("./data/gemini/genre_prediction/billboard_albums_with_genres_gemini.csv")
    generate_summary_report(df)
