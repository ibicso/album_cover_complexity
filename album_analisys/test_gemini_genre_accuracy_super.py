import pandas as pd
import ast
import random
import google.generativeai as genai
import os
import json
import io

GEMINI_API_KEY = "AIzaSyBMC8_LdcQy5fuQGgV8v1mm2rKpOVzC4Rk"

# Configure the API key
genai.configure(api_key=GEMINI_API_KEY)

# The 11 supergenres used in the dataset
SUPERGENRES = [
    'Classical', 'Country & Folk', 'Electronic', 'Hip Hop', 
    'Jazz & Blues', 'Metal', 'Pop', 'R&B', 'Rock', 'Speciality', 'World music'
]


def load_labelled_albums(csv_path, sample_size=500):
    """
    Load the billboard dataset and create a random sample of labelled albums
    
    Args:
        csv_path: Path to the billboard dataset CSV
        sample_size: Number of albums to sample for testing
    
    Returns:
        DataFrame: Random sample of labelled albums
    """
    try:
        df = pd.read_csv(csv_path)
        
        # Filter out albums without genres (empty lists or missing data)
        df_with_genres = df[df['genres'].notna() & (df['genres'] != '[]')].copy()
        
        # Take a random sample
        if len(df_with_genres) > sample_size:
            sample_df = df_with_genres.sample(n=sample_size, random_state=42)
        else:
            sample_df = df_with_genres
            print(f"Warning: Only {len(df_with_genres)} labelled albums available, using all of them.")
        
        print(f"Selected {len(sample_df)} albums for testing")
        return sample_df
        
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return None


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
        prompt += f"Album: {row['album_group_title']} by {row['artist_name']} ({row['first_release_date_group']}), MBID: {row['album_group_mbid']}\n"
    
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
        
        print(f"Processing batch {batch_num}...")
        
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
        print(f"Raw response: {response_text}")
        
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


def test_gemini_accuracy():
    """
    Main function to test Gemini's genre classification accuracy
    """
    # Load the labelled dataset
    dataset_path = './data/billboard_album_final_super_genres.csv'
    test_albums = load_labelled_albums(dataset_path, sample_size=500)
    
    if test_albums is None:
        return
    
    # Prepare the ground truth data
    ground_truth = []
    for _, row in test_albums.iterrows():
        try:
            actual_genres = ast.literal_eval(row['genres'])
        except:
            actual_genres = []
        
        ground_truth.append({
            'album_group_mbid': row['album_group_mbid'],
            'title': row['album_group_title'], 
            'release_group_date': row['first_release_date_group'],
            'actual_genres': actual_genres
        })
    
    # Save ground truth for comparison
    ground_truth_df = pd.DataFrame(ground_truth)
    ground_truth_df.to_csv('./data/gemini/gemini_test_ground_truth.csv', index=False)
    print("Ground truth saved to ./data/gemini/gemini_test_ground_truth.csv")
    
    # Process albums in batches (Gemini has token limits)
    batch_size = 75
    all_predictions = []
    
    for i in range(0, len(test_albums), batch_size):
        batch = test_albums.iloc[i:i+batch_size]
        batch_num = i // batch_size + 1
        
        # Get predictions from Gemini
        gemini_response = get_genres_from_gemini(batch, batch_num)
        
        if gemini_response:
            # Parse the response
            predictions_df = parse_gemini_response(gemini_response)
            
            if predictions_df is not None:
                all_predictions.append(predictions_df)
                print(f"Successfully processed batch {batch_num}")
            else:
                print(f"Failed to parse response for batch {batch_num}")
        else:
            print(f"Failed to get response for batch {batch_num}")
    
    # Combine all predictions
    if all_predictions:
        final_predictions = pd.concat(all_predictions, ignore_index=True)
        
        # Save predictions
        final_predictions.to_csv('./data/gemini/gemini_genre_predictions.csv', index=False)
        print(f"Gemini predictions saved to ./data/gemini/gemini_genre_predictions.csv")
        print(f"Total albums processed: {len(final_predictions)}")
        
        # Display sample of predictions
        print("\nSample predictions:")
        print(final_predictions.head(10))
        
    else:
        print("No successful predictions obtained from Gemini")


if __name__ == "__main__":
    # Create results directory if it doesn't exist
    os.makedirs('./data/gemini', exist_ok=True)
    
    test_gemini_accuracy() 