import pandas as pd
import ast
import random
from google import genai
from google.genai.types import Tool, GenerateContentConfig, GoogleSearch
import os
import json
import io
import numpy as np

GEMINI_API_KEY = "YOUR_GEMINI_API_KEY"

# Configure the API key
os.environ['GOOGLE_API_KEY'] = GEMINI_API_KEY
client = genai.Client()
model_id = "gemini-2.0-flash"

# Configure Google Search tool
google_search_tool = Tool(
    google_search=GoogleSearch()
)


def load_labelled_albums(csv_path, sample_size=500):
    """
    Load the billboard dataset and create a random sample of albums with known release dates
    
    Args:
        csv_path: Path to the billboard dataset CSV
        sample_size: Number of albums to sample for testing
    
    Returns:
        DataFrame: Random sample of albums with valid release dates
    """
    try:
        df = pd.read_csv(csv_path)
        
        # Filter out albums without valid release dates
        df_with_dates = df[df['first_release_date_group'].notna()].copy()
        
        # Further filter to albums with 4-digit years
        df_with_dates = df_with_dates[
            df_with_dates['first_release_date_group'].astype(str).str.len() >= 4
        ].copy()
        
        # Extract year for validation
        df_with_dates['year'] = df_with_dates['first_release_date_group'].astype(str).str[:4]
        df_with_dates = df_with_dates[df_with_dates['year'].str.isdigit()].copy()
        df_with_dates['year'] = df_with_dates['year'].astype(int)
        
        # Filter reasonable years (1900-2025)
        df_with_dates = df_with_dates[
            (df_with_dates['year'] >= 1900) & (df_with_dates['year'] <= 2025)
        ].copy()
        
        # Take a random sample
        if len(df_with_dates) > sample_size:
            sample_df = df_with_dates.sample(n=sample_size, random_state=42)
        else:
            sample_df = df_with_dates
            print(f"Warning: Only {len(df_with_dates)} albums with valid dates available, using all of them.")
        
        print(f"Selected {len(sample_df)} albums for testing")
        print(f"Date range: {sample_df['year'].min()} - {sample_df['year'].max()}")
        return sample_df
        
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return None


def create_gemini_prompt(albums_batch):
    """
    Create a prompt for Gemini to predict album release years using Google Search
    
    Args:
        albums_batch: DataFrame containing a batch of albums
    
    Returns:
        str: Formatted prompt for Gemini
    """

    prompt = f"""
You are a music expert tasked with finding the first release years of albums. Use Google Search to look up each album and find accurate release information.

SEARCH STRATEGY:
- Search for each album using queries like: "[artist name] [album title] release date"
- Look for official sources like Wikipedia, AllMusic, Discogs, or official artist websites
- Cross-reference multiple sources when possible

IMPORTANT INSTRUCTIONS:
1. Return ONLY a CSV format with the following columns: album_group_mbid, title, artist_name, first_release_year, sure
2. The first_release_year should be in YYYY format (e.g., "1973") - use the FIRST release year, not reissues
3. Set "sure" to true only if you find reliable sources confirming the date
4. If you cannot find reliable information, return an empty year (leave blank) and set "sure" to false
5. Do not include any explanations, headers, or additional text - ONLY the CSV data
6. Do not include markdown formatting or code blocks
7. CRITICAL: Use proper CSV escaping - put double quotes around any field that contains commas, quotes, or special characters
8. Example format: "album_id","Title, with comma","Artist Name","1973","true"

Here are the albums to research and find release years for:

"""
    
    # Add album information - intentionally hide the actual release date
    for _, row in albums_batch.iterrows():
        prompt += f"Album: {row['album_group_title']} by {row['artist_name']}, MBID: {row['album_group_mbid']}\n"
    
    prompt += "\nPlease search for each album and return the CSV data with accurate release years:"
    
    return prompt


def get_years_from_gemini(albums_batch, batch_num):
    """
    Get year predictions from Gemini for a batch of albums using Google Search
    
    Args:
        albums_batch: DataFrame containing albums to predict years for
        batch_num: Batch number for progress tracking
    
    Returns:
        str: Gemini's response in CSV format
    """
    prompt = create_gemini_prompt(albums_batch)
    
    try:
        print(f"Processing batch {batch_num} with Google Search...")
        
        # Generate content with Google Search tool
        response = client.models.generate_content(
            model=model_id,
            contents=prompt,
            config=GenerateContentConfig(
                tools=[google_search_tool],
                response_modalities=["TEXT"],
            )
        )
        
        # Extract text from response
        response_text = ""
        for part in response.candidates[0].content.parts:
            if hasattr(part, 'text'):
                response_text += part.text
        
        # Print grounding information if available
        try:
            if hasattr(response.candidates[0], 'grounding_metadata') and response.candidates[0].grounding_metadata:
                print(f"Batch {batch_num}: Used Google Search for grounding")
        except:
            pass
        
        return response_text
        
    except Exception as e:
        print(f"Error calling Gemini API with Google Search for batch {batch_num}: {e}")
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
            csv_lines.insert(0, 'album_group_mbid,title,artist_name,first_release_year,sure')
        
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
            # Split but be careful about commas in quotes
            parts = []
            current_part = ""
            in_quotes = False
            
            for char in line:
                if char == '"':
                    in_quotes = not in_quotes
                elif char == ',' and not in_quotes:
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
                    'artist_name': parts[2],
                    'first_release_year': parts[3],
                    'sure': parts[4]
                })
    
    if parsed_data:
        return pd.DataFrame(parsed_data)
    else:
        raise ValueError("No valid data could be parsed")


def calculate_accuracy_metrics(ground_truth_df, predictions_df):
    """
    Calculate various accuracy metrics for date predictions
    
    Args:
        ground_truth_df: DataFrame with actual years
        predictions_df: DataFrame with predicted years
    
    Returns:
        dict: Dictionary containing accuracy metrics
    """
    # Merge predictions with ground truth
    merged = ground_truth_df.merge(
        predictions_df, 
        on='album_group_mbid', 
        how='inner',
        suffixes=('_actual', '_predicted')
    )
    
    print(f"Successfully matched {len(merged)} albums")
    
    # Clean and convert predicted years to numeric first
    merged['predicted_year'] = pd.to_numeric(
        merged['first_release_year'], 
        errors='coerce'
    )
    
    # Convert actual years to numeric as well (in case they're strings)
    merged['actual_year'] = pd.to_numeric(
        merged['actual_year'], 
        errors='coerce'
    )
    
    # Filter out invalid predictions and actual years
    valid_predictions = merged[
        (merged['predicted_year'].notna()) & 
        (merged['actual_year'].notna())
    ].copy()
    
    if len(valid_predictions) == 0:
        print("No valid year predictions found!")
        return {}
    
    # Convert both to integers for accurate comparison
    valid_predictions['actual_year'] = valid_predictions['actual_year'].astype(int)
    valid_predictions['predicted_year'] = valid_predictions['predicted_year'].astype(int)
    
    print(f"Data types after conversion:")
    print(f"actual_year: {valid_predictions['actual_year'].dtype}")
    print(f"predicted_year: {valid_predictions['predicted_year'].dtype}")
    print(f"Sample comparison:")
    sample_data = valid_predictions[['actual_year', 'predicted_year']].head()
    print(sample_data)
    print(f"Sample equality check: {(sample_data['actual_year'] == sample_data['predicted_year']).tolist()}")
    
    # Calculate metrics
    metrics = {}
    
    # Exact year match
    exact_match_mask = (valid_predictions['actual_year'] == valid_predictions['predicted_year'])
    exact_matches = exact_match_mask.sum()
    metrics['exact_accuracy'] = exact_matches / len(valid_predictions)
    print(f"Exact matches found: {exact_matches} out of {len(valid_predictions)}")
    
    # Debug: Show first 20 wrong predictions
    wrong_predictions = valid_predictions[~exact_match_mask].copy()
    if len(wrong_predictions) > 0:
        print(f"\nFirst 20 WRONG predictions:")
        print("-" * 80)
        debug_cols = ['title', 'artist_name', 'actual_year', 'predicted_year']
        for i, (_, row) in enumerate(wrong_predictions.head(20).iterrows()):
            actual = row['actual_year']
            predicted = row['predicted_year']
            title = row.get('title', row.get('title_actual', 'Unknown'))
            artist = row.get('artist_name', row.get('artist_name_actual', 'Unknown'))
            print(f"{i+1:2d}. {title[:30]:<30} by {artist[:20]:<20} | Actual: {actual} | Predicted: {predicted} | Diff: {abs(actual-predicted)}")
        print("-" * 80)
    
    # Debug: Show first 20 correct predictions
    correct_predictions = valid_predictions[exact_match_mask].copy()
    if len(correct_predictions) > 0:
        print(f"\nFirst 20 CORRECT predictions:")
        print("-" * 80)
        for i, (_, row) in enumerate(correct_predictions.head(20).iterrows()):
            actual = row['actual_year']
            predicted = row['predicted_year']
            title = row.get('title', row.get('title_actual', 'Unknown'))
            artist = row.get('artist_name', row.get('artist_name_actual', 'Unknown'))
            print(f"{i+1:2d}. {title[:30]:<30} by {artist[:20]:<20} | Actual: {actual} | Predicted: {predicted}")
        print("-" * 80)
    
    # Within 1 year
    within_1_year = (abs(valid_predictions['actual_year'] - valid_predictions['predicted_year']) <= 1).sum()
    metrics['within_1_year_accuracy'] = within_1_year / len(valid_predictions)
    
    # Within 2 years
    within_2_years = (abs(valid_predictions['actual_year'] - valid_predictions['predicted_year']) <= 2).sum()
    metrics['within_2_years_accuracy'] = within_2_years / len(valid_predictions)
    
    # Within 5 years
    within_5_years = (abs(valid_predictions['actual_year'] - valid_predictions['predicted_year']) <= 5).sum()
    metrics['within_5_years_accuracy'] = within_5_years / len(valid_predictions)
    
    # Same decade
    valid_predictions['actual_decade'] = (valid_predictions['actual_year'] // 10) * 10
    valid_predictions['predicted_decade'] = (valid_predictions['predicted_year'] // 10) * 10
    same_decade = (valid_predictions['actual_decade'] == valid_predictions['predicted_decade']).sum()
    metrics['same_decade_accuracy'] = same_decade / len(valid_predictions)
    
    # Average absolute error
    metrics['mean_absolute_error'] = abs(valid_predictions['actual_year'] - valid_predictions['predicted_year']).mean()
    
    # Standard deviation of errors
    metrics['std_error'] = abs(valid_predictions['actual_year'] - valid_predictions['predicted_year']).std()
    
    # Total albums processed
    metrics['total_albums'] = len(merged)
    metrics['valid_predictions'] = len(valid_predictions)
    metrics['invalid_predictions'] = len(merged) - len(valid_predictions)
    
    # Confidence analysis - comprehensive breakdown
    if 'sure' in valid_predictions.columns:
        print("\n" + "="*40)
        print("ACCURACY BY CONFIDENCE LEVEL")
        print("="*40)
        
        # Analyze confidence distribution
        confidence_counts = valid_predictions['sure'].astype(str).str.lower().value_counts()
        print(f"Confidence distribution:")
        for confidence, count in confidence_counts.items():
            print(f"  {confidence}: {count} predictions ({count/len(valid_predictions):.1%})")
        
        # Calculate metrics for confident predictions (sure=true)
        confident_predictions = valid_predictions[valid_predictions['sure'].astype(str).str.lower() == 'true']
        if len(confident_predictions) > 0:
            confident_exact = (confident_predictions['actual_year'] == confident_predictions['predicted_year']).sum()
            confident_within_1 = (abs(confident_predictions['actual_year'] - confident_predictions['predicted_year']) <= 1).sum()
            confident_within_2 = (abs(confident_predictions['actual_year'] - confident_predictions['predicted_year']) <= 2).sum()
            confident_within_5 = (abs(confident_predictions['actual_year'] - confident_predictions['predicted_year']) <= 5).sum()
            confident_same_decade = ((confident_predictions['actual_year'] // 10) * 10 == (confident_predictions['predicted_year'] // 10) * 10).sum()
            
            metrics['confident_predictions_count'] = len(confident_predictions)
            metrics['confident_exact_accuracy'] = confident_exact / len(confident_predictions)
            metrics['confident_within_1_accuracy'] = confident_within_1 / len(confident_predictions)
            metrics['confident_within_2_accuracy'] = confident_within_2 / len(confident_predictions)
            metrics['confident_within_5_accuracy'] = confident_within_5 / len(confident_predictions)
            metrics['confident_same_decade_accuracy'] = confident_same_decade / len(confident_predictions)
            metrics['confident_mean_absolute_error'] = abs(confident_predictions['actual_year'] - confident_predictions['predicted_year']).mean()
            
            print(f"\nConfident Predictions (sure=true): {len(confident_predictions)} predictions")
            print("-" * 30)
            print(f"  Exact year match: {metrics['confident_exact_accuracy']:.3f} ({confident_exact}/{len(confident_predictions)})")
            print(f"  Within 1 year: {metrics['confident_within_1_accuracy']:.3f} ({confident_within_1}/{len(confident_predictions)})")
            print(f"  Within 2 years: {metrics['confident_within_2_accuracy']:.3f} ({confident_within_2}/{len(confident_predictions)})")
            print(f"  Within 5 years: {metrics['confident_within_5_accuracy']:.3f} ({confident_within_5}/{len(confident_predictions)})")
            print(f"  Same decade: {metrics['confident_same_decade_accuracy']:.3f} ({confident_same_decade}/{len(confident_predictions)})")
            print(f"  Mean absolute error: {metrics['confident_mean_absolute_error']:.2f} years")
        
        # Calculate metrics for unconfident predictions (sure=false)
        unconfident_predictions = valid_predictions[valid_predictions['sure'].astype(str).str.lower() == 'false']
        if len(unconfident_predictions) > 0:
            unconfident_exact = (unconfident_predictions['actual_year'] == unconfident_predictions['predicted_year']).sum()
            unconfident_within_1 = (abs(unconfident_predictions['actual_year'] - unconfident_predictions['predicted_year']) <= 1).sum()
            unconfident_within_2 = (abs(unconfident_predictions['actual_year'] - unconfident_predictions['predicted_year']) <= 2).sum()
            unconfident_within_5 = (abs(unconfident_predictions['actual_year'] - unconfident_predictions['predicted_year']) <= 5).sum()
            unconfident_same_decade = ((unconfident_predictions['actual_year'] // 10) * 10 == (unconfident_predictions['predicted_year'] // 10) * 10).sum()
            
            metrics['unconfident_predictions_count'] = len(unconfident_predictions)
            metrics['unconfident_exact_accuracy'] = unconfident_exact / len(unconfident_predictions)
            metrics['unconfident_within_1_accuracy'] = unconfident_within_1 / len(unconfident_predictions)
            metrics['unconfident_within_2_accuracy'] = unconfident_within_2 / len(unconfident_predictions)
            metrics['unconfident_within_5_accuracy'] = unconfident_within_5 / len(unconfident_predictions)
            metrics['unconfident_same_decade_accuracy'] = unconfident_same_decade / len(unconfident_predictions)
            metrics['unconfident_mean_absolute_error'] = abs(unconfident_predictions['actual_year'] - unconfident_predictions['predicted_year']).mean()
            
            print(f"\nUnconfident Predictions (sure=false): {len(unconfident_predictions)} predictions")
            print("-" * 30)
            print(f"  Exact year match: {metrics['unconfident_exact_accuracy']:.3f} ({unconfident_exact}/{len(unconfident_predictions)})")
            print(f"  Within 1 year: {metrics['unconfident_within_1_accuracy']:.3f} ({unconfident_within_1}/{len(unconfident_predictions)})")
            print(f"  Within 2 years: {metrics['unconfident_within_2_accuracy']:.3f} ({unconfident_within_2}/{len(unconfident_predictions)})")
            print(f"  Within 5 years: {metrics['unconfident_within_5_accuracy']:.3f} ({unconfident_within_5}/{len(unconfident_predictions)})")
            print(f"  Same decade: {metrics['unconfident_same_decade_accuracy']:.3f} ({unconfident_same_decade}/{len(unconfident_predictions)})")
            print(f"  Mean absolute error: {metrics['unconfident_mean_absolute_error']:.2f} years")
    
    return metrics, valid_predictions


def generate_accuracy_report(metrics, valid_predictions):
    """
    Generate a detailed accuracy report
    
    Args:
        metrics: Dictionary containing accuracy metrics
        valid_predictions: DataFrame with valid predictions for analysis
    """
    report_file = './data/gemini/date_accuracy_report.txt'
    
    with open(report_file, 'w') as f:
        f.write("GEMINI DATE PREDICTION ACCURACY REPORT\n")
        f.write("=" * 50 + "\n\n")
        
        f.write(f"Total albums tested: {metrics.get('total_albums', 0)}\n")
        f.write(f"Valid predictions: {metrics.get('valid_predictions', 0)}\n")
        f.write(f"Invalid predictions: {metrics.get('invalid_predictions', 0)}\n\n")
        
        f.write("ACCURACY METRICS:\n")
        f.write("-" * 20 + "\n")
        f.write(f"Exact year match: {metrics.get('exact_accuracy', 0):.3f} ({metrics.get('exact_accuracy', 0)*100:.1f}%)\n")
        f.write(f"Within 1 year: {metrics.get('within_1_year_accuracy', 0):.3f} ({metrics.get('within_1_year_accuracy', 0)*100:.1f}%)\n")
        f.write(f"Within 2 years: {metrics.get('within_2_years_accuracy', 0):.3f} ({metrics.get('within_2_years_accuracy', 0)*100:.1f}%)\n")
        f.write(f"Within 5 years: {metrics.get('within_5_years_accuracy', 0):.3f} ({metrics.get('within_5_years_accuracy', 0)*100:.1f}%)\n")
        f.write(f"Same decade: {metrics.get('same_decade_accuracy', 0):.3f} ({metrics.get('same_decade_accuracy', 0)*100:.1f}%)\n\n")
        
        f.write("ERROR STATISTICS:\n")
        f.write("-" * 20 + "\n")
        f.write(f"Mean absolute error: {metrics.get('mean_absolute_error', 0):.2f} years\n")
        f.write(f"Standard deviation: {metrics.get('std_error', 0):.2f} years\n\n")
        
        if 'confident_exact_accuracy' in metrics or 'unconfident_exact_accuracy' in metrics:
            f.write("CONFIDENCE ANALYSIS:\n")
            f.write("-" * 20 + "\n")
            
            if 'confident_exact_accuracy' in metrics:
                f.write(f"Confident predictions (sure=true): {metrics.get('confident_predictions_count', 0)}\n")
                f.write(f"  Exact year match: {metrics.get('confident_exact_accuracy', 0):.3f} ({metrics.get('confident_exact_accuracy', 0)*100:.1f}%)\n")
                f.write(f"  Within 1 year: {metrics.get('confident_within_1_accuracy', 0):.3f} ({metrics.get('confident_within_1_accuracy', 0)*100:.1f}%)\n")
                f.write(f"  Within 5 years: {metrics.get('confident_within_5_accuracy', 0):.3f} ({metrics.get('confident_within_5_accuracy', 0)*100:.1f}%)\n")
                f.write(f"  Same decade: {metrics.get('confident_same_decade_accuracy', 0):.3f} ({metrics.get('confident_same_decade_accuracy', 0)*100:.1f}%)\n")
                f.write(f"  Mean absolute error: {metrics.get('confident_mean_absolute_error', 0):.2f} years\n\n")
            
            if 'unconfident_exact_accuracy' in metrics:
                f.write(f"Unconfident predictions (sure=false): {metrics.get('unconfident_predictions_count', 0)}\n")
                f.write(f"  Exact year match: {metrics.get('unconfident_exact_accuracy', 0):.3f} ({metrics.get('unconfident_exact_accuracy', 0)*100:.1f}%)\n")
                f.write(f"  Within 1 year: {metrics.get('unconfident_within_1_accuracy', 0):.3f} ({metrics.get('unconfident_within_1_accuracy', 0)*100:.1f}%)\n")
                f.write(f"  Within 5 years: {metrics.get('unconfident_within_5_accuracy', 0):.3f} ({metrics.get('unconfident_within_5_accuracy', 0)*100:.1f}%)\n")
                f.write(f"  Same decade: {metrics.get('unconfident_same_decade_accuracy', 0):.3f} ({metrics.get('unconfident_same_decade_accuracy', 0)*100:.1f}%)\n")
                f.write(f"  Mean absolute error: {metrics.get('unconfident_mean_absolute_error', 0):.2f} years\n\n")
        
        # Error distribution
        if len(valid_predictions) > 0:
            errors = abs(valid_predictions['actual_year'] - valid_predictions['predicted_year'])
            f.write("ERROR DISTRIBUTION:\n")
            f.write("-" * 20 + "\n")
            f.write(f"0 years off: {(errors == 0).sum()}\n")
            f.write(f"1 year off: {(errors == 1).sum()}\n")
            f.write(f"2 years off: {(errors == 2).sum()}\n")
            f.write(f"3-5 years off: {((errors >= 3) & (errors <= 5)).sum()}\n")
            f.write(f"6-10 years off: {((errors >= 6) & (errors <= 10)).sum()}\n")
            f.write(f"More than 10 years off: {(errors > 10).sum()}\n")
    
    print(f"Detailed accuracy report saved to {report_file}")


def test_gemini_date_accuracy():
    """
    Main function to test Gemini's date prediction accuracy
    """
    # Load the labelled dataset
    dataset_path = './data/merged_dataset_mumu_msdi_final_cleaned.csv'
    test_albums = load_labelled_albums(dataset_path, sample_size=500)
    
    if test_albums is None:
        return
    
    # Prepare the ground truth data
    ground_truth = []
    for _, row in test_albums.iterrows():
        ground_truth.append({
            'album_group_mbid': row['album_group_mbid'],
            'title': row['album_group_title'], 
            'artist_name': row['artist_name'],
            'actual_year': row['year']
        })
    
    # Save ground truth for comparison
    ground_truth_df = pd.DataFrame(ground_truth)
    ground_truth_df.to_csv('./data/gemini/date_test_ground_truth.csv', index=False)
    print("Ground truth saved to ./data/gemini/date_test_ground_truth.csv")
    
    # Process albums in batches (Gemini has token limits)
    batch_size = 75
    all_predictions = []
    
    for i in range(0, len(test_albums), batch_size):
        batch = test_albums.iloc[i:i+batch_size]
        batch_num = i // batch_size + 1
        
        # Get predictions from Gemini
        gemini_response = get_years_from_gemini(batch, batch_num)
        
        if gemini_response:
            # Parse the response
            predictions_df = parse_gemini_response(gemini_response)
            
            if predictions_df is not None:
                all_predictions.append(predictions_df)
                print(f"Successfully processed batch {batch_num}")
                
                # Add a longer delay for Google Search requests to be respectful
                import time
                time.sleep(2)
            else:
                print(f"Failed to parse response for batch {batch_num}")
                print(f"Raw response: {gemini_response[:500]}...")  # Show first 500 chars for debugging
        else:
            print(f"Failed to get response for batch {batch_num}")
    
    # Combine all predictions
    if all_predictions:
        final_predictions = pd.concat(all_predictions, ignore_index=True)
        
        # Save predictions
        final_predictions.to_csv('./data/gemini/gemini_date_predictions_test.csv', index=False)
        print(f"Gemini predictions saved to ./data/gemini/gemini_date_predictions_test.csv")
        print(f"Total albums processed: {len(final_predictions)}")
        
        # Calculate accuracy metrics
        metrics, valid_predictions = calculate_accuracy_metrics(ground_truth_df, final_predictions)
        
        if metrics:
            # Display results
            print("\n" + "="*50)
            print("ACCURACY RESULTS:")
            print("="*50)
            print(f"Exact year match: {metrics['exact_accuracy']:.3f} ({metrics['exact_accuracy']*100:.1f}%)")
            print(f"Within 1 year: {metrics['within_1_year_accuracy']:.3f} ({metrics['within_1_year_accuracy']*100:.1f}%)")
            print(f"Within 2 years: {metrics['within_2_years_accuracy']:.3f} ({metrics['within_2_years_accuracy']*100:.1f}%)")
            print(f"Within 5 years: {metrics['within_5_years_accuracy']:.3f} ({metrics['within_5_years_accuracy']*100:.1f}%)")
            print(f"Same decade: {metrics['same_decade_accuracy']:.3f} ({metrics['same_decade_accuracy']*100:.1f}%)")
            print(f"Mean absolute error: {metrics['mean_absolute_error']:.2f} years")
            
            # Generate detailed report
            generate_accuracy_report(metrics, valid_predictions)
            
            # Save comparison data
            valid_predictions.to_csv('./data/gemini/date_predictions_comparison.csv', index=False)
            print("Detailed comparison saved to ./data/gemini/date_predictions_comparison.csv")
            
        else:
            print("Could not calculate accuracy metrics")
        
    else:
        print("No successful predictions obtained from Gemini")


if __name__ == "__main__":
    # Create results directory if it doesn't exist
    os.makedirs('./data/gemini', exist_ok=True)
    
    # test_gemini_date_accuracy() 
    ground_truth_df = pd.read_csv('./data/gemini/date_test_ground_truth.csv')
    predictions_df = pd.read_csv('./data/gemini/gemini_date_predictions_test.csv')
    
    print("Loading and calculating accuracy metrics...")
    result = calculate_accuracy_metrics(ground_truth_df, predictions_df)
    
    if len(result) == 2:
        metrics, valid_predictions = result
        
        if metrics:
            # Display results
            print("\n" + "="*50)
            print("OVERALL ACCURACY RESULTS:")
            print("="*50)
            print(f"Exact year match: {metrics['exact_accuracy']:.3f} ({metrics['exact_accuracy']*100:.1f}%)")
            print(f"Within 1 year: {metrics['within_1_year_accuracy']:.3f} ({metrics['within_1_year_accuracy']*100:.1f}%)")
            print(f"Within 2 years: {metrics['within_2_years_accuracy']:.3f} ({metrics['within_2_years_accuracy']*100:.1f}%)")
            print(f"Within 5 years: {metrics['within_5_years_accuracy']:.3f} ({metrics['within_5_years_accuracy']*100:.1f}%)")
            print(f"Same decade: {metrics['same_decade_accuracy']:.3f} ({metrics['same_decade_accuracy']*100:.1f}%)")
            print(f"Mean absolute error: {metrics['mean_absolute_error']:.2f} years")
            
            # Display confidence-based summary
            if 'confident_exact_accuracy' in metrics or 'unconfident_exact_accuracy' in metrics:
                print("\n" + "="*50)
                print("SUMMARY BY CONFIDENCE LEVEL:")
                print("="*50)
                
                if 'confident_exact_accuracy' in metrics:
                    print(f"Confident (sure=True) - {metrics['confident_predictions_count']} predictions:")
                    print(f"  Exact: {metrics['confident_exact_accuracy']:.1%}, Within 1yr: {metrics.get('confident_within_1_accuracy', 0):.1%}, Within 5yr: {metrics.get('confident_within_5_accuracy', 0):.1%}")
                    print(f"  Same decade: {metrics.get('confident_same_decade_accuracy', 0):.1%}, MAE: {metrics.get('confident_mean_absolute_error', 0):.2f} years")
                
                if 'unconfident_exact_accuracy' in metrics:
                    print(f"Unconfident (sure=False) - {metrics['unconfident_predictions_count']} predictions:")
                    print(f"  Exact: {metrics['unconfident_exact_accuracy']:.1%}, Within 1yr: {metrics.get('unconfident_within_1_accuracy', 0):.1%}, Within 5yr: {metrics.get('unconfident_within_5_accuracy', 0):.1%}")
                    print(f"  Same decade: {metrics.get('unconfident_same_decade_accuracy', 0):.1%}, MAE: {metrics.get('unconfident_mean_absolute_error', 0):.2f} years")
            
            # Generate detailed report
            generate_accuracy_report(metrics, valid_predictions)
            
            # Save comparison data
            valid_predictions.to_csv('./data/gemini/date_predictions_comparison.csv', index=False)
            print("Detailed comparison saved to ./data/gemini/date_predictions_comparison.csv")
            
        else:
            print("Could not calculate accuracy metrics")
    else:
        print("Unexpected return format from calculate_accuracy_metrics")