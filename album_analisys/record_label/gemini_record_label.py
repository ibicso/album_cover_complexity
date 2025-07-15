import pandas as pd
import os
import time
import json
from google import genai
from google.genai.types import Tool, GenerateContentConfig, GoogleSearch
import io
from datetime import datetime

GEMINI_API_KEY = "AIzaSyBMC8_LdcQy5fuQGgV8v1mm2rKpOVzC4Rk"

# Configure the API key and client
os.environ['GOOGLE_API_KEY'] = GEMINI_API_KEY
client = genai.Client()
model_id = "gemini-2.0-flash"

# Configure Google Search tool
google_search_tool = Tool(
    google_search=GoogleSearch()
)

# Configuration
BATCH_SIZE = 100
OUTPUT_FOLDER = './data/gemini/record_label_prediction'
PROGRESS_FILE = os.path.join(OUTPUT_FOLDER, 'progress_mumu_msdi.json')
FINAL_OUTPUT_FILE = os.path.join(OUTPUT_FOLDER, 'record_label_classifications_mumu_msdi.csv')

def setup_output_folder():
    """Create output folder and initialize progress tracking"""
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    print(f"Output folder created/verified: {OUTPUT_FOLDER}")

def load_unlabeled_albums(csv_path):
    """
    Load the dataset containing album and label information
    
    Args:
        csv_path: Path to the dataset CSV
    
    Returns:
        DataFrame: Albums to be classified with labels
    """
    try:
        df = pd.read_csv(csv_path)
        
        # Filter out rows without label information
        df = df.dropna(subset=['label_name'])
        df = df[df['label_name'].str.strip() != '']
        
        print(f"Loaded {len(df)} albums with label information")
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

def create_gemini_prompt(labels_batch):
    """
    Create a prompt for Gemini to classify record labels using Google Search
    
    Args:
        labels_batch: List of label names to classify
    
    Returns:
        str: Formatted prompt for Gemini
    """
    prompt = """You are a music industry expert. I need you to classify record labels as either "independent" or "major". Use Google Search to look up each label and find accurate information about their ownership and status.

SEARCH STRATEGY:
- Search for each label using queries like: "[label name] record label ownership" or "[label name] record label major or independent"
- Look for official sources like company websites, industry news, Wikipedia, or reliable music industry databases
- Cross-reference multiple sources when possible

A record label is considered MAJOR if it is:
- One of the "Big 3" major labels: Universal Music Group, Sony Music Entertainment, Warner Music Group
- A subsidiary or imprint of the Big 3

A record label is considered INDEPENDENT if it is:
- Not owned by the Big 3 major labels
- A smaller company, even if successful
- Self-released/artist-owned labels
- Regional or niche labels

IMPORTANT INSTRUCTIONS:
1. Return ONLY a CSV format with the following columns: label_name,classification,confidence_score
2. The classification should be either "independent" or "major"
3. The confidence_score should be between 0.0 and 1.0
4. Do not include any explanations, headers, or additional text - ONLY the CSV data
5. Do not include markdown formatting or code blocks
6. CRITICAL: Use proper CSV escaping - put double quotes around any field that contains commas, quotes, or special characters
7. Example format: "Label Name","independent","0.95"

Here are the labels to research and classify:

"""
    
    for label in labels_batch:
        prompt += f"Label: {label}\n"
    
    prompt += "\nPlease search for each label and return the CSV data with accurate classifications:"
    
    return prompt

def get_classifications_from_gemini(labels_batch, batch_num):
    """
    Get label classifications from Gemini for a batch of labels using Google Search
    
    Args:
        labels_batch: List of labels to classify
        batch_num: Batch number for progress tracking
    
    Returns:
        str: Gemini's response in CSV format
    """
    prompt = create_gemini_prompt(labels_batch)
    
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
        if not csv_lines[0].lower().startswith('label_name'):
            csv_lines.insert(0, 'label_name,classification,confidence_score')
        
        # Create DataFrame from CSV lines with proper CSV parsing
        csv_data = '\n'.join(csv_lines)
        df = pd.read_csv(io.StringIO(csv_data), quotechar='"', skipinitialspace=True)
        
        # Convert confidence_score to float
        df['confidence_score'] = pd.to_numeric(df['confidence_score'], errors='coerce')
        # Fill any NaN values with 0.0
        df['confidence_score'] = df['confidence_score'].fillna(0.0)
        
        return df
        
    except Exception as e:
        print(f"Error parsing Gemini response: {e}")
        return None

def save_batch_results(predictions_df, batch_num):
    """Save batch results to individual files"""
    batch_file = os.path.join(OUTPUT_FOLDER, f'batch_{batch_num:04d}_mumu_msdi.csv')
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
        print(f"Total labels processed: {len(final_df)}")
        
        # Generate summary statistics
        generate_summary_report(final_df)
        
        return final_df
    else:
        print("No data to combine")
        return None

def generate_summary_report(final_df):
    """Generate a summary report of the label classifications"""
    summary_file = os.path.join(OUTPUT_FOLDER, 'summary_report.txt')
    
    with open(summary_file, 'w') as f:
        f.write("GEMINI RECORD LABEL CLASSIFICATION SUMMARY REPORT\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Total labels processed: {len(final_df)}\n")
        f.write(f"Processing completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        # Classification distribution
        total_labels = len(final_df)
        independent_labels = len(final_df[final_df['classification'].str.lower() == 'independent'])
        major_labels = len(final_df[final_df['classification'].str.lower() == 'major'])
        
        f.write("CLASSIFICATION DISTRIBUTION:\n")
        f.write("-" * 30 + "\n")
        f.write(f"Independent Labels: {independent_labels} ({independent_labels/total_labels*100:.1f}%)\n")
        f.write(f"Major Labels: {major_labels} ({major_labels/total_labels*100:.1f}%)\n\n")
        
        # Ensure confidence_score is numeric
        final_df['confidence_score'] = pd.to_numeric(final_df['confidence_score'], errors='coerce')
        final_df['confidence_score'] = final_df['confidence_score'].fillna(0.0)
        
        # Confidence score distribution
        f.write("CONFIDENCE SCORE DISTRIBUTION:\n")
        f.write("-" * 30 + "\n")
        f.write(f"Average confidence: {final_df['confidence_score'].mean():.2f}\n")
        f.write(f"High confidence (>0.8): {len(final_df[final_df['confidence_score'] > 0.8])} labels\n")
        f.write(f"Medium confidence (0.5-0.8): {len(final_df[(final_df['confidence_score'] >= 0.5) & (final_df['confidence_score'] <= 0.8)])} labels\n")
        f.write(f"Low confidence (<0.5): {len(final_df[final_df['confidence_score'] < 0.5])} labels\n")
    
    print(f"Summary report saved to {summary_file}")

def classify_labels():
    """
    Main function to classify record labels in the dataset
    """
    # Setup
    setup_output_folder()
    
    # Load the dataset that needs label classification
    # dataset_path = '../data/billboard_album_final_super_genres_with_images.csv'
    dataset_path = '../data/merged_dataset_mumu_msdi_final_cleaned.csv'
    albums_df = load_unlabeled_albums(dataset_path)
    
    if albums_df is None:
        return
    
    # Get unique labels
    unique_labels = albums_df['label_name'].unique().tolist()
    
    # Load progress
    progress = load_progress()
    start_batch = progress['last_completed_batch'] + 1
    
    if progress['start_time'] is None:
        progress['start_time'] = datetime.now().isoformat()
    
    total_batches = (len(unique_labels) + BATCH_SIZE - 1) // BATCH_SIZE
    print(f"Total batches to process: {total_batches}")
    print(f"Starting from batch: {start_batch + 1}")
    
    # Process labels in batches
    successful_batches = 0
    
    for i in range(start_batch * BATCH_SIZE, len(unique_labels), BATCH_SIZE):
        batch = unique_labels[i:i+BATCH_SIZE]
        batch_num = i // BATCH_SIZE + 1
        
        print(f"\n--- Processing batch {batch_num}/{total_batches} ---")
        
        # Get predictions from Gemini
        gemini_response = get_classifications_from_gemini(batch, batch_num)
        
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
                
                print(f"Successfully processed batch {batch_num} ({len(predictions_df)} labels)")
                
                # Add a small delay to be respectful to the API
                time.sleep(2)
                
            else:
                print(f"Failed to parse response for batch {batch_num}")
        else:
            print(f"Failed to get response for batch {batch_num}")
    
    print(f"\n--- Processing Complete ---")
    print(f"Successfully processed {successful_batches} batches")
    
    # Combine all results
    final_df = combine_all_batches()
    
    if final_df is not None:
        print(f"\nLabel classification complete!")
        print(f"Results saved in: {OUTPUT_FOLDER}")
        print(f"Final file: {FINAL_OUTPUT_FILE}")

if __name__ == "__main__":
    classify_labels()
