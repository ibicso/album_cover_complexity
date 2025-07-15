import pandas as pd
import json
import ast
import os
from typing import List, Dict, Any


def load_genre_mapping(mapping_file_path: str = "11_genre_map.txt") -> Dict[str, str]:
    """
    Load the genre mapping from the text file.
    
    Args:
        mapping_file_path: Path to the genre mapping file
        
    Returns:
        Dictionary mapping original genres to target genres
    """
    try:
        with open(mapping_file_path, 'r', encoding='utf-8') as file:
            content = file.read()
            genre_mapping = json.loads(content)
            return genre_mapping
    except Exception as e:
        print(f"Error loading genre mapping: {e}")
        return {}


def parse_genre_list(genre_string: Any) -> List[str]:
    """
    Parse a genre string that represents a list into an actual list.
    
    Args:
        genre_string: String representation of a list or actual list
        
    Returns:
        List of genre strings
    """
    if pd.isna(genre_string):
        return []
    
    if isinstance(genre_string, list):
        return genre_string
    
    if isinstance(genre_string, str):
        try:
            # Try to parse as a Python literal (list)
            parsed = ast.literal_eval(genre_string)
            if isinstance(parsed, list):
                return [str(item) for item in parsed]
            else:
                return [str(parsed)]
        except (ValueError, SyntaxError):
            # If parsing fails, treat as a single genre
            return [genre_string.strip()]
    
    return [str(genre_string)]


def map_genres(genres: List[str], genre_mapping: Dict[str, str]) -> List[str]:
    """
    Map a list of genres to their corresponding target genres.
    
    Args:
        genres: List of original genre names
        genre_mapping: Mapping dictionary
        
    Returns:
        List of mapped genre names (duplicates removed)
    """
    mapped_genres = []
    
    for genre in genres:
        if genre in genre_mapping:
            mapped_genre = genre_mapping[genre]
            if mapped_genre not in mapped_genres:
                mapped_genres.append(mapped_genre)
        else:
            # If genre is not in mapping, keep original
            if genre not in mapped_genres:
                mapped_genres.append(genre)
    
    return mapped_genres


def clean_genres_csv(input_csv_path: str, output_csv_path: str = None, 
                    mapping_file_path: str = None) -> str:
    """
    Process a CSV file and map genres based on the genre mapping file.
    
    Args:
        input_csv_path: Path to the input CSV file
        output_csv_path: Path for the output CSV file (optional)
        mapping_file_path: Path to the genre mapping file (optional)
        
    Returns:
        Path to the output CSV file
    """
    # Set default paths
    if output_csv_path is None:
        base_name = os.path.splitext(input_csv_path)[0]
        output_csv_path = f"{base_name}_cleaned_genres.csv"
    
    if mapping_file_path is None:
        # Assume the mapping file is in the same directory as this script
        script_dir = os.path.dirname(os.path.abspath(__file__))
        mapping_file_path = os.path.join(script_dir, "11_genre_map.txt")
    
    print(f"Loading genre mapping from: {mapping_file_path}")
    genre_mapping = load_genre_mapping(mapping_file_path)
    
    if not genre_mapping:
        print("Warning: No genre mapping loaded. Proceeding without mapping.")
    else:
        print(f"Loaded {len(genre_mapping)} genre mappings")
    
    print(f"Reading CSV file: {input_csv_path}")
    try:
        df = pd.read_csv(input_csv_path)
    except Exception as e:
        print(f"Error reading CSV file: {e}")
        return ""
    
    if 'genres' not in df.columns:
        print("Error: 'genres' column not found in CSV file")
        print(f"Available columns: {list(df.columns)}")
        return ""
    
    print("Processing genres...")
    
    # Process each row's genres
    def process_row_genres(genre_data):
        original_genres = parse_genre_list(genre_data)
        mapped_genres = map_genres(original_genres, genre_mapping)
        return mapped_genres
    
    df['genres'] = df['genres'].apply(process_row_genres)
    
    print(f"Saving cleaned data to: {output_csv_path}")
    try:
        df.to_csv(output_csv_path, index=False)
        print(f"Successfully saved {len(df)} rows to {output_csv_path}")
        return output_csv_path
    except Exception as e:
        print(f"Error saving CSV file: {e}")
        return ""


# Example usage
if __name__ == "__main__":
    
    # clean_genres_csv("./billboard/billboard_album_final_super_genres.csv", "./billboard/billboard_album_final_super_genres.csv")
    clean_genres_csv("./MSD-I/msdi_album_final_super_genres.csv", "./MSD-I/msdi_album_final_super_genres.csv")
