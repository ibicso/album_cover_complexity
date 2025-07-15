import pandas as pd
import ast


def load_comparison_data():
    """
    Load ground truth and prediction data for comparison
    
    Returns:
        tuple: (ground_truth_df, predictions_df)
    """
    try:
        ground_truth = pd.read_csv('./data/gemini/gemini_test_ground_truth.csv')
        predictions = pd.read_csv('./data/gemini/gemini_genre_predictions.csv')
        
        print(f"Loaded {len(ground_truth)} ground truth records")
        print(f"Loaded {len(predictions)} prediction records")
        
        return ground_truth, predictions
        
    except FileNotFoundError as e:
        print(f"Error: Could not find comparison files.")
        print("Expected files:")
        print("- ./data/gemini/gemini_test_ground_truth.csv")
        print("- ./data/gemini/gemini_genre_predictions.csv")
        return None, None
    except Exception as e:
        print(f"Error loading comparison data: {e}")
        return None, None


def parse_genres(genre_str):
    """
    Parse genre string into a list
    
    Args:
        genre_str: String representation of genres
    
    Returns:
        list: List of genres
    """
    try:
        if pd.isna(genre_str) or genre_str == '[]':
            return []
        
        # Try to parse as Python list
        if isinstance(genre_str, str) and genre_str.startswith('['):
            return ast.literal_eval(genre_str)
        
        # If it's already a list, return as is
        if isinstance(genre_str, list):
            return genre_str
        
        # Otherwise, treat as single genre
        return [str(genre_str)]
        
    except:
        return []


def calculate_exact_match_accuracy(ground_truth, predictions):
    """
    Calculate exact match accuracy (all genres must match exactly)
    
    Args:
        ground_truth: DataFrame with actual genres
        predictions: DataFrame with predicted genres
    
    Returns:
        float: Exact match accuracy score
    """
    # Merge data on MBID
    merged = pd.merge(ground_truth, predictions, on='album_group_mbid', how='inner')
    
    exact_matches = 0
    total_comparisons = len(merged)
    
    for _, row in merged.iterrows():
        actual = set(parse_genres(row['actual_genres']))
        predicted = set(parse_genres(row['genres']))
        
        if actual == predicted:
            exact_matches += 1
    
    accuracy = exact_matches / total_comparisons if total_comparisons > 0 else 0
    
    print(f"Exact Match Accuracy: {accuracy:.3f} ({exact_matches}/{total_comparisons})")
    return accuracy


def calculate_containment_accuracy(ground_truth, predictions):
    """
    Calculate containment accuracy (predicted genres are contained in actual genres)
    
    Args:
        ground_truth: DataFrame with actual genres
        predictions: DataFrame with predicted genres
    
    Returns:
        float: Containment accuracy score
    """
    # Merge data on MBID
    merged = pd.merge(ground_truth, predictions, on='album_group_mbid', how='inner')
    
    containment_matches = 0
    total_comparisons = len(merged)
    
    for _, row in merged.iterrows():
        actual = set(parse_genres(row['actual_genres']))
        predicted = set(parse_genres(row['genres']))
        
        # Check if predicted genres are a subset of actual genres
        if predicted.issubset(actual):
            containment_matches += 1
    
    accuracy = containment_matches / total_comparisons if total_comparisons > 0 else 0
    
    print(f"Containment Accuracy: {accuracy:.3f} ({containment_matches}/{total_comparisons})")
    return accuracy


def analyze_accuracy_by_confidence(ground_truth, predictions):
    """
    Analyze accuracy metrics broken down by confidence level (sure column)
    
    Args:
        ground_truth: DataFrame with actual genres
        predictions: DataFrame with predicted genres
    """
    # Merge data on MBID
    merged = pd.merge(ground_truth, predictions, on='album_group_mbid', how='inner')
    
    print("\n" + "="*40)
    print("ACCURACY BY CONFIDENCE LEVEL")
    print("="*40)
    
    # Check if 'sure' column exists
    if 'sure' not in merged.columns:
        print("No 'sure' column found in predictions data")
        return
    
    # Analyze confidence distribution
    confidence_counts = merged['sure'].value_counts()
    print(f"Confidence distribution:")
    for confidence, count in confidence_counts.items():
        print(f"  {confidence}: {count} predictions ({count/len(merged):.1%})")
    
    # Calculate accuracy for each confidence level
    for confidence_level in [True, False]:
        subset = merged[merged['sure'] == confidence_level]
        if len(subset) == 0:
            continue
            
        print(f"\nConfidence Level: {confidence_level} ({len(subset)} predictions)")
        print("-" * 30)
        
        # Exact match accuracy for this confidence level
        exact_matches = 0
        containment_matches = 0
        
        for _, row in subset.iterrows():
            actual = set(parse_genres(row['actual_genres']))
            predicted = set(parse_genres(row['genres']))
            
            if actual == predicted:
                exact_matches += 1
            
            if predicted.issubset(actual):
                containment_matches += 1
        
        exact_acc = exact_matches / len(subset)
        containment_acc = containment_matches / len(subset)
        
        print(f"  Exact Match Accuracy: {exact_acc:.3f} ({exact_matches}/{len(subset)})")
        print(f"  Containment Accuracy: {containment_acc:.3f} ({containment_matches}/{len(subset)})")


def main():
    """
    Main function to analyze Gemini's genre classification accuracy
    """
    print("GEMINI GENRE CLASSIFICATION ACCURACY ANALYSIS")
    print("=" * 50)
    
    # Load data
    ground_truth, predictions = load_comparison_data()
    
    if ground_truth is None or predictions is None:
        return
    
    print(f"\nAnalyzing {len(ground_truth)} albums...")
    
    # Calculate accuracy metrics
    print("\n" + "="*30)
    print("OVERALL ACCURACY METRICS")
    print("="*30)
    
    exact_accuracy = calculate_exact_match_accuracy(ground_truth, predictions)
    containment_accuracy = calculate_containment_accuracy(ground_truth, predictions)
    
    # Analyze accuracy by confidence level
    analyze_accuracy_by_confidence(ground_truth, predictions)
    
    # Summary
    print("\n" + "="*30)
    print("SUMMARY")
    print("="*30)
    print(f"Overall Exact Match Accuracy: {exact_accuracy:.1%}")
    print(f"Overall Containment Accuracy: {containment_accuracy:.1%}")


if __name__ == "__main__":
    main() 