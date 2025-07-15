import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
from collections import defaultdict

def analyze_hiphop_genre_evolution(csv_file_path):
    """
    Analyzes the evolution of hip-hop albums over time, comparing pure hip-hop albums
    versus hip-hop albums with other genre influences.
    
    Args:
        csv_file_path (str): Path to the CSV file containing album data
        
    Returns:
        dict: Dictionary with years as keys and counts of pure vs mixed hip-hop albums
    """
    print(f"Loading data from {csv_file_path}...")
    df = pd.read_csv(csv_file_path)
    
    # Convert string representation of list to actual list
    df['genres'] = df['genres'].apply(lambda x: eval(x) if isinstance(x, str) else x)
    
    # Extract year from release_date
    df['year'] = pd.to_datetime(df['release_date'], errors='raise', format='mixed', yearfirst=True).dt.year
    
    # Filter out rows with NaN years or without genre information
    df = df.dropna(subset=['year', 'genres'])
    print(f"Number of albums: {len(df)}")
    
    # Convert year to integer
    df['year'] = df['year'].astype(int)
    
    # Filter for albums that have hip-hop as a genre
    hiphop_df = df[df['genres'].apply(lambda x: any(genre.lower().strip() == 'hip hop' for genre in x))]
    print(f"Number of hip-hop albums: {len(hiphop_df)}")
    
    # Categorize as pure hip-hop or mixed hip-hop
    hiphop_df['is_pure_hiphop'] = hiphop_df['genres'].apply(
        lambda x: len(x) == 1 and any(genre.lower().strip() == 'hip hop' for genre in x)
    )
    
    # Group by year and count
    result = defaultdict(lambda: {'pure': 0, 'mixed': 0, 'total': 0})
    
    for _, row in hiphop_df.iterrows():
        year = row['year']
        if row['is_pure_hiphop']:
            result[year]['pure'] += 1
        else:
            result[year]['mixed'] += 1
        result[year]['total'] += 1
    
    # Convert defaultdict to regular dict for easier handling
    result_dict = {year: counts for year, counts in result.items()}
    
    return result_dict

def analyze_metal_genre_evolution(csv_file_path):
    """
    Analyzes the evolution of metal albums over time, comparing pure metal albums
    versus metal albums with other genre influences.
    
    Args:
        csv_file_path (str): Path to the CSV file containing album data
        
    Returns:
        dict: Dictionary with years as keys and counts of pure vs mixed metal albums
    """
    print(f"Loading data from {csv_file_path}...")
    df = pd.read_csv(csv_file_path)
    
    # Convert string representation of list to actual list
    df['genres'] = df['genres'].apply(lambda x: eval(x) if isinstance(x, str) else x)
    
    # Extract year from release_date
    df['year'] = pd.to_datetime(df['release_date'], errors='raise', format='mixed', yearfirst=True).dt.year
    
    # Filter out rows with NaN years or without genre information
    df = df.dropna(subset=['year', 'genres'])
    print(f"Number of albums: {len(df)}")
    
    # Convert year to integer
    df['year'] = df['year'].astype(int)
    
    # Filter for albums that have metal as a genre
    metal_df = df[df['genres'].apply(lambda x: any('metal' in genre.lower().strip() for genre in x))]
    print(f"Number of metal albums: {len(metal_df)}")
    
    # Categorize as pure metal or mixed metal
    metal_df['is_pure_metal'] = metal_df['genres'].apply(
        lambda x: len(x) == 1 and any('metal' in genre.lower().strip() for genre in x)
    )
    
    # Group by year and count
    result = defaultdict(lambda: {'pure': 0, 'mixed': 0, 'total': 0})
    
    for _, row in metal_df.iterrows():
        year = row['year']
        if row['is_pure_metal']:
            result[year]['pure'] += 1
        else:
            result[year]['mixed'] += 1
        result[year]['total'] += 1
    
    # Convert defaultdict to regular dict for easier handling
    result_dict = {year: counts for year, counts in result.items()}
    
    return result_dict

def visualize_hiphop_evolution(results, output_dir=None):
    """
    Creates visualizations for hip-hop genre evolution over time.
    
    Args:
        results (dict): Dictionary with years as keys and counts of pure vs mixed hip-hop albums
        output_dir (str, optional): Directory to save the output visualizations
    """
    years = sorted(results.keys())
    pure_counts = [results[year]['pure'] for year in years]
    mixed_counts = [results[year]['mixed'] for year in years]
    total_counts = [results[year]['total'] for year in years]
    
    # Calculate percentages
    pure_percentages = [100 * results[year]['pure'] / results[year]['total'] if results[year]['total'] > 0 else 0 for year in years]
    mixed_percentages = [100 * results[year]['mixed'] / results[year]['total'] if results[year]['total'] > 0 else 0 for year in years]
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
    
    # Plot 1: Absolute counts
    ax1.bar(years, pure_counts, label='Pure Hip Hop', color='blue', alpha=0.7)
    ax1.bar(years, mixed_counts, bottom=pure_counts, label='Hip Hop + Other Genres', color='orange', alpha=0.7)
    ax1.set_ylabel('Number of Albums')
    ax1.set_title('Hip Hop Albums Evolution: Absolute Counts')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Add total count as a line
    ax1_twin = ax1.twinx()
    ax1_twin.plot(years, total_counts, color='red', linestyle='-', marker='o', label='Total Hip Hop Albums')
    ax1_twin.set_ylabel('Total Albums', color='red')
    ax1_twin.tick_params(axis='y', labelcolor='red')
    
    # Plot 2: Percentages
    ax2.bar(years, pure_percentages, label='Pure Hip Hop (%)', color='blue', alpha=0.7)
    ax2.bar(years, mixed_percentages, bottom=pure_percentages, label='Hip Hop + Other Genres (%)', color='orange', alpha=0.7)
    ax2.set_xlabel('Year')
    ax2.set_ylabel('Percentage (%)')
    ax2.set_title('Hip Hop Albums Evolution: Percentage Distribution')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Set x-axis ticks to show years with appropriate intervals
    interval = max(1, len(years) // 20)  # Show at most 20 year labels
    ax2.set_xticks(years[::interval])
    ax2.set_xticklabels(years[::interval], rotation=45)
    
    plt.tight_layout()
    
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        plt.savefig(os.path.join(output_dir, 'hiphop_genre_evolution.png'), dpi=300, bbox_inches='tight')
        print(f"Visualization saved to {os.path.join(output_dir, 'hiphop_genre_evolution.png')}")
    
    plt.show()

def visualize_metal_evolution(results, output_dir=None):
    """
    Creates visualizations for metal genre evolution over time.
    
    Args:
        results (dict): Dictionary with years as keys and counts of pure vs mixed metal albums
        output_dir (str, optional): Directory to save the output visualizations
    """
    years = sorted(results.keys())
    pure_counts = [results[year]['pure'] for year in years]
    mixed_counts = [results[year]['mixed'] for year in years]
    total_counts = [results[year]['total'] for year in years]
    
    # Calculate percentages
    pure_percentages = [100 * results[year]['pure'] / results[year]['total'] if results[year]['total'] > 0 else 0 for year in years]
    mixed_percentages = [100 * results[year]['mixed'] / results[year]['total'] if results[year]['total'] > 0 else 0 for year in years]
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
    
    # Plot 1: Absolute counts
    ax1.bar(years, pure_counts, label='Pure Metal', color='darkred', alpha=0.7)
    ax1.bar(years, mixed_counts, bottom=pure_counts, label='Metal + Other Genres', color='goldenrod', alpha=0.7)
    ax1.set_ylabel('Number of Albums')
    ax1.set_title('Metal Albums Evolution: Absolute Counts')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Add total count as a line
    ax1_twin = ax1.twinx()
    ax1_twin.plot(years, total_counts, color='black', linestyle='-', marker='o', label='Total Metal Albums')
    ax1_twin.set_ylabel('Total Albums', color='black')
    ax1_twin.tick_params(axis='y', labelcolor='black')
    
    # Plot 2: Percentages
    ax2.bar(years, pure_percentages, label='Pure Metal (%)', color='darkred', alpha=0.7)
    ax2.bar(years, mixed_percentages, bottom=pure_percentages, label='Metal + Other Genres (%)', color='goldenrod', alpha=0.7)
    ax2.set_xlabel('Year')
    ax2.set_ylabel('Percentage (%)')
    ax2.set_title('Metal Albums Evolution: Percentage Distribution')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Set x-axis ticks to show years with appropriate intervals
    interval = max(1, len(years) // 20)  # Show at most 20 year labels
    ax2.set_xticks(years[::interval])
    ax2.set_xticklabels(years[::interval], rotation=45)
    
    plt.tight_layout()
    
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        plt.savefig(os.path.join(output_dir, 'metal_genre_evolution.png'), dpi=300, bbox_inches='tight')
        print(f"Visualization saved to {os.path.join(output_dir, 'metal_genre_evolution.png')}")
    
    plt.show()

def main():
    # Path to the dataset
    csv_file_path = './results/unified_album_dataset_with_complexity.csv'
    
    # Analyze metal genre evolution
    results = analyze_metal_genre_evolution(csv_file_path)
    
    # Create output directory if it doesn't exist
    output_dir = './result/genre_evolution'
    os.makedirs(output_dir, exist_ok=True)
    
    # Visualize results
    visualize_metal_evolution(results, output_dir)
    
    # Print summary
    print("\nSummary of Metal Albums Evolution:")
    print("-----------------------------------")
    
    years = sorted(results.keys())
    for year in years:
        total = results[year]['total']
        pure = results[year]['pure']
        mixed = results[year]['mixed']
        pure_pct = 100 * pure / total if total > 0 else 0
        mixed_pct = 100 * mixed / total if total > 0 else 0
        
        print(f"Year {year}: Total={total}, Pure Metal={pure} ({pure_pct:.1f}%), Mixed={mixed} ({mixed_pct:.1f}%)")

if __name__ == "__main__":
    main()
