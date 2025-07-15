import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import ast
from collections import Counter
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

def plot_albums_by_genre(csv_path, figsize=(12, 8)):
    """
    Plot the number of albums by genre from a CSV file.
    
    Parameters:
    csv_path (str): Path to the CSV file
    figsize (tuple): Figure size for the plot
    
    Returns:
    None: Displays the plot
    """
    # Read the CSV file
    df = pd.read_csv(csv_path)
    
    # Extract genres from the 'genres' column
    all_genres = []
    
    for genres_str in df['genres'].dropna():
        try:
            # Convert string representation of list to actual list
            genres_list = ast.literal_eval(genres_str)
            if isinstance(genres_list, list):
                all_genres.extend(genres_list)
            else:
                all_genres.append(genres_list)
        except (ValueError, SyntaxError):
            # If it's not a valid list format, treat as single genre
            all_genres.append(genres_str)
    
    # Count genres
    genre_counts = Counter(all_genres)
    
    # Create the plot
    plt.figure(figsize=figsize)
    bars = plt.bar(range(len(genre_counts)), list(genre_counts.values()), 
                   color='skyblue', edgecolor='navy', alpha=0.7)
    
    plt.xlabel('Genre', fontsize=12, fontweight='bold')
    plt.ylabel('Number of Albums', fontsize=12, fontweight='bold')
    plt.title('Genres by Number of Albums', fontsize=14, fontweight='bold')
    plt.xticks(range(len(genre_counts)), list(genre_counts.keys()), 
               rotation=45, ha='right')
    
    # Add value labels on bars
    for i, bar in enumerate(bars):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                 f'{height}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.show()
    # """
    # Plot the number of albums by genre from a CSV file.
    
    # Parameters:
    # csv_path (str): Path to the CSV file
    # figsize (tuple): Figure size for the plot
    # top_n (int): Number of top genres to display
    
    # Returns:
    # None: Displays the plot
    # """
    # # Read the CSV file
    # df = pd.read_csv(csv_path)
    
    # # Extract genres from the 'genres' column
    # all_genres = []
    
    # for genres_str in df['genres'].dropna():
    #     try:
    #         # Convert string representation of list to actual list
    #         genres_list = ast.literal_eval(genres_str)
    #         if isinstance(genres_list, list):
    #             all_genres.extend(genres_list)
    #         else:
    #             all_genres.append(genres_list)
    #     except (ValueError, SyntaxError):
    #         # If it's not a valid list format, treat as single genre
    #         all_genres.append(genres_str)
    
    # # Count genres
    # genre_counts = Counter(all_genres)
    
    # # Get top N genres
    # top_genres = dict(genre_counts.most_common(top_n))
    
    # # Create the plot
    # plt.figure(figsize=figsize)
    # bars = plt.bar(range(len(top_genres)), list(top_genres.values()), 
    #                color='skyblue', edgecolor='navy', alpha=0.7)
    
    # plt.xlabel('Genre', fontsize=12, fontweight='bold')
    # plt.ylabel('Number of Albums', fontsize=12, fontweight='bold')
    # plt.title(f'Top {top_n} Genres by Number of Albums', fontsize=14, fontweight='bold')
    # plt.xticks(range(len(top_genres)), list(top_genres.keys()), 
    #            rotation=45, ha='right')
    
    # # Add value labels on bars
    # for i, bar in enumerate(bars):
    #     height = bar.get_height()
    #     plt.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
    #             f'{int(height)}', ha='center', va='bottom', fontweight='bold')
    
    # plt.tight_layout()
    # plt.grid(axis='y', alpha=0.3)
    # plt.show()
    
    print(f"Total unique genres: {len(genre_counts)}")
    print(f"Total albums with genre data: {len(all_genres)}")


def plot_albums_by_year(csv_path, figsize=(14, 8), year_range=None):
    """
    Plot the number of albums by year from a CSV file.
    
    Parameters:
    csv_path (str): Path to the CSV file
    figsize (tuple): Figure size for the plot
    year_range (tuple): Optional tuple (start_year, end_year) to filter years
    
    Returns:
    None: Displays the plot
    """
    # Read the CSV file
    df = pd.read_csv(csv_path)
    
    # Extract year from 'first_release_date_group' column
    years = []
    for date_str in df['first_release_date_group'].dropna():
        try:
            # Extract year from date string (format: YYYY-MM-DD or just YYYY)
            year = int(str(date_str).split('-')[0])
            years.append(year)
        except (ValueError, IndexError):
            continue
    
    # Filter by year range if specified
    if year_range:
        start_year, end_year = year_range
        years = [year for year in years if start_year <= year <= end_year]
    
    # Count albums by year
    year_counts = Counter(years)
    
    # Sort by year
    sorted_years = sorted(year_counts.items())
    years_list = [year for year, count in sorted_years]
    counts_list = [count for year, count in sorted_years]
    
    # Create the plot
    plt.figure(figsize=figsize)
    bars = plt.bar(years_list, counts_list, color='lightcoral', 
                   edgecolor='darkred', alpha=0.7, width=0.8)
    
    plt.xlabel('Year', fontsize=12, fontweight='bold')
    plt.ylabel('Number of Albums', fontsize=12, fontweight='bold')
    plt.title('Number of Albums by Release Year', fontsize=14, fontweight='bold')
    
    # Rotate x-axis labels for better readability
    plt.xticks(rotation=45)
    
    
    plt.tight_layout()
    plt.grid(axis='y', alpha=0.3)
    plt.show()
    
    print(f"Year range: {min(years)} - {max(years)}")
    print(f"Total albums with year data: {len(years)}")
    print(f"Peak year: {max(year_counts, key=year_counts.get)} ({year_counts[max(year_counts, key=year_counts.get)]} albums)")


def plot_albums_by_genre_and_year(csv_path, figsize=(16, 10), top_genres=5, year_range=None):
    """
    Plot the number of albums by genre over time (combined visualization).
    
    Parameters:
    csv_path (str): Path to the CSV file
    figsize (tuple): Figure size for the plot
    top_genres (int): Number of top genres to include in the time series
    year_range (tuple): Optional tuple (start_year, end_year) to filter years
    
    Returns:
    None: Displays the plot
    """
    # Read the CSV file
    df = pd.read_csv(csv_path)
    
    # Create genre-year combinations
    genre_year_data = []
    
    for idx, row in df.iterrows():
        try:
            # Extract year
            year = int(str(row['first_release_date_group']).split('-')[0])
            
            # Filter by year range if specified
            if year_range and not (year_range[0] <= year <= year_range[1]):
                continue
                
            # Extract genres
            genres_str = row['genres']
            if pd.isna(genres_str):
                continue
                
            genres_list = ast.literal_eval(genres_str)
            if isinstance(genres_list, list):
                for genre in genres_list:
                    genre_year_data.append({'year': year, 'genre': genre})
            else:
                genre_year_data.append({'year': year, 'genre': genres_list})
                
        except (ValueError, SyntaxError, IndexError):
            continue
    
    # Convert to DataFrame
    genre_year_df = pd.DataFrame(genre_year_data)
    
    # Get top genres
    top_genre_list = genre_year_df['genre'].value_counts().head(top_genres).index.tolist()
    
    # Filter data for top genres
    filtered_df = genre_year_df[genre_year_df['genre'].isin(top_genre_list)]
    
    # Create pivot table
    pivot_df = filtered_df.groupby(['year', 'genre']).size().unstack(fill_value=0)
    
    # Create the plot
    plt.figure(figsize=figsize)
    
    # Plot lines for each genre
    colors = plt.cm.Set3(range(len(top_genre_list)))
    for i, genre in enumerate(top_genre_list):
        if genre in pivot_df.columns:
            plt.plot(pivot_df.index, pivot_df[genre], marker='o', linewidth=2, 
                    label=genre, color=colors[i], markersize=4)
    
    plt.xlabel('Year', fontsize=12, fontweight='bold')
    plt.ylabel('Number of Albums', fontsize=12, fontweight='bold')
    plt.title(f'Album Releases Over Time by Genre (Top {top_genres} Genres)', 
              fontsize=14, fontweight='bold')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    
    print(f"Top {top_genres} genres analyzed over time:")
    for i, genre in enumerate(top_genre_list):
        total = filtered_df[filtered_df['genre'] == genre].shape[0]
        print(f"{i+1}. {genre}: {total} albums")


# Example usage function
def analyze_music_dataset(csv_path):
    """
    Perform complete analysis of the music dataset with all visualizations.
    
    Parameters:
    csv_path (str): Path to the CSV file
    
    Returns:
    None: Displays all plots
    """
    print("=" * 60)
    print("MUSIC DATASET ANALYSIS")
    print("=" * 60)
    
    print("\n1. Albums by Genre Analysis:")
    print("-" * 40)
    plot_albums_by_genre(csv_path)
    
    print("\n2. Albums by Year Analysis:")
    print("-" * 40)
    plot_albums_by_year(csv_path)
    
    print("\n3. Genre Trends Over Time Analysis:")
    print("-" * 40)
    plot_albums_by_genre_and_year(csv_path)
    
    print("\n" + "=" * 60)
    print("ANALYSIS COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    # Example usage
    # billboard_file = "./billboard/billboard_album_final_super_genres.csv"
    # analyze_music_dataset(billboard_file)

    msdi_file = "./MSD-I/msdi_album_final_super_genres.csv"
    analyze_music_dataset(msdi_file)

    mumu_file = "./MuMu/mb_albums/musicbrainz_album_final_super_genres.csv"
    analyze_music_dataset(mumu_file)

    
