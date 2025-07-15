import pandas as pd
import ast
import os
import matplotlib.pyplot as plt

FONT_SIZE = 18


def create_output_directory():
    """Create the output directory for plots"""
    output_dir = 'result/dataset_info'
    os.makedirs(output_dir, exist_ok=True)
    return output_dir


def load_and_clean_data_no_split(csv_file):

    df_results = pd.read_csv(csv_file)
    
    # Remove rows with missing genre values
    df_clean = df_results.dropna(subset=['genres'])
    
    # Extract year from release_date
    df_clean['year'] = pd.to_datetime(df_clean['release_date'], errors='raise', format='mixed', yearfirst=True).dt.year
    
    # Filter out rows with invalid years
    df_clean = df_clean.dropna(subset=['year'])
    df_clean['year'] = df_clean['year'].astype(int)
    
    # Filter reasonable year range (e.g., 1950-2025)
    print(len(df_clean))
    df_clean = df_clean[(df_clean['year'] >= 1950) & (df_clean['year'] <= 2025)]
    print(len(df_clean))
    return df_clean


def load_and_clean_data(wavelet_csv):

    df_results = pd.read_csv(wavelet_csv)
    
    # Remove rows with missing genre values
    df_clean = df_results.dropna(subset=['genres'])
    
    # Extract year from release_date
    df_clean['year'] = pd.to_datetime(df_clean['release_date'], errors='raise', format='mixed', yearfirst=True).dt.year
    
    # Filter out rows with invalid years
    df_clean = df_clean.dropna(subset=['year'])
    df_clean['year'] = df_clean['year'].astype(int)
    
    # Filter reasonable year range (e.g., 1950-2025)
    df_clean = df_clean[(df_clean['year'] >= 1950) & (df_clean['year'] <= 2025)]

    # Split multi-label genres and create separate rows for each genre
    print("Splitting multi-label genres...")
    expanded_rows = []
    
    for _, row in df_clean.iterrows():
        try:
            # Parse string representation of list
            genres_list = ast.literal_eval(row['genres']) if isinstance(row['genres'], str) else row['genres']
            if not isinstance(genres_list, list):
                genres_list = [genres_list]  # Ensure it's a list
        except (ValueError, SyntaxError):
            genres_list = [genre.strip() for genre in str(row['genres']).split(',')]  # Fallback to comma-separated parsing
        
        for genre in genres_list:
            if genre and genre != 'nan':  # Skip empty or invalid genres
                new_row = row.copy()
                new_row['genre'] = genre
                expanded_rows.append(new_row)
    
    # Create new dataframe with expanded genres
    df_expanded = pd.DataFrame(expanded_rows)
    
    print(f"After splitting genres: {len(df_expanded)} genre-album combinations")
    print(f"Unique individual genres: {df_expanded['genre'].nunique()}")
    
    return df_expanded


def plot_simple_album_count_by_year(df, output_dir, title):
    """Plot simple bar chart showing total album counts by year"""
    print("Creating simple bar plot for album counts by year...")
    
    # Count albums by year
    year_counts = df.groupby('year').size().reset_index(name='album_count')
    plt.rcParams['font.size'] = FONT_SIZE
    
    # Create simple bar plot
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Create bar chart
    ax.bar(year_counts['year'], year_counts['album_count'], 
           color='steelblue', alpha=0.7, edgecolor='black', linewidth=0.5)
    
    ax.set_xlabel('Year', fontweight='bold')
    ax.set_ylabel('Number of Albums', fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    
    # Format x-axis to show appropriate year ticks
    year_range = 2025 - 1950
    if year_range > 50:
        tick_spacing = 5  # Every 5 years if range is large
    else:
        tick_spacing = 2  # Every 2 years if range is smaller
    
    year_ticks = range(1950, 2025 + 1, tick_spacing)
    ax.set_xticks(year_ticks)
    ax.set_xticklabels([str(year) for year in year_ticks], rotation=45, ha='right')
    ax.set_ylim(0, 2000)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/{title}_simple_album_count_by_year.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("Saved simple album count by year plot")
    return year_counts

if __name__ == "__main__":
    output_dir = create_output_directory()
    mumu_msdi_df = load_and_clean_data_no_split('./data/merged_dataset_mumu_msdi_final_cleaned_with_labels.csv')
    billboard_df = load_and_clean_data_no_split('./data/billboard_album_final_with_labels.csv')
    
    #dataset
    print("\n" + "="*50)
    plot_simple_album_count_by_year(mumu_msdi_df, output_dir, 'mumu_msdi')
    # print("\n" + "="*50)
    plot_simple_album_count_by_year(billboard_df, output_dir, 'billboard')
    
