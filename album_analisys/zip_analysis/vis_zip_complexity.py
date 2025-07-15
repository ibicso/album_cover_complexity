import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from datetime import datetime
import warnings
import ast
import os
from upsetplot import UpSet, from_indicators
from mpl_toolkits.axes_grid1 import make_axes_locatable

FONT_SIZE = 18


def load_and_clean_data_no_split(csv_file):
    """Load the wavelet results and merge with main dataset to get year information."""
    print("Loading wavelet analysis results...")
    df_results = pd.read_csv(csv_file)
    
    # Remove rows with missing avg_entropy values
    df_clean = df_results.dropna(subset=['compression_ratio'])
    print(f"Dataset loaded: {len(df_results)} total albums, {len(df_clean)} with valid compression_ratio data")
    
    # Remove rows with missing genre values
    df_clean = df_clean.dropna(subset=['genres'])
    
    # Extract year from release_date
    df_clean['year'] = pd.to_datetime(df_clean['release_date'], errors='raise', format='mixed', yearfirst=True).dt.year
    
    # Filter out rows with invalid years
    df_clean = df_clean.dropna(subset=['year'])
    df_clean['year'] = df_clean['year'].astype(int)
    
    # Filter reasonable year range (e.g., 1950-2025)
    df_clean = df_clean[(df_clean['year'] >= 1950) & (df_clean['year'] <= 2050)]

    df_clean = df_clean[df_clean['compression_ratio'] != 0]
    
    return df_clean

def load_and_clean_data(wavelet_csv):
    """Load the wavelet results and merge with main dataset to get year information."""
    print("Loading wavelet analysis results...")
    df_results = pd.read_csv(wavelet_csv)
    
    # Remove rows with missing avg_entropy values
    df_clean = df_results.dropna(subset=['compression_ratio'])
    print(f"Dataset loaded: {len(df_results)} total albums, {len(df_clean)} with valid compression_ratio data")
    
    # Remove rows with missing genre values
    df_clean = df_clean.dropna(subset=['genres'])
    
    # Extract year from release_date
    df_clean['year'] = pd.to_datetime(df_clean['release_date'], errors='raise', format='mixed', yearfirst=True).dt.year
    
    # Filter out rows with invalid years
    df_clean = df_clean.dropna(subset=['year'])
    df_clean['year'] = df_clean['year'].astype(int)
    
    # Filter reasonable year range (e.g., 1950-2025)
    df_clean = df_clean[(df_clean['year'] >= 1950) & (df_clean['year'] <= 2025)]

    df_clean = df_clean[df_clean['compression_ratio'] != 0]
    
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



def plot_entropy_distribution_box(df, output_dir):
    """Plot: Box plot showing entropy distribution for top genres"""
    print("Creating box plot for entropy distribution...")
    plt.rcParams['font.size'] = FONT_SIZE
    # Get top 12 genres by count for better visualization
    top_genres = df.groupby('genre')['compression_ratio'].mean().nsmallest(12).index.tolist()
    df_top = df[df['genre'].isin(top_genres)]
    
    # Calculate number of albums per genre for color scaling
    genre_counts = df_top['genre'].value_counts()
    
    # Create color map based on album counts
    norm = plt.Normalize(genre_counts.min(), genre_counts.max())
    cmap = plt.cm.YlOrRd  # Yellow to Red colormap - more albums = darker red
    
    # Create box plot
    fig, ax = plt.subplots(figsize=(15, 8))
    
    # Prepare data for box plot
    data_for_boxplot = [df_top[df_top['genre'] == genre]['compression_ratio'].values 
                        for genre in top_genres]
    
    # Create box plot
    box_parts = ax.boxplot(data_for_boxplot, 
                          positions=range(len(top_genres)),
                          patch_artist=True,
                          showmeans=True,
                          meanline=True)
    
    # Customize box plot colors based on number of albums
    for i, genre in enumerate(top_genres):
        color = cmap(norm(genre_counts[genre]))
        box_parts['boxes'][i].set_facecolor(color)
        box_parts['boxes'][i].set_alpha(0.7)
        box_parts['boxes'][i].set_edgecolor('black')
        box_parts['boxes'][i].set_linewidth(1)
    
    # Customize other elements
    for element in ['whiskers', 'fliers', 'medians', 'caps']:
        plt.setp(box_parts[element], color='black', linewidth=1)
    
    # Customize means
    plt.setp(box_parts['means'], color='red', linewidth=2)
    
    ax.set_xticks(range(len(top_genres)))
    ax.set_xticklabels([f"{genre}" for genre in top_genres], 
                       rotation=45, ha='right')
    ax.set_ylabel('Complexity Overall Score (ZIPc)', fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add colorbar to show album count scale
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="2%", pad=0.2)
    cbar = plt.colorbar(sm, cax=cax)
    cbar.set_label('Number of Albums', fontweight='bold')
    
    # Add legend to explain elements
    # legend_elements = [
    #     plt.Line2D([0], [0], color='black', linewidth=2, label='Median'),
    #     plt.Line2D([0], [0], color='red', linewidth=2, label='Mean'),
    # ]
    # ax.legend(handles=legend_elements, bbox_to_anchor=(1.15, 1), loc='upper left')

    genre_means = df_top.groupby('genre')['compression_ratio'].mean()
    for genre, mean in genre_means.items():
        print(f"{genre}: Mean Complexity Score (ZIPc) = {mean:.2f}")
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/compression_ratio_distribution_box.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return df_top


def create_dynamic_periods(df, min_albums=3000):
    """Create dynamic time periods based on minimum album count threshold"""
    # Get year counts
    year_counts = df.groupby('year').size().sort_index()
    
    periods = []
    current_period_years = []
    current_period_count = 0
    
    for year, count in year_counts.items():
        # If current year alone has enough albums, make it its own period
        if count >= min_albums:
            # First, close any ongoing period
            if current_period_years:
                if len(current_period_years) == 1:
                    period_label = str(current_period_years[0])
                else:
                    period_label = f"{current_period_years[0]}-{current_period_years[-1]}"
                
                for y in current_period_years:
                    periods.append({'year': y, 'period': period_label})
                
                current_period_years = []
                current_period_count = 0
            
            # Create period for this single year
            periods.append({'year': year, 'period': str(year)})
        else:
            # Add year to current period
            current_period_years.append(year)
            current_period_count += count
            
            # Check if we've reached the threshold
            if current_period_count >= min_albums:
                if len(current_period_years) == 1:
                    period_label = str(current_period_years[0])
                else:
                    period_label = f"{current_period_years[0]}-{current_period_years[-1]}"
                
                for y in current_period_years:
                    periods.append({'year': y, 'period': period_label})
                
                current_period_years = []
                current_period_count = 0
    
    # Handle any remaining years in the last period
    if current_period_years:
        if len(current_period_years) == 1:
            period_label = str(current_period_years[0])
        else:
            period_label = f"{current_period_years[0]}-{current_period_years[-1]}"
        
        for y in current_period_years:
            periods.append({'year': y, 'period': period_label})
    
    # Convert to DataFrame and merge with original data
    periods_df = pd.DataFrame(periods)
    df_with_periods = df.merge(periods_df, on='year', how='left')
    
    return df_with_periods

def plot_all_genres_entropy_comparison_dynamic(df, df_no_split, output_dir, min_albums=3000):
    """Plot comparing dynamic period entropy trends for all top genres on the same plot"""
    print("Creating comparison plot for all genres' dynamic period entropy trends...")
    plt.rcParams['font.size'] = FONT_SIZE
    # Find top 11 genres by total album count for better visualization
    top_genres = df.groupby('genre').size().nlargest(11).index.tolist()
    print(f"Top genres for comparison: {top_genres}")
    
    # Filter data for top genres
    df_top = df[df['genre'].isin(top_genres)]

    # STEP 1: Create dynamic periods using df_no_split (consistent with other plots)
    df_no_split_with_periods = create_dynamic_periods(df_no_split, min_albums)
    
    # STEP 2: Create a mapping from year to period
    year_to_period = df_no_split_with_periods[['year', 'period']].drop_duplicates().set_index('year')['period'].to_dict()
    
    # STEP 3: Apply the period mapping to the genre-split dataframe
    df_top_with_periods = df_top.copy()
    df_top_with_periods['period'] = df_top_with_periods['year'].map(year_to_period)
    
    # Remove any rows where period mapping failed (shouldn't happen if data is consistent)
    df_top_with_periods = df_top_with_periods.dropna(subset=['period'])
    
    # Continue with the rest of your existing code...
    genre_period_stats = df_top_with_periods.groupby(['genre', 'period']).agg({
        'compression_ratio': 'mean',
        'album_group_mbid': 'count',
        'year': 'mean'  # Calculate average year for each period
    }).reset_index()
    
    # Rename columns for consistency
    genre_period_stats.rename(columns={
        'album_group_mbid': 'album_count',
        'year': 'avg_year'  # Average year for plotting
    }, inplace=True)
    
    # Filter combinations with at least 5 albums per genre-period
    genre_period_stats = genre_period_stats[genre_period_stats['album_count'] >= 50]

    # Create the comparison plot
    fig, ax = plt.subplots(figsize=(15, 10))
    
    # Define colors for each genre
    colors = plt.cm.tab20(np.linspace(0, 1, len(top_genres)))
    
    # Store genre lines for legend
    genre_lines = []
    
    # Get all unique periods and sort them chronologically
    all_periods = sorted(genre_period_stats['period'].unique(), 
                        key=lambda x: int(x.split('-')[0]) if '-' in x else int(x))
    
    # Create x-axis positions for periods (categorical)
    period_positions = {period: i for i, period in enumerate(all_periods)}
    
    # Plot each genre's trend using period positions
    for i, genre in enumerate(top_genres):
        genre_data = genre_period_stats[genre_period_stats['genre'] == genre]
        if len(genre_data) < 2:  # Need at least 2 points to plot
            continue
        
        # Sort genre data by period chronologically
        genre_data = genre_data.sort_values('avg_year')
        
        # Map periods to x-axis positions
        x_positions = [period_positions[period] for period in genre_data['period']]
        
        line, = ax.plot(x_positions, genre_data['compression_ratio'], 
                      marker='o', linewidth=2.5, markersize=6, 
                      color=colors[i], alpha=0.8)
        genre_lines.append((line, genre))
    
    ax.set_xlabel('Period', fontweight='bold')
    ax.set_ylabel('Average Complexity (ZIPc)', fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # Set x-axis to show period labels
    ax.set_xticks(range(len(all_periods)))
    ax.set_xticklabels(all_periods, rotation=45, ha='right')
    
    # Create legend for genres
    genre_legend = ax.legend(
        [line for line, _ in genre_lines],
        [genre for _, genre in genre_lines],
        bbox_to_anchor=(1, 1), 
        loc='upper left'
    )
    
    # Set reasonable y-limits
    all_scores = genre_period_stats['compression_ratio']
    y_min = max(0, all_scores.min() - 0.05)
    y_max = all_scores.max() + 0.05
    ax.set_ylim(y_min, y_max)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/all_genres_complexity_comparison_dynamic_periods.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("Saved comparison plot for all genres with dynamic periods")
    return genre_period_stats

def plot_quantile_box_comparison_dynamic(df, output_dir, min_albums=3000):
    """Create box plots using dynamic periods for temporal comparison of compression ratios"""
    print("Creating box plots for temporal distribution comparison with dynamic periods...")
    plt.rcParams['font.size'] = FONT_SIZE
    # Apply dynamic periods
    df_with_periods = create_dynamic_periods(df.copy(), min_albums)
    
    # Filter periods with sufficient data (at least 50 albums)
    album_counts = df_with_periods.groupby('period').size()
    valid_periods = album_counts[album_counts >= 50].index
    df_filtered = df_with_periods[df_with_periods['period'].isin(valid_periods)]
    
    # Create figure with single subplot
    fig, ax = plt.subplots(figsize=(15, 8))
    
    # Sort periods chronologically
    sorted_periods = sorted(valid_periods, 
                          key=lambda x: int(x.split('-')[0]) if '-' in x else int(x))
    
    # Prepare data for box plot by time period
    time_data = [df_filtered[df_filtered['period'] == period]['compression_ratio'].values 
                for period in sorted_periods]
    
    # Create box plot for time periods
    box_parts = ax.boxplot(time_data, 
                          positions=range(len(sorted_periods)),
                          patch_artist=True,
                          showmeans=True,
                          meanline=True,
                          widths=0.6)
    
    # Color boxes with a gradient to show temporal progression
    colors_time = plt.cm.viridis(np.linspace(0, 1, len(sorted_periods)))
    for i, (box, color) in enumerate(zip(box_parts['boxes'], colors_time)):
        box.set_facecolor(color)
        box.set_alpha(0.7)
        box.set_edgecolor('black')
        box.set_linewidth(1)
    
    # Customize other elements
    for element in ['whiskers', 'fliers', 'medians', 'caps']:
        plt.setp(box_parts[element], color='black', linewidth=1)
    plt.setp(box_parts['means'], color='red', linewidth=2)

    # Add album counts to labels
    period_labels = [f"{period}" for period in sorted_periods]
    
    ax.set_xticks(range(len(sorted_periods)))
    ax.set_xticklabels(period_labels, rotation=45, ha='right')
    ax.set_ylabel('Complexity Overall Score (ZIPc)', fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add legend to explain elements
    # legend_elements = [
    #     plt.Line2D([0], [0], color='black', linewidth=2, label='Median'),
    #     plt.Line2D([0], [0], color='red', linewidth=2, label='Mean'),
    # ]
    # ax.legend(handles=legend_elements, bbox_to_anchor=(1.05, 1), loc='upper left')
        
    plt.tight_layout()
    plt.savefig(f'{output_dir}/quantile_box_comparison_dynamic_periods.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Create detailed statistics table
    print("\n" + "="*80)
    print(f"TEMPORAL DISTRIBUTION ANALYSIS (Dynamic Periods, Min {min_albums} Albums)")
    print("="*80)
    
    stats_table = []
    for period in sorted_periods:
        period_data = df_filtered[df_filtered['period'] == period]['compression_ratio']
        stats = {
            'Period': period,
            'Albums': len(period_data),
            'Mean': period_data.mean(),
            'Median': period_data.median(),
            'Std': period_data.std(),
            'Q10': period_data.quantile(0.10),
            'Q25': period_data.quantile(0.25),
            'Q75': period_data.quantile(0.75),
            'Q90': period_data.quantile(0.90),
            'Range_80': period_data.quantile(0.90) - period_data.quantile(0.10),
            'IQR': period_data.quantile(0.75) - period_data.quantile(0.25)
        }
        stats_table.append(stats)
    
    # Print formatted table
    print(f"{'Period':<12} {'Albums':<8} {'Mean':<7} {'Median':<7} {'Std':<7} {'Q10':<7} {'Q25':<7} {'Q75':<7} {'Q90':<7} {'80%Rng':<7} {'IQR':<7}")
    print("-" * 100)
    for stats in stats_table:
        print(f"{stats['Period']:<12} {stats['Albums']:<8} {stats['Mean']:<7.3f} {stats['Median']:<7.3f} "
              f"{stats['Std']:<7.3f} {stats['Q10']:<7.3f} {stats['Q25']:<7.3f} {stats['Q75']:<7.3f} "
              f"{stats['Q90']:<7.3f} {stats['Range_80']:<7.3f} {stats['IQR']:<7.3f}")
    
    return df_filtered, stats_table

def main():
    output_dir = 'result/zip'
    os.makedirs(output_dir, exist_ok=True)

    df = load_and_clean_data("../results/unified_album_dataset_with_complexity.csv")
    df_no_split = load_and_clean_data_no_split("../results/unified_album_dataset_with_complexity.csv")
    
    plot_entropy_distribution_box(df,output_dir)
    plot_all_genres_entropy_comparison_dynamic(df, df_no_split, output_dir, min_albums=3000)
    plot_quantile_box_comparison_dynamic(df_no_split, output_dir, min_albums=3000)
    
if __name__ ==  "__main__":
    main()