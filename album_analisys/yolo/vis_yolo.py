import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import ast
from pathlib import Path
import os
import warnings
from collections import defaultdict
warnings.filterwarnings('ignore')
FONT_SIZE = 18

def load_and_clean_data_no_split(csv_file):
    """Load the object detection results without splitting genres."""
    print("Loading object detection results...")
    df_results = pd.read_csv(csv_file)
    
    # Parse object_classes column (it's stored as string representation of dict)
    def parse_object_classes(obj_str):
        try:
            if obj_str == '{}' or pd.isna(obj_str):
                return {}
            return ast.literal_eval(obj_str)
        except:
            return {}
    
    df_results['object_classes_parsed'] = df_results['object_classes'].apply(parse_object_classes)
    
    # Create total objects per image
    df_results['total_objects'] = df_results['object_classes_parsed'].apply(
        lambda x: sum(x.values()) if x else 0
    )
    
    # Create has_objects flag
    df_results['has_objects'] = df_results['total_objects'] > 0
    
    # Extract year from release_date
    df_results['year'] = pd.to_datetime(df_results['release_date'], errors='coerce', format='mixed', yearfirst=True).dt.year
    
    # Filter out rows with invalid years
    df_clean = df_results.dropna(subset=['year'])
    df_clean['year'] = df_clean['year'].astype(int)
    
    # Filter reasonable year range
    df_clean = df_clean[(df_clean['year'] >= 1950) & (df_clean['year'] <= 2025)]
    
    print(f"Dataset loaded: {len(df_results)} total albums, {len(df_clean)} with valid year data")
    
    return df_clean

def load_and_clean_data(csv_file):
    """Load the object detection results and split multi-label genres."""
    print("Loading object detection results...")
    df_results = pd.read_csv(csv_file)
    
    # Parse object_classes column (it's stored as string representation of dict)
    def parse_object_classes(obj_str):
        try:
            if obj_str == '{}' or pd.isna(obj_str):
                return {}
            return ast.literal_eval(obj_str)
        except:
            return {}
    
    df_results['object_classes_parsed'] = df_results['object_classes'].apply(parse_object_classes)
    
    # Create total objects per image
    df_results['total_objects'] = df_results['object_classes_parsed'].apply(
        lambda x: sum(x.values()) if x else 0
    )

    print("Total objects < 0:")
    print(df_results[df_results['total_objects'] < 0].head(5))
    
    # Create has_objects flag
    df_results['has_objects'] = df_results['total_objects'] > 0
    
    # Extract year from release_date
    df_results['year'] = pd.to_datetime(df_results['release_date'], errors='coerce', format='mixed', yearfirst=True).dt.year
    
    # Filter out rows with invalid years
    df_clean = df_results.dropna(subset=['year'])
    df_clean['year'] = df_clean['year'].astype(int)
    
    # Filter reasonable year range
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

def plot_objects_over_time_by_genre(df, df_no_split, output_dir, min_albums=3000):
    """
    Plot 3: Line plot showing mean number of objects per album over time 
    grouped by genre all in one plot, using dynamic periods.
    """
    print("Creating line plot of objects over time by genre with dynamic periods...")
    plt.rcParams['font.size'] = FONT_SIZE
    # Get top genres by count
    top_genres = df.groupby('genre').size().nlargest(11).index.tolist()
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
    
    # Group by genre and period, calculate mean objects
    genre_period_objects = df_top_with_periods.groupby(['genre', 'period']).agg({
        'total_objects': ['mean', 'count'],
        'year': 'mean'  # Calculate average year for each period
    }).reset_index()
    
    # Flatten column names
    genre_period_objects.columns = ['genre', 'period', 'mean', 'count', 'avg_year']
    
    # Filter combinations with reasonable count (reduced from 50 since we're using dynamic periods)
    genre_period_objects = genre_period_objects[genre_period_objects['count'] >= 20]
    
    # Sort periods chronologically
    def period_sort_key(period):
        if '-' in period:
            return int(period.split('-')[0])
        else:
            return int(period)
    
    genre_period_objects['sort_key'] = genre_period_objects['period'].apply(period_sort_key)
    genre_period_objects = genre_period_objects.sort_values('sort_key')
    
    print("Dynamic periods created:")
    period_counts = df_no_split_with_periods.groupby('period').size()
    for period in sorted(period_counts.index, key=period_sort_key):
        print(f"  {period}: {period_counts[period]} albums")
    
    # Create plot
    fig, ax = plt.subplots(figsize=(14, 10))
    
    # Define colors for each genre
    colors = plt.cm.tab20(np.linspace(0, 1, len(top_genres)))
    
    # Get all unique periods and sort them chronologically
    all_periods = sorted(genre_period_objects['period'].unique(), key=period_sort_key)
    
    # Create x-axis positions for periods (categorical)
    period_positions = {period: i for i, period in enumerate(all_periods)}
    
    # Store genre lines for legend
    genre_lines = []
    
    # Plot each genre's trend using period positions
    for i, genre in enumerate(top_genres):
        genre_data = genre_period_objects[genre_period_objects['genre'] == genre]
        if len(genre_data) < 2:  # Need at least 2 points to plot
            continue
        
        # Sort genre data by period chronologically
        genre_data = genre_data.sort_values('avg_year')
        
        # Map periods to x-axis positions
        x_positions = [period_positions[period] for period in genre_data['period']]
        
        line, = ax.plot(x_positions, genre_data['mean'], 
                      marker='o', linewidth=2, markersize=6, 
                      color=colors[i], alpha=0.8)
        genre_lines.append((line, genre))
    
    # Add labels and title
    # ax.set_title(f'Average Number of Objects in Album Covers by Genre Over Dynamic Periods\n(Min {min_albums} Albums per Period)', 
    #             fontsize=14, fontweight='bold')
    ax.set_xlabel('Time Period', fontweight='bold')
    ax.set_ylabel('Average Number of Objects', fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # Set x-axis to show period labels
    ax.set_xticks(range(len(all_periods)))
    ax.set_xticklabels(all_periods, rotation=45, ha='right')

    ax.set_yticklabels(ax.get_yticks())
    
    # Create legend for genres
    genre_legend = ax.legend(
        [line for line, _ in genre_lines],
        [genre for _, genre in genre_lines],
        bbox_to_anchor=(1, 1), 
        loc='upper left',
    )
    
    ax.set_ylim(0.9, 3)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/objects_over_time_by_genre_dynamic_periods_{min_albums}.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    return genre_period_objects


def get_yolo_class_distribution(df):
    """
    Get the ordered list of YOLO classes with their total occurrences.
    
    Args:
        df: DataFrame with object_classes_parsed column
        
    Returns:
        pandas.Series: Ordered series of class names and their counts
    """
    print("Calculating YOLO class distribution...")
    
    # Dictionary to store class counts
    class_counts = defaultdict(int)
    
    # Iterate through all albums and count object classes
    for _, row in df.iterrows():
        object_classes = row['object_classes_parsed']
        if object_classes:  # If there are objects detected
            for class_name, count in object_classes.items():
                class_counts[class_name] += count
    
    # Convert to pandas Series and sort by count (descending)
    class_series = pd.Series(class_counts).sort_values(ascending=False)
    
    print(f"Found {len(class_series)} unique object classes")
    print(f"Total objects detected: {class_series.sum()}")
    
    # Print top 10 classes
    print("\nTop 10 most frequent object classes:")
    for i, (class_name, count) in enumerate(class_series.head(80).items(), 1):
        print(f"{i:2d}. {class_name}: {count} occurrences")
    
    return class_series


def plot_yolo_class_distribution_by_genre(df, output_dir, top_genres=10):
    """
    Create a stacked bar plot showing YOLO class distribution per genre,
    similar to Figure 5.10 in the attached image.
    
    Args:
        df: DataFrame with genre and object_classes_parsed columns
        output_dir: Directory to save the plot
        top_genres: Number of top genres to include
    """
    print(f"Creating YOLO class distribution plot for top {top_genres} genres...")
    plt.rcParams['font.size'] = FONT_SIZE
    # Get top genres by album count
    top_genre_list = df.groupby('genre').size().nlargest(top_genres).index.tolist()
    df_top = df[df['genre'].isin(top_genre_list)]
    
    # Dictionary to store class counts per genre
    genre_class_counts = defaultdict(lambda: defaultdict(int))
    
    # Count objects by genre and class
    for _, row in df_top.iterrows():
        genre = row['genre']
        object_classes = row['object_classes_parsed']
        if object_classes:  # If there are objects detected
            for class_name, count in object_classes.items():
                genre_class_counts[genre][class_name] += count
    
    # Convert to DataFrame for easier plotting
    # Create a matrix where rows are genres and columns are object classes
    all_classes = set()
    for genre_classes in genre_class_counts.values():
        all_classes.update(genre_classes.keys())
    
    # Create the data matrix
    data_matrix = []
    genre_names = []
    
    for genre in top_genre_list:
        genre_names.append(genre)
        row_data = []
        for class_name in sorted(all_classes):
            row_data.append(genre_class_counts[genre][class_name])
        data_matrix.append(row_data)
    
    # Convert to DataFrame
    df_matrix = pd.DataFrame(data_matrix, 
                            index=genre_names, 
                            columns=sorted(all_classes))
    
    # Calculate total objects per genre for sorting
    genre_totals = df_matrix.sum(axis=1).sort_values(ascending=False)
    df_matrix = df_matrix.loc[genre_totals.index]
    
    # Get the most common classes across all genres for better visualization
    class_totals = df_matrix.sum(axis=0).sort_values(ascending=False)
    top_classes = class_totals.head(15).index.tolist()  # Show top 15 classes
    
    # Filter matrix to only include top classes
    df_matrix_filtered = df_matrix[top_classes]
    
    # Convert to percentages first
    df_matrix_percent = df_matrix_filtered.div(genre_totals, axis=0) * 100
    df_matrix_percent = df_matrix_percent.fillna(0)
    
    # Create the stacked bar plot
    fig, ax = plt.subplots(figsize=(14, 10))
    
    # Generate colors for each class
    colors = plt.cm.tab20(np.linspace(0, 1, len(top_classes)))
    
    # Create stacked bars using percentages
    bottom = np.zeros(len(df_matrix_percent))
    
    for i, class_name in enumerate(top_classes):
        values = df_matrix_percent[class_name].values
        bars = ax.bar(range(len(df_matrix_percent)), values, bottom=bottom, 
                     label=class_name, color=colors[i], alpha=0.8, 
                     edgecolor='white', linewidth=0.5)
        
        # Add text labels for significant values
        for j, (bar, value) in enumerate(zip(bars, values)):
            if value > 2:  # Only show labels for segments > 2%
                ax.text(bar.get_x() + bar.get_width()/2., 
                       bottom[j] + value/2.,
                       str(int(df_matrix_filtered.iloc[j, i])),  # Show absolute count
                       ha='center', va='center', 
                       fontweight='bold',
                       color='white' if value > 5 else 'black')
        
        bottom += values
    
    # Add total object counts as text above bars
    for i, (genre, total) in enumerate(genre_totals.items()):
        ax.text(i, 102, str(int(total)), 
               ha='center', va='bottom', fontweight='bold')
    
    # Customize the plot
    ax.set_xlabel('Genre', fontweight='bold')
    ax.set_ylabel('Percentage', fontweight='bold')
    
    # Set x-axis labels
    ax.set_xticks(range(len(df_matrix_percent.index)))
    ax.set_xticklabels([name.replace(' & ', '\n& ').replace(' and ', '\nand ') 
                       for name in df_matrix_percent.index], 
                      rotation=0, ha='center')
    
    ax.set_ylim(0, 105)
    
    # Create legend
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # Add grid
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/yolo_class_distribution_by_genre.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Print summary statistics
    print(f"\nYOLO Class Distribution Summary (Top {top_genres} Genres):")
    print("=" * 60)
    print(f"{'Genre':<15} {'Total Objects':<12} {'Top Class':<15} {'Count':<8}")
    print("-" * 60)
    
    for genre in df_matrix_filtered.index:
        total_objects = int(genre_totals[genre])
        top_class = df_matrix_filtered.loc[genre].idxmax()
        top_class_count = int(df_matrix_filtered.loc[genre, top_class])
        print(f"{genre:<15} {total_objects:<12} {top_class:<15} {top_class_count:<8}")
    
    return df_matrix_filtered, genre_totals


def plot_albums_with_objects_by_genre(df, output_dir, top_genres=11):
    """
    Create a stacked bar plot showing percentage of albums with/without objects detected by genre.
    
    Args:
        df: DataFrame with genre and object_classes_parsed columns
        output_dir: Directory to save the plot
        top_genres: Number of top genres to include
    """
    print(f"Creating stacked bar plot of albums with/without objects for top {top_genres} genres...")
    plt.rcParams['font.size'] = FONT_SIZE
    # Get top genres by album count
    top_genre_list = df.groupby('genre').size().nlargest(top_genres).index.tolist()
    df_top = df[df['genre'].isin(top_genre_list)]
    
    # Create has_objects flag based on total_objects
    df_top = df_top.copy()
    df_top['has_objects'] = df_top['total_objects'] > 0
    
    # Count albums with/without objects by genre
    genre_object_stats = df_top.groupby(['genre', 'has_objects']).size().unstack(fill_value=0)
    
    # Rename columns for clarity
    if True in genre_object_stats.columns and False in genre_object_stats.columns:
        genre_object_stats = genre_object_stats.rename(columns={
            True: 'With Objects', 
            False: 'Without Objects'
        })
    elif True in genre_object_stats.columns:
        genre_object_stats = genre_object_stats.rename(columns={True: 'With Objects'})
        genre_object_stats['Without Objects'] = 0
    elif False in genre_object_stats.columns:
        genre_object_stats = genre_object_stats.rename(columns={False: 'Without Objects'})
        genre_object_stats['With Objects'] = 0
    else:
        # No data case
        genre_object_stats['With Objects'] = 0
        genre_object_stats['Without Objects'] = 0
    
    # Ensure both columns exist
    if 'With Objects' not in genre_object_stats.columns:
        genre_object_stats['With Objects'] = 0
    if 'Without Objects' not in genre_object_stats.columns:
        genre_object_stats['Without Objects'] = 0
    
    # Calculate total albums per genre and sort by total
    genre_object_stats['Total'] = genre_object_stats['With Objects'] + genre_object_stats['Without Objects']
    genre_object_stats = genre_object_stats.sort_values('Total', ascending=False)
    
    # Calculate percentages
    genre_object_percent = genre_object_stats[['With Objects', 'Without Objects']].div(
        genre_object_stats['Total'], axis=0
    ) * 100
    
    # Create the stacked bar plot
    fig, ax = plt.subplots(figsize=(14, 10))
    
    # Colors: green for with objects, red for without objects
    colors = ['#FFA500', '#800080']  # orange and purple
    
    # Create stacked bars
    genre_object_percent[['With Objects', 'Without Objects']].plot(
        kind='bar', 
        stacked=True, 
        ax=ax, 
        color=colors,
        alpha=0.8,
        edgecolor='white',
        linewidth=1
    )
    
    # Add percentage labels on bars
    for i, genre in enumerate(genre_object_percent.index):
        # Get values
        with_obj_pct = genre_object_percent.loc[genre, 'With Objects']
        without_obj_pct = genre_object_percent.loc[genre, 'Without Objects']
        
        # Get absolute counts
        with_obj_count = genre_object_stats.loc[genre, 'With Objects']
        without_obj_count = genre_object_stats.loc[genre, 'Without Objects']
        total_count = genre_object_stats.loc[genre, 'Total']
        
        # Add percentage labels only if segment is large enough
        # if with_obj_pct > 3:  # Only show if > 3%
        #     ax.text(i, with_obj_pct/2, 
        #            f'{with_obj_pct:.1f}%', 
        #            ha='center', va='center', fontweight='bold', 
        #            color='black', fontsize=FONT_SIZE-10)
        
        # if without_obj_pct > 3:  # Only show if > 3%
        #     ax.text(i, with_obj_pct + without_obj_pct/2, 
        #            f'{without_obj_pct:.1f}%', 
        #            ha='center', va='center', fontweight='bold', 
        #            color='white', fontsize=FONT_SIZE-10)
        

    
    # Customize the plot
    ax.set_xlabel('Genre', fontweight='bold')
    ax.set_ylabel('Percentage of Albums', fontweight='bold')

    
    # Format x-axis labels
    ax.set_xticklabels([name.replace(' & ', '\n& ').replace(' and ', '\nand ') 
                       for name in genre_object_percent.index], 
                      rotation=45, ha='center')
    
    # Set y-axis
    ax.set_ylim(0, 100)
    ax.set_yticklabels([f'{int(y)}%' for y in ax.get_yticks()])
    
    # Customize legend
    ax.legend( 
             bbox_to_anchor=(1, 1), 
             loc='upper left', 
             )
    
    # Add grid
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/albums_with_objects_by_genre_percentage.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Print summary statistics
    print(f"\nAlbums with Object Detection Summary (Top {top_genres} Genres):")
    print("=" * 80)
    print(f"{'Genre':<15} {'Total Albums':<12} {'With Objects':<12} {'Without Objects':<14} {'Detection Rate':<14}")
    print("-" * 80)
    
    for genre in genre_object_percent.index:
        total_albums = int(genre_object_stats.loc[genre, 'Total'])
        with_objects = int(genre_object_stats.loc[genre, 'With Objects'])
        without_objects = int(genre_object_stats.loc[genre, 'Without Objects'])
        detection_rate = genre_object_percent.loc[genre, 'With Objects']
        
        print(f"{genre:<15} {total_albums:<12} {with_objects:<12} {without_objects:<14} {detection_rate:<14.1f}%")
    
    return genre_object_stats, genre_object_percent


def main():
    """Main function to run the visualization."""
    # Create output directory
    output_dir = 'result/objects_analysis'
    os.makedirs(output_dir, exist_ok=True)
    
    # Path to the CSV file with object detection results
    df_csv = "../results/objects/unified_with_objects.csv"

    # Check if files exist
    if not Path(df_csv).exists():
        print(f"Error: One or more input files not found!")
        return
    
    # Load and clean data
    df = load_and_clean_data(df_csv)

    df_no_split = load_and_clean_data_no_split(df_csv)

    
    # Analyze YOLO class distribution
    print("\n" + "="*50)
    print("YOLO CLASS ANALYSIS")
    print("="*50)
    
    # Get ordered list of YOLO classes with occurrences
    # class_distribution = get_yolo_class_distribution(df_no_split)
    
    # Create YOLO class distribution plot by genre
    # yolo_matrix, genre_totals = plot_yolo_class_distribution_by_genre(df, output_dir, top_genres=11)
    
    # Create albums with/without objects detection plot by genre
    object_stats, object_percent = plot_albums_with_objects_by_genre(df, output_dir, top_genres=11)
    
    
    # Generate other plots
    print("\n" + "="*50)
    print("TIME-BASED ANALYSIS")
    print("="*50)
    # plot_objects_by_genre(df, output_dir)
    # plot_objects_over_time(df_no_split, output_dir)
    plot_objects_over_time_by_genre(df, df_no_split, output_dir, min_albums=3000)
    # plot_objects_boxplot_by_time(df_no_split, output_dir)
    # plot_objects_boxplot_by_genre(df, output_dir)

    
    print(f"All plots saved to: {output_dir}")

if __name__ == "__main__":
    main()
