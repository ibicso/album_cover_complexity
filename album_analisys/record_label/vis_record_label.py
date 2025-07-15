import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from pathlib import Path
import ast
import warnings
from adjustText import adjust_text
from scipy import stats
from mpl_toolkits.axes_grid1 import make_axes_locatable

warnings.filterwarnings('ignore')

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")
plt.rcParams.update({
    'font.size': 12,
    'font.family': 'sans-serif',
    'axes.labelweight': 'bold',
    'axes.titleweight': 'bold',
    'axes.titlesize': 14,
    'axes.labelsize': 12,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.grid': True,
    'grid.alpha': 0.3
})

def load_and_process_data_no_split():
    """Load and combine both entropy-complexity datasets - keep albums as single entities"""
    # Load datasets
    combined_df = pd.read_csv('../results/unified_album_dataset_with_complexity.csv')
    
    # Create label_type from is_independent column
    combined_df['label_type'] = combined_df['is_independent'].map({True: 'Independent', False: 'Major'})
    
    # Process date
    combined_df['year'] = pd.to_datetime(combined_df['release_date'], errors='coerce', format='mixed').dt.year
    print("Combined after year:", len(combined_df))
    
    # Clean data
    combined_df = combined_df.dropna(subset=['year', 'label_type'])
    print("Combined after dropna:", len(combined_df))
    combined_df['year'] = combined_df['year'].astype(int)
    combined_df = combined_df.drop_duplicates(subset=['album_group_mbid'])
    
    # Filter reasonable year range
    combined_df = combined_df[(combined_df['year'] >= 1950) & (combined_df['year'] <= 2025)]
    
    print(f"Dataset loaded (no split): {len(combined_df)} total albums")
    print("Label type distribution:")
    print(combined_df['label_type'].value_counts())
    
    return combined_df

def load_and_process_data():
    """Load and combine both entropy-complexity datasets - split genres into individual rows"""
    # First load the base dataset
    combined_df = load_and_process_data_no_split()
    print("Combined after initial load:", len(combined_df))
    print("Columns in combined_df:", combined_df.columns.tolist())
    print("Label type in combined_df:", combined_df['label_type'].value_counts())
    
    # Split multi-label genres and create separate rows for each genre
    print("Splitting multi-label genres...")
    expanded_rows = []
    
    for _, row in combined_df.iterrows():
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
                new_row['genre'] = genre.strip()
                expanded_rows.append(new_row)
    
    # Create new dataframe with expanded genres
    df_expanded = pd.DataFrame(expanded_rows)
    
    print(f"Dataset loaded (with split): {len(df_expanded)} genre-album combinations")
    print(f"Original albums: {len(combined_df)}")
    print(f"Unique individual genres: {df_expanded['genre'].nunique()}")
    print("Label type distribution in expanded data:")
    print(df_expanded['label_type'].value_counts())
    print("Columns in expanded data:", df_expanded.columns.tolist())
    
    return df_expanded

def process_genres(df):
    """Process genres column to extract individual genres"""
    processed_rows = []
    
    for _, row in df.iterrows():
        try:
            # Parse genres string as list
            if pd.notna(row['genres']) and row['genres'] != '':
                genres = ast.literal_eval(row['genres'])
                if isinstance(genres, list):
                    for genre in genres:
                        new_row = row.copy()
                        new_row['genre'] = genre.strip()
                        processed_rows.append(new_row)
                else:
                    # Handle single genre case
                    new_row = row.copy()
                    new_row['genre'] = genres.strip()
                    processed_rows.append(new_row)
        except:
            # Handle comma-separated string case
            if pd.notna(row['genres']):
                genres = [g.strip() for g in str(row['genres']).split(',')]
                for genre in genres:
                    if genre and genre != 'nan':
                        new_row = row.copy()
                        new_row['genre'] = genre
                        processed_rows.append(new_row)
    
    if not processed_rows:
        return pd.DataFrame(columns=df.columns.tolist() + ['genre'])
    
    return pd.DataFrame(processed_rows)

def create_dynamic_periods(df, min_albums=1500):
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

def plot_entropy_complexity_by_dynamic_periods(df, min_albums=3000):
    """Plot 1: Entropy-complexity plane evolution on dynamic periods of independent albums vs major"""
    df_periods = create_dynamic_periods(df.copy(), min_albums=min_albums)
    
    # Calculate means, std, and count by period for SEM calculation
    period_stats = df_periods.groupby(['label_type', 'period'])[['permutation_entropy', 'statistical_complexity']].agg(['mean', 'std', 'count']).reset_index()
    period_stats.columns = ['label_type', 'period', 'entropy_mean', 'entropy_std', 'entropy_count', 'complexity_mean', 'complexity_std', 'complexity_count']
    
    # Sort periods chronologically
    def period_sort_key(period):
        if '-' in period:
            return int(period.split('-')[0])
        else:
            return int(period)
    
    period_stats['sort_key'] = period_stats['period'].apply(period_sort_key)
    period_stats = period_stats.sort_values(['label_type', 'sort_key']).reset_index(drop=True)
    
    # Calculate SEM (Standard Error of the Mean)
    period_stats['entropy_sem'] = period_stats['entropy_std'] / np.sqrt(period_stats['entropy_count'])
    period_stats['complexity_sem'] = period_stats['complexity_std'] / np.sqrt(period_stats['complexity_count'])
    
    print("Dynamic periods created:")
    for _, row in period_stats.iterrows():
        print(f"  {row['label_type']} - {row['period']}: {row['entropy_count']} albums")
    
    fig, ax = plt.subplots(figsize=(14, 10))
    
    # Define colors for each label type
    label_colors = {'Independent': 'blue', 'Major': 'red'}
    
    # Plot each label type separately
    for label_type in ['Independent', 'Major']:
        label_data = period_stats[period_stats['label_type'] == label_type]
        
        # Create color gradient for time periods within each label type
        color_base = label_colors[label_type]
        color_gradient = plt.cm.get_cmap('Blues' if label_type == 'Independent' else 'Reds')
        colors = color_gradient(np.linspace(0.3, 1, len(label_data)))
        
        # Plot with error bars
        for i, (_, row) in enumerate(label_data.iterrows()):
            ax.errorbar(row['entropy_mean'], 
                       row['complexity_mean'],
                       xerr=row['entropy_sem'],
                       yerr=row['complexity_sem'],
                       fmt='o', 
                       markersize=10,
                       capsize=2,
                       capthick=0.5,
                       elinewidth=0.5,
                       alpha=0.8,
                       color=colors[i],
                       markeredgecolor='white',
                       markeredgewidth=0.5,
                       label=f"{label_type} {row['period']}" if i == 0 else None)
        
        # Add connecting line with gradient
        ax.plot(label_data['entropy_mean'], 
               label_data['complexity_mean'],
               '--', alpha=0.7, linewidth=2, 
               color=color_base, zorder=1,
               label=f"{label_type} Trend")
        
        # Add period labels with better styling
        for i, (_, row) in enumerate(label_data.iterrows()):
            # Vary annotation positions to avoid overlap
            angle = i * (360 / len(label_data)) + (0 if label_type == 'Independent' else 180)
            offset_x = 40 * np.cos(np.radians(angle))
            offset_y = 40 * np.sin(np.radians(angle))
            
            # Create label with album count
            label = f"{row['period']}"
            
            ax.annotate(label, 
                       (row['entropy_mean'], row['complexity_mean']),
                       xytext=(offset_x, offset_y), textcoords='offset points', 
                       fontsize=9,
                       ha='center', va='center',
                       color=color_base,
                       arrowprops=dict(arrowstyle='-', color=colors[i], alpha=0.7, lw=0.5))
    
    # Adjust axis limits if needed for better visualization
    # ax.set_ylim(0.1125, 0.14)
    # ax.set_xlim(0.82, 0.87)
    
    ax.set_xlabel('Permutation Entropy', fontweight='bold', fontsize=14)
    ax.set_ylabel('Statistical Complexity', fontweight='bold', fontsize=14)
    ax.set_title(f'Evolution of Album Complexity: Independent vs Major Labels\n(Dynamic Periods, Min {min_albums} Albums ± SEM)', 
                fontweight='bold', fontsize=16, pad=20)
    
    # Add legend with custom styling
    handles, labels = ax.get_legend_handles_labels()
    legend = ax.legend(handles[:4], labels[:4], loc='upper left', frameon=True, 
                      framealpha=0.9, edgecolor='gray')
    legend.get_frame().set_facecolor('#f8f8f8')
    
    # Add subtle background styling
    ax.set_facecolor('#fafafa')
    ax.grid(True, alpha=0.4, linestyle='-', linewidth=0.5)
    
    plt.tight_layout()
    plt.savefig(f'../result/record_label/entropy_complexity_independent_vs_major_dynamic_periods_{min_albums}.png', 
               dpi=300, bbox_inches='tight')
    plt.show()

def plot_entropy_complexity_overall_means(df):
    """Plot 2: Mean of the entropy-complexity of independents and major"""
    # Calculate means, std, and count by label type for SEM calculation
    label_stats = df.groupby('label_type')[['permutation_entropy', 'statistical_complexity']].agg(['mean', 'std', 'count']).reset_index()
    label_stats.columns = ['label_type', 'entropy_mean', 'entropy_std', 'entropy_count', 'complexity_mean', 'complexity_std', 'complexity_count']
    
    # Calculate SEM
    label_stats['entropy_sem'] = label_stats['entropy_std'] / np.sqrt(label_stats['entropy_count'])
    label_stats['complexity_sem'] = label_stats['complexity_std'] / np.sqrt(label_stats['complexity_count'])
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Define colors for each label type
    label_colors = {'Independent': 'blue', 'Major': 'red'}
    
    # Plot with error bars
    for i, (_, row) in enumerate(label_stats.iterrows()):
        ax.errorbar(row['entropy_mean'], 
                   row['complexity_mean'],
                   xerr=row['entropy_sem'],
                   yerr=row['complexity_sem'],
                   fmt='o', 
                   markersize=14,
                   capsize=4,
                   capthick=1.5,
                   elinewidth=1.5,
                   alpha=0.8,
                   color=label_colors[row['label_type']],
                   markeredgecolor='white',
                   markeredgewidth=1.5,
                   label=row['label_type'])
        
        # Add label annotations
        ax.annotate(f"{row['label_type']}\n(n={row['entropy_count']})", 
                   (row['entropy_mean'], row['complexity_mean']),
                   xytext=(0, 30), textcoords='offset points', 
                   fontsize=12, fontweight='bold',
                   ha='center', va='bottom',
                   bbox=dict(boxstyle="round,pad=0.3", 
                            facecolor=label_colors[row['label_type']], 
                            alpha=0.2, 
                            edgecolor=label_colors[row['label_type']]))
    
    # Add statistical significance testing
    # Get data for each group
    independent_data = df[df['label_type'] == 'Independent']
    major_data = df[df['label_type'] == 'Major']
    
    # Perform t-tests
    entropy_ttest = stats.ttest_ind(
        independent_data['permutation_entropy'].dropna(),
        major_data['permutation_entropy'].dropna(),
        equal_var=False  # Welch's t-test for unequal variances
    )
    
    complexity_ttest = stats.ttest_ind(
        independent_data['statistical_complexity'].dropna(),
        major_data['statistical_complexity'].dropna(),
        equal_var=False
    )
    
    # Add p-values to plot
    ax.text(0.02, 0.98, 
           f"Entropy p-value: {entropy_ttest.pvalue:.4f}\nComplexity p-value: {complexity_ttest.pvalue:.4f}",
           transform=ax.transAxes,
           fontsize=10,
           va='top',
           bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8, edgecolor='gray'))
    
    ax.set_xlabel('Permutation Entropy', fontweight='bold', fontsize=14)
    ax.set_ylabel('Statistical Complexity', fontweight='bold', fontsize=14)
    ax.set_title('Album Complexity: Independent vs Major Labels\n(Overall Mean ± SEM)', 
                fontweight='bold', fontsize=16, pad=20)
    
    # Add legend
    ax.legend(frameon=True, framealpha=0.9, edgecolor='gray')
    
    # Add subtle background styling
    ax.set_facecolor('#fafafa')
    ax.grid(True, alpha=0.4, linestyle='-', linewidth=0.5)
    
    plt.tight_layout()
    plt.savefig('../result/record_label/entropy_complexity_independent_vs_major_overall.png', 
               dpi=300, bbox_inches='tight')
    plt.show()

def plot_entropy_complexity_by_genre(df):
    """Plot 3: Mean of the entropy-complexity of independents and major for every genre"""
    # Process genres to split into individual rows
    print("Original dataframe columns:", df.columns.tolist())
    print("Original dataframe shape:", df.shape)
    print("Label type values in original df:", df['label_type'].value_counts())
    
    df_genres = process_genres(df.copy())
    print("Processed genres dataframe columns:", df_genres.columns.tolist())
    print("Processed genres dataframe shape:", df_genres.shape)
    
    # Check if label_type exists
    if 'label_type' not in df_genres.columns:
        print("WARNING: label_type column not found after processing genres!")
        # Create label_type from is_independent column if it exists
        if 'is_independent' in df_genres.columns:
            print("Creating label_type from is_independent column")
            df_genres['label_type'] = df_genres['is_independent'].map({True: 'Independent', False: 'Major'})
        else:
            print("ERROR: Cannot create label_type column, is_independent not found")
            return
    
    print("Label type values after processing:", df_genres['label_type'].value_counts())
    
    # Get top genres by count
    genre_counts = df_genres['genre'].value_counts()
    top_genres = genre_counts[genre_counts >= 100].index.tolist()  # Only include genres with at least 100 albums
    print(f"Found {len(top_genres)} genres with at least 100 albums")
    
    # Filter for top genres
    df_top_genres = df_genres[df_genres['genre'].isin(top_genres)]
    print(f"Filtered to {len(df_top_genres)} albums with top genres")
    print("Label type values in filtered df:", df_top_genres['label_type'].value_counts())
    
    # Calculate means, std, and count by genre and label type for SEM calculation
    genre_label_stats = df_top_genres.groupby(['genre', 'label_type'])[['permutation_entropy', 'statistical_complexity']].agg(['mean', 'std', 'count']).reset_index()
    genre_label_stats.columns = ['genre', 'label_type', 'entropy_mean', 'entropy_std', 'entropy_count', 'complexity_mean', 'complexity_std', 'complexity_count']
    
    # Filter out combinations with too few samples
    min_samples = 20
    genre_label_stats = genre_label_stats[genre_label_stats['entropy_count'] >= min_samples]
    
    # Get list of genres that have both independent and major labels with sufficient samples
    valid_genres = []
    for genre in top_genres:
        genre_data = genre_label_stats[genre_label_stats['genre'] == genre]
        if len(genre_data) == 2:  # Has both independent and major
            valid_genres.append(genre)
    
    print(f"Found {len(valid_genres)} genres with both independent and major labels")
    if len(valid_genres) == 0:
        print("No valid genres found for comparison. Exiting.")
        return
    
    # Calculate SEM
    genre_label_stats['entropy_sem'] = genre_label_stats['entropy_std'] / np.sqrt(genre_label_stats['entropy_count'])
    genre_label_stats['complexity_sem'] = genre_label_stats['complexity_std'] / np.sqrt(genre_label_stats['complexity_count'])
    
    # Create subplots for each valid genre
    n_genres = len(valid_genres)
    n_cols = 2
    n_rows = (n_genres + n_cols - 1) // n_cols  # Ceiling division
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 5 * n_rows))
    axes = axes.flatten() if n_rows > 1 else [axes] if n_cols == 1 else axes
    
    # Define colors for each label type
    label_colors = {'Independent': 'blue', 'Major': 'red'}
    
    for i, genre in enumerate(valid_genres):
        ax = axes[i]
        genre_data = genre_label_stats[genre_label_stats['genre'] == genre]
        
        # Plot with error bars
        for _, row in genre_data.iterrows():
            ax.errorbar(row['entropy_mean'], 
                       row['complexity_mean'],
                       xerr=row['entropy_sem'],
                       yerr=row['complexity_sem'],
                       fmt='o', 
                       markersize=10,
                       capsize=3,
                       capthick=1,
                       elinewidth=1,
                       alpha=0.8,
                       color=label_colors[row['label_type']],
                       markeredgecolor='white',
                       markeredgewidth=1,
                       label=f"{row['label_type']} (n={row['entropy_count']})")
        
        # Perform t-tests for this genre
        genre_df = df_top_genres[df_top_genres['genre'] == genre]
        independent_data = genre_df[genre_df['label_type'] == 'Independent']
        major_data = genre_df[genre_df['label_type'] == 'Major']
        
        if len(independent_data) >= min_samples and len(major_data) >= min_samples:
            entropy_pvalue = stats.ttest_ind(
                independent_data['permutation_entropy'].dropna(),
                major_data['permutation_entropy'].dropna(),
                equal_var=False
            ).pvalue
            
            complexity_pvalue = stats.ttest_ind(
                independent_data['statistical_complexity'].dropna(),
                major_data['statistical_complexity'].dropna(),
                equal_var=False
            ).pvalue
            
            # Add significance stars
            significance_text = ""
            if entropy_pvalue < 0.05 or complexity_pvalue < 0.05:
                significance_text = "* p<0.05"
            if entropy_pvalue < 0.01 or complexity_pvalue < 0.01:
                significance_text = "** p<0.01"
            if entropy_pvalue < 0.001 or complexity_pvalue < 0.001:
                significance_text = "*** p<0.001"
                
            if significance_text:
                ax.text(0.02, 0.98, significance_text,
                       transform=ax.transAxes,
                       fontsize=10, fontweight='bold',
                       va='top')
        
        ax.set_xlabel('Permutation Entropy', fontsize=11)
        ax.set_ylabel('Statistical Complexity', fontsize=11)
        ax.set_title(f'{genre}', fontweight='bold', fontsize=13)
        
        ax.legend(frameon=True, framealpha=0.9, edgecolor='gray', fontsize=9)
        ax.grid(True, alpha=0.4, linestyle='-', linewidth=0.5)
    
    # Hide any unused subplots
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)
    
    plt.suptitle('Album Complexity by Genre: Independent vs Major Labels\n(Mean ± SEM)', 
                fontweight='bold', fontsize=16)
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.92)
    plt.savefig('../result/record_label/entropy_complexity_independent_vs_major_by_genre.png', 
               dpi=300, bbox_inches='tight')
    plt.show()

def order_genres_custom(df, genre_column='genre'):
    """
    Order genres according to a predefined order.
    Returns the dataframe with an additional 'genre_order' column that can be used for sorting.
    """
    # Define the custom order for genres
    genre_order = {
        'Pop': 1,
        'Rock': 2,
        'Speciality': 3,
        'Jazz & Blues': 4,
        'Electronic': 5,
        'World Music': 6,
        'Country & Folk': 7,
        'Metal': 8,
        'R&B': 9,
        'Hip Hop': 10,
        'Classical': 11
    }
    
    # Create a function to map genres to their order, with fallback for unlisted genres
    def get_genre_order(genre):
        # Handle case variations (e.g., "pop" should match "Pop")
        for key in genre_order:
            if genre.lower() == key.lower():
                return genre_order[key]
        # For genres not in the predefined list, place them at the end
        return 999
    
    # Apply the ordering function
    df = df.copy()
    df['genre_order'] = df[genre_column].apply(get_genre_order)
    
    return df

def plot_entropy_complexity_all_genres_by_label(df):
    """Plot all genres in the same entropy-complexity plane for independent vs major labels"""
    # Process genres to split into individual rows
    print("Processing genres for all-genres plot...")
    df_genres = process_genres(df.copy())
    
    # Check if label_type exists
    if 'label_type' not in df_genres.columns:
        print("WARNING: label_type column not found after processing genres!")
        # Create label_type from is_independent column if it exists
        if 'is_independent' in df_genres.columns:
            print("Creating label_type from is_independent column")
            df_genres['label_type'] = df_genres['is_independent'].map({True: 'Independent', False: 'Major'})
        else:
            print("ERROR: Cannot create label_type column, is_independent not found")
            return
    
    # Get top genres by count
    genre_counts = df_genres['genre'].value_counts()
    top_genres = genre_counts[genre_counts >= 100].index.tolist()  # Only include genres with at least 100 albums
    print(f"Found {len(top_genres)} genres with at least 100 albums")
    
    # Filter for top genres
    df_top_genres = df_genres[df_genres['genre'].isin(top_genres)]
    
    # Calculate means, std, and count by genre and label type for SEM calculation
    genre_label_stats = df_top_genres.groupby(['genre', 'label_type'])[['permutation_entropy', 'statistical_complexity']].agg(['mean', 'std', 'count']).reset_index()
    genre_label_stats.columns = ['genre', 'label_type', 'entropy_mean', 'entropy_std', 'entropy_count', 'complexity_mean', 'complexity_std', 'complexity_count']
    
    # Filter out combinations with too few samples
    min_samples = 20
    genre_label_stats = genre_label_stats[genre_label_stats['entropy_count'] >= min_samples]
    
    # Apply custom genre ordering
    genre_label_stats = order_genres_custom(genre_label_stats)
    
    # Calculate SEM
    genre_label_stats['entropy_sem'] = genre_label_stats['entropy_std'] / np.sqrt(genre_label_stats['entropy_count'])
    genre_label_stats['complexity_sem'] = genre_label_stats['complexity_std'] / np.sqrt(genre_label_stats['complexity_count'])
    
    # Sort by custom genre order
    genre_label_stats = genre_label_stats.sort_values(by=['label_type', 'genre_order'])
    
    # Create plot
    fig, ax = plt.subplots(figsize=(16, 12))
    
    # Define colors for each label type
    label_colors = {'Independent': 'blue', 'Major': 'red'}
    
    # Create color maps for each label type
    independent_cmap = plt.cm.Blues
    major_cmap = plt.cm.Reds
    
    # Get unique genres for each label type
    independent_genres = genre_label_stats[genre_label_stats['label_type'] == 'Independent']['genre'].unique()
    major_genres = genre_label_stats[genre_label_stats['label_type'] == 'Major']['genre'].unique()
    
    # Create color gradients
    independent_colors = independent_cmap(np.linspace(0.3, 1, len(independent_genres)))
    major_colors = major_cmap(np.linspace(0.3, 1, len(major_genres)))
    
    # Create color mapping for each genre within each label type
    independent_color_map = dict(zip(independent_genres, independent_colors))
    major_color_map = dict(zip(major_genres, major_colors))
    
    # Plot with error bars
    for label_type in ['Independent', 'Major']:
        label_data = genre_label_stats[genre_label_stats['label_type'] == label_type]
        color_map = independent_color_map if label_type == 'Independent' else major_color_map
        
        for i, (_, row) in enumerate(label_data.iterrows()):
            genre = row['genre']
            color = color_map.get(genre, label_colors[label_type])
            
            ax.errorbar(row['entropy_mean'], 
                       row['complexity_mean'],
                       xerr=row['entropy_sem'],
                       yerr=row['complexity_sem'],
                       fmt='o', 
                       markersize=10,
                       capsize=2,
                       capthick=0.5,
                       elinewidth=0.5,
                       alpha=0.8,
                       color=color,
                       markeredgecolor='white',
                       markeredgewidth=0.5,
                       label=f"{label_type} - {genre}" if i == 0 else None)
    
    # Create text annotations
    texts = []
    for _, row in genre_label_stats.iterrows():
        # Create label with genre and label type
        label = f"{row['genre']} ({row['label_type'][0]})"  # Use first letter of label type (I/M)
        
        # Determine color based on label type
        color = label_colors[row['label_type']]
        
        text = ax.annotate(label, 
                          (row['entropy_mean'], row['complexity_mean']),
                          xytext=(0, 0), textcoords='offset points',
                          fontsize=9, fontweight='bold',
                          ha='center', va='center',
                          color=color)
        texts.append(text)
    
    
    # Add legend for label types
    handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=label_colors[label], 
                         markersize=10, label=label) for label in label_colors]
    ax.legend(handles=handles, loc='upper left', title='Label Type', frameon=True, framealpha=0.9)
    
    # Set axis labels and title
    ax.set_xlabel('Permutation Entropy', fontweight='bold', fontsize=14)
    ax.set_ylabel('Statistical Complexity', fontweight='bold', fontsize=14)
    ax.set_title('Musical Genres in Entropy-Complexity Space: Independent vs Major Labels\n(Mean Values ± SEM)', 
                fontweight='bold', fontsize=16, pad=20)
    
    # Enhanced styling
    ax.set_facecolor('#fafafa')
    ax.grid(True, alpha=0.4, linestyle='-', linewidth=0.5)
    
    # Add subtle border
    for spine in ax.spines.values():
        spine.set_edgecolor('#cccccc')
        spine.set_linewidth(1.5)
    
    plt.tight_layout()
    plt.savefig('../result/record_label/entropy_complexity_all_genres_by_label.png', dpi=300, bbox_inches='tight')
    plt.show()

def plot_complexity_comparison_dynamic_by_label(df, df_no_split, output_dir, min_albums=3000):
    """Plot comparing dynamic period complexity trends for independent vs major labels"""
    print("Creating comparison plot for complexity trends by label type (dynamic periods)...")
    
    # Create dynamic periods using df_no_split (consistent with other plots)
    df_no_split_with_periods = create_dynamic_periods(df_no_split, min_albums)
    
    # Create a mapping from year to period
    year_to_period = df_no_split_with_periods[['year', 'period']].drop_duplicates().set_index('year')['period'].to_dict()
    
    # Apply the period mapping to the genre-split dataframe
    df_with_periods = df.copy()
    df_with_periods['period'] = df_with_periods['year'].map(year_to_period)
    
    # Remove any rows where period mapping failed
    df_with_periods = df_with_periods.dropna(subset=['period'])
    
    # Calculate statistics by label type and period
    label_period_stats = df_with_periods.groupby(['label_type', 'period']).agg({
        'mdlc_complexity_score': 'mean',
        'album_group_mbid': 'count',
        'year': 'mean'  # Calculate average year for each period
    }).reset_index()
    
    # Rename columns for consistency
    label_period_stats.rename(columns={
        'album_group_mbid': 'album_count',
        'year': 'avg_year'  # Average year for plotting
    }, inplace=True)
    
    # Filter combinations with at least 50 albums per label-period
    label_period_stats = label_period_stats[label_period_stats['album_count'] >= 50]
    
    # Create the comparison plot
    fig, ax = plt.subplots(figsize=(15, 10))
    
    # Define colors for each label type
    label_colors = {'Independent': 'blue', 'Major': 'red'}
    
    # Get all unique periods and sort them chronologically
    all_periods = sorted(label_period_stats['period'].unique(), 
                        key=lambda x: int(x.split('-')[0]) if '-' in x else int(x))
    
    # Create x-axis positions for periods (categorical)
    period_positions = {period: i for i, period in enumerate(all_periods)}
    
    # Plot each label type's trend using period positions
    for label_type, color in label_colors.items():
        label_data = label_period_stats[label_period_stats['label_type'] == label_type]
        if len(label_data) < 2:  # Need at least 2 points to plot
            continue
        
        # Sort label data by period chronologically
        label_data = label_data.sort_values('avg_year')
        
        # Map periods to x-axis positions
        x_positions = [period_positions[period] for period in label_data['period']]
        
        # Calculate standard error for error bars
        label_data['sem'] = df_with_periods[df_with_periods['label_type'] == label_type].groupby('period')['mdlc_complexity_score'].sem().reset_index()['mdlc_complexity_score']
        
        # Plot with error bars
        ax.errorbar(x_positions, 
                   label_data['mdlc_complexity_score'],
                   yerr=label_data['sem'],
                   fmt='o-', 
                   linewidth=2.5, 
                   markersize=8,
                   capsize=5,
                   color=color, 
                   alpha=0.8,
                   label=f"{label_type}")
        
        # Add album count annotations
        for i, row in label_data.iterrows():
            ax.annotate(f"n={row['album_count']}", 
                       (period_positions[row['period']], row['mdlc_complexity_score']),
                       xytext=(0, 10 if label_type == 'Independent' else -25), 
                       textcoords='offset points',
                       ha='center', 
                       fontsize=8, 
                       color=color,
                       bbox=dict(boxstyle='round,pad=0.2', fc='white', alpha=0.7))
    
    ax.set_title(f'Complexity Trends: Independent vs Major Labels\n(Dynamic Periods, Min {min_albums} Albums ± SEM)', 
                fontsize=16, fontweight='bold')
    ax.set_xlabel('Period', fontsize=14)
    ax.set_ylabel('Average Complexity Score', fontsize=14)
    ax.grid(True, alpha=0.3)
    
    # Set x-axis to show period labels
    ax.set_xticks(range(len(all_periods)))
    ax.set_xticklabels(all_periods, rotation=45, ha='right')
    
    # Add legend
    ax.legend(fontsize=12, loc='best')
    
    # Set reasonable y-limits
    y_min = max(30, label_period_stats['mdlc_complexity_score'].min() - 2)
    y_max = min(45, label_period_stats['mdlc_complexity_score'].max() + 2)
    ax.set_ylim(y_min, y_max)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/complexity_independent_vs_major_dynamic_periods.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("Saved complexity comparison plot for independent vs major labels with dynamic periods")
    return label_period_stats

def plot_complexity_box_independent(df, output_dir):
    """Create box plots showing complexity distribution over time for independent labels only"""
    print("Creating box plots for temporal complexity distribution for independent labels...")
    
    # Filter for independent labels only
    df_independent = df[df['label_type'] == 'Independent'].copy()
    
    # Use the same grouping logic as quantile regression
    df_independent['year_group'] = (df_independent['year'] // 5) * 5
    
    # Filter periods with sufficient data (at least 20 albums)
    album_counts = df_independent.groupby('year_group').size()
    valid_periods = album_counts[album_counts >= 50].index
    df_filtered = df_independent[df_independent['year_group'].isin(valid_periods)]
    
    # Create figure with single subplot
    fig, ax = plt.subplots(figsize=(15, 8))
    
    # Prepare data for box plot by time period
    time_periods = sorted(valid_periods)
    time_data = [df_filtered[df_filtered['year_group'] == period]['mdlc_complexity_score'].values 
                 for period in time_periods]
    
    # Create time period labels with album counts
    period_counts = [album_counts[period] for period in time_periods]
    time_labels = [f"{int(period)}-{int(period)+4}\n(n={count})" for period, count in zip(time_periods, period_counts)]
    
    # Create box plot for time periods
    box_parts = ax.boxplot(time_data, 
                          positions=range(len(time_periods)),
                          patch_artist=True,
                          showmeans=True,
                          meanline=True,
                          widths=0.6)
    
    # Create color map based on album counts
    norm = plt.Normalize(min(period_counts), max(period_counts))
    cmap = plt.cm.Blues  # Blue colormap for independent labels
    
    # Color boxes with a gradient based on album count
    for i, (box, count) in enumerate(zip(box_parts['boxes'], period_counts)):
        color = cmap(norm(count))
        box.set_facecolor(color)
        box.set_alpha=0.7
        box.set_edgecolor('black')
        box.set_linewidth=1
    
    # Customize other elements
    for element in ['whiskers', 'fliers', 'medians', 'caps']:
        plt.setp(box_parts[element], color='black', linewidth=1)
    plt.setp(box_parts['means'], color='red', linewidth=2)

    ax.set_xticks(range(len(time_periods)))
    ax.set_xticklabels(time_labels, rotation=45, ha='right')
    ax.set_ylabel('Complexity Score', fontsize=12)
    ax.set_title('Complexity Distribution Evolution Over Time - Independent Labels\n(5-Year Periods)', 
                fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add legend to explain elements
    legend_elements = [
        plt.Line2D([0], [0], color='black', linewidth=2, label='Median'),
        plt.Line2D([0], [0], color='red', linewidth=2, label='Mean'),
    ]
    ax.legend(handles=legend_elements, bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
    
    # Add colorbar to show album count scale
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="2%", pad=0.2)
    cbar = plt.colorbar(sm, cax=cax)
    cbar.set_label('Number of Albums', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/complexity_box_independent_labels.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Print statistics
    print("\nIndependent Labels - Complexity Statistics by Period:")
    print("-" * 80)
    print(f"{'Period':<12} {'Count':<8} {'Mean':<8} {'Median':<8} {'Std':<8} {'Min':<8} {'Max':<8}")
    print("-" * 80)
    
    for period in time_periods:
        period_data = df_filtered[df_filtered['year_group'] == period]['mdlc_complexity_score']
        print(f"{f'{int(period)}-{int(period)+4}':<12} {len(period_data):<8} {period_data.mean():<8.2f} "
              f"{period_data.median():<8.2f} {period_data.std():<8.2f} {period_data.min():<8.2f} "
              f"{period_data.max():<8.2f}")
    
    return df_filtered

def plot_complexity_box_major(df, output_dir):
    """Create box plots showing complexity distribution over time for major labels only"""
    print("Creating box plots for temporal complexity distribution for major labels...")
    
    # Filter for major labels only
    df_major = df[df['label_type'] == 'Major'].copy()
    
    # Use the same grouping logic as quantile regression
    df_major['year_group'] = (df_major['year'] // 5) * 5
    
    # Filter periods with sufficient data (at least 20 albums)
    album_counts = df_major.groupby('year_group').size()
    valid_periods = album_counts[album_counts >= 50].index
    df_filtered = df_major[df_major['year_group'].isin(valid_periods)]
    
    # Create figure with single subplot
    fig, ax = plt.subplots(figsize=(15, 8))
    
    # Prepare data for box plot by time period
    time_periods = sorted(valid_periods)
    time_data = [df_filtered[df_filtered['year_group'] == period]['mdlc_complexity_score'].values 
                 for period in time_periods]
    
    # Create time period labels with album counts
    period_counts = [album_counts[period] for period in time_periods]
    time_labels = [f"{int(period)}-{int(period)+4}\n(n={count})" for period, count in zip(time_periods, period_counts)]
    
    # Create box plot for time periods
    box_parts = ax.boxplot(time_data, 
                          positions=range(len(time_periods)),
                          patch_artist=True,
                          showmeans=True,
                          meanline=True,
                          widths=0.6)
    
    # Create color map based on album counts
    norm = plt.Normalize(min(period_counts), max(period_counts))
    cmap = plt.cm.Reds  # Red colormap for major labels
    
    # Color boxes with a gradient based on album count
    for i, (box, count) in enumerate(zip(box_parts['boxes'], period_counts)):
        color = cmap(norm(count))
        box.set_facecolor(color)
        box.set_alpha=0.7
        box.set_edgecolor('black')
        box.set_linewidth=1
    
    # Customize other elements
    for element in ['whiskers', 'fliers', 'medians', 'caps']:
        plt.setp(box_parts[element], color='black', linewidth=1)
    plt.setp(box_parts['means'], color='red', linewidth=2)

    ax.set_xticks(range(len(time_periods)))
    ax.set_xticklabels(time_labels, rotation=45, ha='right')
    ax.set_ylabel('Complexity Score', fontsize=12)
    ax.set_title('Complexity Distribution Evolution Over Time - Major Labels\n(5-Year Periods)', 
                fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add legend to explain elements
    legend_elements = [
        plt.Line2D([0], [0], color='black', linewidth=2, label='Median'),
        plt.Line2D([0], [0], color='red', linewidth=2, label='Mean'),
    ]
    ax.legend(handles=legend_elements, bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
    
    # Add colorbar to show album count scale
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="2%", pad=0.2)
    cbar = plt.colorbar(sm, cax=cax)
    cbar.set_label('Number of Albums', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/complexity_box_major_labels.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Print statistics
    print("\nMajor Labels - Complexity Statistics by Period:")
    print("-" * 80)
    print(f"{'Period':<12} {'Count':<8} {'Mean':<8} {'Median':<8} {'Std':<8} {'Min':<8} {'Max':<8}")
    print("-" * 80)
    
    for period in time_periods:
        period_data = df_filtered[df_filtered['year_group'] == period]['mdlc_complexity_score']
        print(f"{f'{int(period)}-{int(period)+4}':<12} {len(period_data):<8} {period_data.mean():<8.2f} "
              f"{period_data.median():<8.2f} {period_data.std():<8.2f} {period_data.min():<8.2f} "
              f"{period_data.max():<8.2f}")
    
    return df_filtered

def num_label_independent_major():
    """Count the number of independent labels in the dataset"""
    billboard_df = pd.read_csv('./data/gemini/record_label_prediction/record_label_classifications.csv')
    mumu_msdi_df = pd.read_csv('./data/gemini/record_label_prediction/record_label_classifications_mumu_msdi.csv')
    df_independent = billboard_df[billboard_df['classification'] == 'independent'].copy()
    df_major = billboard_df[billboard_df['classification'] == 'major'].copy()
    df_independent_mumu_msdi = mumu_msdi_df[mumu_msdi_df['classification'] == 'independent'].copy()
    df_major_mumu_msdi = mumu_msdi_df[mumu_msdi_df['classification'] == 'major'].copy()
    other_billboard = billboard_df[billboard_df['classification'] != 'major' ].copy()
    other_billboard = other_billboard[other_billboard['classification'] != 'independent'].copy()
    other_mumu_msdi = mumu_msdi_df[mumu_msdi_df['classification'] != 'major'].copy()
    other_mumu_msdi = other_mumu_msdi[other_mumu_msdi['classification'] != 'independent'].copy()
    indipendent = len(df_independent) + len(df_independent_mumu_msdi)
    major = len(df_major) + len(df_major_mumu_msdi)
    print(f"Independent labels: {indipendent}")
    print(f"Major labels: {major}")
    print(f"Total labels: {indipendent + major}")
    other = len(other_billboard) + len(other_mumu_msdi)
    print(f"Other labels: {other_mumu_msdi}")
    print(f"Total labels: {indipendent + major + other}")
    return indipendent, major, other


def main():
    """Main function to create all plots"""
    print("Loading data...")
    # Load data with expanded genres
    df = load_and_process_data()
    # Load data without expanding genres
    df_no_split = load_and_process_data_no_split()
    
    print(f"Loaded {len(df_no_split)} albums with label classifications")
    print(f"Expanded to {len(df)} genre-album combinations")
    
    # Create output directory
    Path("../result/record_label").mkdir(parents=True, exist_ok=True)
    
    print("Creating entropy-complexity plots for independent vs major labels...")
    
    # # Plot 1: Dynamic periods comparison (uses non-expanded data)
    # print("\n--- Creating Plot 1: Dynamic periods comparison ---")
    # plot_entropy_complexity_by_dynamic_periods(df_no_split, min_albums=3000)
    
    # # Plot 2: Overall means comparison (uses non-expanded data)
    # print("\n--- Creating Plot 2: Overall means comparison ---")
    # plot_entropy_complexity_overall_means(df_no_split)
    
    # Plot 3: Genre-specific comparison (uses expanded data with genres)
    # print("\n--- Creating Plot 3: Genre-specific comparison ---")
    # plot_entropy_complexity_by_genre(df)
    
    # Plot 4: All genres in same plot by label type
    print("\n--- Creating Plot 4: All genres in same plot by label type ---")
    plot_entropy_complexity_all_genres_by_label(df)
    
    # Plot 5: Complexity trends by label type (dynamic periods)
    print("\n--- Creating Plot 5: Complexity trends by label type (dynamic periods) ---")
    plot_complexity_comparison_dynamic_by_label(df_no_split, df_no_split, "../result/record_label", min_albums=3000)
    
    # Plot 6: Complexity distribution for independent labels
    print("\n--- Creating Plot 6: Complexity distribution for independent labels ---")
    plot_complexity_box_independent(df_no_split, "../result/record_label")
    
    # Plot 7: Complexity distribution for major labels
    print("\n--- Creating Plot 7: Complexity distribution for major labels ---")
    plot_complexity_box_major(df_no_split, "../result/record_label")
    
    print("All plots saved successfully!")

if __name__ == "__main__":
    # main()
    num_label_independent_major()
