import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from pathlib import Path
import ast
import warnings
from adjustText import adjust_text

warnings.filterwarnings('ignore')
FONT_SIZE = 18
# Set style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")
plt.rcParams.update({
    'font.size': FONT_SIZE,
    'font.family': 'sans-serif',
    'axes.labelweight': 'bold',
    'axes.titleweight': 'bold',
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.grid': True,
    'grid.alpha': 0.3
})

def load_and_process_data_no_split():
    """Load and combine both entropy-complexity datasets - keep albums as single entities"""
    # Load datasets
    combined_df = pd.read_csv('../results/unified_album_dataset_with_complexity.csv')
    combined_df['year'] = pd.to_datetime(combined_df['release_date'], errors='raise', format='mixed', yearfirst=True).dt.year
    print("combined after year", len(combined_df))
    combined_df = combined_df.dropna(subset=['year'])
    print("combined after dropna", len(combined_df))
    combined_df['year'] = combined_df['year'].astype(int)
    combined_df = combined_df.drop_duplicates(subset=['album_group_mbid'])
    
    # Filter reasonable year range
    combined_df = combined_df[(combined_df['year'] >= 1950) & (combined_df['year'] <= 2025)]
    
    print(f"Dataset loaded (no split): {len(combined_df)} total albums")
    
    return combined_df

def load_and_process_data():
    """Load and combine both entropy-complexity datasets - split genres into individual rows"""
    combined_df = pd.read_csv('../results/unified_album_dataset_with_complexity.csv')
    combined_df['year'] = pd.to_datetime(combined_df['release_date'], errors='raise', format='mixed', yearfirst=True).dt.year
    combined_df = combined_df.dropna(subset=['year'])
    combined_df['year'] = combined_df['year'].astype(int)
    print("combined after concat", len(combined_df))
    combined_df = combined_df.drop_duplicates(subset=['album_group_mbid'])
    print("combined after drop duplicates", len(combined_df))
    # Filter reasonable year range
    combined_df = combined_df[(combined_df['year'] >= 1950) & (combined_df['year'] <= 2025)]
    
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
        except:
            continue
    
    return pd.DataFrame(processed_rows)

def create_5year_periods(df):
    """Create 5-year periods from years"""
    df['period'] = (df['year'] // 5) * 5
    return df

def plot_entropy_complexity_by_5years(df):
    """Plot 1: Mean entropy-complexity by 10-year periods"""
    df_periods = create_5year_periods(df.copy())
    
    # Calculate means, std, and count by period for SEM calculation
    period_stats = df_periods.groupby('period')[['permutation_entropy', 'statistical_complexity']].agg(['mean', 'std', 'count']).reset_index()
    period_stats.columns = ['period', 'entropy_mean', 'entropy_std', 'entropy_count', 'complexity_mean', 'complexity_std', 'complexity_count']
    period_stats = period_stats[period_stats['period'] >= 1960]  # Filter reasonable years
    
    # Calculate SEM (Standard Error of the Mean)
    period_stats['entropy_sem'] = period_stats['entropy_std'] / np.sqrt(period_stats['entropy_count'])
    period_stats['complexity_sem'] = period_stats['complexity_std'] / np.sqrt(period_stats['complexity_count'])
    
    fig, ax = plt.subplots(figsize=(12, 9))
    
    # Create color gradient for time periods
    colors = plt.cm.tab10(np.linspace(0, 1, len(period_stats)))
    
    # Plot with error bars
    for i, (_, row) in enumerate(period_stats.iterrows()):
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
                   markeredgewidth=0.5)
    
    # Add connecting line with gradient
    ax.plot(period_stats['entropy_mean'], 
           period_stats['complexity_mean'],
           '--', alpha=0.7, linewidth=2, color='gray', zorder=1)
    
    # Add period labels with better styling and varied positioning
    for i, (_, row) in enumerate(period_stats.iterrows()):
        # Vary annotation positions to avoid overlap
        angle = i * (360 / len(period_stats))  # Distribute around circle
        offset_x = 50 * np.cos(np.radians(angle))
        offset_y = 50 * np.sin(np.radians(angle))
        
        ax.annotate(f"{int(row['period'])}s", 
                   (row['entropy_mean'], row['complexity_mean']),
                   xytext=(offset_x, offset_y), textcoords='offset points', 
                   fontsize=10, fontweight='bold',
                   bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8, edgecolor='gray'),
                   ha='center',
                   arrowprops=dict(arrowstyle='-', color='gray', alpha=0.5, lw=1))
    
    ax.set_xlabel('Permutation Entropy', fontweight='bold')
    ax.set_ylabel('Statistical Complexity', fontweight='bold')
    ax.set_title('Evolution of Album Complexity Through Time\n(10-Year Periods ± SEM)', 
                fontweight='bold', pad=20)
    
    # Add subtle background styling
    ax.set_facecolor('#fafafa')
    ax.grid(True, alpha=0.4, linestyle='-', linewidth=0.5)
    
    plt.tight_layout()
    plt.savefig('../result/entr_compl_plots/entropy_complexity_5year_periods.png', dpi=300, bbox_inches='tight')
    plt.show()

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
    """Plot entropy-complexity by dynamic periods based on minimum album count"""
    df_periods = create_dynamic_periods(df.copy(), min_albums=min_albums)
    
    # Calculate means, std, and count by period for SEM calculation
    period_stats = df_periods.groupby('period')[['permutation_entropy', 'statistical_complexity']].agg(['mean', 'std', 'count']).reset_index()
    period_stats.columns = ['period', 'entropy_mean', 'entropy_std', 'entropy_count', 'complexity_mean', 'complexity_std', 'complexity_count']
    
    # Sort periods chronologically
    def period_sort_key(period):
        if '-' in period:
            return int(period.split('-')[0])
        else:
            return int(period)
    
    period_stats['sort_key'] = period_stats['period'].apply(period_sort_key)
    period_stats = period_stats.sort_values('sort_key').reset_index(drop=True)
    
    # Calculate SEM (Standard Error of the Mean)
    period_stats['entropy_sem'] = period_stats['entropy_std'] / np.sqrt(period_stats['entropy_count'])
    period_stats['complexity_sem'] = period_stats['complexity_std'] / np.sqrt(period_stats['complexity_count'])
    
    print("Dynamic periods created:")
    for _, row in period_stats.iterrows():
        print(f"  {row['period']}: {row['entropy_count']} albums")
    
    fig, ax = plt.subplots(figsize=(14, 10))
    
    # Create color gradient for time periods
    colors = plt.cm.viridis(np.linspace(0, 1, len(period_stats)))
    
    # Plot with error bars
    for i, (_, row) in enumerate(period_stats.iterrows()):
        ax.errorbar(row['entropy_mean'], 
                   row['complexity_mean'],
                   xerr=row['entropy_sem'],
                   yerr=row['complexity_sem'],
                   fmt='o', 
                   markersize=7,
                   capsize=2,
                   capthick=0.5,
                   elinewidth=0.5,
                   alpha=0.8,
                   color=colors[i],
                   markeredgecolor='white',
                   markeredgewidth=0.5)
    
    # Add connecting line with gradient
    ax.plot(period_stats['entropy_mean'], 
           period_stats['complexity_mean'],
           '--', alpha=0.7, linewidth=1, color='gray', zorder=1)
    
    # Add period labels with better styling and varied positioning
    for i, (_, row) in enumerate(period_stats.iterrows()):
        # Vary annotation positions to avoid overlap
        offset_x = 50
        offset_y = 50 

        if(row['period'] == '1990-1993' or row['period'] == '2006-2007' or row['period'] == '2008-2009'):
            offset_x = -50
            offset_y = -50
        elif(row['period'] == '1994-1996'):
            offset_x = 15
            offset_y = 50
        
        # Create label with album count
        label = f"{row['period']}"
        
        ax.annotate(label, 
                   (row['entropy_mean'], row['complexity_mean']),
                   xytext=(offset_x, offset_y), textcoords='offset points', 
                   ha='center', va='center',
                   arrowprops=dict(arrowstyle='-', color=colors[i], alpha=0.7, lw=0.5))
        

    ax.set_ylim(0.1125, 0.14)
    ax.set_xlim(0.82, 0.87)
    
    ax.set_xlabel('Permutation Entropy', fontweight='bold')
    ax.set_ylabel('Statistical Complexity', fontweight='bold')

    
    # Add subtle background styling
    ax.set_facecolor('#fafafa')
    ax.grid(True, alpha=0.4, linestyle='-', linewidth=0.5)
    
    
    plt.tight_layout()
    plt.savefig(f'../result/entr_compl_plots/entropy_complexity_dynamic_periods_{min_albums}.png', dpi=300, bbox_inches='tight')
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

def plot_entropy_complexity_total_genre_means(df):
    """Plot 3: Total mean for each genre with non-overlapping labels"""
    df_genres = process_genres(df.copy())
    
    # Calculate overall means, std, and count by genre for SEM calculation
    genre_stats = df_genres.groupby('genre')[['permutation_entropy', 'statistical_complexity']].agg(['mean', 'std', 'count']).reset_index()
    genre_stats.columns = ['genre', 'entropy_mean', 'entropy_std', 'entropy_count', 'complexity_mean', 'complexity_std', 'complexity_count']
    
    # Filter genres with sufficient data
    min_count = 50
    genre_stats = genre_stats[genre_stats['entropy_count'] >= min_count]
    
    # Apply custom genre ordering
    genre_stats = order_genres_custom(genre_stats)
    
    # Calculate SEM
    genre_stats['entropy_sem'] = genre_stats['entropy_std'] / np.sqrt(genre_stats['entropy_count'])
    genre_stats['complexity_sem'] = genre_stats['complexity_std'] / np.sqrt(genre_stats['complexity_count'])
    
    fig, ax = plt.subplots(figsize=(16, 12))  # Increased figure size
    
    # Sort by custom genre order instead of count
    genre_stats = genre_stats.sort_values(by='genre_order')
    
    # Create a consistent color palette based on the ordered genres
    colors = plt.cm.tab20(np.linspace(0, 1, len(genre_stats)))
    
    # Plot with error bars
    scatter_points = []
    for i, (_, row) in enumerate(genre_stats.iterrows()):
        point = ax.errorbar(row['entropy_mean'], 
                           row['complexity_mean'],
                           xerr=row['entropy_sem'],
                           yerr=row['complexity_sem'],
                           fmt='o', 
                           markersize=12,
                           capsize=2,
                           capthick=0.5,
                           elinewidth=0.5,
                           alpha=0.8,
                           color=colors[i],
                           markeredgecolor='white',
                           markeredgewidth=0.5)
        scatter_points.append(point)
    
    # Create text annotations for adjustText
    texts = []
    for i, (_, row) in enumerate(genre_stats.iterrows()):
        # Vary annotation positions to avoid overlap
        angle = i * (360 / len(genre_stats)) 
        if(row['genre'] == 'R&B'): # Distribute around circle
            angle = 250
        elif(row['genre'] == 'Pop'):
            angle = 120
        elif(row['genre'] == 'Speciality'):
            angle = 340
        elif(row['genre'] == 'Jazz & Blues'):
            angle = 120
        elif(row['genre'] == 'World music'):
            angle = 20
        elif(row['genre'] == 'Country & Folk'):
            angle = 290
        elif(row['genre'] == 'Rock'):
            angle = 250
        elif(row['genre'] == 'Classical'):
            angle = 120

        offset_x = 60 * np.cos(np.radians(angle))
        offset_y = 60 * np.sin(np.radians(angle))
        
        text = ax.annotate(row['genre'], 
                          (row['entropy_mean'], row['complexity_mean']),
                          xytext=(offset_x, offset_y), textcoords='offset points',
                          ha='center', va='center',
                          arrowprops=dict(arrowstyle='-', color=colors[i], alpha=0.6, lw=0.5))
        texts.append(text)
    
    # Use adjustText to prevent overlapping labels
    # adjust_text(texts, 
    #             arrowprops=dict(arrowstyle='-', color='gray', alpha=0.6, lw=1, 
    #                            shrinkA=5, shrinkB=5),  # Prevent arrows from striking through text
    #             expand_points=(10, 10),  # Expand around points
    #             expand_text=(1.2, 1.2),   # Expand around text
    #             force_points=10,         # Force away from points
    #             force_text=0.8,           # Force text away from each other
    #             ax=ax)
    
    
    ax.set_xlabel('Permutation Entropy', fontweight='bold')
    ax.set_ylabel('Statistical Complexity', fontweight='bold')
    
    ax.set_ylim(0.1125, 0.14)
    ax.set_xlim(0.82, 0.87)
    
    # Enhanced styling
    ax.set_facecolor('#fafafa')
    ax.grid(True, alpha=0.4, linestyle='-', linewidth=0.5)
    
    # Add subtle border
    for spine in ax.spines.values():
        spine.set_edgecolor('#cccccc')
        spine.set_linewidth(1.5)
    
    plt.tight_layout()
    plt.savefig('../result/entr_compl_plots/entropy_complexity_genre_means.png', dpi=300, bbox_inches='tight')
    plt.show()


def main():
    """Main function to create all plots"""
    print("Loading data...")
    df = load_and_process_data()
    print(f"Loaded {len(df)} albums")
    df_no_split = load_and_process_data_no_split()
    print(f"Loaded {len(df_no_split)} albums")
    
    # Create output directory
    Path("../result/entr_compl_plots").mkdir(parents=True, exist_ok=True)
    
    print("Creating entropy-complexity plots...")
    
    # Plot 1: 5-year periods
    # plot_entropy_complexity_by_5years(df_no_split)
    
    # Plot 2: Genres by 5-year periods  
    # plot_entropy_complexity_genres_by_5years(df)
    
    # Plot 3: Total genre means
    plot_entropy_complexity_total_genre_means(df)
    
    # Plot 4: Dynamic periods
    plot_entropy_complexity_by_dynamic_periods(df_no_split)
    
    print("All plots saved successfully!")

if __name__ == "__main__":
    main()
