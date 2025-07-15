import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import os
import logging
import ast

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def analyze_dataset():
    # Define file paths
    base_dir = os.path.dirname(os.path.abspath(__file__))
    input_path = os.path.join(base_dir, 'mumu_msdi_final_with_complexity.csv')
    output_dir = os.path.join(base_dir, 'analysis_results')
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Read the dataset
    logging.info("Reading dataset...")
    df = pd.read_csv(input_path)
    
    # Extract year from release_date
    df['year'] = pd.to_datetime(df['release_date'], errors='coerce').dt.year
    
    # Count by genre
    logging.info("Analyzing genre distribution...")
    genre_counts = []
    for idx, row in df.iterrows():
        if pd.isna(row['genres']) or not row['genres']:
            continue
        # Handle genres as string representation of list
        try:
            if isinstance(row['genres'], str):
                # Parse string representation of list
                genres = ast.literal_eval(row['genres'])
            else:
                genres = row['genres']
        except (ValueError, SyntaxError):
            # Fallback to comma-separated parsing if ast fails
            genres = row['genres'].split(',') if isinstance(row['genres'], str) else [row['genres']]
        
        if isinstance(genres, list):
            for genre in genres:
                genre_counts.append(str(genre).strip())
        else:
            genre_counts.append(str(genres).strip())
    
    genre_dist = pd.Series(genre_counts).value_counts()
    
    # Plot genre distribution
    plt.figure(figsize=(15, 8))
    genre_dist.plot(kind='bar')
    plt.title('Genres Distribution')
    plt.xlabel('Genre')
    plt.ylabel('Count')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'genre_distribution.png'))
    plt.close()
    
    # Save genre distribution to CSV
    genre_dist.to_csv(os.path.join(output_dir, 'genre_distribution.csv'))
    
    # Count by year
    logging.info("Analyzing year distribution...")
    year_dist = df['year'].value_counts().sort_index()
    year_dist = year_dist[year_dist.index.notnull()]  # Remove NaN years
    
    # Plot year distribution
    plt.figure(figsize=(15, 8))
    year_dist.plot(kind='line')
    plt.title('Albums Distribution by Year')
    plt.xlabel('Year')
    plt.ylabel('Count')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'year_distribution.png'))
    plt.close()
    
    # Save year distribution to CSV
    year_dist.to_csv(os.path.join(output_dir, 'year_distribution.csv'))
    
    # Count by genre and year
    logging.info("Analyzing genre-year distribution...")
    genre_year_counts = []
    for idx, row in df.iterrows():
        if pd.isna(row['genres']) or pd.isna(row['year']) or not row['genres']:
            continue
        # Handle genres as string representation of list
        try:
            if isinstance(row['genres'], str):
                # Parse string representation of list
                genres = ast.literal_eval(row['genres'])
            else:
                genres = row['genres']
        except (ValueError, SyntaxError):
            # Fallback to comma-separated parsing if ast fails
            genres = row['genres'].split(',') if isinstance(row['genres'], str) else [row['genres']]
        
        if isinstance(genres, list):
            for genre in genres:
                genre_year_counts.append((str(genre).strip(), row['year']))
        else:
            genre_year_counts.append((str(genres).strip(), row['year']))
    
    genre_year_df = pd.DataFrame(genre_year_counts, columns=['genre', 'year'])
    genre_year_dist = genre_year_df.groupby(['genre', 'year']).size().reset_index(name='count')
    
    # Create heatmap for top genres over years
    top_genres = genre_dist.head(10).index
    heatmap_data = genre_year_dist[genre_year_dist['genre'].isin(top_genres)]
    heatmap_pivot = heatmap_data.pivot(index='genre', columns='year', values='count').fillna(0)
    
    plt.figure(figsize=(20, 10))
    sns.heatmap(heatmap_pivot, cmap='YlOrRd')
    plt.title('Top 10 Genres Distribution Over Years')
    plt.xlabel('Year')
    plt.ylabel('Genre')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'genre_year_heatmap.png'))
    plt.close()
    
    # Save genre-year distribution to CSV
    genre_year_dist.to_csv(os.path.join(output_dir, 'genre_year_distribution.csv'), index=False)
    
    # Print summary statistics
    logging.info("\nDataset Summary:")
    logging.info(f"Total number of albums: {len(df)}")
    logging.info(f"Number of unique genres: {len(genre_dist)}")
    logging.info(f"Year range: {year_dist.index.min()} to {year_dist.index.max()}")
    logging.info("\nTop 10 genres:")
    for genre, count in genre_dist.head(10).items():
        logging.info(f"{genre}: {count}")

if __name__ == "__main__":
    analyze_dataset()
