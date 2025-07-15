#!/usr/bin/env python3
"""
Run YOLO object detection analysis on album covers.
This script provides easy-to-use functions for analyzing objects in album artwork.
"""

import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import json
from collections import Counter
from album_object_detector import AlbumObjectDetector

def setup_directories():
    """Create necessary directories for output."""
    os.makedirs("../results", exist_ok=True)
    os.makedirs("object_analysis_results", exist_ok=True)
    os.makedirs("object_analysis_results/plots", exist_ok=True)
    os.makedirs("object_analysis_results/annotated_images", exist_ok=True)

def test_single_image(image_path: str):
    """Test object detection on a single image."""
    print(f"Testing object detection on: {image_path}")
    
    detector = AlbumObjectDetector(confidence_threshold=0.25)
    
    # Detect objects and save annotated image
    result = detector.detect_objects(
        image_path, 
        save_annotated=True, 
        output_path="object_analysis_results/test_annotated.jpg"
    )
    
    print(f"Found {result['num_objects']} objects:")
    for detection in result['detections']:
        print(f"  - {detection['class_name']}: {detection['confidence']:.2f} confidence")
    
    return result

def analyze_sample_dataset(sample_size: int = 100):
    """Analyze a sample of the album cover dataset."""
    print(f"Analyzing sample of {sample_size} album covers...")
    
    detector = AlbumObjectDetector(confidence_threshold=0.25)
    
    # File paths
    csv_file = "../data/billboard_album_final_super_genres_with_images.csv"
    image_directory = "../data/img_all"
    output_file = "object_analysis_results/sample_object_detection_results.csv"
    
    # Check if files exist
    if not os.path.exists(csv_file):
        print(f"Error: CSV file not found at {csv_file}")
        print("Available CSV files:")
        data_dir = "../data"
        if os.path.exists(data_dir):
            csv_files = [f for f in os.listdir(data_dir) if f.endswith('.csv')]
            for f in csv_files:
                print(f"  - {f}")
        return None
    
    if not os.path.exists(image_directory):
        print(f"Error: Image directory not found at {image_directory}")
        return None
    
    # Run analysis
    results_df = detector.analyze_dataset(
        csv_file=csv_file,
        image_directory=image_directory,
        output_file=output_file,
        max_images=sample_size
    )
    
    return results_df

def generate_basic_visualizations(results_df: pd.DataFrame):
    """Generate basic visualizations from object detection results."""
    print("Generating visualizations...")
    
    # 1. Distribution of objects per album
    plt.figure(figsize=(12, 8))
    
    plt.subplot(2, 2, 1)
    plt.hist(results_df['num_objects_detected'], bins=20, alpha=0.7, color='steelblue')
    plt.title('Distribution of Objects per Album')
    plt.xlabel('Number of Objects')
    plt.ylabel('Number of Albums')
    
    # 2. Albums with vs without objects
    plt.subplot(2, 2, 2)
    has_objects = (results_df['num_objects_detected'] > 0).sum()
    no_objects = (results_df['num_objects_detected'] == 0).sum()
    plt.pie([has_objects, no_objects], 
           labels=['With Objects', 'No Objects'], 
           autopct='%1.1f%%',
           colors=['lightgreen', 'lightcoral'])
    plt.title('Albums with Detected Objects')
    
    # 3. Object type presence
    plt.subplot(2, 2, 3)
    object_types = ['has_person', 'has_vehicle', 'has_animal']
    type_counts = [results_df[col].sum() for col in object_types if col in results_df.columns]
    type_labels = [col.replace('has_', '').title() for col in object_types if col in results_df.columns]
    
    if type_counts:
        plt.bar(type_labels, type_counts, color=['red', 'blue', 'orange'])
        plt.title('Presence of Object Categories')
        plt.ylabel('Number of Albums')
    
    # 4. Most common object classes
    plt.subplot(2, 2, 4)
    all_classes = Counter()
    for idx, row in results_df.iterrows():
        if pd.notna(row['object_classes']) and row['object_classes'] != '{}':
            try:
                classes = json.loads(row['object_classes'])
                all_classes.update(classes)
            except:
                continue
    
    if all_classes:
        top_classes = all_classes.most_common(10)
        class_names, counts = zip(*top_classes)
        plt.barh(range(len(class_names)), counts, color='steelblue')
        plt.yticks(range(len(class_names)), class_names)
        plt.title('Top 10 Most Detected Objects')
        plt.xlabel('Number of Detections')
    
    plt.tight_layout()
    plt.savefig('object_analysis_results/plots/object_detection_overview.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("Visualizations saved to object_analysis_results/plots/")

def analyze_by_genre(results_df: pd.DataFrame):
    """Analyze object detection results by music genre."""
    if 'super_genre' not in results_df.columns:
        print("No genre information available for analysis")
        return
    
    print("Analyzing object detection by genre...")
    
    # Calculate genre statistics
    genre_stats = results_df.groupby('super_genre').agg({
        'num_objects_detected': ['mean', 'sum', 'count'],
        'has_person': 'sum',
        'has_vehicle': 'sum',
        'has_animal': 'sum'
    }).round(2)
    
    # Flatten column names
    genre_stats.columns = ['avg_objects', 'total_objects', 'album_count', 'person_count', 'vehicle_count', 'animal_count']
    genre_stats = genre_stats.reset_index()
    
    print("\nObject Detection by Genre:")
    print(genre_stats.to_string(index=False))
    
    # Visualize genre analysis
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Average objects per album by genre
    axes[0, 0].bar(genre_stats['super_genre'], genre_stats['avg_objects'], color='steelblue')
    axes[0, 0].set_title('Average Objects per Album by Genre')
    axes[0, 0].set_ylabel('Average Number of Objects')
    axes[0, 0].tick_params(axis='x', rotation=45)
    
    # Total objects by genre
    axes[0, 1].bar(genre_stats['super_genre'], genre_stats['total_objects'], color='green')
    axes[0, 1].set_title('Total Objects Detected by Genre')
    axes[0, 1].set_ylabel('Total Number of Objects')
    axes[0, 1].tick_params(axis='x', rotation=45)
    
    # Object types by genre (stacked bar)
    width = 0.8
    x_pos = range(len(genre_stats))
    
    axes[1, 0].bar(x_pos, genre_stats['person_count'], width, label='Person', color='red')
    axes[1, 0].bar(x_pos, genre_stats['vehicle_count'], width, 
                   bottom=genre_stats['person_count'], label='Vehicle', color='blue')
    axes[1, 0].bar(x_pos, genre_stats['animal_count'], width,
                   bottom=genre_stats['person_count'] + genre_stats['vehicle_count'], 
                   label='Animal', color='orange')
    
    axes[1, 0].set_title('Object Types by Genre (Stacked)')
    axes[1, 0].set_ylabel('Number of Albums')
    axes[1, 0].set_xticks(x_pos)
    axes[1, 0].set_xticklabels(genre_stats['super_genre'], rotation=45)
    axes[1, 0].legend()
    
    # Heatmap of object presence by genre
    object_cols = ['has_person', 'has_vehicle', 'has_animal']
    available_cols = [col for col in object_cols if col in results_df.columns]
    
    if available_cols:
        heatmap_data = results_df.groupby('super_genre')[available_cols].mean()
        sns.heatmap(heatmap_data.T, annot=True, cmap='YlOrRd', ax=axes[1, 1])
        axes[1, 1].set_title('Object Presence Rate by Genre')
        axes[1, 1].set_ylabel('Object Type')
    
    plt.tight_layout()
    plt.savefig('object_analysis_results/plots/objects_by_genre.png', dpi=300, bbox_inches='tight')
    plt.show()

def main():
    """Main function to run object detection analysis."""
    print("Album Cover Object Detection Analysis")
    print("=" * 40)
    
    # Setup directories
    setup_directories()
    
    # Check if ultralytics is installed
    try:
        import ultralytics
        print("✓ ultralytics is installed")
    except ImportError:
        print("✗ ultralytics not found. Installing...")
        os.system("pip install ultralytics")
    
    # Option 1: Test on a single image
    print("\n1. Testing on a single image...")
    image_files = []
    img_dir = "../data/img_all"
    if os.path.exists(img_dir):
        image_files = [f for f in os.listdir(img_dir) if f.lower().endswith(('.jpg', '.png'))]
    
    if image_files:
        test_image = os.path.join(img_dir, image_files[0])
        print(f"Testing with: {test_image}")
        try:
            result = test_single_image(test_image)
            print("✓ Single image test successful")
        except Exception as e:
            print(f"✗ Single image test failed: {e}")
            return
    else:
        print("No images found for testing")
        return
    
    # Option 2: Analyze sample dataset
    print("\n2. Analyzing sample dataset...")
    try:
        results_df = analyze_sample_dataset(sample_size=50)  # Start with small sample
        if results_df is not None:
            print("✓ Sample analysis successful")
            
            # Generate visualizations
            generate_basic_visualizations(results_df)
            
            # Analyze by genre if available
            analyze_by_genre(results_df)
            
            print(f"\nSummary:")
            print(f"- Analyzed {len(results_df)} albums")
            print(f"- Found objects in {(results_df['num_objects_detected'] > 0).sum()} albums")
            print(f"- Average objects per album: {results_df['num_objects_detected'].mean():.2f}")
            
        else:
            print("✗ Sample analysis failed")
    except Exception as e:
        print(f"✗ Sample analysis failed: {e}")

if __name__ == "__main__":
    main() 