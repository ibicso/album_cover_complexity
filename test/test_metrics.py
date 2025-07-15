from PIL import Image
import numpy as np
import os
import pandas as pd
import sys
import matplotlib.pyplot as plt
from tqdm import tqdm
import time
import importlib.util
import zipfile
import io

# Add parent directory to path to import modules
sys.path.append('..')
from measure_complexity import ComplexityMeasurer
from album_analisys.album_cover_analyzer import AlbumCoverAnalyzer

# Import entropy/complexity functions directly since we can't import from a module with a dash in the name
# We'll import the necessary functions manually
sys.path.append('../album_analisys')
from itertools import permutations

# Path to test images CSV
TEST_IMAGES_CSV = "test_images_paths.csv"
OUTPUT_CSV = "test_metrics_results.csv"

# --- Load functions from modules with problematic names ---

# Load entropy-complexity functions
def image_to_grayscale_matrix(image_path):
    """Convert image to grayscale numpy array"""
    img = Image.open(image_path).convert("L")
    return np.array(img)

def ordinal_pattern_distribution(matrix, dx=2, dy=2):
    """Calculate ordinal pattern distribution of a matrix"""
    patterns = list(permutations(range(dx * dy)))
    pattern_counts = {p: 0 for p in patterns}
    num_total_patterns = len(patterns)

    if matrix.shape[0] < dx or matrix.shape[1] < dy:
        # Not enough data to form any patches
        return np.zeros(num_total_patterns)

    for i in range(matrix.shape[0] - dx + 1):
        for j in range(matrix.shape[1] - dy + 1):
            patch = matrix[i:i+dx, j:j+dy].flatten()
            ranks = tuple(np.argsort(patch))
            pattern_counts[ranks] += 1

    total = sum(pattern_counts.values())
    if total == 0:
        return np.zeros(num_total_patterns)
        
    probs = np.array([count / total for count in pattern_counts.values()])
    return probs

def shannon_entropy(p):
    """Calculate Shannon entropy of a probability distribution"""
    p = p[p > 0]
    return -np.sum(p * np.log(p))

def permutation_entropy(probs):
    """Calculate normalized permutation entropy"""
    n_total_patterns = len(probs)  # n = (dx*dy)!

    if n_total_patterns <= 1:
        return 0.0

    probs_filtered = probs[probs > 0]  # use only non-zero probabilities

    if len(probs_filtered) == 0:  # All patterns have zero probability (empty/too small image)
        return 0.0
    
    normalized_entropy = shannon_entropy(probs_filtered) / np.log(n_total_patterns)
    return normalized_entropy

def statistical_complexity(probs):
    """Calculate statistical complexity measure"""
    uniform = np.ones_like(probs) / len(probs)
    M = 0.5 * (probs + uniform)
    D_JS = shannon_entropy(M) - 0.5 * (shannon_entropy(probs) + shannon_entropy(uniform))
    D_max = -0.5 * ((len(probs)+1)/len(probs)*np.log(len(probs)+1) + np.log(len(probs)) - 2*np.log(2*len(probs)))
    H = permutation_entropy(probs)
    C = D_JS * H / D_max
    return C

# Load ZIP compression functions
def image_to_bitmap_bytes(image_path, target_size=None, preserve_aspect_ratio=True):
    """Convert image to uncompressed bitmap bytes for compression analysis"""
    with Image.open(image_path) as img:
        # Convert to RGB to ensure consistent format
        img = img.convert("RGB")
        
        if target_size is not None:
            target_width, target_height = target_size
            
            if preserve_aspect_ratio:
                # Calculate scaling to fit within target size while preserving aspect ratio
                img_width, img_height = img.size
                scale = min(target_width / img_width, target_height / img_height)
                
                # Resize maintaining aspect ratio
                new_width = int(img_width * scale)
                new_height = int(img_height * scale)
                img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
                
                # Create black canvas of target size and paste resized image
                canvas = Image.new("RGB", target_size, (0, 0, 0))
                offset_x = (target_width - new_width) // 2
                offset_y = (target_height - new_height) // 2
                canvas.paste(img, (offset_x, offset_y))
                img = canvas
            else:
                # Simple resize without preserving aspect ratio
                img = img.resize(target_size, Image.Resampling.LANCZOS)
        
        # Convert to raw bytes (RGB format)
        return img.tobytes()

def calculate_compression_ratio(data_bytes):
    """Calculate ZIP compression ratio for raw data bytes"""
    original_size = len(data_bytes)
    if original_size == 0:
        return np.nan
    
    # Create a temporary in-memory zip file
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, 'w', zipfile.ZIP_DEFLATED, compresslevel=9) as zip_file:
        zip_file.writestr('data', data_bytes)
    
    compressed_size = buffer.tell()
    compression_ratio = compressed_size / original_size
    
    return compression_ratio

def calculate_compression_complexity(image_path, target_size=None, preserve_aspect_ratio=True):
    """Calculate complexity based on ZIP compression of bitmap data"""
    try:
        # Convert image to uncompressed bitmap bytes
        bitmap_bytes = image_to_bitmap_bytes(image_path, target_size, preserve_aspect_ratio)
        
        # Calculate compression ratio
        compression_ratio = calculate_compression_ratio(bitmap_bytes)
        
        return compression_ratio
        
    except Exception as e:
        return np.nan

# --- Metric 1: Album Cover Analyzer (ComplexityMeasurer) ---

def calculate_album_analyzer_complexity(image_path):
    """
    Calculate complexity using the real AlbumCoverAnalyzer implementation
    """
    try:
        # Create analyzer instance
        analyzer = AlbumCoverAnalyzer(image_path=image_path, target_size=(224, 224))
        
        # Calculate complexity scores
        scores = analyzer.analyze_complexity()
        
        # Get overall score
        overall_score = analyzer.get_overall_score()
        
        return scores, overall_score
    except Exception as e:
        print(f"Error processing {image_path} with AlbumCoverAnalyzer: {e}")
        return None, np.nan

# --- Metric 2: Permutation Entropy and Statistical Complexity ---

def calculate_entropy_complexity(image_path):
    """Calculate permutation entropy and statistical complexity for an image"""
    try:
        matrix = image_to_grayscale_matrix(image_path)
        probs = ordinal_pattern_distribution(matrix, dx=2, dy=2)
        entropy = permutation_entropy(probs)
        complexity = statistical_complexity(probs)
        return entropy, complexity
    except Exception as e:
        print(f"Error processing {image_path} for entropy/complexity: {e}")
        return np.nan, np.nan

# --- Metric 3: ZIP Compression Ratio ---

def calculate_zip_complexity(image_path, target_size=(224, 224)):
    """Calculate complexity based on ZIP compression ratio"""
    try:
        return calculate_compression_complexity(image_path, target_size, preserve_aspect_ratio=True)
    except Exception as e:
        print(f"Error processing {image_path} for ZIP complexity: {e}")
        return np.nan

# --- Main Testing Function ---

def test_metrics_on_images():
    """Test all three complexity metrics on the test images"""
    # Load test images CSV
    df = pd.read_csv(TEST_IMAGES_CSV)
    
    # Initialize results
    results = []
    
    # Process each image
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Processing images"):
        image_path = row['image_path']
        image_name = row['image_name']
        
        # Skip if image doesn't exist
        if not os.path.exists(image_path):
            print(f"Warning: Image {image_path} not found. Skipping.")
            continue
        
        print(f"\nProcessing image: {image_name}")
        
        # Start timing
        start_time = time.time()
        
        # Metric 1: Album Cover Analyzer
        print("  Calculating AlbumCoverAnalyzer complexity...")
        complexity_scores, overall_score = calculate_album_analyzer_complexity(image_path)
        
        # Metric 2: Permutation Entropy and Statistical Complexity
        print("  Calculating Permutation Entropy and Statistical Complexity...")
        entropy, stat_complexity = calculate_entropy_complexity(image_path)
        
        # Metric 3: ZIP Compression Ratio
        print("  Calculating ZIP Compression Ratio...")
        zip_ratio = calculate_zip_complexity(image_path)
        
        # Calculate processing time
        process_time = time.time() - start_time
        print(f"  Completed in {process_time:.3f} seconds")
        
        # Create result dictionary
        result = {
            'image_path': image_path,
            'image_name': image_name,
            'permutation_entropy': entropy,
            'statistical_complexity': stat_complexity,
            'zip_compression_ratio': zip_ratio,
            'mdl_complexity': overall_score
        }
        
        # Add to results
        results.append(result)
        
        # Print summary for this image
        print(f"  Results for {image_name}:")
        print(f"    Permutation Entropy: {entropy:.4f}")
        print(f"    Statistical Complexity: {stat_complexity:.4f}")
        print(f"    ZIP Compression Ratio: {zip_ratio:.4f}")
        print(f"    MDL Complexity: {overall_score:.4f}")
    
    # Convert to DataFrame
    results_df = pd.DataFrame(results)
    
    # Save to CSV
    results_df.to_csv(OUTPUT_CSV, index=False)
    print(f"\nResults saved to {OUTPUT_CSV}")
    
    return results_df


def create_rand_noise_img():
    img = np.random.rand(224,224,3)
    img = (img * 255).astype(np.uint8)
    img = Image.fromarray(img)
    img.save('./test_images/rand_noise.png')

if __name__ == "__main__":
    print("Testing complexity metrics on test images...")
    results_df = test_metrics_on_images()
    
    # Display summary statistics
    print("\nSummary Statistics:")
    print(results_df.describe())
    
    # Plot comparison
    plot_metrics_comparison(results_df)
    
    print("\nDone!")
