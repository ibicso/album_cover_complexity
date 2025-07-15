import cv2
import numpy as np
import pandas as pd
import os
import time
from pathlib import Path
import json
from typing import Dict, List, Tuple, Optional
from collections import Counter, defaultdict
from tqdm import tqdm
import multiprocessing as mp
from functools import partial

class AlbumObjectDetector:
    """
    YOLO-based object detection system for album covers.
    Detects objects in album artwork and provides analysis capabilities.
    """
    
    def __init__(self, model_name='yolov8n.pt', confidence_threshold=0.25):
        """
        Initialize the object detector.
        
        Args:
            model_name: YOLO model to use (yolov8n.pt, yolov8s.pt, yolov8m.pt, yolov8l.pt, yolov8x.pt)
            confidence_threshold: Minimum confidence for object detection
        """
        self.model_name = model_name
        self.confidence_threshold = confidence_threshold
        self.model = None
        self.class_names = None
        self._load_model()
        
    def _load_model(self):
        """Load the YOLO model and get class names."""
        try:
            print(f"Loading YOLO model: {self.model_name}")
            # Try importing ultralytics
            try:
                from ultralytics import YOLO
            except ImportError as e:
                print("Error: ultralytics module not found.")
                print("Please install it with: pip install ultralytics")
                print("Or try reinstalling: pip install --upgrade ultralytics")
                raise ImportError(f"ultralytics not available: {e}")
            
            self.model = YOLO(self.model_name)
            
            # Get class names from the model
            self.class_names = self.model.names
            print(f"Model loaded successfully. Available classes: {len(self.class_names)}")
            
        except Exception as e:
            print(f"Error loading YOLO model: {e}")
            raise
    
    def detect_objects(self, image_path: str, save_annotated: bool = False, 
                      output_path: Optional[str] = None) -> Dict:
        """
        Detect objects in a single album cover image.
        
        Args:
            image_path: Path to the album cover image
            save_annotated: Whether to save annotated image
            output_path: Path to save annotated image
            
        Returns:
            Dictionary containing detection results
        """
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")
        
        try:
            # Run detection
            results = self.model(image_path, conf=self.confidence_threshold, verbose=False)
            
            # Extract information from results
            detection_data = self._extract_detection_data(results[0], image_path)
            
            # Save annotated image if requested
            if save_annotated:
                self._save_annotated_image(results[0], image_path, output_path)
            
            return detection_data
            
        except Exception as e:
            print(f"Error detecting objects in {image_path}: {e}")
            return self._empty_detection_result(image_path)
    
    def _extract_detection_data(self, result, image_path: str) -> Dict:
        """Extract structured data from YOLO detection results."""
        
        # Get image dimensions
        img_height, img_width = result.orig_shape
        
        detections = []
        class_counts = Counter()
        total_area_covered = 0
        
        if result.boxes is not None and len(result.boxes) > 0:
            boxes = result.boxes.xyxy.cpu().numpy()
            confidences = result.boxes.conf.cpu().numpy()
            class_ids = result.boxes.cls.cpu().numpy()
            
            for i, (box, conf, cls_id) in enumerate(zip(boxes, confidences, class_ids)):
                x1, y1, x2, y2 = box
                width = x2 - x1
                height = y2 - y1
                area = width * height
                area_percentage = (area / (img_width * img_height)) * 100
                
                class_name = self.class_names[int(cls_id)]
                class_counts[class_name] += 1
                total_area_covered += area_percentage
                
                detection = {
                    'class_id': int(cls_id),
                    'class_name': class_name,
                    'confidence': float(conf),
                    'bbox': [float(x1), float(y1), float(x2), float(y2)],
                    'area_percentage': float(area_percentage),
                    'center_x': float((x1 + x2) / 2),
                    'center_y': float((y1 + y2) / 2)
                }
                detections.append(detection)
        
        return {
            'image_path': image_path,
            'image_id': os.path.basename(image_path).replace('.jpg', '').replace('.png', ''),
            'num_objects': len(detections),
            'detections': detections,
            'class_counts': dict(class_counts),
            'total_area_covered': float(total_area_covered),
            'image_dimensions': {'width': int(img_width), 'height': int(img_height)},
            'processing_timestamp': time.time()
        }
    
    def _empty_detection_result(self, image_path: str) -> Dict:
        """Return empty detection result for failed detections."""
        return {
            'image_path': image_path,
            'image_id': os.path.basename(image_path).replace('.jpg', '').replace('.png', ''),
            'num_objects': 0,
            'detections': [],
            'class_counts': {},
            'total_area_covered': 0.0,
            'image_dimensions': {'width': 0, 'height': 0},
            'processing_timestamp': time.time(),
            'error': True
        }
    
    def _save_annotated_image(self, result, original_path: str, output_path: Optional[str] = None):
        """Save image with object detection annotations."""
        if output_path is None:
            output_dir = os.path.dirname(original_path)
            filename = os.path.basename(original_path)
            name, ext = os.path.splitext(filename)
            output_path = os.path.join(output_dir, f"{name}_annotated{ext}")
        
        # Get annotated image from YOLO result
        annotated_img = result.plot()
        
        # Save the image
        cv2.imwrite(output_path, annotated_img)
    
    def analyze_dataset(self, csv_file: str, image_directory: str, 
                       output_file: str, max_images: Optional[int] = None,
                       save_progress: bool = True, num_processes: Optional[int] = None) -> pd.DataFrame:
        """
        Analyze a dataset of album covers for object detection.
        
        Args:
            csv_file: Path to CSV file with album data
            image_directory: Directory containing album cover images
            output_file: Path for output CSV file with object detection results
            max_images: Maximum number of images to process (None for all)
            save_progress: Whether to save progress periodically
            num_processes: Number of parallel processes (None for auto-detect)
            
        Returns:
            DataFrame with object detection results
        """
        print(f"Loading album data from {csv_file}")
        df = pd.read_csv(csv_file)
        
        if max_images:
            df = df.head(max_images)
            print(f"Limited to {max_images} albums for analysis")
        
        print(f"Analyzing {len(df)} albums...")
        
        # Set up multiprocessing
        if num_processes is None:
            num_processes = min(mp.cpu_count()-1, 7)  # Limit to 7 to avoid memory issues
        
        print(f"Using {num_processes} parallel processes")
        
        start_time = time.time()
        
        # Prepare arguments for multiprocessing
        args_list = [
            (row, image_directory, self.model_name, self.confidence_threshold, start_time)
            for _, row in df.iterrows()
        ]
        
        # Process albums in parallel with progress bar
        results = []
        processed_count = 0
        
        with mp.Pool(processes=num_processes) as pool:
            # Use imap for progress tracking
            with tqdm(total=len(df), desc="Processing albums", unit="album") as pbar:
                for result in pool.imap(process_single_album, args_list):
                    if result is not None:
                        results.append(result)
                        processed_count += 1
                    
                    pbar.update(1)
                    pbar.set_postfix({
                        'processed': processed_count,
                        'rate': f"{processed_count/(time.time()-start_time):.1f}/s"
                    })
                    
                    # Save progress periodically
                    if save_progress and processed_count % 5000 == 0 and processed_count > 0:
                        temp_df = pd.DataFrame(results)
                        temp_output = output_file.replace('.csv', f'_progress_{processed_count}.csv')
                        temp_df.to_csv(temp_output, index=False)
        
        # Create final DataFrame and save
        results_df = pd.DataFrame(results)
        results_df.to_csv(output_file, index=False)
        
        elapsed_time = time.time() - start_time
        print(f"\nAnalysis complete!")
        print(f"Processed {processed_count} albums in {elapsed_time/60:.1f} minutes")
        print(f"Results saved to {output_file}")
        
        return results_df


def process_single_album(args):
    """
    Worker function for multiprocessing. Process a single album.
    
    Args:
        args: Tuple containing (row_data, image_directory, model_name, confidence_threshold, start_time)
    
    Returns:
        Dictionary with processed album data or None if failed
    """
    row, image_directory, model_name, confidence_threshold, start_time = args
    
    try:
        # Initialize YOLO model in this process
        from ultralytics import YOLO
        model = YOLO(model_name)
        
        album_mbid = str(row['album_group_mbid'])
        image_path = os.path.join(image_directory, f"{album_mbid}.jpg")
        
        if not os.path.exists(image_path):
            # Try with .png extension
            image_path = os.path.join(image_directory, f"{album_mbid}.png")
            if not os.path.exists(image_path):
                return None
        
        # Run detection
        results = model(image_path, conf=confidence_threshold, verbose=False)
        
        # Extract detection data
        img_height, img_width = results[0].orig_shape
        detections = []
        class_counts = Counter()
        total_area_covered = 0
        
        if results[0].boxes is not None and len(results[0].boxes) > 0:
            boxes = results[0].boxes.xyxy.cpu().numpy()
            confidences = results[0].boxes.conf.cpu().numpy()
            class_ids = results[0].boxes.cls.cpu().numpy()
            
            for i, (box, conf, cls_id) in enumerate(zip(boxes, confidences, class_ids)):
                x1, y1, x2, y2 = box
                width = x2 - x1
                height = y2 - y1
                area = width * height
                area_percentage = (area / (img_width * img_height)) * 100
                
                class_name = model.names[int(cls_id)]
                class_counts[class_name] += 1
                total_area_covered += area_percentage
        
        # Combine with original album data
        result_row = row.to_dict()
        result_row.update({
            'num_objects_detected': len(detections),
            'total_area_covered': float(total_area_covered),
            'object_classes': json.dumps(dict(class_counts)),
            'processing_time': time.time() - start_time
        })
        
        return result_row
        
    except Exception as e:
        return None


if __name__ == "__main__":
    # Example usage
    detector = AlbumObjectDetector(model_name='yolov8n.pt', confidence_threshold=0.25)
    
    # Analyze a dataset
    csv_file = "../results/unified_album_dataset_with_complexity.csv"
    image_directory = "../data/img_all"
    output_file = "../results/objects/unified_with_objects.csv"
    
    # Process a small sample first to test
    print("Running object detection analysis on album covers...")
    results_df = detector.analyze_dataset(
        csv_file=csv_file,
        image_directory=image_directory,
        output_file=output_file 
    ) 