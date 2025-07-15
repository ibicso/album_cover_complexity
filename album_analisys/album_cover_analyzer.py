import cv2
import numpy as np
import os
import csv
import time
import sys
from datetime import datetime
sys.path.append('..')
from measure_complexity import ComplexityMeasurer
import matplotlib.pyplot as plt


class AlbumCoverAnalyzer:
    def __init__(self, image_path=None, target_size=(224, 224)):
        """
        Initialize the analyzer with optional image resizing for standardized processing
        
        Args:
            image_path: Path to the image file
            target_size: Tuple (width, height) to resize images to. Set to None to disable resizing.
        """
        self.image_path = image_path
        self.target_size = target_size
        self.complexity_measurer = ComplexityMeasurer(
            ncs_to_check=5, 
            n_cluster_inits=1,
            nz=2,
            num_levels=4,
            cluster_model='GMM', 
            info_subsample=0.3, 
        )
        self.complexity_scores = None
    
    def load_image(self, image_path=None):
        """Load and optionally resize image for standardized processing"""
        if image_path:
            self.image_path = image_path
        
        img = cv2.imread(self.image_path)
        
        if img is None:
            raise ValueError(f"Could not load image from {self.image_path}")
        
        # Convert BGR to RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Resize image if target_size is specified
        if self.target_size is not None:
            original_shape = img.shape
            img = cv2.resize(img, self.target_size, interpolation=cv2.INTER_LANCZOS4)
            print(f"Resized image from {original_shape} to {img.shape}")
        
        return img
    
    def analyze_complexity(self, image_path=None):

        img = self.load_image(image_path)
        
        self.complexity_scores = self.complexity_measurer.interpret(img)
        
        return self.complexity_scores
    
    def get_overall_score(self):

        if self.complexity_scores is None:
            raise ValueError("No complexity scores available")
        
        return sum(self.complexity_scores)
    
    def print_results(self):

        if self.complexity_scores is None:
            raise ValueError("No complexity scores available")
        
        print(f"Image: {self.image_path}")
        print("Complexity scores at each level:")
        for i, score in enumerate(self.complexity_scores):
            level_description = "local detail" if i == 0 else "global structure" if i == len(self.complexity_scores)-1 else f"level {i+1}"
            print(f"  Level {i+1} ({level_description}): {score:.4f}")
        
        print(f"Overall complexity score: {self.get_overall_score():.4f}")