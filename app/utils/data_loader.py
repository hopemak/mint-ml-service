"""
Data Loader - Load and preprocess data
"""

import json
import csv
import os
import re
from collections import defaultdict


class DataLoader:
    """Data loading and preprocessing utilities"""
    
    @staticmethod
    def load_json(filepath):
        """Load JSON data from file"""
        if not os.path.exists(filepath):
            return None
        
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    @staticmethod
    def load_csv(filepath):
        """Load CSV data from file"""
        if not os.path.exists(filepath):
            return None
        
        with open(filepath, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            return list(reader)
    
    @staticmethod
    def save_json(data, filepath):
        """Save data to JSON file"""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2)
    
    @staticmethod
    def preprocess_text(text):
        """Preprocess text for ML tasks"""
        if not text:
            return ""
        text = text.lower()
        text = re.sub(r'[^\w\s]', ' ', text)
        text = ' '.join(text.split())
        return text
    
    @staticmethod
    def get_word_counts(texts):
        """Get word frequency counts across texts"""
        counts = defaultdict(int)
        for text in texts:
            words = text.split()
            for word in words:
                counts[word] += 1
        return dict(counts)