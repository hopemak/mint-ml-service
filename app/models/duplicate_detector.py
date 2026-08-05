"""
Duplicate Detector - Jaccard + TF-IDF similarity
Pure Python implementation
"""

import numpy as np
import math
import re
from collections import Counter


class DuplicateDetector:
    """Detect duplicate ideas using text similarity"""
    
    def __init__(self, model_dir='./data/models', threshold=0.6):
        self.model_dir = model_dir
        self.threshold = threshold
        self.idea_corpus = []  # List of (id, text) tuples
    
    def add_idea(self, idea_id, text):
        """Add an idea to the corpus"""
        self.idea_corpus.append((idea_id, text))
    
    def load_corpus(self, ideas):
        """Load multiple ideas into corpus"""
        for idea in ideas:
            self.idea_corpus.append((idea['id'], idea['text']))
    
    def tokenize(self, text):
        """Tokenize text for similarity comparison"""
        if not text:
            return []
        text = text.lower()
        text = re.sub(r'[^\w\s]', ' ', text)
        words = text.split()
        return [w for w in words if len(w) > 2]
    
    def jaccard_similarity(self, tokens1, tokens2):
        """Calculate Jaccard similarity between two token sets"""
        if not tokens1 or not tokens2:
            return 0
        
        set1 = set(tokens1)
        set2 = set(tokens2)
        
        intersection = len(set1.intersection(set2))
        union = len(set1.union(set2))
        
        if union == 0:
            return 0
        
        return intersection / union
    
    def tfidf_similarity(self, tokens1, tokens2, all_tokens=None):
        """Calculate TF-IDF similarity"""
        if not tokens1 or not tokens2:
            return 0
        
        # Count term frequencies
        tf1 = Counter(tokens1)
        tf2 = Counter(tokens2)
        
        # Get all unique terms
        if all_tokens is None:
            all_tokens = set(tokens1).union(set(tokens2))
        
        # Calculate IDF
        doc_freq = {}
        for term in all_tokens:
            doc_freq[term] = 0
            if term in tf1:
                doc_freq[term] += 1
            if term in tf2:
                doc_freq[term] += 1
        
        # Calculate TF-IDF vectors
        def get_tfidf_vector(tf, doc_freq, total_docs):
            vector = {}
            for term, count in tf.items():
                if term in doc_freq:
                    idf = math.log((total_docs + 1) / (doc_freq[term] + 1)) + 1
                    vector[term] = (count / sum(tf.values())) * idf
            return vector
        
        total_docs = 2  # We're comparing two documents
        vec1 = get_tfidf_vector(tf1, doc_freq, total_docs)
        vec2 = get_tfidf_vector(tf2, doc_freq, total_docs)
        
        # Calculate cosine similarity
        dot_product = 0
        norm1 = 0
        norm2 = 0
        
        for term in set(vec1.keys()).union(set(vec2.keys())):
            v1 = vec1.get(term, 0)
            v2 = vec2.get(term, 0)
            dot_product += v1 * v2
            norm1 += v1 ** 2
            norm2 += v2 ** 2
        
        if norm1 == 0 or norm2 == 0:
            return 0
        
        return dot_product / (math.sqrt(norm1) * math.sqrt(norm2))
    
    def calculate_similarity(self, text1, text2):
        """Calculate combined similarity between two texts"""
        tokens1 = self.tokenize(text1)
        tokens2 = self.tokenize(text2)
        
        if not tokens1 or not tokens2:
            return 0.0
        
        # Calculate Jaccard similarity
        jaccard_sim = self.jaccard_similarity(tokens1, tokens2)
        
        # Calculate TF-IDF similarity
        all_tokens = set(tokens1).union(set(tokens2))
        tfidf_sim = self.tfidf_similarity(tokens1, tokens2, all_tokens)
        
        # Combined similarity (weighted average)
        combined = 0.4 * jaccard_sim + 0.6 * tfidf_sim
        
        return combined
    
    def detect(self, text):
        """Detect if text is a duplicate of any idea in corpus"""
        if not self.idea_corpus:
            return {
                'is_duplicate': False,
                'similarity_score': 0,
                'matched_ideas': []
            }
        
        matches = []
        for idea_id, idea_text in self.idea_corpus:
            similarity = self.calculate_similarity(text, idea_text)
            if similarity >= self.threshold:
                matches.append({
                    'idea_id': idea_id,
                    'similarity_score': round(similarity * 100, 2)
                })
        
        # Sort by similarity
        matches.sort(key=lambda x: x['similarity_score'], reverse=True)
        
        if matches:
            return {
                'is_duplicate': True,
                'similarity_score': matches[0]['similarity_score'],
                'matched_ideas': matches[:5]  # Top 5 matches
            }
        else:
            return {
                'is_duplicate': False,
                'similarity_score': 0,
                'matched_ideas': []
            }