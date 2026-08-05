"""
NLP Service - Simple NLP utilities
"""

import re
from collections import Counter

class NLPService:
    def preprocess(self, text):
        if not text:
            return ""
        text = text.lower()
        text = re.sub(r'[^\w\s]', ' ', text)
        return ' '.join(text.split())
    
    def tokenize(self, text):
        return self.preprocess(text).split()
    
    def get_stopwords(self):
        return {'a', 'an', 'the', 'is', 'are', 'was', 'were', 'be', 'been',
                'to', 'for', 'of', 'in', 'on', 'at', 'with', 'by', 'from'}
    
    def extract_keywords(self, text, n=10):
        tokens = self.tokenize(text)
        stopwords = self.get_stopwords()
        filtered = [t for t in tokens if t not in stopwords and len(t) > 2]
        return [word for word, _ in Counter(filtered).most_common(n)]
