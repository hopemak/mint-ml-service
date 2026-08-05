"""
Embedding Service - Simple embedding generation
"""

import numpy as np
import re

class EmbeddingService:
    """Service for generating text embeddings"""
    
    def __init__(self, embedding_dim=50):
        self.embedding_dim = embedding_dim
        self.word_embeddings = {}
        self._init_word_embeddings()
    
    def _init_word_embeddings(self):
        """Initialize word embeddings"""
        common_words = [
            'ai', 'machine', 'learning', 'data', 'analytics', 'cloud', 'software',
            'product', 'market', 'customer', 'innovation', 'technology', 'solution',
            'platform', 'development', 'research', 'science', 'engineering',
            'business', 'startup', 'venture', 'capital', 'funding', 'investor',
            'mentor', 'advisor', 'strategy', 'growth', 'scale', 'launch'
        ]
        
        np.random.seed(42)
        for word in common_words:
            hash_val = hash(word) % 1000
            np.random.seed(hash_val)
            embedding = np.random.normal(0, 1, self.embedding_dim)
            embedding = embedding / np.linalg.norm(embedding)
            self.word_embeddings[word] = embedding
    
    def get_word_embedding(self, word):
        word = word.lower()
        if word in self.word_embeddings:
            return self.word_embeddings[word]
        hash_val = hash(word) % 1000
        np.random.seed(hash_val)
        embedding = np.random.normal(0, 0.1, self.embedding_dim)
        return embedding / np.linalg.norm(embedding)
    
    def get_text_embedding(self, text):
        if not text:
            return np.zeros(self.embedding_dim)
        words = re.findall(r'\b\w+\b', text.lower())
        embeddings = [self.get_word_embedding(word) for word in words]
        if not embeddings:
            return np.zeros(self.embedding_dim)
        mean_embedding = np.mean(embeddings, axis=0)
        norm = np.linalg.norm(mean_embedding)
        return mean_embedding / (norm + 1e-8)
    
    def cosine_similarity(self, text1, text2):
        emb1 = self.get_text_embedding(text1)
        emb2 = self.get_text_embedding(text2)
        if np.linalg.norm(emb1) == 0 or np.linalg.norm(emb2) == 0:
            return 0
        return np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2) + 1e-8)
