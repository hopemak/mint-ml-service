"""
Mentor Matcher - Cosine similarity on skill embeddings
Pure Python implementation
"""

import numpy as np
import pickle
import os
import math
from collections import defaultdict


class MentorMatcher:
    """Mentor matching using cosine similarity"""
    
    def __init__(self, model_dir='./data/models'):
        self.model_dir = model_dir
        self.skill_embeddings = {}
        self.mentor_profiles = []
        self._load_mentor_data()
    
    def _load_mentor_data(self):
        """Load mentor data from storage"""
        # Sample mentor data (in production, this would come from database)
        self.mentor_profiles = [
            {
                'id': 'mentor_001',
                'name': 'Dr. Evelyn Reed',
                'title': 'Professor',
                'expertise': ['ai', 'machine_learning', 'deep_learning', 'nlp'],
                'experience_years': 15,
                'rating': 4.9,
                'available': True,
                'bio': 'Expert in AI and ML with 15 years of experience.'
            },
            {
                'id': 'mentor_002',
                'name': 'Dr. Sarah Chen',
                'title': 'Senior Researcher',
                'expertise': ['biotech', 'pharmaceutical', 'research', 'development'],
                'experience_years': 12,
                'rating': 4.8,
                'available': True,
                'bio': 'Leading researcher in biotechnology innovations.'
            },
            {
                'id': 'mentor_003',
                'name': 'Michael Torres',
                'title': 'Startup Advisor',
                'expertise': ['saas', 'product_management', 'growth', 'marketing'],
                'experience_years': 10,
                'rating': 4.7,
                'available': True,
                'bio': 'Experienced startup advisor with multiple exits.'
            },
            {
                'id': 'mentor_004',
                'name': 'Dr. James Park',
                'title': 'Chief Technology Officer',
                'expertise': ['blockchain', 'fintech', 'software_architecture', 'security'],
                'experience_years': 18,
                'rating': 4.9,
                'available': False,
                'bio': 'CTO with deep expertise in blockchain and fintech.'
            },
            {
                'id': 'mentor_005',
                'name': 'Dr. Emma Wilson',
                'title': 'Investment Partner',
                'expertise': ['finance', 'venture_capital', 'strategy', 'business_development'],
                'experience_years': 14,
                'rating': 4.6,
                'available': True,
                'bio': 'Venture capital partner with focus on early-stage startups.'
            },
            {
                'id': 'mentor_006',
                'name': 'Dr. Ananya Sharma',
                'title': 'AI Research Lead',
                'expertise': ['ai', 'computer_vision', 'robotics', 'automation'],
                'experience_years': 11,
                'rating': 4.8,
                'available': True,
                'bio': 'AI research lead with focus on computer vision and robotics.'
            },
            {
                'id': 'mentor_007',
                'name': 'Robert Kim',
                'title': 'Product Strategist',
                'expertise': ['product_design', 'ux', 'human_computer_interaction', 'innovation'],
                'experience_years': 9,
                'rating': 4.5,
                'available': True,
                'bio': 'Product strategist specializing in user-centered design.'
            },
            {
                'id': 'mentor_008',
                'name': 'Dr. Patricia Diaz',
                'title': 'Sustainability Expert',
                'expertise': ['cleantech', 'sustainability', 'renewable_energy', 'policy'],
                'experience_years': 16,
                'rating': 4.7,
                'available': True,
                'bio': 'Expert in clean technology and sustainability innovation.'
            }
        ]
        
        # Pre-compute skill embeddings (simplified for demo)
        self._build_skill_embeddings()
    
    def _build_skill_embeddings(self):
        """Build skill embeddings using dummy vectors"""
        # In production, use proper embeddings like Word2Vec or BERT
        # Here we create random but deterministic embeddings
        skill_list = [
            'ai', 'machine_learning', 'deep_learning', 'nlp', 'computer_vision',
            'robotics', 'automation', 'biotech', 'pharmaceutical', 'research',
            'development', 'saas', 'product_management', 'growth', 'marketing',
            'blockchain', 'fintech', 'software_architecture', 'security',
            'finance', 'venture_capital', 'strategy', 'business_development',
            'product_design', 'ux', 'human_computer_interaction', 'innovation',
            'cleantech', 'sustainability', 'renewable_energy', 'policy'
        ]
        
        # Create deterministic embeddings using hash functions
        np.random.seed(42)
        for skill in skill_list:
            # Use hash to create a deterministic vector
            hash_val = hash(skill) % 1000
            np.random.seed(hash_val)
            embedding = np.random.normal(0, 1, 50)
            # Normalize
            embedding = embedding / np.linalg.norm(embedding)
            self.skill_embeddings[skill] = embedding
    
    def get_skill_embedding(self, skill):
        """Get embedding for a skill"""
        if skill in self.skill_embeddings:
            return self.skill_embeddings[skill]
        # Return a random embedding for unknown skills
        return np.random.normal(0, 0.1, 50)
    
    def cosine_similarity(self, vec1, vec2):
        """Calculate cosine similarity between two vectors"""
        if np.linalg.norm(vec1) == 0 or np.linalg.norm(vec2) == 0:
            return 0
        return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
    
    def match(self, idea_id, required_skills, startup_data=None):
        """Match mentors based on required skills"""
        if not required_skills:
            required_skills = startup_data.get('required_skills', [])
        
        # Convert required skills to embeddings (average of skill embeddings)
        required_embeddings = []
        for skill in required_skills:
            if isinstance(skill, str):
                emb = self.get_skill_embedding(skill.lower())
                required_embeddings.append(emb)
        
        if not required_embeddings:
            # If no skills provided, use startup industry as proxy
            industry = startup_data.get('industry', 'tech')
            required_embeddings = [self.get_skill_embedding(industry.lower())]
        
        required_vector = np.mean(required_embeddings, axis=0)
        if np.linalg.norm(required_vector) > 0:
            required_vector = required_vector / np.linalg.norm(required_vector)
        
        # Score each mentor
        matches = []
        for mentor in self.mentor_profiles:
            if not mentor.get('available', True):
                continue
            
            # Calculate skill similarity
            skill_vectors = [self.get_skill_embedding(skill) for skill in mentor['expertise']]
            if skill_vectors:
                mentor_vector = np.mean(skill_vectors, axis=0)
                if np.linalg.norm(mentor_vector) > 0:
                    mentor_vector = mentor_vector / np.linalg.norm(mentor_vector)
                
                skill_similarity = self.cosine_similarity(required_vector, mentor_vector)
            else:
                skill_similarity = 0.3
            
            # Calculate experience score
            experience_score = min(mentor['experience_years'] / 20, 1.0)
            
            # Calculate rating score
            rating_score = mentor['rating'] / 5.0
            
            # Combined match score
            match_percentage = (
                skill_similarity * 0.50 +
                experience_score * 0.25 +
                rating_score * 0.25
            ) * 100
            
            # Ensure match percentage is reasonable
            match_percentage = min(max(match_percentage, 10), 98)
            
            matches.append({
                'mentor_id': mentor['id'],
                'name': mentor['name'],
                'title': mentor['title'],
                'expertise': mentor['expertise'],
                'experience_years': mentor['experience_years'],
                'rating': mentor['rating'],
                'bio': mentor['bio'],
                'match_percentage': round(match_percentage, 2),
                'skill_similarity': round(skill_similarity * 100, 2),
                'available': mentor.get('available', True)
            })
        
        # Sort by match percentage
        matches.sort(key=lambda x: x['match_percentage'], reverse=True)
        
        return matches
    
    def add_mentor(self, mentor_data):
        """Add a new mentor to the profiles"""
        self.mentor_profiles.append(mentor_data)
    
    def update_mentor(self, mentor_id, updates):
        """Update mentor profile"""
        for mentor in self.mentor_profiles:
            if mentor['id'] == mentor_id:
                mentor.update(updates)
                return True
        return False
    
    def get_mentor(self, mentor_id):
        """Get mentor by ID"""
        for mentor in self.mentor_profiles:
            if mentor['id'] == mentor_id:
                return mentor
        return None