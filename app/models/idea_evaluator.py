"""
Idea Evaluator - TF-IDF + Naive Bayes for text classification
Pure Python implementation
"""

import numpy as np
import math
import pickle
import os
from collections import Counter, defaultdict
import re


class NaiveBayesClassifier:
    """Naive Bayes classifier with Laplace smoothing (pure Python)"""
    
    def __init__(self, alpha=1.0):
        self.alpha = alpha
        self.class_probs = {}
        self.word_probs = {}
        self.classes = []
        self.vocabulary = set()
        self.doc_counts = {}
        self.word_counts = defaultdict(lambda: defaultdict(float))
        self.total_docs = 0
    
    def fit(self, X, y):
        """Train the classifier"""
        self.classes = list(set(y))
        self.total_docs = len(y)
        
        # Count documents per class
        self.doc_counts = Counter(y)
        
        # Calculate class probabilities
        for c in self.classes:
            self.class_probs[c] = self.doc_counts[c] / self.total_docs
        
        # Count word occurrences per class
        for doc, label in zip(X, y):
            words = self._tokenize(doc)
            for word in words:
                self.word_counts[label][word] += 1
                self.vocabulary.add(word)
        
        # Calculate word probabilities with Laplace smoothing
        vocab_size = len(self.vocabulary)
        for c in self.classes:
            total_words = sum(self.word_counts[c].values())
            self.word_probs[c] = {}
            for word in self.vocabulary:
                count = self.word_counts[c].get(word, 0)
                self.word_probs[c][word] = (count + self.alpha) / (total_words + self.alpha * vocab_size)
    
    def predict(self, X):
        """Predict class for documents"""
        predictions = []
        for doc in X:
            predictions.append(self.predict_single(doc))
        return predictions
    
    def predict_single(self, doc):
        """Predict class for a single document"""
        words = self._tokenize(doc)
        scores = {}
        
        for c in self.classes:
            # Start with log of class probability
            log_score = np.log(self.class_probs.get(c, 1e-10))
            
            # Add log of word probabilities
            for word in words:
                prob = self.word_probs.get(c, {}).get(word, 1e-10)
                log_score += np.log(prob)
            
            scores[c] = log_score
        
        # Return class with highest score
        return max(scores, key=scores.get)
    
    def predict_proba(self, X):
        """Predict class probabilities"""
        probabilities = []
        for doc in X:
            probs = self._predict_proba_single(doc)
            probabilities.append(probs)
        return probabilities
    
    def _predict_proba_single(self, doc):
        """Predict probabilities for a single document"""
        words = self._tokenize(doc)
        scores = {}
        
        for c in self.classes:
            log_score = np.log(self.class_probs.get(c, 1e-10))
            for word in words:
                prob = self.word_probs.get(c, {}).get(word, 1e-10)
                log_score += np.log(prob)
            scores[c] = log_score
        
        # Convert to probabilities using softmax
        exp_scores = {c: np.exp(s) for c, s in scores.items()}
        total = sum(exp_scores.values())
        if total == 0:
            total = 1e-10
        return {c: exp_scores[c] / total for c in exp_scores}
    
    def _tokenize(self, text):
        """Tokenize text into words"""
        if not text:
            return []
        text = text.lower()
        text = re.sub(r'[^\w\s]', ' ', text)
        return text.split()


class TFIDFVectorizer:
    """Pure Python TF-IDF Vectorizer"""
    
    def __init__(self, max_features=10000, ngram_range=(1, 3)):
        self.max_features = max_features
        self.ngram_range = ngram_range
        self.vocabulary = {}
        self.idf = {}
        self.term_freq = {}
        self.doc_count = 0
    
    def fit(self, documents):
        """Fit the vectorizer on documents"""
        self.doc_count = len(documents)
        self.term_freq = defaultdict(float)
        
        for doc in documents:
            tokens = self._get_ngrams(doc)
            unique_tokens = set(tokens)
            for token in unique_tokens:
                self.term_freq[token] += 1
        
        # Build vocabulary (top max_features)
        sorted_terms = sorted(self.term_freq.items(), key=lambda x: x[1], reverse=True)
        if self.max_features:
            sorted_terms = sorted_terms[:self.max_features]
        
        self.vocabulary = {term: idx for idx, (term, _) in enumerate(sorted_terms)}
        
        # Calculate IDF
        for term in self.vocabulary:
            self.idf[term] = np.log((self.doc_count + 1) / (self.term_freq[term] + 1)) + 1
    
    def transform(self, documents):
        """Transform documents to TF-IDF vectors"""
        vectors = []
        for doc in documents:
            tokens = self._get_ngrams(doc)
            token_counts = Counter(tokens)
            
            vector = np.zeros(len(self.vocabulary))
            for token, count in token_counts.items():
                if token in self.vocabulary:
                    tf = count / len(tokens) if len(tokens) > 0 else 0
                    idf = self.idf.get(token, 0)
                    vector[self.vocabulary[token]] = tf * idf
            
            # Normalize
            norm = np.linalg.norm(vector)
            if norm > 0:
                vector = vector / norm
            
            vectors.append(vector)
        
        return np.array(vectors)
    
    def _get_ngrams(self, text):
        """Extract n-grams from text"""
        if not text:
            return []
        text = text.lower()
        text = re.sub(r'[^\w\s]', ' ', text)
        words = text.split()
        
        ngrams = []
        for n in range(self.ngram_range[0], min(self.ngram_range[1], len(words)) + 1):
            for i in range(len(words) - n + 1):
                ngrams.append('_'.join(words[i:i+n]))
        
        if not ngrams:
            ngrams = words
        
        return ngrams


class IdeaEvaluator:
    """Main Idea Evaluator with scoring and SWOT analysis"""
    
    def __init__(self, model_dir='./data/models'):
        self.model_dir = model_dir
        self.classifier = None
        self.vectorizer = None
        self.category_mapping = {}
        
        # Keywords for scoring dimensions
        self.innovation_keywords = {
            'novel': 2, 'breakthrough': 3, 'disruptive': 3, 'innovative': 2,
            'patent': 3, 'unique': 2, 'pioneering': 3, 'revolutionary': 3,
            'cutting_edge': 2, 'first_of_its_kind': 3, 'groundbreaking': 3,
            'transforming': 2, 'next_gen': 2, 'advanced': 1, 'novelty': 2
        }
        
        self.feasibility_keywords = {
            'feasible': 2, 'viable': 2, 'implement': 1, 'prototype': 2,
            'tested': 2, 'proven': 2, 'resources': 1, 'timeline': 1,
            'budget': 1, 'team': 1, 'expertise': 1, 'track_record': 2,
            'achievable': 2, 'realistic': 2, 'practical': 2
        }
        
        self.market_keywords = {
            'market': 2, 'demand': 2, 'revenue': 2, 'growth': 2,
            'tam': 3, 'sam': 3, 'customer': 2, 'trend': 1,
            'competition': 1, 'differentiation': 2, 'product_market_fit': 3,
            'scalable': 2, 'margins': 2, 'market_share': 2
        }
        
        self.social_keywords = {
            'sustainable': 2, 'environmental': 2, 'social': 2,
            'community': 1, 'impact': 2, 'inclusive': 2,
            'diversity': 1, 'accessibility': 2, 'affordable': 2,
            'education': 1, 'health': 2, 'wellbeing': 1,
            'climate': 2, 'green': 2, 'ethical': 2
        }
        
        self.financial_keywords = {
            'profitable': 2, 'margin': 2, 'revenue': 2, 'cost': 1,
            'burn_rate': 2, 'runway': 2, 'roi': 3, 'valuation': 2,
            'investor': 2, 'funding': 2, 'financial': 1, 'break_even': 3,
            'cash_flow': 2, 'projection': 1, 'economics': 1
        }
        
        self.risk_keywords_high = {
            'risk': 2, 'uncertain': 2, 'regulatory': 3, 'compliance': 3,
            'barrier': 2, 'competition': 1, 'technical_challenge': 2,
            'high_cost': 2, 'long_timeline': 2, 'market_risk': 2,
            'execution_risk': 2, 'financial_risk': 2, 'technology_risk': 2
        }
        
        self._load_models()
    
    def _load_models(self):
        """Load pre-trained models if available"""
        classifier_path = os.path.join(self.model_dir, 'classifier.pkl')
        vectorizer_path = os.path.join(self.model_dir, 'vectorizer.pkl')
        
        if os.path.exists(classifier_path) and os.path.exists(vectorizer_path):
            try:
                with open(classifier_path, 'rb') as f:
                    self.classifier = pickle.load(f)
                with open(vectorizer_path, 'rb') as f:
                    self.vectorizer = pickle.load(f)
            except Exception as e:
                print(f"Error loading models: {e}")
                self.classifier = None
                self.vectorizer = None
    
    def save_models(self):
        """Save trained models"""
        os.makedirs(self.model_dir, exist_ok=True)
        if self.classifier:
            with open(os.path.join(self.model_dir, 'classifier.pkl'), 'wb') as f:
                pickle.dump(self.classifier, f)
        if self.vectorizer:
            with open(os.path.join(self.model_dir, 'vectorizer.pkl'), 'wb') as f:
                pickle.dump(self.vectorizer, f)
    
    def evaluate(self, title, description, category=None, industry=None, 
                 target_market=None, business_model=None):
        """Evaluate an idea and return comprehensive scores"""
        combined_text = f"{title} {description} {target_market or ''} {business_model or ''}"
        
        # Calculate dimension scores
        innovation_score = self._score_innovation(combined_text)
        feasibility_score = self._score_feasibility(combined_text)
        market_potential = self._score_market(combined_text)
        social_impact = self._score_social(combined_text)
        financial_viability = self._score_financial(combined_text)
        
        # Calculate technical complexity
        technical_complexity = self._score_technical_complexity(combined_text)
        
        # Calculate risk score
        risk_score = self._score_risk(combined_text)
        
        # Calculate overall score (weighted)
        overall_score = (
            innovation_score * 0.25 +
            feasibility_score * 0.20 +
            market_potential * 0.20 +
            social_impact * 0.15 +
            financial_viability * 0.20
        )
        
        # Generate SWOT analysis
        swot = self._generate_swot(combined_text, category, industry)
        
        # Determine approval status
        approval_status = self._determine_approval(overall_score, risk_score)
        
        # Generate AI feedback
        feedback = self._generate_feedback(
            overall_score, innovation_score, feasibility_score,
            market_potential, social_impact, financial_viability,
            risk_score, swot
        )
        
        # Get classification if model is available
        category_pred = None
        confidence = None
        if self.classifier and self.vectorizer:
            try:
                vector = self.vectorizer.transform([combined_text])
                probs = self.classifier.predict_proba([combined_text])
                if probs:
                    max_prob = max(probs[0].values())
                    category_pred = max(probs[0], key=probs[0].get)
                    confidence = max_prob
            except Exception:
                pass
        
        return {
            'overall_score': round(overall_score, 2),
            'innovation_score': round(innovation_score, 2),
            'feasibility': round(feasibility_score, 2),
            'market_potential': round(market_potential, 2),
            'technical_complexity': round(technical_complexity, 2),
            'social_impact': round(social_impact, 2),
            'financial_viability': round(financial_viability, 2),
            'risk_score': round(risk_score, 2),
            'swot_analysis': swot,
            'approval_status': approval_status,
            'feedback': feedback,
            'category_prediction': category_pred,
            'confidence': confidence
        }
    
    def _score_innovation(self, text):
        """Score innovation based on keyword presence"""
        score = 5.0  # Base score
        text_lower = text.lower()
        
        for keyword, weight in self.innovation_keywords.items():
            if keyword in text_lower:
                score += weight * 2
        
        # Penalize for common/obvious ideas
        common_terms = ['simple', 'basic', 'common', 'ordinary', 'standard']
        for term in common_terms:
            if term in text_lower:
                score -= 1
        
        return max(0, min(100, score))
    
    def _score_feasibility(self, text):
        """Score feasibility based on keyword presence"""
        score = 5.0
        text_lower = text.lower()
        
        for keyword, weight in self.feasibility_keywords.items():
            if keyword in text_lower:
                score += weight * 2
        
        # Presence of concrete details increases score
        detail_terms = ['specific', 'detailed', 'plan', 'strategy', 'roadmap']
        for term in detail_terms:
            if term in text_lower:
                score += 0.5
        
        return max(0, min(100, score))
    
    def _score_market(self, text):
        """Score market potential"""
        score = 5.0
        text_lower = text.lower()
        
        for keyword, weight in self.market_keywords.items():
            if keyword in text_lower:
                score += weight * 2
        
        # Bonus for market size mentions
        size_terms = ['billion', 'million', 'large', 'growing', 'rapid']
        for term in size_terms:
            if term in text_lower:
                score += 0.5
        
        return max(0, min(100, score))
    
    def _score_social(self, text):
        """Score social impact"""
        score = 5.0
        text_lower = text.lower()
        
        for keyword, weight in self.social_keywords.items():
            if keyword in text_lower:
                score += weight * 2
        
        return max(0, min(100, score))
    
    def _score_financial(self, text):
        """Score financial viability"""
        score = 5.0
        text_lower = text.lower()
        
        for keyword, weight in self.financial_keywords.items():
            if keyword in text_lower:
                score += weight * 2
        
        return max(0, min(100, score))
    
    def _score_technical_complexity(self, text):
        """Score technical complexity (higher = more complex)"""
        score = 5.0
        text_lower = text.lower()
        
        complex_terms = ['ai', 'machine learning', 'deep learning', 'blockchain',
                        'quantum', 'biotech', 'nanotech', 'robotics',
                        'complex', 'sophisticated', 'advanced']
        
        for term in complex_terms:
            if term in text_lower:
                score += 3
        
        return max(0, min(100, score))
    
    def _score_risk(self, text):
        """Score risk (higher = more risk)"""
        score = 5.0
        text_lower = text.lower()
        
        for keyword, weight in self.risk_keywords_high.items():
            if keyword in text_lower:
                score += weight * 1.5
        
        return max(0, min(100, score))
    
    def _generate_swot(self, text, category=None, industry=None):
        """Generate SWOT analysis from text patterns"""
        text_lower = text.lower()
        
        # Strengths
        strengths = []
        strength_patterns = [
            ('innovative', 'Novel approach with potential for disruption'),
            ('feasible', 'Practical and achievable implementation path'),
            ('market', 'Clear market demand and growth potential'),
            ('social', 'Positive social and environmental impact'),
            ('financial', 'Strong financial viability and revenue potential'),
        ]
        for pattern, desc in strength_patterns:
            if pattern in text_lower:
                strengths.append(desc)
        
        if not strengths:
            strengths.append('Unique value proposition identified')
            strengths.append('Scalable business model')
        
        # Weaknesses
        weaknesses = []
        if 'complex' in text_lower or 'challenging' in text_lower:
            weaknesses.append('Technical complexity may require specialized expertise')
        if 'uncertain' in text_lower or 'unknown' in text_lower:
            weaknesses.append('Uncertain market acceptance')
        if len(text.split()) < 50:
            weaknesses.append('Limited detail in project description')
        if not weaknesses:
            weaknesses.append('Execution risk remains to be validated')
            weaknesses.append('Team composition needs further assessment')
        
        # Opportunities
        opportunities = []
        if 'growing' in text_lower or 'trend' in text_lower:
            opportunities.append('Growing market trend creates favorable conditions')
        if 'partner' in text_lower or 'collaborate' in text_lower:
            opportunities.append('Partnership opportunities available')
        opportunities.append('Potential for geographic expansion')
        opportunities.append('Opportunity for vertical integration')
        
        # Threats
        threats = []
        if 'competition' in text_lower:
            threats.append('Competitive market landscape')
        if 'regulatory' in text_lower:
            threats.append('Regulatory compliance requirements')
        threats.append('Technology disruption risk')
        threats.append('Market volatility')
        
        return {
            'strengths': strengths[:3],
            'weaknesses': weaknesses[:3],
            'opportunities': opportunities[:3],
            'threats': threats[:3]
        }
    
    def _determine_approval(self, overall_score, risk_score):
        """Determine approval status"""
        if overall_score >= 75 and risk_score < 50:
            return 'Recommended for Funding'
        elif overall_score >= 65 and risk_score < 60:
            return 'Conditionally Approved'
        elif overall_score >= 50:
            return 'Under Review'
        else:
            return 'Not Recommended'
    
    def _generate_feedback(self, overall, innovation, feasibility, market, 
                          social, financial, risk, swot):
        """Generate AI feedback"""
        feedback = []
        
        if overall >= 75:
            feedback.append("This is a promising innovation with strong potential.")
        elif overall >= 60:
            feedback.append("This idea shows good potential with some areas for improvement.")
        else:
            feedback.append("Consider refining the concept and addressing key challenges.")
        
        if innovation >= 80:
            feedback.append("Highly innovative approach with strong novelty.")
        elif innovation >= 60:
            feedback.append("Good innovation level, consider adding more unique elements.")
        else:
            feedback.append("Innovation could be enhanced by introducing more novel concepts.")
        
        if feasibility >= 80:
            feedback.append("Strong feasibility with clear implementation path.")
        elif feasibility >= 60:
            feedback.append("Feasible but requires more detailed planning.")
        else:
            feedback.append("Implementation challenges need to be addressed.")
        
        if market >= 80:
            feedback.append("Excellent market potential identified.")
        elif market >= 60:
            feedback.append("Good market opportunity, consider refining positioning.")
        else:
            feedback.append("Market analysis needs further development.")
        
        if risk >= 60:
            feedback.append("Risk management should be prioritized.")
        else:
            feedback.append("Risk profile is reasonable.")
        
        # Add specific recommendations
        recommendations = []
        if innovation < 60:
            recommendations.append("Enhance novelty by researching existing solutions")
        if feasibility < 60:
            recommendations.append("Develop a more detailed implementation plan")
        if market < 60:
            recommendations.append("Conduct deeper market research")
        if social < 60:
            recommendations.append("Consider strengthening social impact elements")
        if financial < 60:
            recommendations.append("Refine financial projections")
        
        if recommendations:
            feedback.append("Recommendations: " + "; ".join(recommendations))
        
        return " ".join(feedback)