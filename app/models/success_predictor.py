"""
Success Predictor - Logistic Regression with Gradient Descent
Pure Python implementation
"""

import numpy as np
import pickle
import os
import math
from collections import defaultdict


class LogisticRegression:
    """Pure Python Logistic Regression with Gradient Descent"""
    
    def __init__(self, learning_rate=0.01, n_iterations=1000, regularization=0.01):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.regularization = regularization
        self.weights = None
        self.bias = None
        self.loss_history = []
    
    def fit(self, X, y):
        """Train the model using gradient descent"""
        n_samples, n_features = X.shape
        
        # Initialize weights and bias
        self.weights = np.zeros(n_features)
        self.bias = 0
        
        # Gradient descent
        for i in range(self.n_iterations):
            # Linear combination
            linear_model = np.dot(X, self.weights) + self.bias
            
            # Sigmoid function
            y_pred = self._sigmoid(linear_model)
            
            # Compute gradients
            dw = (1 / n_samples) * np.dot(X.T, (y_pred - y)) + self.regularization * self.weights
            db = (1 / n_samples) * np.sum(y_pred - y)
            
            # Update parameters
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db
            
            # Calculate loss
            loss = self._binary_cross_entropy(y, y_pred)
            self.loss_history.append(loss)
    
    def predict(self, X):
        """Predict class labels"""
        probabilities = self.predict_proba(X)
        return (probabilities >= 0.5).astype(int)
    
    def predict_proba(self, X):
        """Predict probabilities"""
        linear_model = np.dot(X, self.weights) + self.bias
        return self._sigmoid(linear_model)
    
    def _sigmoid(self, z):
        """Sigmoid activation function"""
        # Clip to avoid overflow
        z = np.clip(z, -500, 500)
        return 1 / (1 + np.exp(-z))
    
    def _binary_cross_entropy(self, y_true, y_pred):
        """Binary cross entropy loss"""
        eps = 1e-15
        y_pred = np.clip(y_pred, eps, 1 - eps)
        return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
    
    def get_feature_importance(self, feature_names=None):
        """Get feature importance weights"""
        if feature_names is None:
            feature_names = [f"feature_{i}" for i in range(len(self.weights))]
        importance = []
        for name, weight in zip(feature_names, self.weights):
            importance.append({
                'feature': name,
                'weight': float(weight),
                'importance': abs(float(weight))
            })
        return sorted(importance, key=lambda x: x['importance'], reverse=True)


class SuccessPredictor:
    """Main Success Predictor with feature engineering"""
    
    def __init__(self, model_dir='./data/models'):
        self.model_dir = model_dir
        self.model = None
        self.feature_names = [
            'team_size', 'years_experience', 'patents_count', 
            'pivot_count', 'market_size', 'competition_level',
            'funding_raised_ratio', 'revenue_growth', 'user_growth',
            'burn_rate_ratio', 'runway_months', 'product_maturity'
        ]
        self._load_model()
    
    def _load_model(self):
        """Load pre-trained model if available"""
        model_path = os.path.join(self.model_dir, 'success_predictor.pkl')
        if os.path.exists(model_path):
            try:
                with open(model_path, 'rb') as f:
                    self.model = pickle.load(f)
            except Exception as e:
                print(f"Error loading model: {e}")
                self.model = None
    
    def save_model(self):
        """Save trained model"""
        os.makedirs(self.model_dir, exist_ok=True)
        if self.model:
            with open(os.path.join(self.model_dir, 'success_predictor.pkl'), 'wb') as f:
                pickle.dump(self.model, f)
    
    def predict(self, funding, team_size, industry, stage, 
                revenue_growth=0.5, user_growth=0.5, burn_rate_ratio=0.3,
                runway_months=12, competition_level=0.5, market_size=0.5,
                years_experience=5, patents_count=1, pivot_count=0,
                product_maturity=0.5):
        """Predict startup success probability"""
        
        # Feature engineering
        features = self._engineer_features(
            funding, team_size, industry, stage,
            revenue_growth, user_growth, burn_rate_ratio,
            runway_months, competition_level, market_size,
            years_experience, patents_count, pivot_count,
            product_maturity
        )
        
        # If model is not trained, use heuristic scoring
        if self.model is None:
            probability = self._heuristic_predict(features)
            confidence = 0.6
            key_factors = self._get_key_factors(features)
            recommendations = self._generate_recommendations(features)
        else:
            prob = self.model.predict_proba(np.array([features]).reshape(1, -1))
            probability = float(prob[0])
            confidence = 0.8 if probability > 0.7 or probability < 0.3 else 0.6
            key_factors = self._get_key_factors_ml(features)
            recommendations = self._generate_recommendations_ml(features, probability)
        
        return {
            'success_probability': round(probability * 100, 2),
            'confidence': round(confidence, 2),
            'key_factors': key_factors,
            'recommendations': recommendations,
            'risk_level': self._get_risk_level(probability)
        }
    
    def _engineer_features(self, funding, team_size, industry, stage,
                          revenue_growth, user_growth, burn_rate_ratio,
                          runway_months, competition_level, market_size,
                          years_experience, patents_count, pivot_count,
                          product_maturity):
        """Engineer features from raw inputs"""
        
        # Industry multiplier
        industry_multiplier = {
            'tech': 1.2, 'fintech': 1.3, 'healthcare': 1.1,
            'biotech': 0.9, 'clean_tech': 1.1, 'edtech': 0.8,
            'saas': 1.2, 'ai': 1.4, 'default': 1.0
        }.get(industry.lower(), 1.0)
        
        # Stage multiplier
        stage_multiplier = {
            'idea': 0.5, 'prototype': 0.7, 'mvp': 0.8,
            'seed': 0.9, 'series_a': 1.0, 'growth': 1.1
        }.get(stage.lower(), 1.0)
        
        # Normalize funding
        funding_normalized = min(funding / 1000000, 10)  # Cap at $10M
        
        # Team size factor
        team_factor = min(team_size / 10, 2.0)
        
        # Revenue growth factor
        revenue_factor = min(revenue_growth / 100, 2.0)
        
        # User growth factor
        user_factor = min(user_growth / 100, 2.0)
        
        # Burn rate factor (lower is better)
        burn_factor = max(0, 1 - burn_rate_ratio)
        
        # Runway factor
        runway_factor = min(runway_months / 24, 1.5)
        
        # Competition factor (higher competition = lower success)
        competition_factor = max(0, 1 - competition_level)
        
        # Market size factor
        market_factor = min(market_size, 1.0) * 1.5
        
        # Experience factor
        experience_factor = min(years_experience / 10, 1.5)
        
        # Product maturity factor
        maturity_factor = product_maturity * 1.5
        
        # Combine into features array
        features = np.array([
            team_factor,
            experience_factor,
            min(patents_count / 5, 2.0),
            max(0, 1 - pivot_count / 3),
            market_factor,
            competition_factor,
            funding_normalized / 5,
            revenue_factor,
            user_factor,
            burn_factor,
            runway_factor,
            maturity_factor
        ])
        
        # Apply industry and stage multipliers
        features = features * industry_multiplier * stage_multiplier
        
        return features
    
    def _heuristic_predict(self, features):
        """Heuristic prediction when model is not trained"""
        # Weighted sum of normalized features
        weights = np.array([0.15, 0.10, 0.05, 0.05, 0.15, 0.05, 0.10, 0.10, 0.05, 0.10, 0.05, 0.05])
        score = np.sum(features * weights)
        
        # Normalize to probability
        score = np.clip(score / 2, 0, 1)
        
        # Add some randomness to avoid constant values
        score = score * 0.8 + 0.1 * np.random.random()
        
        return min(max(score, 0.1), 0.9)
    
    def _get_key_factors(self, features):
        """Get key success factors"""
        factors = []
        
        # Team factors
        if features[0] > 1.0:
            factors.append({'factor': 'Strong team size and composition', 'impact': 'positive'})
        elif features[0] < 0.5:
            factors.append({'factor': 'Small team size may limit execution capacity', 'impact': 'negative'})
        
        # Market factors
        if features[4] > 1.0:
            factors.append({'factor': 'Large addressable market opportunity', 'impact': 'positive'})
        elif features[4] < 0.5:
            factors.append({'factor': 'Limited market size', 'impact': 'negative'})
        
        # Funding factors
        if features[6] > 0.5:
            factors.append({'factor': 'Adequate funding for growth stage', 'impact': 'positive'})
        elif features[6] < 0.3:
            factors.append({'factor': 'Limited funding may constrain growth', 'impact': 'negative'})
        
        # Revenue factors
        if features[7] > 0.5:
            factors.append({'factor': 'Strong revenue growth trajectory', 'impact': 'positive'})
        elif features[7] < 0.3:
            factors.append({'factor': 'Revenue growth needs acceleration', 'impact': 'negative'})
        
        return factors[:5]
    
    def _get_key_factors_ml(self, features):
        """Get key factors using ML model interpretation"""
        return self._get_key_factors(features)  # Simplified fallback
    
    def _generate_recommendations(self, features):
        """Generate recommendations based on features"""
        recommendations = []
        
        if features[0] < 0.5:
            recommendations.append("Consider expanding the team with key hires")
        if features[2] < 0.5:
            recommendations.append("Focus on building intellectual property portfolio")
        if features[5] > 1.5:
            recommendations.append("Develop strong differentiation strategy")
        if features[6] < 0.3:
            recommendations.append("Explore additional funding opportunities")
        if features[7] < 0.3:
            recommendations.append("Focus on customer acquisition and revenue generation")
        if features[9] < 0.5:
            recommendations.append("Optimize burn rate to extend runway")
        if features[11] < 0.5:
            recommendations.append("Accelerate product development cycle")
        
        if not recommendations:
            recommendations.append("Maintain current growth trajectory")
            recommendations.append("Focus on market expansion")
        
        return recommendations[:4]
    
    def _generate_recommendations_ml(self, features, probability):
        """Generate recommendations using ML interpretation"""
        recommendations = self._generate_recommendations(features)
        
        if probability > 0.7:
            recommendations.append("Consider aggressive market expansion")
        elif probability < 0.3:
            recommendations.append("Re-evaluate business model and market fit")
        
        return recommendations[:4]
    
    def _get_risk_level(self, probability):
        """Get risk level based on probability"""
        if probability >= 0.7:
            return 'Low Risk'
        elif probability >= 0.5:
            return 'Medium Risk'
        elif probability >= 0.3:
            return 'High Risk'
        else:
            return 'Very High Risk'