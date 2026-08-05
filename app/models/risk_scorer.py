"""
Risk Scorer - Comprehensive risk assessment
Pure Python implementation
"""

import numpy as np
import re
from collections import Counter


class RiskScorer:
    """Risk assessment for ideas and startups"""
    
    def __init__(self):
        self.risk_categories = {
            'market': {
                'weight': 0.25,
                'keywords': ['competition', 'market', 'demand', 'customer', 'adoption'],
                'high_risk': ['high competition', 'saturated', 'declining market', 'niche']
            },
            'technical': {
                'weight': 0.20,
                'keywords': ['complex', 'technical', 'development', 'integration', 'infrastructure'],
                'high_risk': ['complex', 'unproven', 'integration challenge', 'legacy']
            },
            'financial': {
                'weight': 0.20,
                'keywords': ['cost', 'revenue', 'funding', 'budget', 'capital'],
                'high_risk': ['high cost', 'low margin', 'limited funding', 'burn rate']
            },
            'operational': {
                'weight': 0.15,
                'keywords': ['team', 'operations', 'process', 'scale', 'execution'],
                'high_risk': ['small team', 'inexperienced', 'scaling challenge', 'process']
            },
            'regulatory': {
                'weight': 0.10,
                'keywords': ['regulatory', 'compliance', 'legal', 'policy', 'governance'],
                'high_risk': ['strict regulation', 'compliance risk', 'legal challenge', 'policy']
            },
            'competitive': {
                'weight': 0.10,
                'keywords': ['competitor', 'alternative', 'substitute', 'barrier'],
                'high_risk': ['strong competitors', 'low barrier', 'copycat', 'displacement']
            }
        }
        
        self.risk_levels = {
            'low': (0, 30),
            'medium': (30, 60),
            'high': (60, 80),
            'critical': (80, 100)
        }
    
    def assess_idea(self, text, category=None):
        """Assess risks for an idea"""
        text_lower = text.lower()
        tokens = text_lower.split()
        
        risk_breakdown = {}
        overall_risk = 0
        
        for category, config in self.risk_categories.items():
            score = self._calculate_category_risk(text_lower, config)
            risk_breakdown[category] = {
                'score': round(score, 2),
                'level': self._get_risk_level(score),
                'weight': config['weight']
            }
            overall_risk += score * config['weight']
        
        # Overall risk level
        overall_level = self._get_risk_level(overall_risk)
        
        return {
            'overall_risk_score': round(overall_risk, 2),
            'overall_risk_level': overall_level,
            'risk_breakdown': risk_breakdown,
            'risk_factors': self._identify_risk_factors(text_lower),
            'mitigations': self._suggest_mitigations(risk_breakdown)
        }
    
    def _calculate_category_risk(self, text, config):
        """Calculate risk for a category"""
        score = 10  # Base score
        
        # Check for keywords
        for keyword in config.get('keywords', []):
            if keyword in text:
                score += 2
        
        # Check for high risk indicators
        for indicator in config.get('high_risk', []):
            if indicator in text:
                score += 5
        
        # Cap the score
        return min(score, 100)
    
    def _get_risk_level(self, score):
        """Get risk level from score"""
        for level, (low, high) in self.risk_levels.items():
            if low <= score < high:
                return level
        return 'critical'
    
    def _identify_risk_factors(self, text):
        """Identify specific risk factors"""
        factors = []
        
        risk_indicators = {
            'market': ['market risk', 'demand uncertainty', 'customer acquisition'],
            'technical': ['technical debt', 'development complexity', 'integration issues'],
            'financial': ['funding gap', 'high burn rate', 'revenue uncertainty'],
            'operational': ['team capacity', 'scaling issues', 'operational overhead'],
            'regulatory': ['compliance burden', 'legal complexity', 'policy changes'],
            'competitive': ['competitive pressure', 'market entry barriers', 'substitute products']
        }
        
        for category, indicators in risk_indicators.items():
            for indicator in indicators:
                if indicator in text:
                    factors.append({
                        'category': category,
                        'factor': indicator,
                        'severity': 'medium'
                    })
        
        return factors[:10]  # Top 10 factors
    
    def _suggest_mitigations(self, risk_breakdown):
        """Suggest mitigations for risks"""
        mitigations = []
        
        for category, data in risk_breakdown.items():
            if data['score'] > 60:
                if category == 'market':
                    mitigations.append("Conduct thorough market research and validation")
                    mitigations.append("Develop strong go-to-market strategy")
                elif category == 'technical':
                    mitigations.append("Implement rigorous testing and quality assurance")
                    mitigations.append("Consider phased development approach")
                elif category == 'financial':
                    mitigations.append("Create detailed financial projections")
                    mitigations.append("Secure adequate funding runway")
                elif category == 'operational':
                    mitigations.append("Build scalable operational processes")
                    mitigations.append("Consider strategic partnerships")
                elif category == 'regulatory':
                    mitigations.append("Engage legal counsel early")
                    mitigations.append("Build compliance into product roadmap")
                elif category == 'competitive':
                    mitigations.append("Develop strong differentiation strategy")
                    mitigations.append("Build defensible moat")
        
        # Remove duplicates
        unique_mitigations = []
        for m in mitigations:
            if m not in unique_mitigations:
                unique_mitigations.append(m)
        
        return unique_mitigations[:8]