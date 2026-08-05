"""
Simple ML Service - Single File
Run with: python simple_service.py
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import json
import re
import math
from datetime import datetime

app = Flask(__name__)
CORS(app)

# Simple keyword-based scoring
def evaluate_idea(title, description, category="", industry=""):
    text = f"{title} {description}".lower()
    
    # Innovation score
    innovation_score = 50
    innovation_keywords = ['novel', 'breakthrough', 'disruptive', 'innovative', 'patent', 'unique', 'pioneering', 'revolutionary']
    for kw in innovation_keywords:
        if kw in text:
            innovation_score += 10
    
    # Feasibility
    feasibility = 50
    feasibility_keywords = ['feasible', 'viable', 'implement', 'prototype', 'tested', 'proven', 'resources']
    for kw in feasibility_keywords:
        if kw in text:
            feasibility += 10
    
    # Market potential
    market_potential = 50
    market_keywords = ['market', 'demand', 'revenue', 'growth', 'customer', 'scalable']
    for kw in market_keywords:
        if kw in text:
            market_potential += 10
    
    # Social impact
    social_impact = 50
    social_keywords = ['sustainable', 'environmental', 'social', 'community', 'impact', 'health']
    for kw in social_keywords:
        if kw in text:
            social_impact += 10
    
    # Financial viability
    financial_viability = 50
    financial_keywords = ['profitable', 'margin', 'revenue', 'roi', 'investor', 'funding']
    for kw in financial_keywords:
        if kw in text:
            financial_viability += 10
    
    # Technical complexity
    technical_complexity = 50
    complex_terms = ['ai', 'machine learning', 'blockchain', 'quantum', 'biotech', 'complex']
    for term in complex_terms:
        if term in text:
            technical_complexity += 8
    
    # Risk score
    risk_score = 30
    risk_keywords = ['risk', 'uncertain', 'regulatory', 'compliance', 'challenge']
    for kw in risk_keywords:
        if kw in text:
            risk_score += 8
    
    # Cap scores
    innovation_score = min(innovation_score, 100)
    feasibility = min(feasibility, 100)
    market_potential = min(market_potential, 100)
    social_impact = min(social_impact, 100)
    financial_viability = min(financial_viability, 100)
    technical_complexity = min(technical_complexity, 100)
    risk_score = min(risk_score, 100)
    
    # Overall score
    overall_score = (
        innovation_score * 0.25 +
        feasibility * 0.20 +
        market_potential * 0.20 +
        social_impact * 0.15 +
        financial_viability * 0.20
    )
    
    # SWOT
    swot = {
        'strengths': ['Innovative approach', 'Market potential identified'],
        'weaknesses': ['Requires further validation', 'Resource requirements need assessment'],
        'opportunities': ['Growing market demand', 'Potential partnerships'],
        'threats': ['Competitive landscape', 'Regulatory considerations']
    }
    
    # Approval status
    if overall_score >= 75 and risk_score < 50:
        approval_status = 'Recommended for Funding'
    elif overall_score >= 65:
        approval_status = 'Conditionally Approved'
    elif overall_score >= 50:
        approval_status = 'Under Review'
    else:
        approval_status = 'Not Recommended'
    
    return {
        'overall_score': round(overall_score, 2),
        'innovation_score': round(innovation_score, 2),
        'feasibility': round(feasibility, 2),
        'market_potential': round(market_potential, 2),
        'technical_complexity': round(technical_complexity, 2),
        'social_impact': round(social_impact, 2),
        'financial_viability': round(financial_viability, 2),
        'risk_score': round(risk_score, 2),
        'swot_analysis': swot,
        'approval_status': approval_status,
        'feedback': f"This idea shows {'promising' if overall_score > 70 else 'moderate'} potential. Innovation score: {round(innovation_score)}/100, Market potential: {round(market_potential)}/100."
    }


@app.route('/health', methods=['GET'])
def health():
    return jsonify({
        'status': 'healthy',
        'service': 'ml-ai-service',
        'version': '1.0.0',
        'timestamp': datetime.utcnow().isoformat()
    })


@app.route('/api/ml/evaluate', methods=['POST'])
def evaluate():
    data = request.get_json()
    if not data:
        return jsonify({'error': 'No data provided'}), 400
    
    title = data.get('title', '')
    description = data.get('description', '')
    
    if not title or not description:
        return jsonify({'error': 'Title and description are required'}), 400
    
    result = evaluate_idea(
        title=title,
        description=description,
        category=data.get('category', ''),
        industry=data.get('industry', '')
    )
    
    return jsonify({'success': True, 'data': result})


@app.route('/api/ml/predict', methods=['POST'])
def predict():
    data = request.get_json()
    if not data:
        return jsonify({'error': 'No data provided'}), 400
    
    funding = float(data.get('funding', 100000))
    team_size = int(data.get('team_size', 3))
    industry = data.get('industry', 'tech')
    stage = data.get('stage', 'seed')
    
    # Simple heuristic for success probability
    base_prob = 0.5
    if funding > 500000:
        base_prob += 0.15
    if team_size >= 3:
        base_prob += 0.10
    if industry in ['ai', 'tech', 'fintech']:
        base_prob += 0.05
    if stage in ['series_a', 'growth']:
        base_prob += 0.10
    elif stage in ['seed', 'mvp']:
        base_prob -= 0.05
    
    probability = min(max(base_prob, 0.1), 0.9)
    
    return jsonify({
        'success': True,
        'data': {
            'success_probability': round(probability * 100, 2),
            'confidence': 0.75,
            'key_factors': [
                {'factor': 'Adequate funding', 'impact': 'positive'},
                {'factor': 'Team composition', 'impact': 'positive'}
            ],
            'recommendations': ['Focus on customer acquisition', 'Optimize burn rate'],
            'risk_level': 'Medium Risk' if probability < 0.7 else 'Low Risk'
        }
    })


@app.route('/api/ml/match', methods=['POST'])
def match():
    data = request.get_json()
    required_skills = data.get('required_skills', [])
    
    # Sample mentors
    mentors = [
        {'name': 'Dr. Evelyn Reed', 'title': 'Professor', 'expertise': ['AI', 'ML'], 'match_percentage': 92},
        {'name': 'Dr. Sarah Chen', 'title': 'Senior Researcher', 'expertise': ['Biotech', 'Research'], 'match_percentage': 85},
        {'name': 'Michael Torres', 'title': 'Startup Advisor', 'expertise': ['SaaS', 'Product'], 'match_percentage': 78},
    ]
    
    return jsonify({'success': True, 'data': {'matches': mentors}})


if __name__ == '__main__':
    print("🚀 Starting ML Service on port 5000")
    print("📍 Health: http://localhost:5000/health")
    print("📋 POST /api/ml/evaluate")
    app.run(host='0.0.0.0', port=5000, debug=True)
