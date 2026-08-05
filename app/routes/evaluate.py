"""
Evaluation Routes
"""

from flask import Blueprint, request, jsonify
from flask_cors import cross_origin
import random

evaluate_bp = Blueprint('evaluate', __name__, url_prefix='/api')

@evaluate_bp.route('/public/evaluate', methods=['POST', 'OPTIONS'])
@cross_origin(origins=["http://localhost:5173", "*"])
def evaluate_public():
    if request.method == 'OPTIONS':
        return '', 200
    
    try:
        data = request.json
        print(f"📥 Evaluating: {data.get('title', 'Unknown')}")
        
        # Extract scores
        innovation = int(data.get('innovation_score', random.randint(60, 95)))
        market = int(data.get('market_potential', random.randint(60, 95)))
        feasibility = int(data.get('feasibility', random.randint(60, 95)))
        financial = int(data.get('financial_plan', random.randint(60, 95)))
        social = int(data.get('social_impact', random.randint(60, 95)))
        team = int(data.get('team_capability', random.randint(60, 95)))
        
        # Calculate overall
        overall = int((innovation + market + feasibility + financial + social + team) / 6)
        
        # Generate rating
        if overall >= 85:
            rating = "Excellent"
        elif overall >= 70:
            rating = "Good"
        elif overall >= 50:
            rating = "Average"
        else:
            rating = "Needs Improvement"
        
        risk = 100 - overall
        
        response = {
            'success': True,
            'data': {
                'overall_score': overall,
                'rating': rating,
                'innovation_score': innovation,
                'market_potential': market,
                'feasibility': feasibility,
                'financial_plan': financial,
                'social_impact': social,
                'team_capability': team,
                'technical_complexity': int((innovation + feasibility) / 2),
                'financial_viability': int((financial + market) / 2),
                'risk_score': risk,
                'approval_status': 'Approved' if overall >= 70 else 'Under Review',
                'swot_analysis': {
                    'strengths': ['Strong technical team', 'Clear market need', 'Innovative solution'],
                    'weaknesses': ['Limited resources', 'Market competition', 'Regulatory challenges'],
                    'opportunities': ['Growing market', 'Partnership potential', 'Technology trends'],
                    'threats': ['Economic uncertainty', 'New competitors', 'Regulatory changes']
                },
                'feedback': [
                    '🌟 Your idea shows strong potential!',
                    '💡 Consider refining your go-to-market strategy.',
                    '📈 Great social impact potential.'
                ]
            }
        }
        
        print(f"✅ Evaluation complete: {overall}%")
        return jsonify(response)
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500
