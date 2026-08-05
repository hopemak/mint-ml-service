"""
Success Prediction API Route
"""

from flask import Blueprint, request, jsonify, current_app
from app.models.success_predictor import SuccessPredictor

predict_bp = Blueprint('predict', __name__)

# Initialize predictor
predictor = SuccessPredictor()


@predict_bp.route('/predict', methods=['POST'])
def predict_success():
    """Predict startup success probability"""
    data = request.get_json()
    
    if not data:
        return jsonify({'error': 'No data provided'}), 400
    
    # Extract fields with defaults
    funding = float(data.get('funding', 100000))
    team_size = int(data.get('team_size', 3))
    industry = data.get('industry', 'tech')
    stage = data.get('stage', 'seed')
    
    revenue_growth = float(data.get('revenue_growth', 0.5))
    user_growth = float(data.get('user_growth', 0.5))
    burn_rate_ratio = float(data.get('burn_rate_ratio', 0.3))
    runway_months = float(data.get('runway_months', 12))
    competition_level = float(data.get('competition_level', 0.5))
    market_size = float(data.get('market_size', 0.5))
    years_experience = float(data.get('years_experience', 5))
    patents_count = int(data.get('patents_count', 0))
    pivot_count = int(data.get('pivot_count', 0))
    product_maturity = float(data.get('product_maturity', 0.5))
    
    try:
        result = predictor.predict(
            funding=funding,
            team_size=team_size,
            industry=industry,
            stage=stage,
            revenue_growth=revenue_growth,
            user_growth=user_growth,
            burn_rate_ratio=burn_rate_ratio,
            runway_months=runway_months,
            competition_level=competition_level,
            market_size=market_size,
            years_experience=years_experience,
            patents_count=patents_count,
            pivot_count=pivot_count,
            product_maturity=product_maturity
        )
        
        current_app.logger.info(
            f"Success prediction: {industry} startup | Probability: {result['success_probability']}%"
        )
        
        return jsonify({
            'success': True,
            'data': result
        })
        
    except Exception as e:
        current_app.logger.error(f"Error predicting success: {str(e)}")
        return jsonify({'error': str(e)}), 500