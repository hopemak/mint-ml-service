```python
import os
import json
import traceback

from flask import Flask, request, jsonify
from flask_cors import CORS
from groq import Groq
from dotenv import load_dotenv

# Load environment variables from .env
load_dotenv()

app = Flask(__name__)

CORS(
    app,
    origins=[
        "http://localhost:5173",
        "http://localhost:5174",
        "http://localhost:5175",
        "http://localhost:5176",
        "*"
    ]
)

# Read the API key from the .env file
api_key = os.getenv("GROQ_API_KEY")

if not api_key:
    raise RuntimeError(
        "GROQ_API_KEY is missing. Please add it to your .env file."
    )

client = Groq(api_key=api_key)
MODEL = "llama-3.3-70b-versatile"


@app.route('/api/health', methods=['GET'])
def health():
    return {
        'status': 'OK',
        'message': 'ML service is running!'
    }


@app.route('/api/public/evaluate', methods=['POST', 'OPTIONS'])
def evaluate():
    if request.method == 'OPTIONS':
        return '', 200

    try:
        data = request.get_json(silent=True) or {}

        innovation = int(data.get('innovation_score', 75))
        market = int(data.get('market_potential', 75))
        feasibility = int(data.get('feasibility', 75))
        financial = int(data.get('financial_plan', 75))
        social = int(data.get('social_impact', 75))
        team = int(data.get('team_capability', 75))

        overall = int(
            (innovation + market + feasibility + financial + social + team) / 6
        )

        rating = (
            "Excellent" if overall >= 85
            else "Good" if overall >= 70
            else "Average"
        )

        return jsonify({
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
                'risk_score': 100 - overall,
                'approval_status': 'Approved' if overall >= 70 else 'Under Review',
                'swot_analysis': {
                    'strengths': ['Strong team', 'Clear market need'],
                    'weaknesses': ['Limited resources', 'Competition'],
                    'opportunities': ['Growing market', 'Partnerships'],
                    'threats': ['Competitors', 'Regulatory changes']
                },
                'feedback': (
                    '🌟 Your idea shows strong potential! 💡 '
                    'Consider refining your strategy.'
                )
            }
        })

    except (TypeError, ValueError) as e:
        return jsonify({
            'success': False,
            'error': f'Invalid evaluation score: {str(e)}'
        }), 400

    except Exception as e:
        app.logger.exception("Evaluation error")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/api/chat', methods=['POST', 'OPTIONS'])
def chat():
    if request.method == 'OPTIONS':
        return '', 200

    try:
        data = request.get_json(silent=True) or {}

        query = data.get('query', '')
        context = data.get('context', {})

        if not isinstance(query, str) or not query.strip():
            return jsonify({
                'success': False,
                'error': 'Please provide a query.'
            }), 400

        print(f"💬 Chat Query: {query}")

        # Build context string safely
        context_parts = []

        if isinstance(context, dict):
            if context.get('title'):
                context_parts.append(f"Idea: {context['title']}")

            if context.get('overallScore') is not None:
                context_parts.append(
                    f"Score: {context['overallScore']}%"
                )

            scores = context.get('scores')

            if isinstance(scores, dict):
                score_str = ", ".join(
                    f"{key}: {value}%"
                    for key, value in scores.items()
                )
                context_parts.append(f"Scores: {score_str}")

        context_msg = " | ".join(context_parts)

        messages = [
            {
                "role": "system",
                "content": (
                    "You are an AI assistant for the Innovation Incubator "
                    "Platform. Help entrepreneurs understand their evaluation "
                    "scores and provide actionable advice. Be helpful, "
                    "specific, and encouraging. Keep responses concise "
                    "(2-3 paragraphs)."
                )
            }
        ]

        if context_msg:
            messages.append({
                "role": "system",
                "content": f"Current context: {context_msg}"
            })

        messages.append({
            "role": "user",
            "content": query
        })

        completion = client.chat.completions.create(
            model=MODEL,
            messages=messages,
            temperature=0.7,
            max_tokens=500
        )

        reply = completion.choices[0].message.content

        print("✅ Response sent")

        return jsonify({
            'success': True,
            'data': {
                'response': reply,
                'query': query
            }
        })

    except Exception as e:
        app.logger.exception("Chat error")

        return jsonify({
            'success': False,
            'error': 'Unable to process the chat request.',
            'data': {
                'response': 'Sorry, an error occurred while processing your request.'
            }
        }), 500


if __name__ == '__main__':
    print("🚀 ML Service on http://localhost:5001")
    app.run(debug=True, port=5001, host='0.0.0.0')
```