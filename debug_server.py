load_dotenv()
from dotenv import load_dotenv
import os
from flask import Flask, request, jsonify
from flask_cors import CORS
from groq import Groq
import traceback
import json

app = Flask(__name__)
CORS(app, origins=["http://localhost:5173", "http://localhost:5174", "http://localhost:5175", "http://localhost:5176", "*"])

client = Groq(api_key=os.getenv("GROQ_API_KEY"))
MODEL = "llama-3.3-70b-versatile"

@app.route('/api/health', methods=['GET'])
def health():
    return {'status': 'OK', 'message': 'ML service is running!'}

@app.route('/api/public/evaluate', methods=['POST', 'OPTIONS'])
def evaluate():
    if request.method == 'OPTIONS':
        return '', 200
    try:
        data = request.json
        print(f"📥 Evaluating: {data.get('title', 'Unknown')}")
        
        innovation = int(data.get('innovation_score', 75))
        market = int(data.get('market_potential', 75))
        feasibility = int(data.get('feasibility', 75))
        financial = int(data.get('financial_plan', 75))
        social = int(data.get('social_impact', 75))
        team = int(data.get('team_capability', 75))
        
        overall = int((innovation + market + feasibility + financial + social + team) / 6)
        rating = "Excellent" if overall >= 85 else "Good" if overall >= 70 else "Average"
        
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
                'feedback': '🌟 Your idea shows strong potential! 💡 Consider refining your strategy.'
            }
        })
    except Exception as e:
        print(f"❌ Evaluate Error: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/chat', methods=['POST', 'OPTIONS'])
def chat():
    if request.method == 'OPTIONS':
        return '', 200
    
    try:
        data = request.json
        query = data.get('query', '')
        context = data.get('context', {})
        
        print("=" * 50)
        print(f"💬 Chat Query: {query}")
        print(f"📋 Context received: {context}")
        print(f"📋 Context type: {type(context)}")
        
        # Build context safely
        context_msg = ""
        if context and isinstance(context, dict):
            parts = []
            if context.get('title'):
                parts.append(f"Idea: {context['title']}")
            if context.get('overallScore'):
                parts.append(f"Score: {context['overallScore']}%")
            if context.get('scores') and isinstance(context['scores'], dict):
                score_items = []
                for k, v in context['scores'].items():
                    score_items.append(f"{k}: {v}%")
                if score_items:
                    parts.append(f"Scores: {', '.join(score_items)}")
            context_msg = " | ".join(parts)
        
        print(f"📋 Context message: {context_msg}")
        
        # Build messages
        messages = [
            {
                "role": "system", 
                "content": """You are an AI assistant for the Innovation Incubator Platform. 
Help entrepreneurs understand their evaluation scores and provide actionable advice.
Be helpful, specific, and encouraging. Keep responses concise (2-3 paragraphs)."""
            }
        ]
        
        if context_msg:
            messages.append({
                "role": "system",
                "content": f"Current context: {context_msg}"
            })
        
        messages.append({"role": "user", "content": query})
        
        print(f"📤 Sending to Groq with model: {MODEL}")
        print(f"📤 Messages: {messages[:2]}...")
        
        # Call Groq
        completion = client.chat.completions.create(
            model=MODEL,
            messages=messages,
            temperature=0.7,
            max_tokens=500
        )
        
        reply = completion.choices[0].message.content
        print(f"✅ Response sent: {reply[:100]}...")
        print("=" * 50)
        
        return jsonify({
            'success': True,
            'data': {'response': reply, 'query': query}
        })
        
    except Exception as e:
        print(f"❌ Chat Error: {e}")
        print(f"❌ Full traceback:")
        traceback.print_exc()
        print("=" * 50)
        return jsonify({
            'success': False,
            'error': str(e),
            'data': {'response': f"Error: {str(e)}"}
        }), 500

if __name__ == '__main__':
    print("🚀 Debug ML Service on http://localhost:5001")
    print("📊 Health: http://localhost:5001/api/health")
    print("💬 Chat: http://localhost:5001/api/chat")
    app.run(debug=True, port=5001, host='0.0.0.0')
