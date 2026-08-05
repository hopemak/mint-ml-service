"""
Real AI Chat using OpenAI API
"""

from flask import Blueprint, request, jsonify
from flask_cors import cross_origin
from openai import OpenAI
import os

chat_bp = Blueprint('chat_openai', __name__, url_prefix='/api')

# Initialize OpenAI client
client = OpenAI(
    api_key=os.environ.get('OPENAI_API_KEY', 'your-api-key-here')
)

# System prompt - defines the chatbot's behavior
SYSTEM_PROMPT = """You are an AI assistant for the Innovation Incubator Platform. 
Your role is to help entrepreneurs and innovators with:

1. Understanding their evaluation scores (Innovation, Feasibility, Market Potential, Technical Complexity, Social Impact, Financial Viability)
2. Providing actionable advice to improve their ideas
3. Explaining the platform's features (funding, mentorship, resources)
4. Offering guidance on entrepreneurship and innovation
5. Answering questions about the evaluation process

Be helpful, professional, and encouraging. Always provide specific, actionable advice based on the user's context.

If the user provides evaluation context (scores, idea title, etc.), use it to give personalized advice.
If the user asks about something unrelated to the platform, politely redirect them to platform-related topics.
"""

@chat_bp.route('/chat', methods=['POST', 'OPTIONS'])
@cross_origin(origins=["http://localhost:5173", "*"])
def chat():
    """AI-powered chat using OpenAI"""
    if request.method == 'OPTIONS':
        return '', 200
    
    try:
        data = request.json
        query = data.get('query', '')
        context = data.get('context', {})
        
        print(f"💬 Real AI Chat: {query}")
        
        # Build the messages
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT}
        ]
        
        # Add context if available
        if context:
            context_text = build_context(context)
            messages.append({
                "role": "system",
                "content": f"Current evaluation context: {context_text}"
            })
        
        # Add user query
        messages.append({"role": "user", "content": query})
        
        # Call OpenAI API
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",  # or "gpt-4" for better responses
            messages=messages,
            temperature=0.7,
            max_tokens=500
        )
        
        reply = response.choices[0].message.content
        
        print(f"✅ AI Response: {reply[:100]}...")
        
        return jsonify({
            'success': True,
            'data': {
                'response': reply,
                'query': query
            }
        })
        
    except Exception as e:
        print(f"❌ Chat error: {e}")
        return jsonify({
            'success': False,
            'error': str(e),
            'data': {
                'response': "I'm having trouble connecting to the AI service. Please try again later."
            }
        }), 500

def build_context(context):
    """Build a text description of the evaluation context"""
    parts = []
    
    if context.get('title'):
        parts.append(f"Idea: {context['title']}")
    
    if context.get('overallScore'):
        parts.append(f"Overall Score: {context['overallScore']}%")
    
    if context.get('scores'):
        scores = context['scores']
        score_text = ", ".join([f"{k}: {v}%" for k, v in scores.items()])
        parts.append(f"Scores: {score_text}")
    
    if context.get('feedback') and isinstance(context['feedback'], list):
        parts.append(f"Feedback: {', '.join(context['feedback'][:2])}")
    
    return " | ".join(parts)
