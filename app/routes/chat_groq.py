load_dotenv()
from dotenv import load_dotenv
"""
Real AI Chat using Groq API
"""

from flask import Blueprint, request, jsonify
from flask_cors import cross_origin
from groq import Groq
import os

chat_bp = Blueprint('chat_groq', __name__, url_prefix='/api')

# Initialize Groq client with your API key
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

SYSTEM_PROMPT = """You are an AI assistant for the Innovation Incubator Platform. 
Your role is to help entrepreneurs and innovators with:

1. Understanding their evaluation scores (Innovation, Feasibility, Market Potential, Technical Complexity, Social Impact, Financial Viability)
2. Providing actionable advice to improve their ideas
3. Explaining the platform's features (funding, mentorship, resources)
4. Offering guidance on entrepreneurship and innovation
5. Answering questions about the evaluation process

Be helpful, professional, and encouraging. Always provide specific, actionable advice based on the user's context.
If the user provides evaluation context (scores, idea title, etc.), use it to give personalized advice.
Keep responses concise but informative (2-3 paragraphs maximum).
If the user asks about something unrelated to the platform, politely redirect them to platform-related topics.

Current Date: August 2026
"""

@chat_bp.route('/chat', methods=['POST', 'OPTIONS'])
@cross_origin(origins=["http://localhost:5173", "http://localhost:5001", "*"])
def chat():
    """AI-powered chat using Groq API"""
    if request.method == 'OPTIONS':
        return '', 200
    
    try:
        data = request.json
        query = data.get('query', '')
        context = data.get('context', {})
        
        print(f"💬 Groq Chat Query: {query}")
        
        # Build context
        context_text = build_context(context)
        
        # Build messages
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT}
        ]
        
        if context_text:
            messages.append({
                "role": "system", 
                "content": f"Current evaluation context: {context_text}"
            })
        
        messages.append({"role": "user", "content": query})
        
        # Call Groq API
        completion = client.chat.completions.create(
            model="mixtral-8x7b-32768",
            messages=messages,
            temperature=0.7,
            max_tokens=500,
            top_p=1
        )
        
        reply = completion.choices[0].message.content
        
        print(f"✅ Groq Response: {reply[:100]}...")
        
        return jsonify({
            'success': True,
            'data': {
                'response': reply,
                'query': query
            }
        })
        
    except Exception as e:
        print(f"❌ Groq Error: {e}")
        return jsonify({
            'success': False,
            'error': str(e),
            'data': {
                'response': "I'm having trouble connecting to the AI service. Please try again later."
            }
        }), 500

def build_context(context):
    """Build a text description of the evaluation context"""
    if not context:
        return ""
    
    parts = []
    
    if context.get('title'):
        parts.append(f"Idea: {context['title']}")
    
    if context.get('overallScore'):
        parts.append(f"Overall Score: {context['overallScore']}%")
    
    if context.get('rating'):
        parts.append(f"Rating: {context['rating']}")
    
    if context.get('scores'):
        scores = context['scores']
        score_text = ", ".join([f"{k}: {v}%" for k, v in scores.items()])
        parts.append(f"Scores: {score_text}")
    
    if context.get('feedback'):
        if isinstance(context['feedback'], list):
            parts.append(f"Key Feedback: {', '.join(context['feedback'][:2])}")
        else:
            parts.append(f"Feedback: {context['feedback']}")
    
    return " | ".join(parts)
