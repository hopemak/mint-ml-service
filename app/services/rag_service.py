"""
RAG Service - Simple RAG implementation, now context-aware.

generate_response() previously accepted `context` but never read it, so every
answer came from a 3-sentence static knowledge base regardless of what idea
the user was asking about. This version actually uses `context` (the current
idea's title/scores/feedback, sent from the frontend) to answer
score/risk/improvement questions with the real numbers, and only falls back
to the generic knowledge-base lookup for general platform questions.
"""

import re
import numpy as np
from .embedding_service import EmbeddingService


class RAGService:
    def __init__(self, knowledge_base_dir='./data/knowledge'):
        self.embedding_service = EmbeddingService()
        self.knowledge_docs = [
            {'id': 'doc_001', 'title': 'Platform Overview',
             'content': 'The Innovation Incubator Platform is a comprehensive ecosystem for nurturing innovation.'},
            {'id': 'doc_002', 'title': 'Idea Evaluation',
             'content': 'The platform uses NLP and ML for idea classification and scoring.'},
            {'id': 'doc_003', 'title': 'Mentor Matching',
             'content': 'The system matches startups with mentors based on expertise and compatibility.'},
        ]

    def retrieve_documents(self, query, top_k=3):
        if not query:
            return []
        scored = []
        for doc in self.knowledge_docs:
            sim = self.embedding_service.cosine_similarity(query, doc['content'])
            scored.append((doc, sim))
        scored.sort(key=lambda x: x[1], reverse=True)
        return [doc for doc, _ in scored[:top_k]]

    # --- NEW: real, context-aware answers about the current idea ---

    DIMENSION_ALIASES = {
        'innovation': 'innovation_score',
        'novelty': 'innovation_score',
        'feasib': 'feasibility',
        'market': 'market_potential',
        'social': 'social_impact',
        'impact': 'social_impact',
        'financ': 'financial_viability',
        'technical': 'technical_complexity',
        'complex': 'technical_complexity',
        'risk': 'risk_score',
    }

    DIMENSION_LABELS = {
        'innovation_score': 'Innovation',
        'feasibility': 'Feasibility',
        'market_potential': 'Market Potential',
        'social_impact': 'Social Impact',
        'financial_viability': 'Financial Viability',
        'technical_complexity': 'Technical Complexity',
        'risk_score': 'Risk',
    }

    def _find_mentioned_dimension(self, query_lower):
        for alias, key in self.DIMENSION_ALIASES.items():
            if alias in query_lower:
                return key
        return None

    def _answer_from_context(self, query, context):
        """Return a real answer built from the idea's actual scores/feedback,
        or None if the question doesn't look like it's about the idea."""
        if not context:
            return None

        query_lower = query.lower()
        title = context.get('title', 'this idea')
        scores = context.get('scores', {})
        overall = context.get('overallScore')
        feedback = context.get('feedback', '')

        # "why is my X score low/what is my X score" style questions
        dim_key = self._find_mentioned_dimension(query_lower)
        if dim_key and (scores.get(dim_key) is not None or context.get(dim_key) is not None):
            value = scores.get(dim_key, context.get(dim_key))
            label = self.DIMENSION_LABELS.get(dim_key, dim_key)
            if value is None:
                return None
            if value >= 80:
                tone = f"That's a strong score — {label} is one of the stronger dimensions of \"{title}\"."
            elif value >= 60:
                tone = f"That's a moderate score for {label} on \"{title}\" — there's room to strengthen it."
            else:
                tone = f"That's a low score for {label} on \"{title}\", which is worth addressing before resubmitting."
            return f"Your {label} score is {value}/100. {tone}"

        # Overall score / summary questions
        if any(w in query_lower for w in ['overall', 'total score', 'how did i do', 'summary', 'rating']):
            if overall is not None:
                return f'"{title}" scored {overall}/100 overall. {feedback}'.strip()

        # "how can I improve" / recommendation questions
        if any(w in query_lower for w in ['improve', 'better', 'increase', 'recommend', 'suggestion', 'weak']):
            if feedback:
                # feedback string ends with "Recommendations: A; B; C" when present
                match = re.search(r'Recommendations:\s*(.+)$', feedback)
                if match:
                    return f"Based on \"{title}\"'s evaluation, here's what to focus on: {match.group(1)}"
                return feedback

        return None

    def generate_response(self, query, user_id, context=None):
        # 1. Try to answer directly from the real idea data first.
        context_answer = self._answer_from_context(query, context)
        if context_answer:
            return {
                'response': context_answer,
                'citations': [],
                'confidence': 0.95,
                'source': 'idea_context',
            }

        # 2. Fall back to the generic knowledge-base lookup for platform questions.
        docs = self.retrieve_documents(query)
        if not docs:
            return {
                'response': "I don't have specific information about that. Try asking about your idea's scores (e.g. \"why is my feasibility score low?\") or general platform questions.",
                'citations': [],
                'confidence': 0.5,
                'source': 'none',
            }
        response = f"Based on our knowledge base: {docs[0]['content']}"
        return {
            'response': response,
            'citations': [{'title': d['title'], 'id': d['id']} for d in docs[:3]],
            'confidence': 0.85,
            'source': 'knowledge_base',
        }