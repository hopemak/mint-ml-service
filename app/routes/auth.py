"""
Authentication Routes - Fixed Version
"""

from flask import Blueprint, request, jsonify
import bcrypt
import uuid
from datetime import datetime

from app.utils.jwt_helper import create_token, token_required

auth_bp = Blueprint('auth', __name__)

# In-memory user storage (replace with database)
users = {}

@auth_bp.route('/register', methods=['POST'])
def register():
    """Register a new user"""
    data = request.get_json()
    
    # Debug: print what we received
    print(f"📥 Registration request received: {data}")
    
    if not data:
        return jsonify({'error': 'No data provided'}), 400
    
    # Check required fields (support both camelCase and snake_case)
    email = data.get('email')
    password = data.get('password')
    full_name = data.get('full_name') or data.get('fullName')
    role = data.get('role', 'founder')
    
    if not email:
        return jsonify({'error': 'Email is required'}), 400
    
    if not password:
        return jsonify({'error': 'Password is required'}), 400
    
    if not full_name:
        return jsonify({'error': 'Full name is required'}), 400
    
    email = email.lower().strip()
    
    # Check if user already exists
    if email in users:
        return jsonify({'error': 'User already exists'}), 409
    
    # Hash password
    hashed = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt())
    
    user_id = str(uuid.uuid4())
    user = {
        'id': user_id,
        'email': email,
        'password': hashed.decode('utf-8'),
        'full_name': full_name.strip(),
        'role': role,
        'created_at': datetime.utcnow().isoformat()
    }
    
    users[email] = user
    
    # Create JWT token
    token = create_token(user_id, email, user['role'])
    
    # Return user without password
    user_data = {k: v for k, v in user.items() if k != 'password'}
    
    print(f"✅ User registered: {email}")
    
    return jsonify({
        'success': True,
        'message': 'User registered successfully',
        'data': {
            'user': user_data,
            'token': token
        }
    }), 201

@auth_bp.route('/login', methods=['POST'])
def login():
    """Login a user"""
    data = request.get_json()
    
    print(f"📥 Login request received: {data}")
    
    if not data:
        return jsonify({'error': 'No data provided'}), 400
    
    email = data.get('email', '').lower().strip()
    password = data.get('password', '')
    
    if not email or not password:
        return jsonify({'error': 'Email and password required'}), 400
    
    if email not in users:
        return jsonify({'error': 'Invalid credentials'}), 401
    
    user = users[email]
    
    # Verify password
    if not bcrypt.checkpw(password.encode('utf-8'), user['password'].encode('utf-8')):
        return jsonify({'error': 'Invalid credentials'}), 401
    
    # Create new token
    token = create_token(user['id'], email, user['role'])
    
    user_data = {k: v for k, v in user.items() if k != 'password'}
    
    return jsonify({
        'success': True,
        'message': 'Login successful',
        'data': {
            'user': user_data,
            'token': token
        }
    }), 200

@auth_bp.route('/me', methods=['GET'])
@token_required
def get_current_user():
    """Get current user info"""
    user_payload = getattr(request, 'user', None)
    
    if not user_payload:
        return jsonify({'error': 'User not found'}), 404
    
    email = user_payload.get('email')
    if email not in users:
        return jsonify({'error': 'User not found'}), 404
    
    user = users[email]
    user_data = {k: v for k, v in user.items() if k != 'password'}
    
    return jsonify({
        'success': True,
        'data': {
            'user': user_data
        }
    }), 200

@auth_bp.route('/logout', methods=['POST'])
@token_required
def logout():
    """Logout user (client-side token removal)"""
    return jsonify({
        'success': True,
        'message': 'Logged out successfully'
    }), 200

@auth_bp.route('/refresh', methods=['POST'])
@token_required
def refresh_token():
    """Refresh JWT token"""
    user_payload = getattr(request, 'user', None)
    
    if not user_payload:
        return jsonify({'error': 'User not found'}), 404
    
    user_id = user_payload.get('user_id')
    email = user_payload.get('email')
    role = user_payload.get('role')
    
    # Create new token
    new_token = create_token(user_id, email, role)
    
    return jsonify({
        'success': True,
        'data': {
            'token': new_token
        }
    }), 200
