"""
Document Upload Routes
"""

from flask import Blueprint, request, jsonify
from werkzeug.utils import secure_filename
import os
import uuid
from datetime import datetime

documents_bp = Blueprint('documents', __name__)

UPLOAD_FOLDER = './uploads'
ALLOWED_EXTENSIONS = {'pdf', 'doc', 'docx', 'ppt', 'pptx', 'xls', 'xlsx', 'png', 'jpg', 'jpeg', 'txt'}

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@documents_bp.route('/upload', methods=['POST'])
def upload_file():
    """Upload a document"""
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        if not allowed_file(file.filename):
            return jsonify({'error': f'File type not allowed. Allowed: {", ".join(ALLOWED_EXTENSIONS)}'}), 400
        
        # Secure filename
        filename = secure_filename(file.filename)
        unique_id = str(uuid.uuid4())[:8]
        timestamp = datetime.utcnow().strftime('%Y%m%d_%H%M%S')
        new_filename = f"{timestamp}_{unique_id}_{filename}"
        
        filepath = os.path.join(UPLOAD_FOLDER, new_filename)
        file.save(filepath)
        
        return jsonify({
            'success': True,
            'message': 'File uploaded successfully',
            'data': {
                'filename': filename,
                'saved_as': new_filename,
                'path': filepath,
                'size': os.path.getsize(filepath)
            }
        }), 201
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@documents_bp.route('/uploads', methods=['GET'])
def list_uploads():
    """List all uploaded files"""
    try:
        files = []
        for f in os.listdir(UPLOAD_FOLDER):
            filepath = os.path.join(UPLOAD_FOLDER, f)
            if os.path.isfile(filepath):
                files.append({
                    'filename': f,
                    'size': os.path.getsize(filepath),
                    'uploaded': datetime.fromtimestamp(os.path.getctime(filepath)).isoformat()
                })
        return jsonify({'success': True, 'data': files}), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500
