"""
ML AI Service Main Entry Point
"""

import sys
import os
from pathlib import Path

# Add the parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from app import create_app

app = create_app()

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    debug = os.environ.get('FLASK_ENV') == 'development'
    
    print(f"Starting ML AI Service on port {port}")
    app.run(host='0.0.0.0', port=port, debug=debug)