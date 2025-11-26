# server.py
"""
Flask server for web scraping endpoint
Configured for ScriptBees.com
"""

from flask import Flask, request, jsonify
from scrape.scraper import run_scrape
from flask_cors import CORS

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})  # Dev only; restrict in production

@app.route('/scrape', methods=['POST'])
def scrape_endpoint():
    """
    POST JSON: { 
        "start_url": "https://scriptbees.com", 
        "max_pages": 50, 
        "delay": 1.5, 
        "use_js_render": true 
    }
    
    Blocks until scrape finishes. For long jobs, implement background job and return job id.
    """
    data = request.get_json() or {}
    start_url = data.get('start_url', 'https://scriptbees.com')  # Default to ScriptBees
    
    if not start_url:
        return jsonify({'error': 'start_url required'}), 400

    try:
        max_pages = int(data.get('max_pages', 50))  # Increased default for ScriptBees
        delay = float(data.get('delay', 1.5))
        use_js = bool(data.get('use_js_render', False))
    except Exception:
        return jsonify({'error': 'invalid parameters'}), 400

    print(f"\n🐝 Scraping ScriptBees: {start_url}")
    print(f"   Max pages: {max_pages}")
    print(f"   Delay: {delay}s")
    print(f"   JS Render: {use_js}")

    # Run the scraper synchronously (blocking). For production, run in background worker.
    pages, output_path, sample = run_scrape(
        start_url, 
        max_pages=max_pages, 
        output_dir='content', 
        delay=delay, 
        use_js_render=use_js
    )
    
    return jsonify({
        'success': True,
        'pages_count': len(pages),
        'output_path': output_path,
        'sample_pages': sample,
        'message': f'Successfully scraped {len(pages)} pages from ScriptBees'
    })

@app.route('/health', methods=['GET'])
def health():
    return jsonify({
        'status': 'ok',
        'service': 'ScriptBees Scraper',
        'version': '1.0'
    })

@app.route('/', methods=['GET'])
def index():
    return jsonify({
        'service': 'ScriptBees Web Scraper',
        'endpoints': {
            'scrape': '/scrape (POST)',
            'health': '/health (GET)'
        },
        'example_request': {
            'start_url': 'https://scriptbees.com',
            'max_pages': 50,
            'delay': 1.5,
            'use_js_render': False
        }
    })

if __name__ == '__main__':
    print("\n" + "="*60)
    print("🐝 SCRIPTBEES WEB SCRAPER SERVER")
    print("="*60)
    print("Endpoints:")
    print("  POST /scrape - Scrape website")
    print("  GET  /health - Health check")
    print("="*60 + "\n")
    
    app.run(host='0.0.0.0', port=5000, debug=True)