from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from werkzeug.utils import secure_filename
import os
import logging
from utils.text_extractor import TextExtractor
from utils.chat_generator import ChatGenerator
from config import Config

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)
app.config.from_object(Config)

# Initialize components with proper error handling
try:
    # Try to auto-detect Tesseract
    tesseract_path = TextExtractor.auto_detect_tesseract()
    if tesseract_path:
        logger.info(f"Tesseract found at: {tesseract_path}")
    else:
        logger.warning("Tesseract executable not found. OCR may not work correctly.")
    
    text_extractor = TextExtractor(tesseract_path=tesseract_path)
    chat_generator = ChatGenerator()
    logger.info("Components initialized successfully")
except Exception as e:
    logger.error(f"Error initializing components: {e}")
    raise

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in Config.ALLOWED_EXTENSIONS

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/extract-text', methods=['POST'])
def extract_text():
    if 'image' not in request.files:
        logger.warning("No image file provided in request")
        return jsonify({'error': 'No image file provided'}), 400
    
    file = request.files['image']
    if file.filename == '':
        logger.warning("Empty filename provided")
        return jsonify({'error': 'No selected file'}), 400
    
    if file and allowed_file(file.filename):
        try:
            # Create uploads folder if it doesn't exist
            os.makedirs(Config.UPLOAD_FOLDER, exist_ok=True)
            
            # Save the file temporarily
            filename = secure_filename(file.filename)
            filepath = os.path.join(Config.UPLOAD_FOLDER, filename)
            file.save(filepath)
            
            logger.info(f"Processing image: {filename}")
            
            # Process the chat image
            result = text_extractor.process_chat_image(filepath)
            
            # Return the processed messages - FIXED ACCESS PATTERN
            return jsonify({
                'success': True,
                'messages': result['messages'],
                'structured_chat': result['structured_chat']
            })
        except Exception as e:
            logger.error(f"Error processing image: {e}")
            return jsonify({'error': str(e)}), 500
    
    logger.warning(f"Invalid file type: {file.filename}")
    return jsonify({'error': 'Invalid file type'}), 400

@app.route('/api/generate-responses', methods=['POST'])
def generate_response():
    data = request.json
    if not data or 'chat_text' not in data:
        logger.warning("No chat text provided")
        return jsonify({'success': False, 'error': 'No chat text provided'}), 400
    
    try:
        chat_text = data['chat_text']
        style = data.get('style', 'friendly')
        message_length = data.get('message_length', 'medium')
        
        logger.info(f"Generating 1 response with style: {style}, length: {message_length}")
        
        # Parse chat history from text (handling both string and list inputs)
        if isinstance(chat_text, str):
            chat_history = chat_generator.parse_chat_text(chat_text)
        else:
            chat_history = chat_text
        
        # Generate a single response instead of multiple
        response = chat_generator.generate_response(
            chat_history, 
            style=style,
            message_length=message_length
        )
        
        # Return as a list with one item for consistency with frontend
        return jsonify({'success': True, 'responses': [response]})
    except Exception as e:
        logger.error(f"Error generating response: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/upload-base64', methods=['POST'])
@app.route('/api/upload-base64', methods=['POST'])
def upload_base64():
    """Handle base64 encoded image uploads"""
    data = request.json
    if not data or 'image_data' not in data:
        logger.warning("No base64 image data provided")
        return jsonify({'error': 'No image data provided'}), 400
    
    try:
        base64_data = data['image_data']
        logger.info("Processing base64 image data")
        
        # Process the base64 image directly
        result = TextExtractor.from_base64(base64_data)
        
        return jsonify({
            'success': True,
            'messages': result['messages'],
            'structured_chat': result['structured_chat']
        })
    except Exception as e:
        logger.error(f"Error processing base64 image: {e}")
        return jsonify({'error': str(e)}), 500
# Add a new endpoint for direct text input without image
@app.route('/api/process-text', methods=['POST'])
def process_text():
    data = request.json
    if not data or 'text' not in data:
        logger.warning("No text provided")
        return jsonify({'success': False, 'error': 'No text provided'}), 400
    
    try:
        text = data['text']
        logger.info("Processing direct text input")
        
        # Parse the text into messages
        messages = chat_generator.parse_chat_text(text)
        
        # For structured format, we'll create a simple version
        structured_chat = []
        for i, msg in enumerate(messages):
            sender = "User" if i % 2 == 0 else "Friend"
            structured_chat.append({"sender": sender, "content": msg})
        
        return jsonify({
            'success': True,
            'messages': messages,
            'structured_chat': structured_chat
        })
    except Exception as e:
        logger.error(f"Error processing text: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

if __name__ == '__main__':
    # Create uploads folder if it doesn't exist
    os.makedirs(Config.UPLOAD_FOLDER, exist_ok=True)
    logger.info(f"Starting Flask server on http://127.0.0.1:5000")
    app.run(debug=True)