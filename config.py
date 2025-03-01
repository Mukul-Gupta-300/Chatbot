import os

class Config:
    # Upload folder for temporary image storage
    UPLOAD_FOLDER = os.path.join(os.getcwd(), 'uploads')
    
    # Allowed file extensions for image uploads
    ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp', 'tiff', 'webp'}
    
    # Maximum content length for uploads (10MB)
    MAX_CONTENT_LENGTH = 10 * 1024 * 1024
    
    # Default style for responses
    DEFAULT_STYLE = 'friendly'
    
    # Flask secret key (change this in production!)
    SECRET_KEY = 'dev-secret-key-change-this-in-production'