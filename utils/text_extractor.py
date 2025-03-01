import pytesseract
from PIL import Image
import re
import logging
import cv2
import numpy as np
from typing import List, Optional, Dict, Tuple
import os
import io
import base64

class TextExtractor:
    def __init__(self, tesseract_path: Optional[str] = None):
        """Initialize the WhatsApp chat extractor with optional tesseract path."""
        self.tesseract = pytesseract
        if tesseract_path:
            self.tesseract.pytesseract.tesseract_cmd = tesseract_path

        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # Regular expressions for pattern matching
        self.time_pattern = re.compile(r'\d{1,2}:\d{2}(?:\s*[AP]M)?')
        self.date_pattern = re.compile(r'\d{1,2}/\d{1,2}/\d{2,4}')
        self.sender_pattern = re.compile(r'([^:]+):\s')

    def preprocess_image(self, image_path_or_data) -> Image.Image:
        """
        Enhanced preprocessing with multiple techniques for optimal OCR results.
        
        Args:
            image_path_or_data: Either a file path or base64 encoded image data
            
        Returns:
            Processed PIL Image
        """
        try:
            # Handle either file path or base64 data
            if isinstance(image_path_or_data, str):
                if image_path_or_data.startswith('data:image'):
                    # Extract base64 data from data URL
                    base64_data = image_path_or_data.split(',')[1]
                    image_data = base64.b64decode(base64_data)
                    image = Image.open(io.BytesIO(image_data))
                else:
                    # Regular file path
                    image = Image.open(image_path_or_data)
            else:
                # Assume bytes or BytesIO object
                image = Image.open(io.BytesIO(image_path_or_data))
            
            # Convert to OpenCV format
            img_array = np.array(image)
            
            # Check if image is grayscale or color
            if len(img_array.shape) == 3:
                # Color image
                img_cv = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
                gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
            else:
                # Already grayscale
                gray = img_array
            
            # Create multiple preprocessing variations for better OCR results
            processed_images = []
            
            # Version 1: Basic denoising with adaptive threshold
            denoised = cv2.fastNlMeansDenoising(gray, None, h=10, templateWindowSize=7, searchWindowSize=21)
            thresh1 = cv2.adaptiveThreshold(
                denoised, 255,
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY, 11, 2
            )
            processed_images.append(thresh1)
            
            # Version 2: Sharpening for better text definition
            kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
            sharpened = cv2.filter2D(gray, -1, kernel)
            thresh2 = cv2.adaptiveThreshold(
                sharpened, 255,
                cv2.ADAPTIVE_THRESH_MEAN_C,
                cv2.THRESH_BINARY, 11, 2
            )
            processed_images.append(thresh2)
            
            # Version 3: CLAHE (Contrast Limited Adaptive Histogram Equalization)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            enhanced = clahe.apply(gray)
            _, thresh3 = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            processed_images.append(thresh3)
            
            # Return the first processed image (we'll use all versions during OCR)
            return [Image.fromarray(img) for img in processed_images]
            
        except Exception as e:
            self.logger.error(f"Preprocessing error: {e}")
            raise

    def extract_text(self, images: List[Image.Image]) -> str:
        """
        Extract text using multiple processing techniques and select the best result.
        
        Args:
            images: List of preprocessed images to extract text from
            
        Returns:
            Extracted text string
        """
        try:
            best_text = ""
            max_score = 0
            
            configs = [
                r'--oem 3 --psm 6 -l eng+hin+ara+spa+fra -c preserve_interword_spaces=1 --dpi 300',
                r'--oem 3 --psm 1 -l eng+hin+ara+spa+fra',
                r'--oem 3 --psm 4 -l eng+hin+ara+spa+fra'
            ]
            
            for image in images:
                for config in configs:
                    text = self.tesseract.image_to_string(image, config=config)
                    
                    # Score the quality of extraction based on patterns we expect
                    score = self._score_extraction(text)
                    
                    if score > max_score:
                        max_score = score
                        best_text = text
            
            return best_text
            
        except Exception as e:
            self.logger.error(f"OCR extraction error: {e}")
            raise

    def _score_extraction(self, text: str) -> int:
        """
        Score the extraction quality based on WhatsApp message patterns.
        
        Args:
            text: The extracted text to score
            
        Returns:
            Score value (higher is better)
        """
        score = 0
        
        # Check for timestamps (common in WhatsApp messages)
        timestamps = self.time_pattern.findall(text)
        score += len(timestamps) * 5
        
        # Check for message patterns (sender: message)
        sender_matches = self.sender_pattern.findall(text)
        score += len(sender_matches) * 10
        
        # Penalize very short texts that might be noise
        if len(text) < 20:
            score -= 20
            
        # Favor texts with multiple lines (likely contains multiple messages)
        lines = [line for line in text.split('\n') if line.strip()]
        score += min(len(lines), 10) * 3
        
        return score

    def process_content(self, text: str) -> Dict[str, List[str]]:
        """
        Process the extracted text into structured messages with sender information.
        
        Args:
            text: Raw extracted text
            
        Returns:
            Dictionary with 'messages' and 'structured_chat' keys
        """
        raw_messages = []
        structured_chat = []
        current_sender = None
        
        # Split into lines and process each
        lines = text.split('\n')
        for i, line in enumerate(lines):
            line = line.strip()
            if not line:
                continue
                
            # Try to detect timestamp and sender pattern
            sender_match = self.sender_pattern.search(line)
            
            if sender_match:
                # This line starts a new message
                sender = sender_match.group(1).strip()
                content = line[sender_match.end():].strip()
                
                # Clean the sender name from potential OCR errors
                sender = self._clean_text(sender)
                current_sender = sender
                
                if content:  # Only add if there's actual content
                    raw_messages.append(content)
                    structured_chat.append({"sender": sender, "content": content})
            elif current_sender and i > 0:
                # This is likely a continuation of the previous message
                # Check if previous line had a sender and this is a continuation
                content = self._clean_text(line)
                
                if content and len(content) > 3:
                    # Append to the previous message if it exists
                    if structured_chat:
                        # Check if this might actually be a new message without clear sender
                        if self._is_likely_new_message(content, structured_chat[-1]["content"]):
                            structured_chat.append({"sender": "Unknown", "content": content})
                            raw_messages.append(content)
                        else:
                            structured_chat[-1]["content"] += f" {content}"
                            raw_messages[-1] += f" {content}"
        
        # Final cleaning pass to remove noise and duplicates
        cleaned_chat = self._remove_duplicates_and_noise(structured_chat)
        
        return {
            "messages": [msg["content"] for msg in cleaned_chat],
            "structured_chat": cleaned_chat
        }

    def _is_likely_new_message(self, current_text: str, previous_text: str) -> bool:
        """
        Determine if text is likely a new message rather than continuation.
        
        Args:
            current_text: Current line of text
            previous_text: Previous message content
            
        Returns:
            Boolean indicating if this is likely a new message
        """
        # If current text starts with capital letter and previous ends with punctuation
        if (current_text[0].isupper() and 
            previous_text[-1] in '.!?'):
            return True
            
        # If current text is much longer than what would be a typical continuation
        if len(current_text) > 30:
            return True
            
        # If current text contains typical message starters
        starters = ['hi', 'hello', 'hey', 'ok', 'yes', 'no', 'thanks']
        if any(current_text.lower().startswith(s) for s in starters):
            return True
            
        return False

    def _clean_text(self, text: str) -> str:
        """
        Clean text from OCR artifacts and normalize content.
        
        Args:
            text: Text to clean
            
        Returns:
            Cleaned text
        """
        # Remove timestamps and non-essential metadata
        text = re.sub(r'\d{1,2}:\d{2}(?:\s*[AP]M)?', '', text)
        
        # Fix common OCR errors
        replacements = {
            r'\bBnalenge\b': 'Banalenge',
            r'\banaeyaat\b': 'anayata',
            r'\beS\b': '',
            r'\bNu\b': '',
            r'\bQ a ee\b': '',
            r'\bII O\b': '',
            r'\b0k\b': 'OK',
            r'\bl\b': 'I',
            r'\blo\b': 'to',
            r'\b1\b': 'I',
            r'\bjusi\b': 'just',
            r'\bwhai\b': 'what',
            r'\binsiead\b': 'instead'
        }
        
        for pattern, replacement in replacements.items():
            text = re.sub(pattern, replacement, text)
        
        # Remove strange punctuation and character repetitions
        text = re.sub(r'[^\w\s\.\,\!\?\:\;\(\)\[\]\{\}\-\"\'\–\—]', '', text)
        text = re.sub(r'\.{2,}', '...', text)  # normalize ellipsis
        
        # Fix spacing issues
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text

    def _remove_duplicates_and_noise(self, messages: List[Dict]) -> List[Dict]:
        """
        Remove duplicate messages and noise from the structured chat.
        
        Args:
            messages: List of message dictionaries
            
        Returns:
            Cleaned list of message dictionaries
        """
        if not messages:
            return []
            
        cleaned = []
        seen_content = set()
        
        for msg in messages:
            content = msg["content"].strip()
            
            # Skip if too short or already seen
            if len(content) < 3 or content.lower() in seen_content:
                continue
                
            # Skip if it looks like OCR garbage (too many non-alphanumeric chars)
            alpha_ratio = sum(c.isalnum() for c in content) / len(content) if content else 0
            if alpha_ratio < 0.5 and len(content) < 10:
                continue
                
            cleaned.append(msg)
            seen_content.add(content.lower())
            
        return cleaned

    def save_to_file(self, messages: Dict, output_path: str):
        """
        Save processed messages to file in both raw and structured format.
        
        Args:
            messages: Dictionary with 'messages' and 'structured_chat' keys
            output_path: Path to save the output file
        """
        base_path, ext = os.path.splitext(output_path)
        raw_path = f"{base_path}_raw{ext}"
        structured_path = f"{base_path}_structured{ext}"
        
        # Save raw messages
        with open(raw_path, 'w', encoding='utf-8') as f:
            for msg in messages["messages"]:
                f.write(f"{msg}\n")
                
        # Save structured chat (JSON-like format)
        with open(structured_path, 'w', encoding='utf-8') as f:
            for msg in messages["structured_chat"]:
                f.write(f"{msg['sender']}: {msg['content']}\n")
                
        self.logger.info(f"Saved {len(messages['messages'])} messages to {raw_path} and {structured_path}")
        
        return {
            "raw_path": raw_path,
            "structured_path": structured_path
        }

    def process_chat_image(self, image_path_or_data, output_path: Optional[str] = None) -> Dict:
        """
        Full processing pipeline from image to structured chat data.
        
        Args:
            image_path_or_data: Path to image or base64 encoded image data
            output_path: Optional path to save output files
            
        Returns:
            Dictionary with processed chat data
        """
        self.logger.info(f"Processing chat image")
        
        try:
            # Preprocess image to get multiple versions
            preprocessed_images = self.preprocess_image(image_path_or_data)
            
            # Extract text using multiple techniques
            raw_text = self.extract_text(preprocessed_images)
            
            # Process content into structured format
            processed_messages = self.process_content(raw_text)
            
            # Save to file if output path provided
            file_paths = {}
            if output_path:
                file_paths = self.save_to_file(processed_messages, output_path)
            
            # Return all the data
            return {
                "raw_text": raw_text,
                "processed": processed_messages,
                "file_paths": file_paths
            }
            
        except Exception as e:
            self.logger.error(f"Processing failed: {e}")
            raise

    @staticmethod
    def from_base64(base64_data: str, output_path: Optional[str] = None, tesseract_path: Optional[str] = None) -> Dict:
        """
        Static method to process base64 encoded image data.
        
        Args:
            base64_data: Base64 encoded image data
            output_path: Optional path to save output files
            tesseract_path: Optional path to tesseract executable
            
        Returns:
            Dictionary with processed chat data
        """
        extractor = TextExtractor(tesseract_path=tesseract_path)
        return extractor.process_chat_image(base64_data, output_path)

    @staticmethod
    def auto_detect_tesseract() -> Optional[str]:
        """
        Automatically detect Tesseract installation path based on OS.
        
        Returns:
            Path to tesseract executable or None if not found
        """
        common_paths = [
            # Windows paths
            r'C:\Program Files\Tesseract-OCR\tesseract.exe',
            r'C:\Program Files (x86)\Tesseract-OCR\tesseract.exe',
            # Linux paths
            '/usr/bin/tesseract',
            '/usr/local/bin/tesseract',
            # macOS paths
            '/usr/local/Cellar/tesseract/*/bin/tesseract',
            '/opt/homebrew/bin/tesseract'
        ]
        
        for path in common_paths:
            if '*' in path:
                # Handle wildcard paths (for homebrew)
                import glob
                for matched_path in glob.glob(path):
                    if os.path.exists(matched_path) and os.access(matched_path, os.X_OK):
                        return matched_path
            elif os.path.exists(path):
                return path
                
        return None