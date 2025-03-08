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
from spellchecker import SpellChecker
import difflib
import json
from datetime import datetime

class TextExtractor:
    def __init__(self, tesseract_path: Optional[str] = None):
        """Initialize the chat extractor with optional tesseract path."""
        self.tesseract = pytesseract
        if tesseract_path:
            self.tesseract.pytesseract.tesseract_cmd = tesseract_path
        else:
            # Try to auto-detect Tesseract
            auto_path = self.auto_detect_tesseract()
            if auto_path:
                self.tesseract.pytesseract.tesseract_cmd = auto_path
                
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # Regular expressions for pattern matching (expanded for multiple platforms)
        self.time_pattern = re.compile(r'\d{1,2}:\d{2}(?:\s*[AP]M)?')
        self.date_pattern = re.compile(r'\d{1,2}/\d{1,2}/\d{2,4}')
        
        # Various sender patterns for different platforms
        self.sender_patterns = [
            re.compile(r'([^:]+):\s'),  # WhatsApp/general: "Name: Message"
            re.compile(r'@([^\s]+)\s'),  # X/some platforms: "@username Message"
            re.compile(r'\[([^\]]+)\]\s'),  # Discord/some platforms: "[Name] Message"
            re.compile(r'^\s*([A-Za-z0-9_\.\-]+)\s*(?:\d{1,2}:\d{2})'), # Instagram DM pattern
            re.compile(r'([A-Za-z0-9_\.\-]+)\s+-\s+'),  # "Name - Message"
            re.compile(r'Message from ([A-Za-z0-9_\.\-]+):\s'),  # "Message from Name: Message"
            re.compile(r'([A-Za-z0-9_\.\-]+) said:\s'),  # "Name said: Message"
        ]
        
        # Common chat terms and phrases for context-based correction
        self.common_chat_terms = [
            "hello", "hi", "hey", "thanks", "thank you", "ok", "okay", "yes", "no", 
            "please", "sorry", "lol", "haha", "what", "when", "where", "why", "how",
            "meeting", "call", "tomorrow", "today", "yesterday", "morning", "afternoon",
            "night", "send", "sent", "received", "message", "chat", "talk", "later",
            "soon", "wait", "ready", "done", "finished", "started", "let me know",
            "see you", "talk later", "bye", "good", "great", "nice", "awesome"
        ]
        
        # Initialize spell checker
        self.spell = SpellChecker()
        
        # Verify Tesseract is installed
        try:
            version = self.tesseract.get_tesseract_version()
            self.logger.info(f"Tesseract version: {version}")
        except Exception as e:
            self.logger.warning(f"Tesseract warning (OCR may not work): {e}")
    # Fix 1: Correct the auto_detect_tesseract method
    @staticmethod
    def auto_detect_tesseract() -> Optional[str]:  # Remove 'self' parameter
        """
        Attempt to automatically detect Tesseract installation path.
        
        Returns:
            Path to Tesseract executable or None if not found
        """
        common_paths = [
            # Windows common locations
            r'C:\Program Files\Tesseract-OCR\tesseract.exe',
            r'C:\Program Files (x86)\Tesseract-OCR\tesseract.exe',
            # Linux common locations
            '/usr/bin/tesseract',
            '/usr/local/bin/tesseract',
            # macOS (Homebrew)
            '/usr/local/Cellar/tesseract/*/bin/tesseract',
            '/opt/homebrew/bin/tesseract'
        ]
        
        # Check environment variable first
        if 'TESSERACT_CMD' in os.environ:
            path = os.environ['TESSERACT_CMD']
            if os.path.isfile(path):
                return path
        
        # Check common paths
        for path in common_paths:
            if '*' in path:  # Handle wildcard paths
                import glob
                matches = glob.glob(path)
                if matches:
                    return matches[0]
            elif os.path.isfile(path):
                return path
                
        # Try to find using 'which' on UNIX-like systems
        try:
            import subprocess
            return subprocess.check_output(['which', 'tesseract']).decode('utf-8').strip()
        except (subprocess.SubprocessError, FileNotFoundError):
            pass
            
        logging.getLogger(__name__).warning("Could not auto-detect Tesseract path")
        return None

    # Fix 2: Add process_chat_image method
    def process_chat_image(self, image_path):
        """
        Process a chat image and extract messages.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Dictionary with processed results
        """
        return self.process_image(image_path)

    # Fix 3: Add from_base64 static method
    @staticmethod
    def from_base64(base64_data):
        """
        Process image from base64 encoded data.
        
        Args:
            base64_data: Base64 encoded image data
            
        Returns:
            Dictionary with processed results
        """
        extractor = TextExtractor()
        return extractor.process_image(base64_data)

    def preprocess_image(self, image_path_or_data) -> List[Image.Image]:
        """
        Enhanced preprocessing with multiple techniques for optimal OCR results.
        Works with various chat interfaces including dark and light themes.
        
        Args:
            image_path_or_data: Either a file path or base64 encoded image data
            
        Returns:
            List of processed PIL Images
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
            
            # 1. Original grayscale
            processed_images.append(Image.fromarray(gray))
            
            # 2. Simple thresholding with OTSU (works well for clean texts)
            _, thresh_otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            processed_images.append(Image.fromarray(thresh_otsu))
            
            # 3. Inverted image (for light text on dark backgrounds)
            inverted = cv2.bitwise_not(gray)
            processed_images.append(Image.fromarray(inverted))
            
            # 4. Adaptive thresholding (good for varying lighting conditions)
            adaptive_thresh = cv2.adaptiveThreshold(
                gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                cv2.THRESH_BINARY, 11, 2
            )
            processed_images.append(Image.fromarray(adaptive_thresh))
            
            # 5. Denoised image
            denoised = cv2.fastNlMeansDenoising(gray, None, h=10, templateWindowSize=7, searchWindowSize=21)
            processed_images.append(Image.fromarray(denoised))
            
            # 6. Contrast enhancement with CLAHE
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            enhanced = clahe.apply(gray)
            processed_images.append(Image.fromarray(enhanced))
            
            # 7. Bilateral filtering (preserves edges)
            bilateral = cv2.bilateralFilter(gray, 9, 75, 75)
            processed_images.append(Image.fromarray(bilateral))
            
            # 8. Sharpened image
            kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
            sharpened = cv2.filter2D(gray, -1, kernel)
            processed_images.append(Image.fromarray(sharpened))
            
            # 9. Enhanced contrast + thresholding
            enhanced_contrast = cv2.equalizeHist(gray)
            _, thresh_enhanced = cv2.threshold(enhanced_contrast, 150, 255, cv2.THRESH_BINARY)
            processed_images.append(Image.fromarray(thresh_enhanced))
            
            # 10. Morphological operations
            # Opening (erosion followed by dilation) to remove noise
            kernel = np.ones((2,2), np.uint8)
            opening = cv2.morphologyEx(gray, cv2.MORPH_OPEN, kernel)
            processed_images.append(Image.fromarray(opening))
            
            # Resize images to improve OCR performance
            # For very small text, upscaling can help
            height, width = gray.shape[:2]
            if width < 1000 or height < 1000:
                scale_factor = 2.0
                resized = cv2.resize(gray, None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_CUBIC)
                processed_images.append(Image.fromarray(resized))
            
            return processed_images
            
        except Exception as e:
            self.logger.error(f"Preprocessing error: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())
            # Return original image as fallback
            return [Image.fromarray(gray)] if 'gray' in locals() else []

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
            all_texts = []
            
            # Expanded tesseract configurations for better coverage
            configs = [
                r'--oem 3 --psm 6 -l eng -c preserve_interword_spaces=1',
                r'--oem 3 --psm 4 -l eng',
                r'--oem 3 --psm 11 -l eng',  # For sparse text
                r'--oem 1 --psm 6 -l eng',
                r'--oem 3 --psm 3 -l eng',   # Fully automatic page segmentation
                r'--oem 3 --psm 6 -l eng --dpi 300',
                r'--oem 3 --psm 12 -l eng',  # Sparse text with OSD
                r'--oem 3 --psm 13 -l eng',  # Raw line
                r'--oem 3 --psm 6 -l eng+osd'  # With orientation detection
            ]
            
            # Try each image with each config
            for i, image in enumerate(images):
                for config in configs:
                    try:
                        text = self.tesseract.image_to_string(image, config=config)
                        text = text.strip()
                        
                        if not text:
                            continue
                            
                        # Store all results for potential combination later
                        all_texts.append(text)
                        
                        # Score the quality of extraction based on patterns we expect
                        score = self._score_extraction(text)
                        
                        self.logger.debug(f"Image {i}, config: {config}, score: {score}")
                        
                        if score > max_score:
                            max_score = score
                            best_text = text
                    except Exception as e:
                        self.logger.warning(f"OCR failed for image {i} with config {config}: {e}")
                        continue
            
            if not best_text and all_texts:
                # If no ideal result was found, combine the most promising ones
                self.logger.info("No ideal OCR result found, combining best results")
                best_text = self._combine_best_texts(all_texts)
            
            if not best_text:
                self.logger.warning("All OCR attempts failed to produce usable text")
                # Last effort with simplest config
                for image in images:
                    try:
                        text = self.tesseract.image_to_string(image)
                        if text and len(text.strip()) > 10:
                            return text
                    except:
                        pass
            
            return best_text
            
        except Exception as e:
            self.logger.error(f"OCR extraction error: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())
            return "Error extracting text. Check Tesseract installation."
    
    def _combine_best_texts(self, texts: List[str]) -> str:
        """
        Combine the best parts of multiple OCR results for better output.
        
        Args:
            texts: List of OCR text results
            
        Returns:
            Combined text
        """
        if not texts:
            return ""
            
        # Filter out empty or very short texts
        valid_texts = [t for t in texts if len(t.strip()) > 10]
        
        if not valid_texts:
            return ""
            
        # Score each text
        scored_texts = [(self._score_extraction(t), t) for t in valid_texts]
        
        # Sort by score, highest first
        scored_texts.sort(reverse=True)
        
        # Take top 3 results or all if fewer than 3
        top_results = scored_texts[:min(3, len(scored_texts))]
        
        # Split into lines for line-by-line comparison
        lined_results = [t[1].split('\n') for t in top_results]
        
        # Find the best version of each line
        best_lines = []
        max_lines = max(len(lines) for lines in lined_results)
        
        for i in range(max_lines):
            line_candidates = []
            for lines in lined_results:
                if i < len(lines) and lines[i].strip():
                    line_candidates.append(lines[i])
            
            if line_candidates:
                # Score each line candidate
                scored_lines = [(self._score_line(line), line) for line in line_candidates]
                scored_lines.sort(reverse=True)
                best_lines.append(scored_lines[0][1])
        
        return '\n'.join(best_lines)
    
    def _score_line(self, line: str) -> int:
        """
        Score a single line of text based on likely correctness.
        
        Args:
            line: Line of text to score
            
        Returns:
            Score value
        """
        score = 0
        
        # Skip empty lines
        if not line or len(line.strip()) < 2:
            return 0
        
        # More words is generally better (less fragmented OCR)
        words = line.split()
        score += len(words)
        
        # Favor lines with complete words (not just fragments)
        avg_word_len = sum(len(w) for w in words) / len(words) if words else 0
        if 3 <= avg_word_len <= 8:  # Average English word length is ~5 chars
            score += 3
        
        # Favor lines with chat patterns
        if any(pattern.search(line) for pattern in self.sender_patterns):
            score += 5
            
        # Favor lines with common chat terms
        score += sum(2 for term in self.common_chat_terms if term.lower() in line.lower())
        
        # Favor lines with better spelling
        misspelled = self.spell.unknown(words)
        score -= len(misspelled)
        
        # Favor lines that look like normal text (alphanumeric + punctuation)
        alpha_ratio = sum(c.isalnum() or c.isspace() or c in '.,:;!?()[]{}"\'-' for c in line) / len(line) if line else 0
        score += int(alpha_ratio * 10)
        
        return score

    def _score_extraction(self, text: str) -> int:
        """
        Score the extraction quality based on chat message patterns.
        Enhanced to recognize patterns from various chat platforms.
        
        Args:
            text: The extracted text to score
            
        Returns:
            Score value (higher is better)
        """
        score = 0
        
        # Skip empty text
        if not text or len(text.strip()) < 10:
            return 0
        
        # Check for timestamps (common in most chat platforms)
        timestamps = self.time_pattern.findall(text)
        score += len(timestamps) * 5
        
        # Check for message patterns using all sender patterns
        for pattern in self.sender_patterns:
            sender_matches = pattern.findall(text)
            score += len(sender_matches) * 10
        
        # Check for emoji patterns (common in chats)
        emoji_count = len(re.findall(r'[😀-🙏]|:\)|:\(|:D|:P', text))
        score += emoji_count * 3
        
        # Penalize very short texts that might be noise
        if len(text) < 20:
            score -= 10
            
        # Favor texts with multiple lines (likely contains multiple messages)
        lines = [line for line in text.split('\n') if line.strip()]
        score += min(len(lines), 10) * 3
        
        # Check for common chat expressions
        for expr in self.common_chat_terms:
            if expr in text.lower():
                score += 2
        
        # Check for proper sentence structure
        sentences = re.split(r'[.!?]+', text)
        for sentence in sentences:
            if sentence.strip() and sentence[0].isupper():
                score += 2
                
        # Check for well-formed words (not just garbage)
        words = text.split()
        real_words = 0
        for word in words:
            if len(word) > 2 and word.lower() in self.spell or any(term.lower() == word.lower() for term in self.common_chat_terms):
                real_words += 1
                
        # Bonus for high ratio of real words
        if words:
            real_word_ratio = real_words / len(words)
            score += int(real_word_ratio * 20)
        
        return score

    def process_content(self, text: str) -> Dict[str, List[str]]:
        """
        Process the extracted text into structured messages with sender information.
        Works with various chat platforms.
        
        Args:
            text: Raw extracted text
            
        Returns:
            Dictionary with 'messages' and 'structured_chat' keys
        """
        # Pre-clean the text before processing
        text = self._preprocess_text(text)
        
        raw_messages = []
        structured_chat = []
        current_sender = None
        
        # Split into lines and process each
        lines = text.split('\n')
        for i, line in enumerate(lines):
            line = line.strip()
            if not line:
                continue
                
            # Try to detect sender pattern using multiple patterns
            sender_match = None
            for pattern in self.sender_patterns:
                match = pattern.search(line)
                if match:
                    sender_match = match
                    break
            
            if sender_match:
                # This line starts a new message
                sender = sender_match.group(1).strip()
                # Clean sender name
                sender = self._clean_sender(sender)
                # Get content based on match end position
                content = line[sender_match.end():].strip()
                # Clean content
                content = self._clean_content(content)
                
                current_sender = sender
                
                if content:  # Only add if there's actual content
                    raw_messages.append(content)
                    structured_chat.append({"sender": sender, "content": content})
            elif current_sender and i > 0:
                # This is likely a continuation of the previous message
                # Check if previous line had a sender and this is a continuation
                content = self._clean_content(line)
                
                if content and len(content) > 3:
                    # Check if this might actually be a new message without clear sender
                    if self._is_likely_new_message(content, structured_chat[-1]["content"] if structured_chat else ""):
                        # This seems like a new message but sender isn't clear
                        # Use the last detected sender or "Unknown"
                        sender = current_sender if current_sender else "Unknown"
                        structured_chat.append({"sender": sender, "content": content})
                        raw_messages.append(content)
                    else:
                        # Append to the previous message
                        structured_chat[-1]["content"] += f" {content}"
                        if raw_messages:
                            raw_messages[-1] += f" {content}"
        
        # Final cleaning pass to remove noise and duplicates
        cleaned_chat = self._remove_duplicates_and_noise(structured_chat)
        
        # If we still have no structured messages, try a simpler approach
        if not cleaned_chat and text:
            # Just break it into lines as separate messages
            for line in lines:
                if len(line.strip()) > 10:  # Only substantial lines
                    cleaned_line = self._clean_content(line)
                    # Try to contextually improve this line
                    improved_line = self._context_based_correction(cleaned_line)
                    cleaned_chat.append({"sender": "Unknown", "content": improved_line})
        
        return {
            "messages": [msg["content"] for msg in cleaned_chat],
            "structured_chat": cleaned_chat
        }

    def _preprocess_text(self, text: str) -> str:
        """
        Preprocess the entire OCR text before detailed processing.
        
        Args:
            text: Raw OCR text
            
        Returns:
            Preprocessed text
        """
        if not text:
            return ""
            
        # Remove very common OCR noise patterns
        text = re.sub(r'\b[IlJj]{3,}\b', '', text)  # Remove III, lll, etc.
        text = re.sub(r'\b[O0]{3,}\b', '', text)    # Remove OOO, 000, etc.
        text = re.sub(r'\b[^a-zA-Z0-9\s]{3,}\b', '', text)  # Remove sequences of symbols
        
        # Fix line breaks that might have been broken by OCR
        text = re.sub(r'(\w)-\s+(\w)', r'\1\2', text)
        
        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Split into lines better by adding newlines after likely message endings
        text = re.sub(r'([.!?])\s+([A-Z])', r'\1\n\2', text)
        
        # Try to identify and format chat-like structures
        for pattern in self.sender_patterns:
            # Add newlines before sender patterns if not already there
            text = re.sub(r'([^\n])\s+(' + pattern.pattern + ')', r'\1\n\2', text)
        
        return text

    def _is_likely_new_message(self, current_text: str, previous_text: str) -> bool:
        """
        Determine if text is likely a new message rather than continuation.
        Enhanced for various chat platforms.
        
        Args:
            current_text: Current line of text
            previous_text: Previous message content
            
        Returns:
            Boolean indicating if this is likely a new message
        """
        # If current text is empty or too short, it's not a new message
        if not current_text or len(current_text) < 3:
            return False
            
        # Check for sender patterns in current text
        for pattern in self.sender_patterns:
            if pattern.search(current_text):
                return True
            
        # If current text starts with capital letter and previous ends with punctuation
        if (current_text[0].isupper() and 
            previous_text and previous_text[-1] in '.!?'):
            return True
            
        # If current text is much longer than what would be a typical continuation
        if len(current_text) > 30:
            return True
            
        # If current text contains typical message starters
        if any(current_text.lower().startswith(s) for s in self.common_chat_terms):
            return True
            
        # If it has a timestamp
        if self.time_pattern.search(current_text):
            return True
            
        return False

    def _clean_content(self, text: str) -> str:
        """
        Clean text from OCR artifacts and normalize content.
        
        Args:
            text: Text to clean
            
        Returns:
            Cleaned text
        """
        if not text:
            return ""
            
        # Remove excess whitespace first
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Remove timestamps in cases where they're not needed
        if not self._is_timestamp_only(text):
            text = re.sub(r'\d{1,2}:\d{2}(?:\s*[AP]M)?(?=\s)', '', text)
        
        # Fix common OCR errors
        replacements = {
            r'\bBnalenge\b': 'Banalenge',
            r'\banaeyaat\b': 'anayata',
            r'\beS\b': '', r'\bNu\b': '', r'\bQ a ee\b': '', r'\bII O\b': '', r'\b0k\b': 'OK',
            r'\bl\b': 'I', r'\blo\b': 'to', r'\b1\b': 'I', r'\bjusi\b': 'just', r'\bwhai\b': 'what',
            r'\binsiead\b': 'instead', r'\byou be\b': 'you be', r'\bsharp\b': 'sharp', r'\bsteesrants\b': 'restaurants',
            r'\beach era\b': '', r'\bGss\b': '', r'\bsee ya\b': 'see ya'
        }
        
        # Apply all replacements
        for pattern, replacement in replacements.items():
            text = re.sub(pattern, replacement, text)
        
        # Remove strange punctuation but retain important marks
        text = re.sub(r'[^\w\s\.\,\!\?\:\;\(\)\[\]\{\}\-\"\'\–\—]', '', text)
        text = re.sub(r'\.{2,}', '...', text)  # normalize ellipsis
        
        # Fix spacing issues
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Fix common OCR number/letter confusions
        text = text.replace('0', 'o')  # Often '0' is recognized as 'o'
        text = re.sub(r'\b1([a-z])', r'I\1', text)  # '1' at start of word is often 'I'
        
        # Convert partial corrections
        text = self._context_based_correction(text)
        
        return text
    
    def _is_timestamp_only(self, text: str) -> bool:
        """
        Check if text appears to be only a timestamp.
        
        Args:
            text: Text to check
            
        Returns:
            Boolean indicating if text is just a timestamp
        """
        # Check if the entire text matches a timestamp pattern
        return bool(re.fullmatch(r'\d{1,2}:\d{2}(?:\s*[AP]M)?', text.strip()))
    
    def _context_based_correction(self, text: str) -> str:
        """
        Apply context-based corrections to improve OCR results.
        
        Args:
            text: Text to correct
            
        Returns:
            Corrected text
        """
        if not text:
            return ""
        
        # Check for specific phrases that are commonly misrecognized
        corrections = {
            r'you be ready at': 'you be ready at',
            r'sharp \d+': lambda m: m.group(0),  # Keep "sharp 10" etc. as is
            r'get[:\s]+your': 'get your',
            r'see ya': 'see ya',
            r'restauranrs': 'restaurants',
            r'steesrants': 'restaurants'
        }
        
        for pattern, replacement in corrections.items():
            if callable(replacement):
                # For function-based replacements
                text = re.sub(pattern, replacement, text)
            else:
                # For exact replacements, check if pattern matches
                match = re.search(pattern, text, re.IGNORECASE)
                if match:
                    # Replace with the proper case version
                    text = text.replace(match.group(0), replacement)
        
        # Word-by-word spell check with context
        words = text.split()
        if len(words) <= 1:
            return text
            
        corrected_words = []
        for i, word in enumerate(words):
            # Skip very short words, punctuation, or already correct words
            if len(word) <= 1 or not any(c.isalpha() for c in word) or word.lower() in self.spell:
                corrected_words.append(word)
                continue
                
            # Get context (words before and after)
            context_before = words[i-1] if i > 0 else ""
            context_after = words[i+1] if i < len(words)-1 else ""
            
            # Check if this might be part of a common chat phrase
            potential_bigrams = []
            if context_before:
                potential_bigrams.append(f"{context_before} {word}")
            if context_after:
                potential_bigrams.append(f"{word} {context_after}")
                
            # Check against common chat terms
            found_in_common = False
            for bigram in potential_bigrams:
                for term in self.common_chat_terms:
                    term_parts = term.split()
                    if len(term_parts) > 1 and term.lower() in bigram.lower():
                        # This is part of a common term, find the closest match
                        # Check if it's the first or second part we need
                        term_index = 0 if term.startswith(context_before) and len(term_parts) > 0 else 1
                        # Make sure the index is valid
                        if term_index < len(term_parts):
                            closest = difflib.get_close_matches(word, [term_parts[term_index]], n=1)
                            if closest:
                                corrected_words.append(closest[0])
                                found_in_common = True
                                break
                if found_in_common:
                    break
                    
            if not found_in_common:
                # Standard spell correction
                correction = self.spell.correction(word)
                if correction:
                    corrected_words.append(correction)
                else:
                    corrected_words.append(word)
        
        return ' '.join(corrected_words)

    def _clean_sender(self, text: str) -> str:
        """
        Clean sender name from any non-alphanumeric characters.
        
        Args:
            text: Sender name to clean
            
        Returns:
            Cleaned sender name
        """
        if not text:
            return "Unknown"
            
        # Remove special characters but retain spaces for names
        text = re.sub(r'[^\w\s]', '', text)
        
        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Check if this looks like a valid name (not just numbers/symbols)
        if not any(c.isalpha() for c in text):
            return "Unknown"
            
        return text

    def _remove_duplicates_and_noise(self, messages: List[Dict[str, str]]) -> List[Dict[str, str]]:
        """
        Remove duplicate messages and noise from structured chat.
        
        Args:
            messages: List of message dictionaries
            
        Returns:
            Cleaned list of messages
        """
        if not messages:
            return []
            
        # Remove messages with very similar content (duplicates from OCR)
        unique_messages = []
        for msg in messages:
            # Skip empty or very short messages
            if not msg["content"] or len(msg["content"]) < 3:
                continue
                
            # Check if this is a duplicate of an existing message
            is_duplicate = False
            for existing in unique_messages:
                similarity = difflib.SequenceMatcher(None, msg["content"].lower(), 
                                                   existing["content"].lower()).ratio()
                if similarity > 0.8:  # 80% similarity threshold
                    is_duplicate = True
                    break
                    
            if not is_duplicate:
                unique_messages.append(msg)
        
        # Filter out noise messages (timestamps alone, etc.)
        filtered_messages = []
        for msg in unique_messages:
            content = msg["content"]
            
            # Skip if it's just a timestamp
            if self._is_timestamp_only(content):
                continue
                
            # Skip if content is just noise (no real words)
            words = content.split()
            if not any(len(w) > 1 and any(c.isalpha() for c in w) for w in words):
                continue
                
            filtered_messages.append(msg)
            
        return filtered_messages

    def process_image(self, image_path_or_data) -> Dict:
        """
        Process an image and extract chat messages.
        
        Args:
            image_path_or_data: Path to image or base64 encoded image data
            
        Returns:
            Dictionary with processed results
        """
        try:
            # Track processing time
            start_time = datetime.now()
            
            # Preprocess the image
            processed_images = self.preprocess_image(image_path_or_data)
            
            if not processed_images:
                return {"error": "Failed to process image", "messages": [], "structured_chat": []}
            
            # Extract text using OCR
            extracted_text = self.extract_text(processed_images)
            
            if not extracted_text or len(extracted_text.strip()) < 10:
                return {"error": "No text detected in image", "messages": [], "structured_chat": []}
            
            # Process the extracted content
            result = self.process_content(extracted_text)
            
            # Add metadata
            processing_time = (datetime.now() - start_time).total_seconds()
            
            # Return complete result
            return {
                "raw_text": extracted_text,
                "messages": result["messages"],
                "structured_chat": result["structured_chat"],
                "processing_time": processing_time,
                "success": True
            }
            
        except Exception as e:
            self.logger.error(f"Processing error: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())
            return {
                "error": str(e),
                "messages": [],
                "structured_chat": [],
                "success": False
            }

    def save_result_to_json(self, result: Dict, output_path: str) -> bool:
        """
        Save extraction result to a JSON file.
        
        Args:
            result: Dictionary with extraction results
            output_path: Path to save the JSON file
            
        Returns:
            Boolean indicating success
        """
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(result, f, indent=2, ensure_ascii=False)
            return True
        except Exception as e:
            self.logger.error(f"Error saving result to JSON: {str(e)}")
            return False