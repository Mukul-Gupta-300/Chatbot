import os
import logging
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import re
from typing import List, Dict, Optional

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

class ChatResponseGenerator:
    """Enhanced chat response generator with multiple personality styles"""
    
    def __init__(self, model_name: str = "facebook/blenderbot-400M-distill"):
        """Initialize the chat response generator with specified model"""
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logging.info(f"Loading model: {model_name} on {self.device}")
        
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForCausalLM.from_pretrained(model_name).to(self.device)
            logging.info("Model loaded successfully")
        except Exception as e:
            logging.error(f"Error loading model: {e}")
            raise
            
        # Define personality styles with specific prompts
        self.styles = {
            "flirty": {
                "prompt": "Generate a flirty and charming reply that shows interest while being respectful:",
                "temp": 0.9, 
                "examples": [
                    "I'm really enjoying getting to know you. Your interests in [topic] are fascinating.",
                    "You seem like someone who's both fun and thoughtful. I'd love to hear more about your day."
                ]
            },
            "confident": {
                "prompt": "Generate a confident and direct reply that shows self-assurance without being arrogant:",
                "temp": 0.8,
                "examples": [
                    "I know exactly what you mean. I've had similar experiences with [topic].",
                    "Let's meet up this weekend. I know a great place we'd both enjoy."
                ]
            },
            "friendly": {
                "prompt": "Generate a warm and friendly reply that shows genuine interest without romantic undertones:",
                "temp": 0.7,
                "examples": [
                    "That's such a thoughtful perspective! I enjoy having these kinds of conversations.",
                    "I appreciate you sharing that story. It reminds me of when I..."
                ]
            },
            "sympathy": {
                "prompt": "Generate a supportive and understanding reply that shows empathy:",
                "temp": 0.6,
                "examples": [
                    "I'm really sorry to hear that happened. It sounds like a difficult situation to navigate.",
                    "That's tough, and I understand why you'd feel that way. Take whatever time you need."
                ]
            },
            "playful": {
                "prompt": "Generate a light-hearted and playful reply with gentle teasing:",
                "temp": 0.9,
                "examples": [
                    "Oh, so you think you're an expert on [topic]? I might have to challenge you on that! 😉",
                    "I can't believe you just said that! Now I'm definitely curious about your taste in movies."
                ]
            },
            "funny": {
                "prompt": "Generate a humorous reply that's witty and lighthearted:",
                "temp": 0.95,
                "examples": [
                    "If my life were a movie based on this conversation, the title would be '[Funny Title]' 😂",
                    "Well, that's one way to break the ice! My coffee nearly came out my nose reading that."
                ]
            }
        }
        
        # Chat history file
        self.chat_file = "chat_output.txt"
    
    def load_chat_history(self, file_path: Optional[str] = None) -> List[str]:
        """Load chat history from file with error handling"""
        file_path = file_path or self.chat_file
        
        if not os.path.exists(file_path):
            logging.warning("Chat history file not found. Starting fresh conversation.")
            return []

        try:
            with open(file_path, "r", encoding="utf-8") as file:
                chat_lines = [line.strip() for line in file.readlines() if line.strip()]
            
            logging.info(f"Successfully loaded {len(chat_lines)} chat messages.")
            return chat_lines

        except Exception as e:
            logging.error(f"Error reading chat history: {e}")
            return []
    
    def _detect_language(self, text: str) -> str:
        """Detect language of the conversation and return language code"""
        # Basic language detection based on common words
        # In a production system, you would use a proper language detection library
        common_words = {
            'en': ['the', 'and', 'is', 'in', 'it', 'you', 'that', 'was', 'for', 'on'],
            'es': ['el', 'la', 'que', 'en', 'y', 'a', 'los', 'del', 'se', 'las'],
            'fr': ['le', 'la', 'et', 'les', 'des', 'en', 'un', 'du', 'une', 'est'],
            'de': ['der', 'die', 'und', 'den', 'von', 'zu', 'das', 'mit', 'sich', 'des'],
            'hi': ['है', 'का', 'में', 'की', 'एक', 'को', 'और', 'से', 'पर', 'कर']
        }
        
        text_lower = text.lower()
        word_count = {lang: 0 for lang in common_words}
        
        for lang, words in common_words.items():
            for word in words:
                pattern = r'\b' + re.escape(word) + r'\b'
                matches = re.findall(pattern, text_lower)
                word_count[lang] += len(matches)
        
        detected_lang = max(word_count, key=word_count.get)
        logging.info(f"Detected language: {detected_lang}")
        return detected_lang
    
    def _analyze_context(self, chat_history: List[str]) -> Dict:
        """Analyze chat to extract context like topics, sentiment, and pace"""
        combined_text = " ".join(chat_history[-5:] if len(chat_history) > 5 else chat_history)
        
        # Basic topic extraction (would be more sophisticated in production)
        common_topics = [
            "movies", "music", "food", "travel", "work", "study", 
            "hobbies", "sports", "books", "family", "weather", "weekend"
        ]
        
        found_topics = []
        for topic in common_topics:
            if topic in combined_text.lower():
                found_topics.append(topic)
        
        # Basic sentiment analysis
        positive_words = ["happy", "good", "great", "love", "enjoy", "excited", "fun", "amazing"]
        negative_words = ["sad", "bad", "upset", "worry", "sorry", "disappointed", "struggle"]
        
        sentiment_score = 0
        for word in positive_words:
            if word in combined_text.lower():
                sentiment_score += 1
        for word in negative_words:
            if word in combined_text.lower():
                sentiment_score -= 1
                
        # Message pace/length analysis
        avg_length = sum(len(msg) for msg in chat_history) / max(1, len(chat_history))
        
        return {
            "language": self._detect_language(combined_text),
            "topics": found_topics[:3],  # Top 3 topics
            "sentiment": "positive" if sentiment_score > 0 else "negative" if sentiment_score < 0 else "neutral",
            "message_length": "short" if avg_length < 50 else "medium" if avg_length < 100 else "long"
        }
    
    def generate_response(self, chat_history: List[str], style: str = "friendly") -> str:
        """Generate contextually appropriate response based on chat history and style"""
        if not chat_history:
            return "Hi there! How's your day going? 😊"
        
        # Get style configuration (default to friendly if style not found)
        style_config = self.styles.get(style.lower(), self.styles["friendly"])
        
        # Analyze conversation context
        context = self._analyze_context(chat_history)
        
        # Parse conversation into context
        conversation = "\n".join([
            f"{'User' if i % 2 == 0 else 'Friend'}: {line}" 
            for i, line in enumerate(chat_history[-6:])  # Last 6 messages for context
        ])
        
        # Build enhanced prompt with context-aware instructions
        topics_str = ", ".join(context["topics"]) if context["topics"] else "any relevant topic"
        
        enhanced_prompt = (
            f"{style_config['prompt']}\n\n"
            f"Conversation language: {context['language']}\n"
            f"Conversation topics: {topics_str}\n"
            f"Conversation sentiment: {context['sentiment']}\n"
            f"Examples of this style:\n"
        )
        
        # Add examples
        for example in style_config["examples"][:2]:
            enhanced_prompt += f"- {example}\n"
            
        enhanced_prompt += f"\nConversation:\n{conversation}\n\nResponse:"
        
        logging.info(f"Generated Prompt:\n{enhanced_prompt}")

        # Tokenize input
        input_ids = self.tokenizer(enhanced_prompt, return_tensors="pt").input_ids.to(self.device)
        
        # Generate response with dynamic parameters based on style
        output = self.model.generate(
            input_ids,
            max_length=min(300, len(conversation.split()) + 60),  # Adaptive length
            min_length=20,
            temperature=style_config.get("temp", 0.7),
            top_k=50,
            top_p=0.92,
            repetition_penalty=1.2,
            no_repeat_ngram_size=2,
            do_sample=True,
        )
        
        # Decode response
        response = self.tokenizer.decode(output[0], skip_special_tokens=True)
        
        # Extract only the generated response part (after the prompt)
        try:
            response = response.split("Response:")[-1].strip()
        except:
            pass
            
        # Cleanup and formatting
        response = self._format_response(response, style, context)
        
        # Fallback if no valid response
        if not response.strip():
            if style == "flirty":
                response = "I've been smiling ever since I saw your message. So, what's something you're passionate about? 😊"
            elif style == "confident":
                response = "I'd definitely like to continue this conversation. What are your plans for the weekend?"
            else:
                response = "That's really interesting! Tell me more about yourself. What do you enjoy doing in your free time?"
        
        logging.info(f"Generated Response:\n{response}")
        
        return response
    
    def _format_response(self, response: str, style: str, context: Dict) -> str:
        """Format and enhance the response based on style and context"""
        # Remove any system-like text that might have been generated
        response = re.sub(r'(System:|AI:|Assistant:|Friend:)', '', response).strip()
        
        # Add style-specific touches
        if style == "flirty":
            if not any(emoji in response for emoji in ["😊", "😉", "😘", "💕", "🙈"]):
                emojis = ["😊", "😉", "💕"]
                response += f" {emojis[len(response) % 3]}"
                
        elif style == "funny":
            if not any(emoji in response for emoji in ["😂", "🤣", "😆", "😜"]):
                response += " 😂"
                
        elif style == "sympathy":
            if not any(emoji in response for emoji in ["❤️", "🤗", "💙"]):
                response += " 🤗"
        
        # Ensure response length is appropriate
        if context["message_length"] == "short" and len(response) > 100:
            # Try to find a natural breakpoint
            sentences = re.split(r'(?<=[.!?])\s+', response)
            response = sentences[0]
            if len(sentences) > 1 and len(response) < 30:
                response += " " + sentences[1]
                
        return response
        
    def save_to_history(self, message: str, is_user: bool = False):
        """Save a message to the chat history file"""
        prefix = "User: " if is_user else "AI: "
        
        try:
            with open(self.chat_file, "a", encoding="utf-8") as file:
                file.write(f"\n{prefix}{message}")
            logging.info(f"Message saved to history: {prefix}{message[:30]}...")
        except Exception as e:
            logging.error(f"Error saving to chat history: {e}")

# Example usage
if __name__ == "__main__":
    try:
        # Initialize generator
        generator = ChatResponseGenerator()
        
        # Load existing chat history
        chat_history = generator.load_chat_history()
        
        if not chat_history:
            # For testing, create a sample conversation
            chat_history = [
                "Hi there! How's your day going?",
                "Pretty good! Just finished a movie marathon. How about you?",
                "I'm doing well too. What movies did you watch?"
            ]
            
        print("Chat History:")
        for msg in chat_history:
            print(f"- {msg}")
            
        # Test different styles
        styles = ["flirty", "confident", "friendly", "sympathy", "playful", "funny"]
        
        for style in styles:
            print(f"\n\n{style.upper()} RESPONSE:")
            response = generator.generate_response(chat_history, style)
            print(response)
            
    except Exception as e:
        logging.error(f"Main execution error: {e}")