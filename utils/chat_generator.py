import os
import logging
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, pipeline
import re
from typing import List, Dict, Optional
import numpy as np
from sentence_transformers import SentenceTransformer
import json
import time
from collections import deque
import random
import spacy
from transformers import pipeline


# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

class ChatGenerator:
    def __init__(self, model_name: str = "facebook/blenderbot-3B"):
        """
        Initialize the chat generator with a specified model.
        
        Args:
            model_name: The name of the model to use for generating responses
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logging.info(f"Using device: {self.device}")
        self.nlp = spacy.load("en_core_web_sm")
        self.sentiment_analysis = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")

        # Try loading a larger model that can handle more context
        try:
            logging.info(f"Loading primary model: {model_name}")
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(self.device)
            
            # Get model's max context window
            self.max_length = self.tokenizer.model_max_length
            logging.info(f"Model loaded successfully. Max context length: {self.max_length}")
            
            # Load an embedding model for conversation summarization
            self.embedding_model = None
            try:
                self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
                logging.info("Embedding model loaded successfully for context management")
            except Exception as e:
                logging.warning(f"Could not load embedding model: {e}. Will use simpler context management.")
                
        except Exception as e:
            logging.error(f"Error loading primary model: {e}")
            # Fallback to a simpler model
            fallback_model = "facebook/blenderbot-400M-distill"
            logging.info(f"Attempting to load fallback model: {fallback_model}")
            self.tokenizer = AutoTokenizer.from_pretrained(fallback_model)
            self.model = AutoModelForSeq2SeqLM.from_pretrained(fallback_model).to(self.device)
            self.max_length = 128  # Default for smaller model
            logging.info("Fallback model loaded successfully")
            
        # Define personality styles with specific prompts
        self.styles = {
            "flirty": {
            "prompt": "As a flirty AI, I need to generate a response that is charming and engaging for a dating context. The response should be personalized based on the conversation history and the current conversation stage. If the conversation is in the initial stages, be friendly and inviting. As the conversation deepens, become more personal and slightly suggestive, helping the user to build rapport and attraction.",
            "temp": 0.9,
            "examples": ["I'm really enjoying getting to know you.", "You have such a great smile, it's making me smile too.", "I can't help but be curious about what makes you tick."]
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
            },
            "curious": {
                "prompt": "Generate a curious and engaging reply that asks thoughtful questions:",
                "temp": 0.7,
                "examples": [
                    "That's fascinating! What made you first get interested in [topic]?",
                    "I'd love to know more about your perspective on that. What influenced your thinking?"
                ]
            },
            "deep": {
                "prompt": "Generate a thoughtful and deep reply that shows emotional intelligence:",
                "temp": 0.6,
                "examples": [
                    "I think those moments of vulnerability are what make connections meaningful. Have you felt that way before?",
                    "It's interesting how our experiences shape our worldview. I've been reflecting on that lately too."
                ]
            }
        }
        
        # Chat history file
        self.chat_file = "chat_output.txt"
        
        # Memory for important conversational context
        self.memory = {}
        
        # Create conversation cache for recent interactions
        self.conversation_cache = {}
    
    def load_chat_history(self, file_path: Optional[str] = None) -> List[str]:
        """
        Load chat history from file with error handling
        
        Args:
            file_path: Path to the chat history file
            
        Returns:
            List of chat messages
        """
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
    
    def parse_chat_text(self, chat_text: str) -> List[str]:
        """
        Parse chat text into a list of messages.
        This function extracts messages from text that might come from OCR.
        
        Args:
            chat_text: Raw chat text to parse
            
        Returns:
            List of individual messages
        """
        if isinstance(chat_text, list):
            return chat_text  # Already in list format
            
        messages = []
        
        # Split by line breaks
        lines = chat_text.split('\n')
        current_message = ""
        
        for line in lines:
            line = line.strip()
            if not line:
                # Empty line might be a message separator
                if current_message:
                    messages.append(current_message)
                    current_message = ""
                continue
                
            # Check if this line has a sender pattern (Name: Message)
            sender_match = re.search(r'([^:]+):\s*(.*)', line)
            if sender_match:
                # Save previous message if exists
                if current_message:
                    messages.append(current_message)
                    
                # Extract just the message part
                message = sender_match.group(2).strip()
                if message:
                    current_message = message
                else:
                    current_message = ""  # Empty message, reset
            else:
                # This might be a continuation of the previous message
                if current_message:
                    current_message += " " + line
                else:
                    # Or a standalone message
                    current_message = line
                
        # Add the last message if exists
        if current_message:
            messages.append(current_message)
                
        return messages
    
    def _detect_language(self, text: str) -> str:
        """
        Detect language of the conversation and return language code
        
        Args:
            text: Text to analyze
            
        Returns:
            Language code (en, es, fr, etc.)
        """
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
    
    def _extract_entities(self, text: str) -> Dict[str, List[str]]:
        doc = self.nlp(text)
        entities = {
            "names": [],
            "places": [],
            "interests": [],
            "events": []
        }
        
        # Extract names using spaCy's PERSON label
        for ent in doc.ents:
            if ent.label_ == "PERSON":
                entities["names"].append(ent.text)
        
        # Extract places using GPE and LOC labels
        for ent in doc.ents:
            if ent.label_ in ["GPE", "LOC"]:
                entities["places"].append(ent.text)
        
        # Extract events using DATE and TIME labels
        for ent in doc.ents:
            if ent.label_ in ["DATE", "TIME"]:
                entities["events"].append(ent.text)
        
        # Extract interests using custom logic with spaCy
        interests = []
        
        # Verb-based interests
        interest_verbs = ["like", "love", "enjoy"]
        for token in doc:
            if token.lemma_ in interest_verbs:
                # Check for dobj
                for child in token.children:
                    if child.dep_ == "dobj":
                        for chunk in doc.noun_chunks:
                            if child in chunk:
                                interests.append(chunk.text)
                                break
                # Check for xcomp
                for child in token.children:
                    if child.dep_ == "xcomp":
                        start = child.left_edge.i
                        end = child.right_edge.i + 1
                        interests.append(doc[start:end].text)
        
        # Noun-based interests
        interest_nouns = ["hobby", "interest", "passion"]
        for token in doc:
            if token.lemma_ in interest_nouns:
                for next_token in token.rights:
                    if next_token.dep_ == "cop":
                        for complement_token in next_token.children:
                            if complement_token.dep_ in ["attr", "acomp"]:
                                start = complement_token.left_edge.i
                                end = complement_token.right_edge.i + 1
                                interests.append(doc[start:end].text)
        
        # Phrase-based interests
        interest_phrases = {
            ("fan", "NOUN"): ["of"],
            ("excited", "ADJ"): ["about"]
        }
        for token in doc:
            key = (token.lemma_, token.pos_)
            if key in interest_phrases:
                for child in token.children:
                    if child.dep_ == "prep" and child.text.lower() in interest_phrases[key]:
                        for grandchild in token.children:
                            if grandchild.dep_ == "pobj":
                                for chunk in doc.noun_chunks:
                                    if grandchild in chunk:
                                        interests.append(chunk.text)
                                        break
        
        # Deduplicate and limit to top 5 unique entities
        for key in entities:
            entities[key] = list(set(entities[key]))[:5]
        
        entities["interests"] = list(set(interests))[:5]
        
        return entities
    
    def _analyze_sentiment(self, text):
        result = self.sentiment_analysis(text)
        return result[0]["label"], result[0]["score"]
    
    def _analyze_context(self, chat_history: List[str], user_id: str = "default") -> Dict:
        """
        Analyze chat to extract context like topics, sentiment, and pace
        
        Args:
            chat_history: List of chat messages
            user_id: Identifier for the user
            
        Returns:
            Dictionary with context information
        """
        # Get cache key
        cache_key = f"{user_id}_{hash(str(chat_history))}"
        
        # Check if we have a cached analysis
        if cache_key in self.conversation_cache:
            cached = self.conversation_cache[cache_key]
            # Check if cache is fresh (less than 5 minutes old)
            if time.time() - cached['timestamp'] < 300:
                logging.info("Using cached conversation analysis")
                return cached['analysis']
        
        # Process new or stale cache
        recent_messages = chat_history[-10:] if len(chat_history) > 10 else chat_history
        combined_text = " ".join(recent_messages)
        
        # Basic topic extraction (would be more sophisticated in production)
        common_topics = [
            "movies", "music", "food", "travel", "work", "study", 
            "hobbies", "sports", "books", "family", "weather", "weekend",
            "dating", "relationship", "career", "education", "health", "fitness",
            "technology", "art", "gaming", "politics", "science", "fashion"
        ]
        
        found_topics = []
        for topic in common_topics:
            if topic in combined_text.lower():
                found_topics.append(topic)
        
                
        # Message pace/length analysis
        avg_length = sum(len(msg) for msg in recent_messages) / max(1, len(recent_messages))
        
        # Extract entities
        entities = self._extract_entities(combined_text)
        
        # Update memory with entities if user_id is provided
        if user_id != "default":
            if user_id not in self.memory:
                self.memory[user_id] = {"entities": {}}
            
            # Merge new entities with existing ones
            for entity_type, values in entities.items():
                if entity_type not in self.memory[user_id]["entities"]:
                    self.memory[user_id]["entities"][entity_type] = []
                
                # Add new entities
                self.memory[user_id]["entities"][entity_type] = list(set(
                    self.memory[user_id]["entities"][entity_type] + values
                ))[:10]  # Keep top 10
        
        # Analyze conversation stage
        stage = self._determine_conversation_stage(chat_history)
        
        # Create context analysis
        analysis = {
            "language": self._detect_language(combined_text),
            "topics": found_topics[:5],  # Top 5 topics
            "message_length": "short" if avg_length < 50 else "medium" if avg_length < 100 else "long",
            "entities": entities,
            "stage": stage
        }

         # Basic sentiment analysis
        combined_text = " ".join(recent_messages)
        label, score = self._analyze_sentiment(combined_text)

        # Map the label to match existing structure, handling neutral cases
        if label == "POSITIVE" and score >= 0.5:
            analysis["sentiment"] = "positive"
        elif label == "NEGATIVE" and score >= 0.5:
            analysis["sentiment"] = "negative"
        else:
            analysis["sentiment"] = "neutral"
        analysis["sentiment_score"] = score
        
        # Cache the analysis
        self.conversation_cache[cache_key] = {
            'timestamp': time.time(),
            'analysis': analysis
        }
        
        return analysis
    
    def _determine_conversation_stage(self, chat_history: List[str]) -> str:
        """
        Determine the stage of the conversation
        
        Args:
            chat_history: List of chat messages
            
        Returns:
            Conversation stage as string
        """
        if not chat_history:
            return "initial_greeting"
            
        message_count = len(chat_history)
        
        if message_count <= 2:
            return "initial_greeting"
        elif message_count <= 6:
            return "getting_to_know"
        elif message_count <= 12:
            return "building_rapport"
        elif message_count <= 20:
            return "deepening_connection"
        else:
            return "established_conversation"
    
    def _summarize_conversation(self, chat_history: List[str], max_messages: int = 5) -> str:
        """
        Summarize older parts of conversation to reduce context length
        
        Args:
            chat_history: List of chat messages
            max_messages: Maximum number of recent messages to keep in full
            
        Returns:
            Summarized conversation text
        """
        if len(chat_history) <= max_messages:
            return "\n".join(chat_history)
            
        # Keep recent messages intact
        recent_messages = chat_history[-max_messages:]
        older_messages = chat_history[:-max_messages]
        
        # Simple summarization for older parts
        topics = []
        sentiment = "neutral"
        
        # Extract key topics
        combined_older = " ".join(older_messages)
        common_topics = ["movies", "music", "food", "travel", "work", "hobbies", "dating", "family"]
        
        for topic in common_topics:
            if topic in combined_older.lower():
                topics.append(topic)
                
        # Basic sentiment
        positive = ["happy", "good", "great", "love", "enjoy", "excited"]
        negative = ["sad", "bad", "upset", "worry", "sorry", "disappointed"]
        
        sentiment_score = 0
        for word in positive:
            if word in combined_older.lower():
                sentiment_score += 1
        for word in negative:
            if word in combined_older.lower():
                sentiment_score -= 1
                
        sentiment = "positive" if sentiment_score > 0 else "negative" if sentiment_score < 0 else "neutral"
        
        # Build summary
        topics_str = ", ".join(topics[:3]) if topics else "general topics"
        summary = f"[Earlier conversation summary: They discussed {topics_str} with a {sentiment} tone.]"
        
        # Combine summary with recent messages
        return summary + "\n" + "\n".join(recent_messages)
    
    def _chunk_conversation(self, chat_history: List[str], max_tokens: int = 500) -> List[str]:
        """
        Chunk conversation to fit within token limits
        
        Args:
            chat_history: List of chat messages
            max_tokens: Maximum tokens per chunk
            
        Returns:
            Chunked conversation
        """
        if not chat_history:
            return []
            
        # Estimate tokens (rough approximation)
        def estimate_tokens(text):
            return len(text.split())
            
        chunks = []
        current_chunk = []
        current_token_count = 0
        
        for message in reversed(chat_history):  # Process most recent first
            message_tokens = estimate_tokens(message)
            
            if current_token_count + message_tokens <= max_tokens:
                current_chunk.insert(0, message)  # Insert at beginning to maintain order
                current_token_count += message_tokens
            else:
                # This message would exceed the limit, save current chunk and start new one
                if current_chunk:
                    chunks.insert(0, current_chunk)  # Insert at beginning to maintain order
                current_chunk = [message]
                current_token_count = message_tokens
                
        # Add the last chunk if not empty
        if current_chunk:
            chunks.insert(0, current_chunk)
            
        return chunks
    
    def generate_response(self, chat_history: List[str], style: str = "friendly", 
                         message_length: str = "medium", user_id: str = "default") -> str:
        """
        Generate contextually appropriate response based on chat history and style
        
        Args:
            chat_history: List of previous chat messages
            style: Response style to use
            message_length: Desired length of responses
            user_id: Identifier for the user
            
        Returns:
            Generated response
        """
        if not chat_history:
            return "Hi there! How's your day going? 😊"
        
        # Get style configuration (default to friendly if style not found)
        style_config = self.styles.get(style.lower(), self.styles["friendly"])
        
        # Analyze conversation context
        context = self._analyze_context(chat_history, user_id)
        
        # Override message length if specified
        if message_length in ["short", "medium", "long"]:
            context["message_length"] = message_length
        
        # Check if conversation is too long for model context
        if len(" ".join(chat_history[-10:])) > self.max_length * 2:  # Conservative estimate
            # Use chunking approach
            logging.info("Conversation too long, using chunking approach")
            return self._generate_with_chunking(chat_history, style, context, user_id)
        
        # Process conversation into context
        # Use the most recent messages that fit in context window
        recent_len = min(6, len(chat_history))  # Default to last 6 messages
        
        # Format conversation as alternating turns
        conversation = "\n".join([
            f"{'User' if i % 2 == 0 else 'Friend'}: {line}" 
            for i, line in enumerate(chat_history[-recent_len:])
        ])
        
        # Add memory context if available
        memory_context = ""
        if user_id in self.memory:
            entity_info = []
            for entity_type, values in self.memory[user_id].get("entities", {}).items():
                if values:
                    entity_info.append(f"{entity_type.capitalize()}: {', '.join(values[:3])}")
            
            if entity_info:
                memory_context = f"Background information: {'; '.join(entity_info)}\n"
        
        # Build enhanced prompt with context-aware instructions
        topics_str = ", ".join(context["topics"]) if context["topics"] else "any relevant topic"
        
        enhanced_prompt = (
            f"{style_config['prompt']}\n\n"
            f"Recent message:\n{chat_history[-1]}\n\n"
            f"Respond in a {style} way, keeping it relevant to {', '.join(context['topics'])}."
        )
        
        # Add examples
        for example in style_config["examples"][:2]:
            enhanced_prompt += f"- {example}\n"
            
        enhanced_prompt += f"\nConversation:\n{conversation}\n\nResponse:"
        
        logging.info(f"Generated Prompt:\n{enhanced_prompt}")

        # Tokenize input with truncation
        input_ids = self.tokenizer(
            enhanced_prompt, 
            return_tensors="pt",
            truncation=True,
            max_length=self.max_length
        ).input_ids.to(self.device)
        
        try:
            # Generate response with dynamic parameters based on style
            output = self.model.generate(
                input_ids,
                max_new_tokens=150,  # Reduce from 200 to keep it concise
                min_length=20,
                temperature=1.0,  # Crank it up for more flair
                top_k=50,
                top_p=0.95,  # Slightly higher for diversity
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
                
        except Exception as e:
            logging.error(f"Error in model generation: {e}")
            # Fallback response
            response = "I'd like to continue our conversation, but I'm having trouble processing right now. Could you share more about your interests?"
            
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
    
    def _generate_with_chunking(self, chat_history: List[str], style: str, context: Dict, user_id: str = "default") -> str:
        """
        Generate response for long conversations by processing in chunks
        
        Args:
            chat_history: List of chat messages
            style: Response style to use
            context: Conversation context
            user_id: Identifier for the user
            
        Returns:
            Generated response
        """
        # Keep last message separate as the immediate context
        last_message = chat_history[-1]
        older_history = chat_history[:-1]
        
        # Summarize older parts of the conversation
        summary = self._summarize_conversation(older_history, max_messages=3)
        
        # Create a new condensed history
        condensed_history = [summary, last_message]
        
        # Get style configuration
        style_config = self.styles.get(style.lower(), self.styles["friendly"])
        
        # Build enhanced prompt with context-aware instructions
        topics_str = ", ".join(context["topics"]) if context["topics"] else "any relevant topic"
        
        # Add memory context if available
        memory_context = ""
        if user_id in self.memory:
            entity_info = []
            for entity_type, values in self.memory[user_id].get("entities", {}).items():
                if values:
                    entity_info.append(f"{entity_type.capitalize()}: {', '.join(values[:3])}")
            
            if entity_info:
                memory_context = f"Background information: {'; '.join(entity_info)}\n"
        
        enhanced_prompt = (
            f"{style_config['prompt']}\n\n"
            f"{memory_context}"
            f"Conversation language: {context['language']}\n"
            f"Conversation topics: {topics_str}\n"
            f"Conversation sentiment: {context['sentiment']}\n"
            f"Earlier conversation summary: {summary}\n"
            f"Examples of this style:\n"
        )
        
        # Add examples (just one to save tokens)
        enhanced_prompt += f"- {style_config['examples'][0]}\n"
            
        enhanced_prompt += f"\nMost recent message:\n{last_message}\n\nResponse:"
        
        logging.info(f"Generated Chunked Prompt:\n{enhanced_prompt}")

        # Tokenize input with truncation
        input_ids = self.tokenizer(
            enhanced_prompt, 
            return_tensors="pt",
            truncation=True,
            max_length=self.max_length
        ).input_ids.to(self.device)
        
        try:
            # Generate response
            output = self.model.generate(
                input_ids,
                max_new_tokens=200,
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
            
            # Extract only the generated response part
            try:
                response = response.split("Response:")[-1].strip()
            except:
                pass
                
        except Exception as e:
            logging.error(f"Error in chunked generation: {e}")
            # Fallback response
            response = "I'd like to continue our conversation, but I'm having trouble processing right now. Could you share more about what you're thinking?"
            
        # Cleanup and formatting
        response = self._format_response(response, style, context)
        
        # Fallback if needed
        if not response.strip():
            response = "I'd love to hear more about that. What other thoughts do you have on this topic?"
        
        logging.info(f"Generated Chunked Response:\n{response}")
        
        return response
    
    def generate_multiple_responses(self, chat_history: List[str], style: str = "friendly", 
                                  count: int = 3, message_length: str = "medium",
                                  user_id: str = "default") -> List[str]:
        """
        Generate multiple different responses to provide options.
        
        Args:
            chat_history: List of previous chat messages
            style: Response style to use
            count: Number of responses to generate
            message_length: Desired length of responses
            user_id: Identifier for the user
            
        Returns:
            List of generated responses
        """
        responses = []
        
        # Add a bit of randomness by slightly adjusting temperature for each generation
        style_config = self.styles.get(style.lower(), self.styles["friendly"])
        base_temp = style_config.get("temp", 0.7)
        
        for i in range(count):
            try:
                # Slightly adjust temperature for each generation to get variety
                temp_adjustment = 0.05 * (i - count//2)  # Will give variety around the base temperature
                style_config["temp"] = max(0.5, min(0.95, base_temp + temp_adjustment))
                
                response = self.generate_response(chat_history, style, message_length, user_id)
                responses.append(response)
                
                # Ensure we're not duplicating responses
                if i > 0 and response in responses[:-1]:
                    # Try once more with a higher temperature
                    style_config["temp"] = min(0.95, base_temp + 0.15)
                    response = self.generate_response(chat_history, style, message_length, user_id)
                    responses[-1] = response
            except Exception as e:
                logging.error(f"Error generating response {i+1}: {e}")
                responses.append(f"I'd love to continue our conversation. What else would you like to talk about?")
        
        return responses
    
    def _format_response(self, response: str, style: str, context: Dict) -> str:
        """
        Format and clean up generated response
        
        Args:
            response: Raw model response
            style: Response style
            context: Conversation context
            
        Returns:
            Formatted response
        """
        # Remove any prefixes that might have been generated
        prefixes_to_remove = ["User:", "Friend:", "Response:", "AI:", "Assistant:"]
        
        for prefix in prefixes_to_remove:
            if response.startswith(prefix):
                response = response[len(prefix):].strip()
        
        # Format based on style and message length preference
        if context.get("message_length") == "short":
            # Keep it brief, remove extra sentences if needed
            sentences = re.split(r'(?<=[.!?])\s+', response)
            if len(sentences) > 2:
                response = " ".join(sentences[:2])
        elif context.get("message_length") == "long" and len(response) < 100:
            # Add more content if response is too short
            if style == "friendly":
                response += " I'd love to hear more about your thoughts on this too!"
            elif style == "flirty":
                response += " By the way, I've been enjoying getting to know you through our conversation."
            elif style == "deep":
                response += " I find discussions like this really help us understand each other better."
        
        # Add emojis for certain styles
        if style in ["friendly", "flirty", "playful"] and "😊" not in response and "😉" not in response:
            if random.random() < 0.7:  # 70% chance to add emoji
                emojis = {
                    "friendly": ["😊", "👍", "🙂"],
                    "flirty": ["😉", "😊", "💕"],
                    "playful": ["😜", "😂", "🤗"]
                }
                selected_emoji = random.choice(emojis.get(style, ["😊"]))
                
                # Add to end if there's a sentence ending
                if response.rstrip()[-1] in ['.', '!', '?']:
                    response = response.rstrip() + " " + selected_emoji
                else:
                    response = response.rstrip() + ". " + selected_emoji
        
        # Ensure response doesn't start with whitespace
        response = response.lstrip()
        
        # Ensure appropriate capitalization
        if response and response[0].islower():
            response = response[0].upper() + response[1:]
            
        return response

    def save_chat_history(self, chat_history: List[str], file_path: Optional[str] = None) -> bool:
        """
        Save chat history to file
        
        Args:
            chat_history: List of chat messages
            file_path: Path to save the chat history to
            
        Returns:
            True if successful, False otherwise
        """
        file_path = file_path or self.chat_file
        
        try:
            with open(file_path, "w", encoding="utf-8") as file:
                for message in chat_history:
                    file.write(message + "\n")
            
            logging.info(f"Successfully saved {len(chat_history)} chat messages to {file_path}")
            return True
            
        except Exception as e:
            logging.error(f"Error saving chat history: {e}")
            return False
    
    def save_conversation_memory(self, user_id: str, memory_file: str = "conversation_memory.json") -> bool:
        """
        Save user conversation memory to file
        
        Args:
            user_id: User identifier
            memory_file: File to save memory to
            
        Returns:
            True if successful, False otherwise
        """
        if user_id not in self.memory:
            logging.warning(f"No memory found for user {user_id}")
            return False
            
        try:
            # Load existing memory if file exists
            existing_memory = {}
            if os.path.exists(memory_file):
                with open(memory_file, "r", encoding="utf-8") as file:
                    existing_memory = json.load(file)
            
            # Update with current user memory
            existing_memory[user_id] = self.memory[user_id]
            
            # Save back to file
            with open(memory_file, "w", encoding="utf-8") as file:
                json.dump(existing_memory, file, indent=2)
                
            logging.info(f"Successfully saved memory for user {user_id}")
            return True
            
        except Exception as e:
            logging.error(f"Error saving conversation memory: {e}")
            return False
    
    def load_conversation_memory(self, memory_file: str = "conversation_memory.json") -> bool:
        """
        Load user conversation memory from file
        
        Args:
            memory_file: File to load memory from
            
        Returns:
            True if successful, False otherwise
        """
        if not os.path.exists(memory_file):
            logging.warning(f"Memory file {memory_file} not found")
            return False
            
        try:
            with open(memory_file, "r", encoding="utf-8") as file:
                loaded_memory = json.load(file)
            
            # Update memory with loaded data
            self.memory.update(loaded_memory)
            
            logging.info(f"Successfully loaded memory for {len(loaded_memory)} users")
            return True
            
        except Exception as e:
            logging.error(f"Error loading conversation memory: {e}")
            return False
    
    def add_personality_style(self, style_name: str, prompt: str, examples: List[str], temperature: float = 0.7) -> bool:
        """
        Add a new personality style
        
        Args:
            style_name: Name of the style
            prompt: Prompt template for the style
            examples: Example responses for the style
            temperature: Temperature parameter (0.0-1.0)
            
        Returns:
            True if successful, False otherwise
        """
        try:
            style_name = style_name.lower()
            
            # Validate inputs
            if not style_name or not prompt or not examples:
                logging.error("Style name, prompt and examples are required")
                return False
                
            if temperature < 0.1 or temperature > 1.0:
                logging.warning(f"Temperature {temperature} outside recommended range (0.1-1.0)")
                temperature = max(0.1, min(1.0, temperature))
            
            # Add or update the style
            self.styles[style_name] = {
                "prompt": prompt,
                "temp": temperature,
                "examples": examples[:5]  # Keep up to 5 examples
            }
            
            logging.info(f"Added personality style: {style_name}")
            return True
            
        except Exception as e:
            logging.error(f"Error adding personality style: {e}")
            return False
    
    def get_available_styles(self) -> List[str]:
        """
        Get list of available personality styles
        
        Returns:
            List of style names
        """
        return list(self.styles.keys())
    
    def clear_memory(self, user_id: Optional[str] = None) -> bool:
        """
        Clear conversation memory for a user or all users
        
        Args:
            user_id: User identifier (None to clear all)
            
        Returns:
            True if successful, False otherwise
        """
        try:
            if user_id:
                if user_id in self.memory:
                    del self.memory[user_id]
                    logging.info(f"Cleared memory for user {user_id}")
                else:
                    logging.warning(f"No memory found for user {user_id}")
                    return False
            else:
                self.memory = {}
                logging.info("Cleared all user memory")
                
            return True
            
        except Exception as e:
            logging.error(f"Error clearing memory: {e}")
            return False
    
    def simulate_conversation(self, initial_message: str, num_turns: int = 5, 
                           style: str = "friendly", user_id: str = "test_user") -> List[Dict[str, str]]:
        """
        Simulate a conversation for testing
        
        Args:
            initial_message: Starting message
            num_turns: Number of conversation turns
            style: Personality style to use
            user_id: User identifier
            
        Returns:
            List of conversation turns
        """
        conversation = []
        chat_history = [initial_message]
        
        try:
            # Add initial user message
            conversation.append({"role": "user", "message": initial_message})
            
            for i in range(num_turns):
                # Generate bot response
                bot_response = self.generate_response(chat_history, style, user_id=user_id)
                conversation.append({"role": "bot", "message": bot_response})
                chat_history.append(bot_response)
                
                # Generate simulated user response if not the last turn
                if i < num_turns - 1:
                    # Simple simulation of user response
                    user_templates = [
                        "That's interesting. Tell me more about {topic}.",
                        "I like {topic} too. What do you think about {related_topic}?",
                        "I had a similar experience with {topic} last week.",
                        "Have you ever tried {related_topic}?",
                        "What would you recommend for someone interested in {topic}?"
                    ]
                    
                    # Extract a topic from the conversation
                    context = self._analyze_context(chat_history, user_id)
                    topics = context.get("topics", ["chatting"])
                    topic = random.choice(topics) if topics else "chatting"
                    
                    # Generate related topic
                    related_topics = {
                        "movies": ["TV shows", "directors", "actors"],
                        "music": ["concerts", "instruments", "albums"],
                        "food": ["cooking", "restaurants", "recipes"],
                        "travel": ["destinations", "hotels", "flights"],
                        "work": ["career", "office", "colleagues"],
                        "dating": ["relationships", "romance", "meetups"],
                        "chatting": ["conversation", "communication", "meeting new people"]
                    }
                    
                    related_topic = random.choice(related_topics.get(topic, ["chatting"]))
                    
                    # Create user message
                    template = random.choice(user_templates)
                    user_response = template.format(topic=topic, related_topic=related_topic)
                    
                    conversation.append({"role": "user", "message": user_response})
                    chat_history.append(user_response)
            
            return conversation
            
        except Exception as e:
            logging.error(f"Error in conversation simulation: {e}")
            return conversation
    
    def _extract_keywords(self, text: str, max_keywords: int = 5) -> List[str]:
        """
        Extract keywords from text
        
        Args:
            text: Input text
            max_keywords: Maximum number of keywords to extract
            
        Returns:
            List of extracted keywords
        """
        # Simple keyword extraction based on word frequency
        # In production, you would use a more sophisticated approach
        
        # Lowercase and tokenize
        words = re.findall(r'\b[a-zA-Z]{3,15}\b', text.lower())
        
        # Filter out common stopwords
        stopwords = [
            "the", "and", "is", "in", "it", "to", "that", "was", "for", "on",
            "with", "are", "as", "this", "be", "by", "an", "not", "have", "has",
            "had", "do", "does", "did", "can", "could", "will", "would", "should",
            "but", "or", "because", "if", "from", "what", "who", "when", "where",
            "which", "how", "why", "all", "any", "both", "each", "few", "more",
            "most", "other", "some", "such", "than", "then", "too", "very",
            "just", "now", "also", "like", "so", "than", "only", "its"
        ]
        
        filtered_words = [word for word in words if word not in stopwords]
        
        # Count frequency
        word_counts = {}
        for word in filtered_words:
            word_counts[word] = word_counts.get(word, 0) + 1
            
        # Sort by frequency
        sorted_words = sorted(word_counts.items(), key=lambda x: x[1], reverse=True)
        
        # Return top keywords
        return [word for word, count in sorted_words[:max_keywords]]


# Example usage
if __name__ == "__main__":
    import random
    
    # Initialize the chat generator
    generator = ChatGenerator()
    
    # Example chat history
    chat_history = [
        "Hi there! How's it going?",
        "I'm doing well, thanks for asking! How about you?",
        "Pretty good! I've been enjoying the nice weather lately.",
        "That's great! I love spending time outdoors when it's nice out. Do you have any favorite outdoor activities?",
        "I enjoy hiking and photography. There's a beautiful trail near my place with great views!"
    ]
    
    # Generate response with different styles
    styles = ["friendly", "flirty", "confident", "deep"]
    
    print("Generating responses with different styles:")
    for style in styles:
        response = generator.generate_response(chat_history, style=style)
        print(f"\n{style.upper()} STYLE:")
        print(response)
    
    # Simulate a conversation
    print("\n\nSimulated conversation:")
    conversation = generator.simulate_conversation("Hey! I'm new here. What do you like to do for fun?", num_turns=3)
    
    for turn in conversation:
        role = turn["role"].upper()
        message = turn["message"]
        print(f"\n{role}: {message}")
    
    # Generate multiple response options
    print("\n\nMultiple response options:")
    responses = generator.generate_multiple_responses(chat_history, count=3)
    
    for i, response in enumerate(responses, 1):
        print(f"\nOption {i}:")
        print(response)