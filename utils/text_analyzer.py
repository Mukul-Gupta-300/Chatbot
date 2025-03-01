# Add this to utils/text_analyzer.py

import re
from textblob import TextBlob

class ChatAnalyzer:
    """Analyzes chat messages to provide context-aware suggestions."""
    
    def analyze_conversation(self, messages):
        """
        Analyzes a conversation for sentiment, topics, and context.
        
        Args:
            messages (list): List of messages in the conversation
            
        Returns:
            dict: Analysis results including sentiment, topics, and context
        """
        # Join all messages for overall analysis
        full_text = " ".join(messages)
        
        # Perform sentiment analysis
        blob = TextBlob(full_text)
        sentiment_score = blob.sentiment.polarity
        
        # Determine overall sentiment
        if sentiment_score > 0.2:
            sentiment = "positive"
        elif sentiment_score < -0.2:
            sentiment = "negative"
        else:
            sentiment = "neutral"
            
        # Extract potential topics
        topics = self._extract_topics(full_text)
        
        # Detect question patterns
        questions = self._detect_questions(messages)
        
        # Detect conversation phase
        phase = self._determine_conversation_phase(messages)
        
        return {
            "sentiment": sentiment,
            "sentiment_score": sentiment_score,
            "topics": topics,
            "questions": questions,
            "phase": phase
        }
    
    def _extract_topics(self, text):
        """Extract potential topics from conversation."""
        # Simple topic extraction based on noun phrases
        blob = TextBlob(text)
        return [phrase.string for phrase in blob.noun_phrases]
    
    def _detect_questions(self, messages):
        """Detect if the last few messages contain questions."""
        # Focus on the last 3 messages
        recent_messages = messages[-3:] if len(messages) >= 3 else messages
        
        questions = []
        for msg in recent_messages:
            # Check for question marks
            if '?' in msg:
                # Extract the question
                question_parts = msg.split('?')
                for part in question_parts:
                    if part.strip():
                        # Find the start of the question
                        sentence = part.strip().split('.')[-1].strip()
                        if sentence:
                            questions.append(sentence + '?')
            
            # Check for question patterns without question marks
            elif re.search(r'\b(what|who|where|when|why|how)\b', msg.lower()):
                questions.append(msg)
                
        return questions
    
    def _determine_conversation_phase(self, messages):
        """Determine the current phase of the conversation."""
        num_messages = len(messages)
        
        if num_messages <= 3:
            return "introduction"
        elif num_messages <= 10:
            return "getting_to_know"
        else:
            return "established_conversation"