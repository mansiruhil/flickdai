"""
NLP-based vibe classification module
Analyzes captions and hashtags to classify fashion vibes
"""

import re
import spacy
from transformers import pipeline
import json
from collections import Counter

class VibeClassifier:
    def __init__(self):
        print(" Initializing Vibe Classifier...")
        
        # Define supported vibes and their keywords
        self.vibe_keywords = {
            'Coquette': [
                'bow', 'ribbon', 'pink', 'feminine', 'delicate', 'sweet', 'cute', 
                'girly', 'romantic', 'soft', 'pastel', 'lace', 'floral', 'pearl'
            ],
            'Clean Girl': [
                'minimal', 'natural', 'effortless', 'simple', 'fresh', 'dewy',
                'no-makeup', 'slicked', 'glass skin', 'minimalist', 'clean'
            ],
            'Cottagecore': [
                'cottage', 'rural', 'vintage', 'floral', 'prairie', 'countryside',
                'garden', 'rustic', 'pastoral', 'nature', 'wildflower', 'meadow'
            ],
            'Streetcore': [
                'street', 'urban', 'edgy', 'grunge', 'punk', 'alternative',
                'rebel', 'cool', 'attitude', 'underground', 'raw', 'gritty'
            ],
            'Y2K': [
                'y2k', '2000s', 'metallic', 'holographic', 'cyber', 'futuristic',
                'tech', 'digital', 'neon', 'chrome', 'space', 'millennium'
            ],
            'Boho': [
                'bohemian', 'free-spirit', 'hippie', 'flowing', 'earthy', 'natural',
                'fringe', 'ethnic', 'wanderlust', 'festival', 'artistic', 'eclectic'
            ],
            'Party Glam': [
                'glam', 'sparkle', 'sequin', 'glitter', 'party', 'night out',
                'dramatic', 'bold', 'statement', 'luxury', 'elegant', 'chic'
            ]
        }
        
        # Load spaCy model for NLP processing
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except OSError:
            print(" spaCy model not found, using basic processing")
            self.nlp = None
        
        # Load sentiment analysis pipeline
        try:
            self.sentiment_analyzer = pipeline("sentiment-analysis", 
                                             model="cardiffnlp/twitter-roberta-base-sentiment-latest")
        except Exception:
            print(" Sentiment analyzer not available")
            self.sentiment_analyzer = None
        
        print(" Vibe Classifier initialized")
    
    def preprocess_text(self, text):
        """Clean and preprocess text"""
        if not text:
            return ""
        
        # Convert to lowercase
        text = text.lower()
        
        # Extract hashtags
        hashtags = re.findall(r'#\w+', text)
        
        # Remove special characters but keep hashtags
        text = re.sub(r'[^\w\s#]', ' ', text)
        
        # Remove extra whitespace
        text = ' '.join(text.split())
        
        return text, hashtags
    
    def extract_keywords(self, text):
        """Extract relevant keywords from text"""
        if not text:
            return []
        
        processed_text, hashtags = self.preprocess_text(text)
        
        # Combine text and hashtags
        all_text = processed_text + ' ' + ' '.join(hashtags)
        
        keywords = []
        
        if self.nlp:
            # Use spaCy for advanced processing
            doc = self.nlp(all_text)
            
            # Extract nouns, adjectives, and hashtags
            for token in doc:
                if (token.pos_ in ['NOUN', 'ADJ'] and 
                    len(token.text) > 2 and 
                    not token.is_stop):
                    keywords.append(token.lemma_)
        else:
            # Basic keyword extraction
            words = all_text.split()
            keywords = [word for word in words if len(word) > 2]
        
        return keywords
    
    def classify_vibes(self, text, max_vibes=3):
        """Classify vibes based on text content"""
        print(f" Classifying vibes from text: '{text[:50]}...'")
        
        if not text:
            return ['Clean Girl']  # Default vibe
        
        keywords = self.extract_keywords(text)
        vibe_scores = {}
        
        # Score each vibe based on keyword matches
        for vibe, vibe_keywords in self.vibe_keywords.items():
            score = 0
            matches = []
            
            for keyword in keywords:
                for vibe_keyword in vibe_keywords:
                    if vibe_keyword in keyword or keyword in vibe_keyword:
                        score += 1
                        matches.append(keyword)
            
            if score > 0:
                vibe_scores[vibe] = {
                    'score': score,
                    'matches': matches
                }
        
        # Sort vibes by score
        sorted_vibes = sorted(vibe_scores.items(), 
                            key=lambda x: x[1]['score'], 
                            reverse=True)
        
        # Return top vibes
        result_vibes = []
        for vibe, data in sorted_vibes[:max_vibes]:
            if data['score'] > 0:
                result_vibes.append(vibe)
        
        # If no vibes detected, return default based on sentiment
        if not result_vibes:
            if self.sentiment_analyzer:
                try:
                    sentiment = self.sentiment_analyzer(text)[0]
                    if sentiment['label'] == 'POSITIVE':
                        result_vibes = ['Clean Girl']
                    else:
                        result_vibes = ['Streetcore']
                except:
                    result_vibes = ['Clean Girl']
            else:
                result_vibes = ['Clean Girl']
        
        print(f" Classified vibes: {result_vibes}")
        return result_vibes
    
    def analyze_hashtags(self, hashtags):
        """Analyze hashtags for vibe classification"""
        hashtag_text = ' '.join(hashtags).replace('#', '')
        return self.classify_vibes(hashtag_text)
    
    def get_vibe_confidence(self, text, vibe):
        """Get confidence score for a specific vibe"""
        keywords = self.extract_keywords(text)
        vibe_keywords = self.vibe_keywords.get(vibe, [])
        
        matches = 0
        for keyword in keywords:
            for vibe_keyword in vibe_keywords:
                if vibe_keyword in keyword or keyword in vibe_keyword:
                    matches += 1
        
        # Calculate confidence as percentage
        if len(keywords) > 0:
            confidence = min(matches / len(keywords), 1.0)
        else:
            confidence = 0.0
        
        return confidence

# Example usage
if __name__ == "__main__":
    classifier = VibeClassifier()
    
    # Test with different captions
    test_captions = [
        "Getting ready for date night  #glam #sparkle #nightout",
        "Soft girl vibes with pink bows and pearls  #coquette #feminine #soft",
        "Minimal makeup, slicked back hair #cleangirl #natural #effortless",
        "Cottage core aesthetic with floral dress #cottagecore #vintage #nature",
        "Y2K metallic top and cyber sunglasses #y2k #futuristic #2000s"
    ]
    
    print(" Testing vibe classification:")
    for caption in test_captions:
        vibes = classifier.classify_vibes(caption)
        print(f"Caption: {caption}")
        print(f"Vibes: {vibes}")
        print("-" * 50)
