"""
Production Vibe Classifier using NLP
Classifies fashion vibes from captions and hashtags
"""

import re
import json
from pathlib import Path
import numpy as np
from collections import Counter
import logging

logger = logging.getLogger(__name__)

class ProductionVibeClassifier:
    def __init__(self, vibes_config_path=None):
        """
        Initialize production vibe classifier
        
        Args:
            vibes_config_path: Path to vibes configuration JSON
        """
        print("Initializing Production Vibe Classifier...")
        
        if vibes_config_path and Path(vibes_config_path).exists():
            self.load_vibes_config(vibes_config_path)
        else:
            self.create_production_vibes_config()
        
        self.compile_patterns()
        
        print("Production Vibe Classifier initialized")
    
    def create_production_vibes_config(self):
        """Create production vibes configuration"""
        print("Creating production vibes configuration...")
        
        self.vibes_config = {
            "Coquette": {
                "keywords": [
                    "bow", "ribbon", "pink", "feminine", "delicate", "sweet", "cute", 
                    "girly", "romantic", "soft", "pastel", "lace", "floral", "pearl",
                    "coquette", "dainty", "pretty", "princess", "ballet", "tulle"
                ],
                "hashtags": [
                    "#coquette", "#coquetteaesthetic", "#bow", "#pink", "#feminine",
                    "#girly", "#romantic", "#soft", "#pastel", "#princess"
                ],
                "weight": 1.0,
                "description": "Feminine, delicate aesthetic with bows, ribbons, and soft colors"
            },
            "Clean Girl": {
                "keywords": [
                    "minimal", "natural", "effortless", "simple", "fresh", "dewy",
                    "no-makeup", "slicked", "glass skin", "minimalist", "clean",
                    "bare", "glowing", "healthy", "radiant", "subtle"
                ],
                "hashtags": [
                    "#cleangirl", "#minimal", "#natural", "#effortless", "#nomakeup",
                    "#glowingskin", "#minimalist", "#fresh", "#healthy", "#simple"
                ],
                "weight": 1.0,
                "description": "Minimal, natural, effortless beauty and style"
            },
            "Cottagecore": {
                "keywords": [
                    "cottage", "rural", "vintage", "floral", "prairie", "countryside",
                    "garden", "rustic", "pastoral", "nature", "wildflower", "meadow",
                    "farmhouse", "cozy", "homemade", "traditional", "earthy"
                ],
                "hashtags": [
                    "#cottagecore", "#vintage", "#floral", "#countryside", "#nature",
                    "#rustic", "#cozy", "#farmhouse", "#traditional", "#pastoral"
                ],
                "weight": 1.0,
                "description": "Rural, vintage-inspired aesthetic celebrating countryside life"
            },
            "Streetcore": {
                "keywords": [
                    "street", "urban", "edgy", "grunge", "punk", "alternative",
                    "rebel", "cool", "attitude", "underground", "raw", "gritty",
                    "skate", "hip-hop", "oversized", "baggy", "distressed"
                ],
                "hashtags": [
                    "#streetcore  "distressed"
                ],
                "hashtags": [
                    "#streetcore", "#streetwear", "#urban", "#edgy", "#grunge", "#punk",
                    "#alternative", "#skate", "#hiphop", "#oversized", "#underground"
                ],
                "weight": 1.0,
                "description": "Urban, edgy street style with attitude"
            },
            "Y2K": {
                "keywords": [
                    "y2k", "2000s", "metallic", "holographic", "cyber", "futuristic",
                    "tech", "digital", "neon", "chrome", "space", "millennium",
                    "iridescent", "shiny", "reflective", "matrix", "techno"
                ],
                "hashtags": [
                    "#y2k", "#2000s", "#metallic", "#holographic", "#cyber", "#futuristic",
                    "#neon", "#chrome", "#techno", "#millennium", "#digital"
                ],
                "weight": 1.0,
                "description": "Early 2000s futuristic aesthetic with metallic and tech elements"
            },
            "Boho": {
                "keywords": [
                    "bohemian", "free-spirit", "hippie", "flowing", "earthy", "natural",
                    "fringe", "ethnic", "wanderlust", "festival", "artistic", "eclectic",
                    "layered", "textured", "tribal", "gypsy", "nomadic"
                ],
                "hashtags": [
                    "#boho", "#bohemian", "#freespirit", "#hippie", "#festival",
                    "#wanderlust", "#ethnic", "#tribal", "#eclectic", "#artistic"
                ],
                "weight": 1.0,
                "description": "Bohemian free-spirit aesthetic with flowing, artistic elements"
            },
            "Party Glam": {
                "keywords": [
                    "glam", "sparkle", "sequin", "glitter", "party", "night out",
                    "dramatic", "bold", "statement", "luxury", "elegant", "chic",
                    "glamorous", "dazzling", "shimmery", "festive", "celebration"
                ],
                "hashtags": [
                    "#partyglam", "#glam", "#sparkle", "#sequin", "#glitter", "#nightout",
                    "#dramatic", "#luxury", "#elegant", "#glamorous", "#festive"
                ],
                "weight": 1.0,
                "description": "Glamorous, sparkly aesthetic for special occasions and nightlife"
            }
        }
        
        config_path = Path("data/vibes_config.json")
        config_path.parent.mkdir(exist_ok=True)
        
        with open(config_path, 'w') as f:
            json.dump(self.vibes_config, f, indent=2)
        
        print(f"Saved vibes configuration to {config_path}")
    
    def load_vibes_config(self, config_path):
        """Load vibes configuration from JSON"""
        print(f"Loading vibes config from {config_path}")
        
        with open(config_path, 'r') as f:
            self.vibes_config = json.load(f)
        
        print(f"Loaded {len(self.vibes_config)} vibe categories")
    
    def compile_patterns(self):
        """Compile regex patterns for efficient matching"""
        self.keyword_patterns = {}
        self.hashtag_patterns = {}
        
        for vibe, config in self.vibes_config.items():
            keywords = [re.escape(kw) for kw in config['keywords']]
            self.keyword_patterns[vibe] = re.compile(
                r'\b(?:' + '|'.join(keywords) + r')\b', 
                re.IGNORECASE
            )
            hashtags = [re.escape(ht) for ht in config['hashtags']]
            self.hashtag_patterns[vibe] = re.compile(
                r'(?:' + '|'.join(hashtags) + r')', 
                re.IGNORECASE
            )
    
    def preprocess_text(self, text):
        """Preprocess text for analysis"""
        if not text:
            return "", []
        text = text.lower()
        hashtags = re.findall(r'#\w+', text)
        cleaned_text = re.sub(r'[^\w\s#]', ' ', text)
        cleaned_text = ' '.join(cleaned_text.split())
        
        return cleaned_text, hashtags
    
    def classify_vibes(self, text, max_vibes=2):
        """
        Classify vibes from text content
        
        Args:
            text: Input text (caption, hashtags, etc.)
            max_vibes: Maximum number of vibes to return
        
        Returns:
            List of classified vibes
        """
        if not text:
            return ["Clean Girl"]  
        
        cleaned_text, hashtags = self.preprocess_text(text)
        vibe_scores = {}
        
        for vibe, config in self.vibes_config.items():
            score = 0
            matches = []
            keyword_matches = self.keyword_patterns[vibe].findall(cleaned_text)
            score += len(keyword_matches) * 2 
            matches.extend(keyword_matches)
            hashtag_text = ' '.join(hashtags)
            hashtag_matches = self.hashtag_patterns[vibe].findall(hashtag_text)
            score += len(hashtag_matches) * 3  
            matches.extend(hashtag_matches)
            score *= config.get('weight', 1.0)
            
            if score > 0:
                vibe_scores[vibe] = {
                    'score': score,
                    'matches': matches,
                    'confidence': min(score / 10.0, 1.0)  # Normalize to 0-1
                }
        
        sorted_vibes = sorted(
            vibe_scores.items(), 
            key=lambda x: x[1]['score'], 
            reverse=True
        )
        
        result_vibes = []
        for vibe, data in sorted_vibes[:max_vibes]:
            if data['score'] > 0:
                result_vibes.append(vibe)
        
        if not result_vibes:
            result_vibes = ["Clean Girl"]
        
        return result_vibes
    
    def get_vibe_confidence(self, text, vibe):
        """Get confidence score for a specific vibe"""
        if vibe not in self.vibes_config:
            return 0.0
        
        cleaned_text, hashtags = self.preprocess_text(text)
        keyword_matches = len(self.keyword_patterns[vibe].findall(cleaned_text))
        hashtag_matches = len(self.hashtag_patterns[vibe].findall(' '.join(hashtags)))
        
        total_score = keyword_matches * 2 + hashtag_matches * 3
        confidence = min(total_score / 10.0, 1.0)
        
        return round(confidence, 3)
    
    def analyze_text_detailed(self, text):
        """Detailed analysis of text for debugging"""
        cleaned_text, hashtags = self.preprocess_text(text)
        
        analysis = {
            'original_text': text,
            'cleaned_text': cleaned_text,
            'hashtags': hashtags,
            'vibe_analysis': {}
        }
        
        for vibe, config in self.vibes_config.items():
            keyword_matches = self.keyword_patterns[vibe].findall(cleaned_text)
            hashtag_matches = self.hashtag_patterns[vibe].findall(' '.join(hashtags))
            
            analysis['vibe_analysis'][vibe] = {
                'keyword_matches': keyword_matches,
                'hashtag_matches': hashtag_matches,
                'total_score': len(keyword_matches) * 2 + len(hashtag_matches) * 3,
                'confidence': self.get_vibe_confidence(text, vibe)
            }
        
        return analysis
    
    def get_supported_vibes(self):
        """Get list of supported vibes"""
        return list(self.vibes_config.keys())
    
    def get_vibe_info(self, vibe):
        """Get detailed information about a specific vibe"""
        return self.vibes_config.get(vibe, {})

if __name__ == "__main__":
    classifier = ProductionVibeClassifier()
    test_texts = [
        "Getting ready for date night with sparkly dress #glam #party #nightout",
        "Soft girl aesthetic with pink bows and pearls #coquette #feminine #soft",
        "Minimal makeup, slicked back hair #cleangirl #natural #effortless",
        "Cottage core vibes with floral dress #cottagecore #vintage #nature",
        "Y2K metallic top and cyber sunglasses #y2k #futuristic #2000s",
        "Street style urban outfit #streetcore #urban #edgy",
        "Bohemian festival look #boho #freespirit #festival"
    ]
    print("Testing Production Vibe Classifier:")
    for text in test_texts:
        vibes = classifier.classify_vibes(text)
        print(f"\nText: {text}")
        print(f"Vibes: {vibes}")
    print(f"\nSupported vibes: {classifier.get_supported_vibes()}")
    print("Production Vibe Classifier ready")