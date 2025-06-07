"""
Real product matching using CLIP embeddings and FAISS
Matches detected fashion items to actual product catalog
"""

import numpy as np
import pandas as pd
from PIL import Image
import torch
import clip
import faiss
import json
import requests
from io import BytesIO
from pathlib import Path
import time
import logging

logger = logging.getLogger(__name__)

class RealProductMatcher:
    def __init__(self, catalog_df=None, embedding_cache_path="embeddings_cache.npy"):
        """
        Initialize product matcher with CLIP and FAISS
        
        Args:
            catalog_df: Product catalog DataFrame
            embedding_cache_path: Path to save/load embeddings cache
        """
        print("Initializing Real Product Matcher...")
        
        # Load CLIP model
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"🖥️ Using device: {self.device}")
        
        try:
            self.model, self.preprocess = clip.load("ViT-B/32", device=self.device)
            print("CLIP ViT-B/32 model loaded successfully")
        except Exception as e:
            print(f"Error loading CLIP model: {e}")
            raise
        
        # Initialize FAISS index
        self.embedding_dim = 512  
        self.index = faiss.IndexFlatIP(self.embedding_dim)  
        
        # Product catalog and embeddings
        self.catalog_df = catalog_df
        self.product_embeddings = []
        self.embedding_cache_path = embedding_cache_path
        
        # Similarity thresholds
        self.thresholds = {
            'exact': 0.90,
            'similar': 0.75,
            'no_match': 0.75
        }
        
        if self.catalog_df is not None:
            self.build_product_index()
        
        print("Real Product Matcher initialized")
    
    def load_image_from_url(self, url, timeout=10):
        """Load image from URL with error handling"""
        try:
            response = requests.get(url, timeout=timeout)
            response.raise_for_status()
            image = Image.open(BytesIO(response.content)).convert('RGB')
            return image
        except Exception as e:
            logger.warning(f"Failed to load image from {url}: {e}")
            return None
    
    def get_image_embedding(self, image):
        """Generate CLIP embedding for an image"""
        try:
            if isinstance(image, np.ndarray):
                # Convert numpy array to PIL Image
                image = Image.fromarray(image).convert('RGB')
            elif isinstance(image, str):
                # Load from URL
                image = self.load_image_from_url(image)
                if image is None:
                    return None
            
            # Preprocess and encode
            image_input = self.preprocess(image).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                image_features = self.model.encode_image(image_input)
                # Normalize for cosine similarity
                image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            
            return image_features.cpu().numpy().flatten()
            
        except Exception as e:
            logger.error(f"Error generating embedding: {e}")
            return None
    
    def build_product_index(self, force_rebuild=False):
        """Build FAISS index from product catalog"""
        print("Building product index...")
        
        # Check if cached embeddings exist
        cache_path = Path(self.embedding_cache_path)
        if cache_path.exists() and not force_rebuild:
            print("Loading cached embeddings...")
            try:
                cached_data = np.load(cache_path, allow_pickle=True).item()
                self.product_embeddings = cached_data['embeddings']
                
                # Rebuild FAISS index
                embeddings_array = np.array(self.product_embeddings)
                self.index.add(embeddings_array)
                
                print(f"Loaded {len(self.product_embeddings)} cached embeddings")
                return
            except Exception as e:
                print(f"Error loading cache: {e}, rebuilding...")
        
        # Generate embeddings for all products
        self.product_embeddings = []
        successful_embeddings = 0
        
        print(f"Processing {len(self.catalog_df)} products...")
        
        for idx, row in self.catalog_df.iterrows():
            print(f"Processing {idx+1}/{len(self.catalog_df)}: {row.get('name', 'Unknown')}")
            
            # Get image URL
            image_url = row.get('image_url', '')
            if not image_url:
                print(f"No image URL for product {idx}")
                # Use zero embedding as placeholder
                embedding = np.zeros(self.embedding_dim, dtype=np.float32)
            else:
                embedding = self.get_image_embedding(image_url)
                if embedding is None:
                    print(f"Failed to generate embedding for product {idx}")
                    embedding = np.zeros(self.embedding_dim, dtype=np.float32)
                else:
                    successful_embeddings += 1
                    print(f"Generated embedding")
            
            self.product_embeddings.append(embedding)
        
        # Add embeddings to FAISS index
        embeddings_array = np.array(self.product_embeddings)
        self.index.add(embeddings_array)
        
        # Cache embeddings
        try:
            cache_data = {
                'embeddings': self.product_embeddings,
                'catalog_size': len(self.catalog_df)
            }
            np.save(cache_path, cache_data)
            print(f"Cached embeddings to {cache_path}")
        except Exception as e:
            print(f"Failed to cache embeddings: {e}")
        
        print(f"Product index built: {successful_embeddings}/{len(self.catalog_df)} successful embeddings")
    
    def match_item_to_catalog(self, detected_item, top_k=5):
        """
        Match a detected item to products in catalog
        
        Args:
            detected_item: Dictionary with detection info and cropped image
            top_k: Number of top matches to return
        """
        print(f"Matching detected {detected_item.get('fashion_type', 'item')}...")
        
        # Get embedding for detected item
        cropped_image = detected_item.get('cropped_image')
        if cropped_image is None:
            print("No cropped image available")
            return self.create_no_match_result(detected_item)
        
        item_embedding = self.get_image_embedding(cropped_image)
        if item_embedding is None:
            print("Failed to generate embedding for detected item")
            return self.create_no_match_result(detected_item)
        
        # Search in FAISS index
        similarities, indices = self.index.search(
            item_embedding.reshape(1, -1).astype('float32'), 
            k=min(top_k, len(self.product_embeddings))
        )
        
        # Get best match
        best_match_idx = indices[0][0]
        best_similarity = float(similarities[0][0])
        
        # Determine match type
        if best_similarity >= self.thresholds['exact']:
            match_type = 'exact'
        elif best_similarity >= self.thresholds['similar']:
            match_type = 'similar'
        else:
            match_type = 'no_match'
        
        # Get matched product info
        matched_product = self.catalog_df.iloc[best_match_idx]
        
        result = {
            'type': self.map_fashion_type(detected_item.get('fashion_type', 'unknown')),
            'color': detected_item.get('estimated_color', 'unknown'),
            'match_type': match_type,
            'matched_product_id': matched_product.get('product_id', f'prod_{best_match_idx}'),
            'matched_product_name': matched_product.get('name', 'Unknown Product'),
            'matched_brand': matched_product.get('brand', 'Unknown Brand'),
            'confidence': round(best_similarity, 3),
            'bounding_box': detected_item.get('bounding_box', {}),
            'frame_number': detected_item.get('frame_number', 0),
            'timestamp': detected_item.get('timestamp', 0.0)
        }
        
        print(f"Match found: {match_type} (similarity: {best_similarity:.3f})")
        print(f"  Product: {result['matched_product_name']} by {result['matched_brand']}")
        
        return result
    
    def create_no_match_result(self, detected_item):
        """Create result for items with no match"""
        return {
            'type': self.map_fashion_type(detected_item.get('fashion_type', 'unknown')),
            'color': detected_item.get('estimated_color', 'unknown'),
            'match_type': 'no_match',
            'matched_product_id': None,
            'matched_product_name': None,
            'matched_brand': None,
            'confidence': 0.0,
            'bounding_box': detected_item.get('bounding_box', {}),
            'frame_number': detected_item.get('frame_number', 0),
            'timestamp': detected_item.get('timestamp', 0.0)
        }
    
    def map_fashion_type(self, fashion_type):
        """Map internal fashion types to output categories"""
        mapping = {
            'clothing': 'dress',
            'bag': 'bag',
            'accessory': 'accessory',
            'jewelry': 'jewelry',
            'shoes': 'shoes'
        }
        return mapping.get(fashion_type.lower(), fashion_type.lower())
    
    def batch_match(self, detections):
        """Match multiple detected items"""
        print(f"Batch matching {len(detections)} items...")
        
        matches = []
        for i, detection in enumerate(detections):
            print(f"\nMatching item {i+1}/{len(detections)}")
            match = self.match_item_to_catalog(detection)
            matches.append(match)
        
        valid_matches = [m for m in matches if m['match_type'] != 'no_match']
        
        print(f"Batch matching complete: {len(valid_matches)}/{len(detections)} successful matches")
        return valid_matches
    
    def get_match_statistics(self, matches):
        """Generate statistics for matches"""
        if not matches:
            return {}
        
        total_matches = len(matches)
        exact_matches = len([m for m in matches if m['match_type'] == 'exact'])
        similar_matches = len([m for m in matches if m['match_type'] == 'similar'])
        
        avg_confidence = np.mean([m['confidence'] for m in matches if m['confidence'] > 0])
        
        return {
            'total_matches': total_matches,
            'exact_matches': exact_matches,
            'similar_matches': similar_matches,
            'match_rate': round(total_matches / len(matches) * 100, 1) if matches else 0,
            'average_confidence': round(avg_confidence, 3) if not np.isnan(avg_confidence) else 0
        }

if __name__ == "__main__":
    mock_catalog = pd.DataFrame([
        {
            'product_id': 'SHEIN_001',
            'name': 'Black Mini Dress',
            'category': 'dress',
            'color': 'black',
            'brand': 'SHEIN',
            'price': 25.99,
            'image_url': 'https://img.ltwebstatic.com/images3_pi/2023/10/17/16/1697526842c8a4e4c5c5f5e5e5e5e5e5e5e5.jpg'
        },
        {
            'product_id': 'ZARA_045',
            'name': 'Gold Hoop Earrings',
            'category': 'earrings',
            'color': 'gold',
            'brand': 'ZARA',
            'price': 15.99,
            'image_url': 'https://static.zara.net/photos///2023/V/1/1/p/1234/567/800/2/w/1920/1234567800_1_1_1.jpg'
        }
    ])
    
    print("Testing Real Product Matcher...")
    matcher = RealProductMatcher(catalog_df=mock_catalog)
    
    mock_detection = {
        'fashion_type': 'clothing',
        'estimated_color': 'black',
        'confidence': 0.89,
        'bounding_box': {'x': 120, 'y': 80, 'w': 200, 'h': 300},
        'frame_number': 15,
        'timestamp': 0.5,
        'cropped_image': np.random.randint(0, 255, (100, 80, 3), dtype=np.uint8)  
    }
    
    result = matcher.match_item_to_catalog(mock_detection)
    print("\nMatch result:")
    print(json.dumps(result, indent=2))
    print("\nReal Product Matcher ready for production")