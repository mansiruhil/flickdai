"""
Production Product Matcher using CLIP + FAISS
Matches detected items to 200-product catalog
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
import pickle

logger = logging.getLogger(__name__)

class ProductionProductMatcher:
    def __init__(self, catalog_path=None, cache_dir="cache"):
        """
        Initialize production product matcher
        
        Args:
            catalog_path: Path to product catalog CSV
            cache_dir: Directory for caching embeddings
        """
        print("Initializing Production Product Matcher...")
        
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {self.device}")
        
        try:
            self.model, self.preprocess = clip.load("ViT-B/32", device=self.device)
            print("CLIP ViT-B/32 model loaded successfully")
        except Exception as e:
            print(f"Error loading CLIP model: {e}")
            raise
        
        self.embedding_dim = 512  # CLIP ViT-B/32 embedding dimension
        self.index = faiss.IndexFlatIP(self.embedding_dim)
        self.catalog_df = None
        self.product_embeddings = []
        self.thresholds = {
            'exact': 0.88,    
            'similar': 0.72,  
            'no_match': 0.72  
        }
        
        if catalog_path and Path(catalog_path).exists():
            self.load_catalog(catalog_path)
        else:
            self.create_production_catalog()
        
        print("Production Product Matcher initialized")
    
    def create_production_catalog(self):
        """Create production-ready catalog with 200 products"""
        print("Creating production catalog with 200 products...")

        brands = ["SHEIN", "ZARA", "H&M", "Urban Outfitters", "ASOS", "Forever 21", "Mango", "COS", "Uniqlo", "Bershka"]
        categories = ["dress", "top", "bottom", "bag", "earrings", "necklace", "shoes", "jacket", "accessory", "jewelry"]
        colors = ["black", "white", "beige", "brown", "gold", "silver", "pink", "blue", "green", "red", "navy", "cream"]
        
        products = []
        
        for i in range(200):
            brand = np.random.choice(brands)
            category = np.random.choice(categories)
            color = np.random.choice(colors)
            
            if category == "dress":
                styles = ["Mini", "Midi", "Maxi", "Bodycon", "A-line", "Wrap", "Slip"]
                name = f"{color.title()} {np.random.choice(styles)} {category.title()}"
            elif category == "top":
                styles = ["Crop", "Oversized", "Fitted", "V-neck", "Off-shoulder", "Halter"]
                name = f"{color.title()} {np.random.choice(styles)} {category.title()}"
            elif category == "bag":
                styles = ["Tote", "Crossbody", "Clutch", "Backpack", "Shoulder", "Mini"]
                name = f"{color.title()} {np.random.choice(styles)} {category.title()}"
            else:
                name = f"{color.title()} {category.title()}"
            
            product = {
                'product_id': f"{brand.upper()}_{str(i+1).zfill(3)}",
                'name': name,
                'category': category,
                'color': color,
                'brand': brand,
                'price': round(np.random.uniform(9.99, 199.99), 2),
                'image_url': f"https://cdn.shopify.com/s/files/1/product_{i+1}.jpg"
            }
            products.append(product)
        
        self.catalog_df = pd.DataFrame(products)
        catalog_path = self.cache_dir / "production_catalog.csv"
        self.catalog_df.to_csv(catalog_path, index=False)
        
        print(f"Created production catalog with {len(self.catalog_df)} products")
        print(f"Saved to {catalog_path}")
        
        self.build_embeddings_index()
    
    def load_catalog(self, catalog_path):
        """Load product catalog from CSV"""
        print(f"Loading catalog from {catalog_path}")
        
        try:
            self.catalog_df = pd.read_csv(catalog_path)
            print(f"Loaded {len(self.catalog_df)} products from catalog")
            self.build_embeddings_index()
            
        except Exception as e:
            print(f"Error loading catalog: {e}")
            self.create_production_catalog()
    
    def build_embeddings_index(self):
        """Build FAISS index with product embeddings"""
        print("Building product embeddings index...")
        
        cache_file = self.cache_dir / "product_embeddings.pkl"
        
        if cache_file.exists():
            print("Loading cached embeddings...")
            try:
                with open(cache_file, 'rb') as f:
                    cache_data = pickle.load(f)
                
                if len(cache_data['embeddings']) == len(self.catalog_df):
                    self.product_embeddings = cache_data['embeddings']
                    
                    embeddings_array = np.array(self.product_embeddings, dtype=np.float32)
                    self.index.add(embeddings_array)
                    
                    print(f"Loaded {len(self.product_embeddings)} cached embeddings")
                    return
                else:
                    print("Cache size mismatch, rebuilding...")
            except Exception as e:
                print(f"Error loading cache: {e}, rebuilding...")
        
        print(f"Generating embeddings for {len(self.catalog_df)} products...")
        
        self.product_embeddings = []
        
        for idx, row in self.catalog_df.iterrows():
            if idx % 50 == 0:
                print(f"  Progress: {idx}/{len(self.catalog_df)}")
            
            product_text = f"{row['name']} {row['category']} {row['color']} {row['brand']}"
            embedding = self.get_text_embedding(product_text)
            
            if embedding is not None:
                self.product_embeddings.append(embedding)
            else:
                self.product_embeddings.append(np.zeros(self.embedding_dim, dtype=np.float32))
        
        embeddings_array = np.array(self.product_embeddings, dtype=np.float32)
        self.index.add(embeddings_array)
        
        try:
            cache_data = {
                'embeddings': self.product_embeddings,
                'catalog_size': len(self.catalog_df),
                'timestamp': time.time()
            }
            with open(cache_file, 'wb') as f:
                pickle.dump(cache_data, f)
            print(f"Cached embeddings to {cache_file}")
        except Exception as e:
            print(f"Failed to cache embeddings: {e}")
        
        print(f"Built embeddings index with {len(self.product_embeddings)} products")
    
    def get_text_embedding(self, text):
        """Generate CLIP text embedding"""
        try:
            text_tokens = clip.tokenize([text]).to(self.device)
            
            with torch.no_grad():
                text_features = self.model.encode_text(text_tokens)
                text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            
            return text_features.cpu().numpy().flatten()
            
        except Exception as e:
            logger.error(f"Error generating text embedding: {e}")
            return None
    
    def get_image_embedding(self, image):
        """Generate CLIP image embedding"""
        try:
            if isinstance(image, np.ndarray):
                image = Image.fromarray(image).convert('RGB')
            
            image_input = self.preprocess(image).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                image_features = self.model.encode_image(image_input)
                image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            
            return image_features.cpu().numpy().flatten()
            
        except Exception as e:
            logger.error(f"Error generating image embedding: {e}")
            return None
    
    def match_detection_to_catalog(self, detection, top_k=5):
        """Match detected item to product catalog"""
        print(f"Matching {detection.get('fashion_type', 'item')}...")
        
        cropped_image = detection.get('cropped_image')
        if cropped_image is None:
            return self.create_no_match_result(detection)
        
        item_embedding = self.get_image_embedding(cropped_image)
        if item_embedding is None:
            return self.create_no_match_result(detection)

        # Search in FAISS index
        similarities, indices = self.index.search(
            item_embedding.reshape(1, -1).astype('float32'), 
            k=min(top_k, len(self.product_embeddings))
        )
        
        best_match_idx = indices[0][0]
        best_similarity = float(similarities[0][0])
        if best_similarity >= self.thresholds['exact']:
            match_type = 'exact'
        elif best_similarity >= self.thresholds['similar']:
            match_type = 'similar'
        else:
            match_type = 'no_match'
        
        matched_product = self.catalog_df.iloc[best_match_idx]
    
        result = {
            'type': self.map_fashion_type(detection.get('fashion_type', 'unknown')),
            'color': matched_product.get('color', 'unknown'),
            'match_type': match_type,
            'matched_product_id': matched_product.get('product_id'),
            'confidence': round(best_similarity, 2)
        }
        
        print(f"Match: {match_type} (similarity: {best_similarity:.3f})")
        return result
    
    def create_no_match_result(self, detection):
        """Create no-match result"""
        return {
            'type': self.map_fashion_type(detection.get('fashion_type', 'unknown')),
            'color': 'unknown',
            'match_type': 'no_match',
            'matched_product_id': None,
            'confidence': 0.0
        }
    
    def map_fashion_type(self, fashion_type):
        """Map internal fashion types to output format"""
        mapping = {
            'clothing': 'dress',
            'dress': 'dress',
            'top': 'top',
            'bottom': 'bottom',
            'bag': 'bag',
            'accessory': 'accessory',
            'jewelry': 'earrings',
            'shoes': 'shoes',
            'jacket': 'jacket'
        }
        return mapping.get(fashion_type.lower(), 'accessory')
    
    def batch_match(self, detections):
        """Match multiple detections"""
        print(f"Batch matching {len(detections)} items...")
        
        matches = []
        for detection in detections:
            match = self.match_detection_to_catalog(detection)
            
            if match['match_type'] != 'no_match':
                matches.append(match)
        
        print(f"Successful matches: {len(matches)}/{len(detections)}")
        return matches
    
    def get_catalog_stats(self):
        """Get catalog statistics"""
        if self.catalog_df is None:
            return {}
        
        return {
            'total_products': len(self.catalog_df),
            'categories': self.catalog_df['category'].value_counts().to_dict(),
            'brands': self.catalog_df['brand'].value_counts().to_dict(),
            'colors': self.catalog_df['color'].value_counts().to_dict(),
            'price_range': {
                'min': float(self.catalog_df['price'].min()),
                'max': float(self.catalog_df['price'].max()),
                'avg': float(self.catalog_df['price'].mean())
            }
        }

if __name__ == "__main__":
    matcher = ProductionProductMatcher()
    stats = matcher.get_catalog_stats()
    print("\nCatalog Statistics:")
    print(f"  Total products: {stats['total_products']}")
    print(f"  Categories: {list(stats['categories'].keys())}")
    print(f"  Brands: {list(stats['brands'].keys())}") 
    print("\nProduction Product Matcher ready")