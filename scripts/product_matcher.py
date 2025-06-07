"""
Product matching module using CLIP embeddings and FAISS
Matches detected fashion items to product catalog
"""

import numpy as np
import pandas as pd
from PIL import Image
import torch
import clip
import faiss
import json
from pathlib import Path
import requests
from io import BytesIO

class ProductMatcher:
    def __init__(self, catalog_path=None):
        print("Initializing Product Matcher...")
        
        # Load CLIP model
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model, self.preprocess = clip.load("ViT-B/32", device=self.device)
        
        # Initialize FAISS index
        self.embedding_dim = 512  
        self.index = faiss.IndexFlatIP(self.embedding_dim)  
        
        # Product catalog
        self.products = []
        self.product_embeddings = []
        
        if catalog_path:
            self.load_catalog(catalog_path)
        else:
            self.create_mock_catalog()
        
        print("Product Matcher initialized")
    
    def create_mock_catalog(self):
        """Create a mock product catalog for demonstration"""
        print("Creating mock product catalog...")
        
        mock_products = [
            {
                'product_id': 'prod_001',
                'name': 'Black Mini Dress',
                'category': 'dress',
                'color': 'black',
                'image_url': 'https://example.com/dress1.jpg'
            },
            {
                'product_id': 'prod_002',
                'name': 'Gold Hoop Earrings',
                'category': 'earrings',
                'color': 'gold',
                'image_url': 'https://example.com/earrings1.jpg'
            },
            {
                'product_id': 'prod_003',
                'name': 'Leather Handbag',
                'category': 'bag',
                'color': 'brown',
                'image_url': 'https://example.com/bag1.jpg'
            },
            {
                'product_id': 'prod_004',
                'name': 'White Crop Top',
                'category': 'top',
                'color': 'white',
                'image_url': 'https://example.com/top1.jpg'
            },
            {
                'product_id': 'prod_005',
                'name': 'High Waisted Jeans',
                'category': 'bottom',
                'color': 'blue',
                'image_url': 'https://example.com/jeans1.jpg'
            }
        ]
        
        self.products = mock_products
        
        # Create mock embeddings (in real implementation, these would be generated from actual images)
        for product in self.products:
            # Generate random embedding for demonstration
            mock_embedding = np.random.randn(self.embedding_dim).astype('float32')
            mock_embedding = mock_embedding / np.linalg.norm(mock_embedding) 
            self.product_embeddings.append(mock_embedding)
        
        # Add embeddings to FAISS index
        embeddings_array = np.array(self.product_embeddings)
        self.index.add(embeddings_array)
        
        print(f"Mock catalog created with {len(self.products)} products")
    
    def load_catalog(self, catalog_path):
        """Load product catalog from CSV file"""
        print(f"Loading catalog from {catalog_path}")
        
        df = pd.read_csv(catalog_path)
        self.products = df.to_dict('records')
        
        # Generate embeddings for each product image
        for product in self.products:
            embedding = self.get_image_embedding(product['image_url'])
            self.product_embeddings.append(embedding)
        
        # Add to FAISS index
        embeddings_array = np.array(self.product_embeddings)
        self.index.add(embeddings_array)
        
        print(f"Catalog loaded with {len(self.products)} products")
    
    def get_image_embedding(self, image_url_or_array):
        """Generate CLIP embedding for an image"""
        try:
            if isinstance(image_url_or_array, str):
                # Load image from URL
                response = requests.get(image_url_or_array)
                image = Image.open(BytesIO(response.content))
            else:
                # Convert numpy array to PIL Image
                image = Image.fromarray(image_url_or_array)
            
            # Preprocess and encode
            image_input = self.preprocess(image).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                image_features = self.model.encode_image(image_input)
                image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            
            return image_features.cpu().numpy().flatten()
        
        except Exception as e:
            print(f"Error processing image: {e}")
            # Return random embedding as fallback
            return np.random.randn(self.embedding_dim).astype('float32')
    
    def match_product(self, detected_item):
        """Match a detected item to products in catalog"""
        print(f"Matching detected {detected_item.get('class_name', 'item')}...")
        
        # Get embedding for detected item
        if 'cropped_image' in detected_item:
            item_embedding = self.get_image_embedding(detected_item['cropped_image'])
        else:
            # Use mock embedding for demonstration
            item_embedding = np.random.randn(self.embedding_dim).astype('float32')
            item_embedding = item_embedding / np.linalg.norm(item_embedding)
        
        # Search in FAISS index
        similarities, indices = self.index.search(
            item_embedding.reshape(1, -1), 
            k=3  
        )
        
        best_match_idx = indices[0][0]
        best_similarity = similarities[0][0]
        
        # Determine match type based on similarity threshold
        if best_similarity > 0.9:
            match_type = 'exact'
        elif best_similarity > 0.75:
            match_type = 'similar'
        else:
            match_type = 'no_match'
        
        matched_product = self.products[best_match_idx]
        
        result = {
            'type': self.map_class_to_type(detected_item.get('class_name', 'unknown')),
            'color': matched_product.get('color', 'unknown'),
            'match_type': match_type,
            'matched_product_id': matched_product['product_id'],
            'confidence': float(best_similarity),
            'bounding_box': detected_item.get('bounding_box', {})
        }
        
        print(f"Match found: {match_type} ({best_similarity:.3f} similarity)")
        return result
    
    def map_class_to_type(self, class_name):
        """Map YOLO class names to product types"""
        mapping = {
            'person': 'clothing',
            'handbag': 'bag',
            'tie': 'accessory',
            'suitcase': 'bag',
            'backpack': 'bag'
        }
        return mapping.get(class_name.lower(), class_name.lower())
    
    def batch_match(self, detections):
        """Match multiple detected items"""
        print(f"Batch matching {len(detections)} items...")
        
        matches = []
        for detection in detections:
            match = self.match_product(detection)
            matches.append(match)
        
        print(f"Batch matching complete")
        return matches

if __name__ == "__main__":
    matcher = ProductMatcher()
    mock_detection = {
        'class_name': 'person',
        'confidence': 0.89,
        'bounding_box': {'x': 120, 'y': 80, 'w': 200, 'h': 300}
    }
    result = matcher.match_product(mock_detection)
    print("Match result:")
    print(json.dumps(result, indent=2))