"""
Dataset loader for Flickd AI Engine
Loads and processes the Google Drive dataset files
"""

import pandas as pd
import json
import os
from pathlib import Path
import requests
from PIL import Image
from io import BytesIO

class FlickdDatasetLoader:
    def __init__(self, data_dir="data"):
        self.data_dir = Path(data_dir)
        self.products_df = None
        self.vibes_list = None
        self.video_files = []
        
        print("Initializing Flickd Dataset Loader...")
        self.load_all_data()
    
    def load_product_catalog(self):
        """Load product catalog from Excel/CSV file"""
        print("Loading product catalog...")
        
        # Try to load from Excel first then CSV
        product_file_xlsx = self.data_dir / "product_data.xlsx"
        product_file_csv = self.data_dir / "images.csv"
        
        if product_file_xlsx.exists():
            print(f"Loading from {product_file_xlsx}")
            self.products_df = pd.read_excel(product_file_xlsx)
        elif product_file_csv.exists():
            print(f"Loading from {product_file_csv}")
            self.products_df = pd.read_csv(product_file_csv)
        else:
            print("No product catalog found, creating mock data...")
            self.create_mock_product_catalog()
            return
        
        print(f"Loaded {len(self.products_df)} products from catalog")
        
        print("\nSample products:")
        for idx, row in self.products_df.head(3).iterrows():
            print(f"  - {row.get('name', 'Unknown')}: {row.get('category', 'N/A')}")
    
    def create_mock_product_catalog(self):
        """Create mock product catalog based on typical fashion items"""
        print("Creating mock product catalog...")
        
        mock_products = [
            {"product_id": "SHEIN_001", "name": "Black Mini Dress", "category": "dress", "color": "black", "brand": "SHEIN", "price": 25.99, "image_url": "https://img.ltwebstatic.com/images3_pi/2023/10/17/16/1697526842c8a4e4c5c5f5e5e5e5e5e5e5e5.jpg"},
            {"product_id": "ZARA_045", "name": "Gold Hoop Earrings", "category": "earrings", "color": "gold", "brand": "ZARA", "price": 15.99, "image_url": "https://static.zara.net/photos///2023/V/1/1/p/1234/567/800/2/w/1920/1234567800_1_1_1.jpg"},
            {"product_id": "H&M_123", "name": "White Crop Top", "category": "top", "color": "white", "brand": "H&M", "price": 12.99, "image_url": "https://lp2.hm.com/hmgoepprod?set=quality%5B79%5D%2Csource%5B%2F12%2F34%2F1234567890abcdef.jpg%5D"},
            {"product_id": "URBAN_789", "name": "High Waisted Jeans", "category": "bottom", "color": "blue", "brand": "Urban Outfitters", "price": 89.99, "image_url": "https://images.urbndata.com/is/image/UrbanOutfitters/12345678_040_b"},
            {"product_id": "ASOS_456", "name": "Leather Handbag", "category": "bag", "color": "brown", "brand": "ASOS", "price": 45.99, "image_url": "https://images.asos-media.com/products/asos-design-structured-handbag/12345678-1-brown"},
            {"product_id": "SHEIN_002", "name": "Pink Bow Hair Clip", "category": "accessory", "color": "pink", "brand": "SHEIN", "price": 3.99, "image_url": "https://img.ltwebstatic.com/images3_pi/2023/09/15/12/1694765432abc123def456.jpg"},
            {"product_id": "ZARA_046", "name": "Metallic Silver Top", "category": "top", "color": "silver", "brand": "ZARA", "price": 35.99, "image_url": "https://static.zara.net/photos///2023/V/1/1/p/5678/901/040/2/w/1920/5678901040_1_1_1.jpg"},
            {"product_id": "H&M_124", "name": "Floral Midi Dress", "category": "dress", "color": "floral", "brand": "H&M", "price": 29.99, "image_url": "https://lp2.hm.com/hmgoepprod?set=quality%5B79%5D%2Csource%5B%2F56%2F78%2F5678901234abcdef.jpg%5D"},
            {"product_id": "URBAN_790", "name": "Chunky Gold Chain", "category": "jewelry", "color": "gold", "brand": "Urban Outfitters", "price": 25.99, "image_url": "https://images.urbndata.com/is/image/UrbanOutfitters/87654321_040_b"},
            {"product_id": "ASOS_457", "name": "Platform Sneakers", "category": "shoes", "color": "white", "brand": "ASOS", "price": 65.99, "image_url": "https://images.asos-media.com/products/asos-design-platform-sneakers/87654321-1-white"}
        ]
        
        self.products_df = pd.DataFrame(mock_products)
        print(f"Created mock catalog with {len(self.products_df)} products")
    
    def load_vibes_taxonomy(self):
        """Load supported vibes from JSON file"""
        print("Loading vibes taxonomy...")
        
        vibes_file = self.data_dir / "vibeslist.json"
        
        if vibes_file.exists():
            with open(vibes_file, 'r') as f:
                self.vibes_list = json.load(f)
            print(f"Loaded {len(self.vibes_list)} supported vibes")
        else:
            print("Vibes file not found, using default taxonomy...")
            self.vibes_list = [
                "Coquette", "Clean Girl", "Cottagecore", 
                "Streetcore", "Y2K", "Boho", "Party Glam"
            ]
        
        print(f"Supported vibes: {', '.join(self.vibes_list)}")
    
    def scan_video_files(self):
        """Scan for video files in the dataset"""
        print("Scanning for video files...")
        
        video_dir = self.data_dir / "videos"
        if not video_dir.exists():
            print("Videos directory not found")
            return
        
        video_extensions = ['.mp4', '.mov', '.avi', '.mkv', '.webm']
        
        for ext in video_extensions:
            videos = list(video_dir.glob(f"*{ext}"))
            self.video_files.extend(videos)
        
        print(f"Found {len(self.video_files)} video files")
        
        if self.video_files:
            print("\n🎥 Sample videos:")
            for video in self.video_files[:3]:
                file_size = video.stat().st_size / (1024 * 1024)  # MB
                print(f"  - {video.name}: {file_size:.1f} MB")
    
    def load_all_data(self):
        """Load all dataset components"""
        try:
            self.load_product_catalog()
            self.load_vibes_taxonomy()
            self.scan_video_files()
            print("\nDataset loading complete")
            
        except Exception as e:
            print(f"Error loading dataset: {e}")
    
    def get_product_by_id(self, product_id):
        """Get product details by ID"""
        if self.products_df is None:
            return None
        
        product = self.products_df[self.products_df['product_id'] == product_id]
        if not product.empty:
            return product.iloc[0].to_dict()
        return None
    
    def search_products_by_category(self, category):
        """Search products by category"""
        if self.products_df is None:
            return []
        
        matches = self.products_df[self.products_df['category'].str.lower() == category.lower()]
        return matches.to_dict('records')
    
    def get_random_products(self, n=5):
        """Get random products for testing"""
        if self.products_df is None:
            return []
        
        return self.products_df.sample(min(n, len(self.products_df))).to_dict('records')
    
    def validate_dataset(self):
        """Validate dataset integrity"""
        print("Validating dataset...")
        
        issues = []
        
        if self.products_df is None or self.products_df.empty:
            issues.append("Product catalog is empty")
        else:
            required_cols = ['product_id', 'name', 'category']
            missing_cols = [col for col in required_cols if col not in self.products_df.columns]
            if missing_cols:
                issues.append(f"Missing columns in product catalog: {missing_cols}")
        
        if not self.vibes_list:
            issues.append("No vibes taxonomy loaded")
        
        if not self.video_files:
            issues.append("No video files found")
        
        if issues:
            print("Dataset validation issues:")
            for issue in issues:
                print(f"  {issue}")
        else:
            print("Dataset validation passed!")
        
        return len(issues) == 0

if __name__ == "__main__":

    loader = FlickdDatasetLoader()

    is_valid = loader.validate_dataset()
    
    if is_valid:
        print("\nDataset Statistics:")
        print(f"  Products: {len(loader.products_df) if loader.products_df is not None else 0}")
        print(f"  Vibes: {len(loader.vibes_list) if loader.vibes_list else 0}")
        print(f"  Videos: {len(loader.video_files)}")

        print("\nSample Products:")
        sample_products = loader.get_random_products(3)
        for product in sample_products:
            print(f"  - {product['name']} ({product['category']}) - {product['brand']}")
        
        if loader.products_df is not None:
            print("\nCategory Distribution:")
            category_counts = loader.products_df['category'].value_counts()
            for category, count in category_counts.head().items():
                print(f"  - {category}: {count} items")
    
    print("\nDataset loader ready for use")