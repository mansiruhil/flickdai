"""
Dataset Manager for Flickd AI Engine
Manages the 10 sample videos and 200 product catalog
"""

import pandas as pd
import json
import os
from pathlib import Path
import shutil
import requests
from urllib.parse import urlparse
import logging

logger = logging.getLogger(__name__)

class DatasetManager:
    def __init__(self, data_dir="data"):
        """Initialize dataset manager"""
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True)

        self.videos_dir = self.data_dir / "videos"
        self.frames_dir = self.data_dir / "frames"
        self.models_dir = self.data_dir / "models"
        
        for dir_path in [self.videos_dir, self.frames_dir, self.models_dir]:
            dir_path.mkdir(exist_ok=True)
        
        print(f"Dataset manager initialized at {self.data_dir}")
    
    def setup_sample_videos(self):
        """Setup 10 sample creator videos"""
        print("Setting up sample creator videos...")
        
        sample_videos = [
            {
                "video_id": "video_001",
                "filename": "coquette_outfit.mp4",
                "duration": 8.5,
                "description": "Soft pink dress with bow accessories",
                "creator": "@fashionista_emma",
                "hashtags": ["#coquette", "#pink", "#feminine", "#bow"],
                "expected_vibes": ["Coquette"]
            },
            {
                "video_id": "video_002", 
                "filename": "clean_girl_routine.mp4",
                "duration": 12.3,
                "description": "Minimal makeup and slicked hair",
                "creator": "@minimal_maya",
                "hashtags": ["#cleangirl", "#natural", "#minimal"],
                "expected_vibes": ["Clean Girl"]
            },
            {
                "video_id": "video_003",
                "filename": "party_glam_look.mp4",
                "duration": 15.0,
                "description": "Sparkly dress for night out",
                "creator": "@glam_goddess",
                "hashtags": ["#glam", "#party", "#sparkle", "#nightout"],
                "expected_vibes": ["Party Glam"]
            },
            {
                "video_id": "video_004",
                "filename": "streetwear_fit.mp4",
                "duration": 9.8,
                "description": "Urban street style with oversized jacket",
                "creator": "@street_style_sam",
                "hashtags": ["#streetcore", "#urban", "#oversized"],
                "expected_vibes": ["Streetcore"]
            },
            {
                "video_id": "video_005",
                "filename": "cottagecore_dress.mp4",
                "duration": 11.2,
                "description": "Floral vintage dress in garden setting",
                "creator": "@cottage_claire",
                "hashtags": ["#cottagecore", "#vintage", "#floral"],
                "expected_vibes": ["Cottagecore"]
            },
            {
                "video_id": "video_006",
                "filename": "y2k_metallic.mp4",
                "duration": 7.9,
                "description": "Metallic top with futuristic accessories",
                "creator": "@cyber_chic",
                "hashtags": ["#y2k", "#metallic", "#futuristic"],
                "expected_vibes": ["Y2K"]
            },
            {
                "video_id": "video_007",
                "filename": "boho_festival.mp4",
                "duration": 13.5,
                "description": "Flowing bohemian outfit with fringe",
                "creator": "@boho_bella",
                "hashtags": ["#boho", "#festival", "#freespirit"],
                "expected_vibes": ["Boho"]
            },
            {
                "video_id": "video_008",
                "filename": "mixed_aesthetic.mp4",
                "duration": 10.1,
                "description": "Coquette meets party glam",
                "creator": "@style_mixer",
                "hashtags": ["#coquette", "#glam", "#pink", "#sparkle"],
                "expected_vibes": ["Coquette", "Party Glam"]
            },
            {
                "video_id": "video_009",
                "filename": "casual_chic.mp4",
                "duration": 6.7,
                "description": "Effortless everyday style",
                "creator": "@everyday_emma",
                "hashtags": ["#casual", "#chic", "#effortless"],
                "expected_vibes": ["Clean Girl"]
            },
            {
                "video_id": "video_010",
                "filename": "grunge_street.mp4",
                "duration": 14.2,
                "description": "Grunge streetwear with distressed denim",
                "creator": "@grunge_girl",
                "hashtags": ["#grunge", "#streetcore", "#distressed"],
                "expected_vibes": ["Streetcore"]
            }
        ]
        
        videos_metadata_path = self.data_dir / "videos_metadata.json"
        with open(videos_metadata_path, 'w') as f:
            json.dump(sample_videos, f, indent=2)
        
        print(f"Created metadata for {len(sample_videos)} sample videos")
        print(f"Saved to {videos_metadata_path}")
        
        return sample_videos
    
    def create_product_catalog(self):
        """Create 200-product catalog CSV"""
        print("Creating 200-product catalog...")
        
        brands = [
            "SHEIN", "ZARA", "H&M", "Urban Outfitters", "ASOS", "Forever 21", 
            "Mango", "COS", "Uniqlo", "Bershka", "Pull & Bear", "Stradivarius",
            "Massimo Dutti", "& Other Stories", "Monki", "Weekday"
        ]
        
        categories = [
            "dress", "top", "bottom", "bag", "earrings", "necklace", 
            "shoes", "jacket", "accessory", "jewelry", "hat", "sunglasses"
        ]
        
        colors = [
            "black", "white", "beige", "brown", "gold", "silver", "pink", 
            "blue", "green", "red", "navy", "cream", "grey", "purple", "orange"
        ]
        
        products = []
        
        for i in range(200):
            brand = np.random.choice(brands)
            category = np.random.choice(categories)
            color = np.random.choice(colors)
            
            product_name = self.generate_product_name(category, color)
            
            image_url = f"https://cdn.shopify.com/s/files/1/{brand.lower()}/products/{category}_{color}_{i+1:03d}.jpg"
            
            product = {
                'Product ID': f"{brand.upper()}_{i+1:03d}",
                'Product Name': product_name,
                'Category': category,
                'Color': color,
                'Brand': brand,
                'Price': round(np.random.uniform(9.99, 299.99), 2),
                'Shopify Image URL': image_url,
                'Description': f"{color.title()} {category} from {brand}",
                'In Stock': np.random.choice([True, False], p=[0.8, 0.2]),
                'Rating': round(np.random.uniform(3.5, 5.0), 1)
            }
            
            products.append(product)
        
        catalog_df = pd.DataFrame(products)
        catalog_path = self.data_dir / "product_catalog.csv"
        catalog_df.to_csv(catalog_path, index=False)
        
        print(f"Created catalog with {len(products)} products")
        print(f"Saved to {catalog_path}")
        
        print("\nSample products:")
        for _, product in catalog_df.head(5).iterrows():
            print(f"  - {product['Product Name']} ({product['Brand']}) - ${product['Price']}")
        
        return catalog_df
    
    def generate_product_name(self, category, color):
        """Generate realistic product names"""
        import numpy as np
        
        if category == "dress":
            styles = ["Mini", "Midi", "Maxi", "Bodycon", "A-line", "Wrap", "Slip", "Shirt", "Sweater"]
            return f"{color.title()} {np.random.choice(styles)} Dress"
        elif category == "top":
            styles = ["Crop", "Oversized", "Fitted", "V-neck", "Off-shoulder", "Halter", "Tank", "Blouse"]
            return f"{color.title()} {np.random.choice(styles)} Top"
        elif category == "bottom":
            styles = ["High-waisted", "Skinny", "Wide-leg", "Straight", "Flare", "Cargo", "Mom"]
            types = ["Jeans", "Pants", "Shorts", "Skirt"]
            return f"{color.title()} {np.random.choice(styles)} {np.random.choice(types)}"
        elif category == "bag":
            styles = ["Tote", "Crossbody", "Clutch", "Backpack", "Shoulder", "Mini", "Bucket"]
            return f"{color.title()} {np.random.choice(styles)} Bag"
        elif category == "shoes":
            styles = ["Platform", "Heeled", "Flat", "Ankle", "Knee-high", "Chunky", "Strappy"]
            types = ["Boots", "Sandals", "Sneakers", "Pumps", "Loafers"]
            return f"{color.title()} {np.random.choice(styles)} {np.random.choice(types)}"
        elif category == "jacket":
            styles = ["Denim", "Leather", "Bomber", "Blazer", "Puffer", "Trench", "Cropped"]
            return f"{color.title()} {np.random.choice(styles)} Jacket"
        elif category in ["earrings", "necklace", "jewelry"]:
            styles = ["Hoop", "Stud", "Drop", "Chain", "Statement", "Delicate", "Chunky"]
            return f"{color.title()} {np.random.choice(styles)} {category.title()}"
        else:
            return f"{color.title()} {category.title()}"
    
    def create_vibes_taxonomy(self):
        """Create vibes taxonomy JSON"""
        print("Creating vibes taxonomy...")
        
        vibes_list = [
            "Coquette",
            "Clean Girl", 
            "Cottagecore",
            "Streetcore",
            "Y2K",
            "Boho",
            "Party Glam"
        ]
        
        vibes_path = self.data_dir / "vibeslist.json"
        with open(vibes_path, 'w') as f:
            json.dump(vibes_list, f, indent=2)
        
        print(f"Created vibes taxonomy with {len(vibes_list)} vibes")
        print(f"Saved to {vibes_path}")
        
        return vibes_list
    
    def setup_complete_dataset(self):
        """Setup complete dataset with all components"""
        print("Setting up complete Flickd dataset...")
        
        videos = self.setup_sample_videos()
        catalog = self.create_product_catalog()
        vibes = self.create_vibes_taxonomy()
        
        summary = {
            "dataset_info": {
                "name": "Flickd AI Hackathon Dataset",
                "version": "1.0",
                "created": "2025-01-06",
                "description": "Complete dataset for fashion video analysis"
            },
            "components": {
                "videos": {
                    "count": len(videos),
                    "total_duration": sum(v['duration'] for v in videos),
                    "formats": ["mp4"],
                    "resolution": "1080p",
                    "fps": 30
                },
                "products": {
                    "count": len(catalog),
                    "categories": catalog['Category'].nunique(),
                    "brands": catalog['Brand'].nunique(),
                    "price_range": {
                        "min": float(catalog['Price'].min()),
                        "max": float(catalog['Price'].max())
                    }
                },
                "vibes": {
                    "count": len(vibes),
                    "supported": vibes
                }
            },
            "file_structure": {
                "videos/": "Sample creator videos (10 files)",
                "frames/": "Extracted video frames",
                "models/": "ML model files and cache",
                "product_catalog.csv": "200 product entries",
                "vibeslist.json": "Supported vibe taxonomy",
                "videos_metadata.json": "Video metadata and descriptions"
            }
        }
        summary_path = self.data_dir / "dataset_summary.json"
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"\nComplete dataset setup finished")
        print(f"Dataset summary:")
        print(f" - {len(videos)} sample videos")
        print(f" - {len(catalog)} products in catalog")
        print(f" - {len(vibes)} supported vibes")
        print(f"Summary saved to {summary_path}")
        
        return summary
    
    def validate_dataset(self):
        """Validate dataset integrity"""
        print("Validating dataset...")
        
        issues = []
        
        required_files = [
            "product_catalog.csv",
            "vibeslist.json", 
            "videos_metadata.json",
            "dataset_summary.json"
        ]
        
        for file_name in required_files:
            file_path = self.data_dir / file_name
            if not file_path.exists():
                issues.append(f"Missing file: {file_name}")
        
        required_dirs = ["videos", "frames", "models"]
        for dir_name in required_dirs:
            dir_path = self.data_dir / dir_name
            if not dir_path.exists():
                issues.append(f"Missing directory: {dir_name}")
        try:
            catalog_path = self.data_dir / "product_catalog.csv"
            if catalog_path.exists():
                catalog_df = pd.read_csv(catalog_path)
                required_columns = ['Product ID', 'Product Name', 'Shopify Image URL']
                missing_cols = [col for col in required_columns if col not in catalog_df.columns]
                if missing_cols:
                    issues.append(f"Missing catalog columns: {missing_cols}")
                
                if len(catalog_df) != 200:
                    issues.append(f"Expected 200 products, found {len(catalog_df)}")
        except Exception as e:
            issues.append(f"Error reading catalog: {e}")
        
        if issues:
            print("Dataset validation issues:")
            for issue in issues:
                print(f"  {issue}")
            return False
        else:
            print("Dataset validation passed")
            return True

if __name__ == "__main__":
    import numpy as np
    manager = DatasetManager()
    summary = manager.setup_complete_dataset()
    is_valid = manager.validate_dataset()
    if is_valid:
        print("\nDataset ready for Flickd AI Engine")
        print("Files created in ./data/ directory")
        print("Ready to process videos and match products")
    else:
        print("\nDataset validation failed. Please check the issues above")