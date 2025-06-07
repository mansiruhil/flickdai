"""
FastAPI server for Flickd AI Engine
Provides REST API endpoints for video processing
"""

from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import json
import tempfile
import os
from pathlib import Path
import uuid
from typing import List, Optional

class MockFlickdEngine:
    def process_video(self, video_path, caption="", hashtags=None):
        """Mock video processing"""
        import time
        import random
        
        time.sleep(2)
        
        vibes = ['Coquette', 'Party Glam'] if 'party' in caption.lower() else ['Clean Girl']
        
        products = [
            {
                'type': 'dress',
                'color': 'black',
                'match_type': 'similar',
                'matched_product_id': f'prod_{random.randint(100, 999)}',
                'confidence': round(random.uniform(0.7, 0.95), 2),
                'bounding_box': {
                    'x': random.randint(50, 150),
                    'y': random.randint(50, 100),
                    'w': random.randint(150, 250),
                    'h': random.randint(200, 350)
                }
            }
        ]
        
        return {
            'video_id': f'video_{uuid.uuid4().hex[:8]}',
            'vibes': vibes,
            'products': products,
            'processing_time': round(random.uniform(1.5, 3.5), 2)
        }

app = FastAPI(
    title="Flickd AI Engine API",
    description="Smart Tagging & Vibe Classification Engine for Fashion Videos",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

engine = MockFlickdEngine()

@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "message": "Flickd AI Engine API",
        "status": "running",
        "version": "1.0.0"
    }

@app.post("/process-video")
async def process_video(
    video: UploadFile = File(...),
    caption: Optional[str] = Form(""),
    hashtags: Optional[str] = Form("")
):
    """
    Process a video file for fashion item detection and vibe classification
    
    Args:
        video: Video file (MP4, MOV, AVI)
        caption: Optional video caption
        hashtags: Optional hashtags (comma-separated)
    
    Returns:
        JSON with detected items, matched products, and classified vibes
    """
    
    if not video.content_type.startswith('video/'):
        raise HTTPException(status_code=400, detail="File must be a video")
    
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as temp_file:
            content = await video.read()
            temp_file.write(content)
            temp_path = temp_file.name
        
        hashtag_list = []
        if hashtags:
            hashtag_list = [tag.strip() for tag in hashtags.split(',')]
        
        result = engine.process_video(
            video_path=temp_path,
            caption=caption,
            hashtags=hashtag_list
        )
        os.unlink(temp_path)
        
        return JSONResponse(content=result)
        
    except Exception as e:
        if 'temp_path' in locals():
            try:
                os.unlink(temp_path)
            except:
                pass
        
        raise HTTPException(status_code=500, detail=f"Processing error: {str(e)}")

@app.post("/classify-vibe")
async def classify_vibe(
    text: str = Form(...),
    max_vibes: Optional[int] = Form(3)
):
    """
    Classify vibes from text/caption only
    
    Args:
        text: Caption or description text
        max_vibes: Maximum number of vibes to return
    
    Returns:
        JSON with classified vibes
    """
    
    vibes = []
    text_lower = text.lower()
    
    if any(word in text_lower for word in ['glam', 'party', 'sparkle', 'night']):
        vibes.append('Party Glam')
    if any(word in text_lower for word in ['bow', 'pink', 'feminine', 'cute']):
        vibes.append('Coquette')
    if any(word in text_lower for word in ['minimal', 'clean', 'natural']):
        vibes.append('Clean Girl')
    if any(word in text_lower for word in ['street', 'urban', 'edgy']):
        vibes.append('Streetcore')
    if any(word in text_lower for word in ['cottage', 'floral', 'vintage']):
        vibes.append('Cottagecore')
    if any(word in text_lower for word in ['y2k', '2000s', 'metallic']):
        vibes.append('Y2K')
    if any(word in text_lower for word in ['boho', 'bohemian', 'free']):
        vibes.append('Boho')
    
    if not vibes:
        vibes = ['Clean Girl']
    
    return {
        'text': text,
        'vibes': vibes[:max_vibes],
        'confidence': 0.85
    }

@app.get("/supported-vibes")
async def get_supported_vibes():
    """Get list of supported fashion vibes"""
    return {
        'vibes': [
            'Coquette',
            'Clean Girl', 
            'Cottagecore',
            'Streetcore',
            'Y2K',
            'Boho',
            'Party Glam'
        ],
        'total': 7
    }

@app.get("/health")
async def health_check():
    """Detailed health check"""
    return {
        'status': 'healthy',
        'components': {
            'video_processor': 'ready',
            'product_matcher': 'ready',
            'vibe_classifier': 'ready'
        },
        'timestamp': '2025-01-06T12:20:06Z'
    }

if __name__ == "__main__":
    import uvicorn
    
    print("Starting Flickd AI Engine API server...")
    print("API Documentation: http://localhost:8000/docs")
    print("Interactive API: http://localhost:8000/redoc")
    
    uvicorn.run(
        "api_server:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )
