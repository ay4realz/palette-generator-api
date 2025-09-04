"""
Palette Generator API - Main Application
Deployed on Railway
"""
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
import tensorflow as tf
import joblib
import numpy as np
import json
import os
import logging
from datetime import datetime
import asyncio
from utils.color_utils import (
    hex_to_rgb, rgb_to_hex, rgb_to_lab, lab_to_rgb, 
    generate_traditional_palette, calculate_contrast_ratio
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="AI Palette Generator API",
    description="Advanced color palette generation using AI and traditional methods",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure this properly for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global model storage
models = {}
model_metadata = {}

# Pydantic models for request validation
class PaletteRequest(BaseModel):
    method: str = Field(default="gmm", description="Generation method: 'gmm', 'deep', or traditional methods")
    num_colors: int = Field(default=5, ge=3, le=8, description="Number of colors to generate")
    seed: Optional[int] = Field(default=None, description="Random seed for reproducibility")
    base_color: Optional[str] = Field(default=None, description="Base color for traditional methods (hex)")
    style: Optional[str] = Field(default="balanced", description="Style preference: balanced, vibrant, muted, pastel")

class ContrastCheckRequest(BaseModel):
    colors: List[str] = Field(description="List of hex colors to check")
    background: str = Field(default="#FFFFFF", description="Background color for contrast checking")

class PaletteResponse(BaseModel):
    palette: List[str]
    method: str
    num_colors: int
    metadata: Dict[str, Any]
    generation_time: float

# Startup event to load models
@app.on_event("startup")
async def startup_event():
    """Load all models and metadata on startup"""
    logger.info(f"Current working directory: {os.getcwd()}")
    logger.info(f"Files in /app/models: {os.listdir('models') if os.path.exists('models') else 'No models dir'}")

    try:
        logger.info("Loading models...")
        
        # Check if model files exist
        model_files = {
            'gmm': 'models/gmm_model.sav',
            'scaler': 'models/scaler.sav',
            'encoder': 'models/encoder_model.h5',
            'decoder': 'models/decoder_model.h5',
            'metadata': 'models/metadata.json'
        }
        
        missing_files = []
        for name, path in model_files.items():
            if not os.path.exists(path):
                missing_files.append(path)
        
        if missing_files:
            logger.error(f"Missing model files: {missing_files}")
            raise FileNotFoundError(f"Missing model files: {missing_files}")
        
        # Load models
        models['gmm'] = joblib.load(model_files['gmm'])
        models['scaler'] = joblib.load(model_files['scaler'])
        
        # Load TensorFlow models with error handling
        try:
            models['encoder'] = tf.keras.models.load_model(model_files['encoder'])
            models['decoder'] = tf.keras.models.load_model(model_files['decoder'])
            logger.info("TensorFlow models loaded successfully")
        except Exception as e:
            logger.warning(f"Failed to load TensorFlow models: {e}")
            models['encoder'] = None
            models['decoder'] = None
        
        # Load metadata
        with open(model_files['metadata'], 'r') as f:
            global model_metadata
            model_metadata = json.load(f)
        
        logger.info("All models loaded successfully!")
        
    except Exception as e:
        logger.error(f"Failed to load models: {e}")
        # Set fallback models to None so we can still serve traditional palettes
        models['gmm'] = None
        models['encoder'] = None
        models['decoder'] = None

def apply_style_modifications(lab_colors: np.ndarray, style: str) -> np.ndarray:
    """Apply style modifications to LAB colors"""
    if style == "vibrant":
        # Increase saturation (a and b channels)
        lab_colors[:, 1] *= 1.3
        lab_colors[:, 2] *= 1.3
    elif style == "muted":
        # Decrease saturation
        lab_colors[:, 1] *= 0.7
        lab_colors[:, 2] *= 0.7
    elif style == "pastel":
        # Increase lightness, decrease saturation
        lab_colors[:, 0] = np.clip(lab_colors[:, 0] * 1.2, 0, 100)
        lab_colors[:, 1] *= 0.5
        lab_colors[:, 2] *= 0.5
    
    return lab_colors

async def generate_ai_palette(method: str, num_colors: int, style: str, seed: Optional[int] = None) -> List[str]:
    """Generate palette using AI models"""
    start_time = datetime.now()
    
    if seed is not None:
        np.random.seed(seed)
    
    try:
        if method == "gmm" and models['gmm'] is not None:
            # Add some randomness by sampling more and shuffling
            samples, _ = models['gmm'].sample(num_colors * 3)
            np.random.shuffle(samples)
            samples = samples[:num_colors]
            lab_colors = apply_style_modifications(samples, style)

            
        elif method == "deep" and models['encoder'] is not None and models['decoder'] is not None:
            # Generate using deep learning model
            latent_dim = 64  # From your model architecture
            latent_sample = np.random.normal(0, 1, (1, latent_dim))
            generated = models['decoder'].predict(latent_sample, verbose=0)
            generated = models['scaler'].inverse_transform(generated)
            samples = generated.reshape(6, 3)[:num_colors]
            lab_colors = apply_style_modifications(samples, style)
            
        else:
            raise ValueError(f"Model for method '{method}' not available")
        
        # Convert LAB to hex colors
        palette = []
        for lab in lab_colors:
            try:
                rgb = lab_to_rgb(lab.tolist())
                hex_color = rgb_to_hex(rgb)
                palette.append(hex_color)
            except Exception as e:
                logger.warning(f"Failed to convert LAB {lab} to hex: {e}")
                # Fallback to a neutral color
                palette.append("#808080")
        
        return palette[:num_colors]
        
    except Exception as e:
        logger.error(f"AI palette generation failed: {e}")
        raise HTTPException(status_code=500, detail=f"AI generation failed: {str(e)}")

@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "message": "AI Palette Generator API",
        "version": "1.0.0",
        "status": "active",
        "endpoints": {
            "generate": "/generate-palette",
            "contrast": "/check-contrast",
            "health": "/health",
            "docs": "/docs"
        },
        "methods": ["gmm", "deep", "complementary", "triadic", "analogous", "monochromatic"]
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    model_status = {
        "gmm_loaded": models.get('gmm') is not None,
        "deep_models_loaded": all([
            models.get('encoder') is not None,
            models.get('decoder') is not None,
            models.get('scaler') is not None
        ])
    }
    
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "models": model_status,
        "metadata": model_metadata if model_metadata else "Not loaded"
    }

@app.post("/generate-palette", response_model=PaletteResponse)
async def generate_palette(request: PaletteRequest):
    """Generate a color palette using specified method"""
    start_time = datetime.now()
    
    try:
        # AI methods
        if request.method in ["gmm", "deep"]:
            palette = await generate_ai_palette(
                request.method, 
                request.num_colors, 
                request.style,
                request.seed
            )
            
        # Traditional color theory methods
        elif request.method in ["complementary", "triadic", "analogous", "monochromatic"]:
            if not request.base_color:
                # Generate a random base color if none provided
                base_rgb = np.random.randint(0, 256, 3)
                base_color = rgb_to_hex(base_rgb)
            else:
                base_color = request.base_color
            
            palette = generate_traditional_palette(
                base_color, 
                request.method, 
                request.num_colors
            )
            
        else:
            raise HTTPException(
                status_code=400, 
                detail=f"Unknown method: {request.method}"
            )
        
        generation_time = (datetime.now() - start_time).total_seconds()
        
        return PaletteResponse(
            palette=palette,
            method=request.method,
            num_colors=len(palette),
            metadata={
                "style": request.style,
                "seed": request.seed,
                "base_color": request.base_color if request.method in ["complementary", "triadic", "analogous", "monochromatic"] else None,
                "model_info": {
                    "gmm_components": model_metadata.get("gmm_components") if model_metadata else None,
                    "training_samples": model_metadata.get("training_samples") if model_metadata else None
                }
            },
            generation_time=generation_time
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Palette generation error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/check-contrast")
async def check_contrast(request: ContrastCheckRequest):
    """Check WCAG contrast ratios for a set of colors"""
    try:
        results = []
        
        for i, color in enumerate(request.colors):
            # Check contrast with background
            bg_contrast = calculate_contrast_ratio(color, request.background)
            
            # Check contrast with other colors in the palette
            color_contrasts = []
            for j, other_color in enumerate(request.colors):
                if i != j:
                    contrast = calculate_contrast_ratio(color, other_color)
                    color_contrasts.append({
                        "with_color": other_color,
                        "ratio": round(contrast, 2),
                        "aa_normal": contrast >= 4.5,
                        "aa_large": contrast >= 3.0,
                        "aaa_normal": contrast >= 7.0,
                        "aaa_large": contrast >= 4.5
                    })
            
            results.append({
                "color": color,
                "background_contrast": {
                    "ratio": round(bg_contrast, 2),
                    "aa_normal": bg_contrast >= 4.5,
                    "aa_large": bg_contrast >= 3.0,
                    "aaa_normal": bg_contrast >= 7.0,
                    "aaa_large": bg_contrast >= 4.5
                },
                "palette_contrasts": color_contrasts
            })
        
        return {
            "results": results,
            "background_color": request.background,
            "overall_accessibility": {
                "aa_compliant": all(r["background_contrast"]["aa_normal"] for r in results),
                "aaa_compliant": all(r["background_contrast"]["aaa_normal"] for r in results)
            }
        }
        
    except Exception as e:
        logger.error(f"Contrast check error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/methods")
async def get_available_methods():
    """Get all available palette generation methods"""
    ai_methods = []
    if models.get('gmm'):
        ai_methods.append("gmm")
    if models.get('encoder') and models.get('decoder'):
        ai_methods.append("deep")
    
    return {
        "ai_methods": ai_methods,
        "traditional_methods": ["complementary", "triadic", "analogous", "monochromatic"],
        "styles": ["balanced", "vibrant", "muted", "pastel"]
    }

# Error handlers
@app.exception_handler(404)
async def not_found_handler(request, exc):
    return JSONResponse(
        status_code=404,
        content={"detail": "Endpoint not found", "available_endpoints": ["/", "/generate-palette", "/check-contrast", "/health", "/docs"]}
    )

@app.exception_handler(500)
async def internal_error_handler(request, exc):
    logger.error(f"Internal server error: {exc}")
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error", "message": "Please try again later"}
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
