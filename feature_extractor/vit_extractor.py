import torch
import numpy as np
from PIL import Image
from transformers import ViTImageProcessor, ViTModel
from .base_extractor import BaseFeatureExtractor
import warnings
warnings.filterwarnings("ignore")

class ViTFeatureExtractor(BaseFeatureExtractor):
    """
    Vision Transformer (ViT) feature extractor.
    Uses the ViT model from Hugging Face transformers library.
    """
    
    def __init__(self, model_name: str = "google/vit-base-patch16-224", 
                 db_path: str = "visual_search_dataset.db"):
        # ViT base model has 768-dimensional features
        super().__init__(model_name=model_name, feature_dim=768, db_path=db_path)
        # Force CPU for ViT to avoid MPS compatibility issues
        self.device = torch.device("cpu")
        print(f"🔧 Using device: {self.device}")
    
    def load_model(self):
        """Load ViT model and processor"""
        print(f"📥 Loading ViT model: {self.model_name}")
        
        try:
            self.processor = ViTImageProcessor.from_pretrained(self.model_name)
            self.model = ViTModel.from_pretrained(self.model_name)
            self.model.to(self.device)
            self.model.eval()
            print(f"✅ ViT model loaded successfully")
        except Exception as e:
            print(f"❌ Error loading ViT model: {e}")
            raise
    
    def preprocess_image(self, image_path: str) -> torch.Tensor:
        """Preprocess image for ViT model"""
        try:
            # Load and preprocess image
            image = Image.open(image_path).convert('RGB')
            inputs = self.processor(images=image, return_tensors="pt")
            return inputs
        except Exception as e:
            print(f"⚠️ Error preprocessing {image_path}: {e}")
            raise
    
    def extract_features(self, image_path: str) -> np.ndarray:
        """Extract features from a single image using ViT"""
        try:
            # Preprocess image
            inputs = self.preprocess_image(image_path)
            
            # Move inputs to device
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Extract features
            with torch.no_grad():
                outputs = self.model(**inputs)
                # Use the [CLS] token representation (first token)
                features = outputs.last_hidden_state[:, 0, :].cpu().numpy()
            
            return features.flatten()
            
        except Exception as e:
            print(f"⚠️ Error extracting features from {image_path}: {e}")
            # Return zero vector for failed extractions
            return np.zeros(self.feature_dim)
    
    def extract_batch_features(self, image_paths: list) -> np.ndarray:
        """Extract features from a batch of images (optimized for ViT)"""
        features = []
        
        for i, image_path in enumerate(image_paths):
            try:
                feature = self.extract_features(image_path)
                features.append(feature)
                
                if (i + 1) % 10 == 0:
                    print(f"  ✅ Processed {i + 1}/{len(image_paths)} images")
                    
            except Exception as e:
                print(f"  ⚠️ Error processing {image_path}: {e}")
                features.append(np.zeros(self.feature_dim))
        
        return np.array(features)
    
    def get_model_info(self) -> dict:
        """Get detailed information about the ViT model"""
        return {
            'model_name': self.model_name,
            'feature_dim': self.feature_dim,
            'model_type': 'Vision Transformer (ViT)',
            'architecture': 'Transformer-based',
            'patch_size': 16,
            'image_size': 224,
            'device': str(self.device),
            'description': 'ViT uses a transformer architecture to process images as sequences of patches'
        } 