#!/usr/bin/env python3
"""
Safe ViT-only Gradio Web Interface for Visual Search Engine
"""

import os
import sys
import gradio as gr
from PIL import Image
import numpy as np
from pathlib import Path
import torch

# Add current directory to path
sys.path.append(os.path.dirname(__file__))

from visual_search_engine import VisualSearchEngine

class SafeViTOnlyVisualSearchApp:
    """Safe ViT-only Gradio app for visual search engine"""
    
    def __init__(self, db_path: str):
        self.db_path = db_path
        self.engine = None
        self.stats = {}
        
        # Set environment variables for safety
        os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
        
        # Force CPU for ViT
        torch.set_num_threads(1)
        
        # Initialize ViT engine
        self._initialize_vit_engine()
    
    def _initialize_vit_engine(self):
        """Initialize the ViT search engine with safety measures"""
        try:
            print(f"🚀 Initializing ViT Visual Search Engine...")
            
            # Force CPU usage
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            self.engine = VisualSearchEngine(self.db_path, "vit")
            
            # Build FAISS index
            self.engine.build_faiss_index()
            
            # Get stats
            self.stats = self.engine.get_database_stats()
            
            print(f"✅ ViT search engine initialized successfully!")
            
        except Exception as e:
            print(f"❌ Error initializing ViT search engine: {e}")
            raise
    
    def search_images(self, uploaded_image, top_k: int = 10):
        """Search for similar images using ViT"""
        if uploaded_image is None:
            return [], "Please upload an image first."
        
        try:
            # Save uploaded image temporarily
            temp_path = "temp_upload.jpg"
            uploaded_image.save(temp_path)
            
            # Search for similar images
            results = self.engine.search_similar_images(temp_path, top_k)
            
            # Clean up temp file
            if os.path.exists(temp_path):
                os.remove(temp_path)
            
            if not results:
                return [], "No similar images found."
            
            # Prepare results for display
            images = []
            info_text = f"Found {len(results)} similar images using ViT:\n\n"
            
            for result in results:
                # Load and resize image for display
                try:
                    img = Image.open(result['image_path'])
                    # Resize for display (maintain aspect ratio)
                    img.thumbnail((200, 200), Image.Resampling.LANCZOS)
                    images.append(img)
                    
                    # Get image info
                    img_info = self.engine.get_image_info(result['image_id'])
                    category = img_info['category'] if img_info else "Unknown"
                    
                    info_text += f"#{result['rank']}: {result['filename']}\n"
                    info_text += f"   Category: {category}\n"
                    info_text += f"   Similarity: {result['similarity']:.3f}\n\n"
                    
                except Exception as e:
                    print(f"⚠️ Error loading image {result['image_path']}: {e}")
                    continue
            
            return images, info_text
            
        except Exception as e:
            return [], f"Error during search: {str(e)}"
    
    def get_database_info(self):
        """Get database information for display"""
        if not self.stats:
            return "Database not loaded."
        
        info = f"""
📊 **Database Statistics**

🖼️ **Images:**
- Total images: {self.stats['total_images']}
- Images with features: {self.stats['images_with_features']}
- Index size: {self.stats['index_size']}

📂 **Categories:** {self.stats['total_categories']}

🧠 **Model:** ViT (CPU)

⚡ **Search Engine:** FAISS Index
        """
        return info
    
    def get_model_info(self):
        """Get current model information"""
        if not hasattr(self, 'engine') or self.engine is None:
            return "Model not loaded."
        
        return f"""
**Model:** ViT
**Feature Extractor:** {self.engine.feature_extractor.model_name}
**Feature Dimension:** {self.engine.feature_extractor.feature_dim}
**Device:** CPU (forced)
**Similarity Metric:** Cosine Similarity
**Index Type:** FAISS Flat Index
        """
    
    def create_interface(self):
        """Create the Gradio interface"""
        
        # Custom CSS for better styling
        css = """
        .gradio-container {
            max-width: 1200px !important;
        }
        .results-gallery {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 10px;
            margin-top: 20px;
        }
        """
        
        with gr.Blocks(css=css, title="Visual Search Engine - ViT (Safe)") as interface:
            
            gr.Markdown("# 🔍 Visual Search Engine - ViT (Safe)")
            gr.Markdown("Upload an image to find similar images in the database using ViT model (CPU optimized).")
            
            with gr.Row():
                with gr.Column(scale=1):
                    # Upload section
                    gr.Markdown("### 📤 Upload Image")
                    upload_input = gr.Image(
                        label="Upload an image",
                        type="pil",
                        height=300
                    )
                    
                    # Search parameters
                    gr.Markdown("### ⚙️ Search Parameters")
                    top_k_slider = gr.Slider(
                        minimum=1,
                        maximum=20,
                        value=10,
                        step=1,
                        label="Number of results (Top-K)"
                    )
                    
                    search_button = gr.Button(
                        "🔍 Search Similar Images",
                        variant="primary",
                        size="lg"
                    )
                
                with gr.Column(scale=1):
                    # Database info
                    gr.Markdown("### 📊 Database Info")
                    db_info = gr.Markdown(self.get_database_info())
                    
                    # Model info
                    gr.Markdown("### 🧠 Model Info")
                    model_info = gr.Markdown(self.get_model_info())
            
            # Results section
            gr.Markdown("### 📋 Search Results")
            
            with gr.Row():
                results_gallery = gr.Gallery(
                    label="Similar Images",
                    show_label=True,
                    elem_id="results-gallery",
                    columns=5,
                    rows=2,
                    height=400
                )
            
            results_info = gr.Textbox(
                label="Results Information",
                lines=10,
                interactive=False
            )
            
            # Event handlers
            search_button.click(
                fn=self.search_images,
                inputs=[upload_input, top_k_slider],
                outputs=[results_gallery, results_info]
            )
            
            # Auto-search on upload
            upload_input.change(
                fn=self.search_images,
                inputs=[upload_input, top_k_slider],
                outputs=[results_gallery, results_info]
            )
        
        return interface

def main():
    """Launch the safe ViT-only Gradio app"""
    db_path = "data/datasets/visual_search_dataset.db"
    
    # Create and launch app
    app = SafeViTOnlyVisualSearchApp(db_path)
    interface = app.create_interface()
    
    print("🚀 Launching Safe ViT-only Visual Search Engine...")
    print("📱 Open your browser to access the web interface")
    
    # Launch with custom settings
    interface.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True
    )

if __name__ == "__main__":
    main() 