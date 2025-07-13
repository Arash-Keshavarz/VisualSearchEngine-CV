#!/usr/bin/env python3
"""
Visual Search Engine with FAISS Indexing and Similarity Search
"""

import os
import sys
import numpy as np
import sqlite3
import pickle
import faiss
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import gradio as gr
from PIL import Image
import torch

# Add current directory to path
sys.path.append(os.path.dirname(__file__))

from feature_extractor.vit_extractor import ViTFeatureExtractor
from feature_extractor.clip_extractor import CLIPFeatureExtractor

class VisualSearchEngine:
    """
    Complete visual search engine with FAISS indexing and similarity search
    """
    
    def __init__(self, db_path: str, model_name: str = "clip"):
        self.db_path = db_path
        self.model_name = model_name
        self.feature_extractor = None
        self.index = None
        self.image_paths = []
        self.image_ids = []
        
        # Initialize feature extractor
        self._initialize_extractor()
        
    def _initialize_extractor(self):
        """Initialize the appropriate feature extractor"""
        if self.model_name.lower() == "clip":
            self.feature_extractor = CLIPFeatureExtractor(db_path=self.db_path)
        elif self.model_name.lower() == "vit":
            self.feature_extractor = ViTFeatureExtractor(db_path=self.db_path)
        else:
            raise ValueError(f"Unknown model: {self.model_name}")
        
        print(f"🧠 Initialized {self.model_name.upper()} feature extractor")
        self.feature_extractor.load_model()
    
    def build_faiss_index(self, force_rebuild: bool = False):
        """Build FAISS index from database features"""
        print(f"🔍 Building FAISS index for {self.model_name}...")
        
        # Check if index already exists
        index_file = f"faiss_index_{self.model_name}.pkl"
        if os.path.exists(index_file) and not force_rebuild:
            print(f"📥 Loading existing FAISS index from {index_file}")
            self._load_faiss_index(index_file)
            return
        
        # Get all features from database
        features, image_paths, image_ids = self._get_all_features_from_db()
        
        if len(features) == 0:
            print("❌ No features found in database. Please run feature extraction first.")
            return
        
        # Convert to numpy array
        features_array = np.array(features, dtype=np.float32)
        
        # Normalize features for cosine similarity
        faiss.normalize_L2(features_array)
        
        # Create FAISS index
        dimension = features_array.shape[1]
        self.index = faiss.IndexFlatIP(dimension)  # Inner product for cosine similarity
        self.index.add(features_array)
        
        # Store metadata
        self.image_paths = image_paths
        self.image_ids = image_ids
        
        # Save index
        self._save_faiss_index(index_file)
        
        print(f"✅ FAISS index built successfully!")
        print(f"📊 Index size: {len(features)} vectors")
        print(f"📊 Feature dimension: {dimension}")
    
    def _get_all_features_from_db(self) -> Tuple[List, List, List]:
        """Get all features from database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT i.id, i.filepath, f.feature_vector
            FROM images i
            JOIN image_features f ON i.id = f.image_id
            WHERE f.model_name = ?
            ORDER BY i.id
        ''', (self.feature_extractor.model_name,))
        
        features = []
        image_paths = []
        image_ids = []
        
        for row in cursor.fetchall():
            image_id, filepath, feature_blob = row
            
            # Check if file exists
            if os.path.exists(filepath):
                feature = pickle.loads(feature_blob)
                features.append(feature)
                image_paths.append(filepath)
                image_ids.append(image_id)
        
        conn.close()
        
        print(f"📸 Found {len(features)} valid features in database")
        return features, image_paths, image_ids
    
    def _save_faiss_index(self, filename: str):
        """Save FAISS index and metadata"""
        index_data = {
            'index': self.index,
            'image_paths': self.image_paths,
            'image_ids': self.image_ids,
            'model_name': self.model_name
        }
        
        with open(filename, 'wb') as f:
            pickle.dump(index_data, f)
        
        print(f"💾 FAISS index saved to: {filename}")
    
    def _load_faiss_index(self, filename: str):
        """Load FAISS index and metadata"""
        with open(filename, 'rb') as f:
            index_data = pickle.load(f)
        
        self.index = index_data['index']
        self.image_paths = index_data['image_paths']
        self.image_ids = index_data['image_ids']
        
        print(f"📥 FAISS index loaded from: {filename}")
        print(f"📊 Index size: {len(self.image_paths)} vectors")
    
    def search_similar_images(self, query_image_path: str, top_k: int = 10) -> List[Dict]:
        """Search for similar images using FAISS"""
        if self.index is None:
            print("❌ FAISS index not built. Please run build_faiss_index() first.")
            return []
        
        try:
            # Extract features from query image
            query_features = self.feature_extractor.extract_features(query_image_path)
            
            if np.all(query_features == 0):
                print("❌ Failed to extract features from query image")
                return []
            
            # Normalize query features
            query_features = query_features.reshape(1, -1).astype(np.float32)
            faiss.normalize_L2(query_features)
            
            # Search in FAISS index
            similarities, indices = self.index.search(query_features, top_k)
            
            # Prepare results
            results = []
            for i, (similarity, idx) in enumerate(zip(similarities[0], indices[0])):
                if idx < len(self.image_paths):  # Valid index
                    results.append({
                        'rank': i + 1,
                        'image_path': self.image_paths[idx],
                        'image_id': self.image_ids[idx],
                        'similarity': float(similarity),
                        'filename': os.path.basename(self.image_paths[idx])
                    })
            
            return results
            
        except Exception as e:
            print(f"❌ Error in similarity search: {e}")
            return []
    
    def get_image_info(self, image_id: int) -> Optional[Dict]:
        """Get image information from database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT i.filename, i.filepath, i.file_size, i.width, i.height, c.name as category
            FROM images i
            JOIN categories c ON i.category_id = c.id
            WHERE i.id = ?
        ''', (image_id,))
        
        row = cursor.fetchone()
        conn.close()
        
        if row:
            return {
                'filename': row[0],
                'filepath': row[1],
                'file_size': row[2],
                'width': row[3],
                'height': row[4],
                'category': row[5]
            }
        return None
    
    def get_database_stats(self) -> Dict:
        """Get database statistics"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Total images
        cursor.execute('SELECT COUNT(*) FROM images')
        total_images = cursor.fetchone()[0]
        
        # Images with features
        cursor.execute('''
            SELECT COUNT(*) FROM image_features 
            WHERE model_name = ?
        ''', (self.feature_extractor.model_name,))
        images_with_features = cursor.fetchone()[0]
        
        # Categories
        cursor.execute('SELECT COUNT(*) FROM categories')
        total_categories = cursor.fetchone()[0]
        
        conn.close()
        
        return {
            'total_images': total_images,
            'images_with_features': images_with_features,
            'total_categories': total_categories,
            'model_name': self.model_name,
            'index_size': len(self.image_paths) if self.image_paths else 0
        }

def main():
    """Test the visual search engine"""
    db_path = "data/datasets/visual_search_dataset.db"
    
    # Initialize search engine
    engine = VisualSearchEngine(db_path, model_name="clip")
    
    # Build FAISS index
    engine.build_faiss_index()
    
    # Get stats
    stats = engine.get_database_stats()
    print(f"\n📊 Database Statistics:")
    print(f"  Total images: {stats['total_images']}")
    print(f"  Images with features: {stats['images_with_features']}")
    print(f"  Categories: {stats['total_categories']}")
    print(f"  Index size: {stats['index_size']}")
    
    # Test search with a sample image
    if len(engine.image_paths) > 0:
        sample_image = engine.image_paths[0]
        print(f"\n🔍 Testing search with: {sample_image}")
        
        results = engine.search_similar_images(sample_image, top_k=5)
        
        print(f"\n📋 Search Results:")
        for result in results:
            print(f"  {result['rank']}. {result['filename']} (similarity: {result['similarity']:.3f})")

if __name__ == "__main__":
    main() 