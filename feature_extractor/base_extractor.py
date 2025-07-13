import os
import numpy as np
import sqlite3
from abc import ABC, abstractmethod
from typing import List, Dict, Tuple, Optional
from pathlib import Path
import pickle
import json
from datetime import datetime

class BaseFeatureExtractor(ABC):
    """
    Base class for all feature extractors.
    Defines the interface and common functionality.
    """
    
    def __init__(self, model_name: str, feature_dim: int, db_path: str = "visual_search_dataset.db"):
        self.model_name = model_name
        self.feature_dim = feature_dim
        self.db_path = db_path
        self.model = None
        self.preprocessor = None
        self.features_cache = {}
        
    @abstractmethod
    def load_model(self):
        """Load the pre-trained model and preprocessor"""
        pass
    
    @abstractmethod
    def preprocess_image(self, image_path: str) -> np.ndarray:
        """Preprocess image for the specific model"""
        pass
    
    @abstractmethod
    def extract_features(self, image_path: str) -> np.ndarray:
        """Extract features from a single image"""
        pass
    
    def extract_batch_features(self, image_paths: List[str]) -> np.ndarray:
        """Extract features from a batch of images"""
        features = []
        for i, image_path in enumerate(image_paths):
            try:
                feature = self.extract_features(image_path)
                features.append(feature)
                if (i + 1) % 10 == 0:
                    print(f"  ✅ Processed {i + 1}/{len(image_paths)} images")
            except Exception as e:
                print(f"  ⚠️ Error processing {image_path}: {e}")
                # Add zero vector for failed images
                features.append(np.zeros(self.feature_dim))
        
        return np.array(features)
    
    def get_database_connection(self):
        """Get SQLite database connection"""
        return sqlite3.connect(self.db_path)
    
    def get_all_images_from_db(self) -> List[Dict]:
        """Get all images from database with their metadata"""
        conn = self.get_database_connection()
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT i.id, i.filename, i.filepath, i.file_size, i.width, i.height, 
                   c.name as category
            FROM images i
            JOIN categories c ON i.category_id = c.id
            ORDER BY i.id
        ''')
        
        images = []
        for row in cursor.fetchall():
            # Check if file exists
            filepath = row[2]
            if os.path.exists(filepath):
                images.append({
                    'id': row[0],
                    'filename': row[1],
                    'filepath': filepath,
                    'file_size': row[3],
                    'width': row[4],
                    'height': row[5],
                    'category': row[6]
                })
            else:
                print(f"⚠️ File not found: {filepath}")
        
        conn.close()
        return images
    
    def get_images_by_category(self, category: str) -> List[Dict]:
        """Get images from a specific category"""
        conn = self.get_database_connection()
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT i.id, i.filename, i.filepath, i.file_size, i.width, i.height, 
                   c.name as category
            FROM images i
            JOIN categories c ON i.category_id = c.id
            WHERE c.name = ?
            ORDER BY i.id
        ''', (category,))
        
        images = []
        for row in cursor.fetchall():
            images.append({
                'id': row[0],
                'filename': row[1],
                'filepath': row[2],
                'file_size': row[3],
                'width': row[4],
                'height': row[5],
                'category': row[6]
            })
        
        conn.close()
        return images
    
    def save_features_to_db(self, image_id: int, features: np.ndarray):
        """Save extracted features to database"""
        conn = self.get_database_connection()
        cursor = conn.cursor()
        
        # Create features table if it doesn't exist
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS image_features (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                image_id INTEGER UNIQUE NOT NULL,
                model_name TEXT NOT NULL,
                feature_vector BLOB NOT NULL,
                feature_dim INTEGER NOT NULL,
                extracted_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (image_id) REFERENCES images (id)
            )
        ''')
        
        # Save features as blob
        feature_blob = pickle.dumps(features)
        
        cursor.execute('''
            INSERT OR REPLACE INTO image_features 
            (image_id, model_name, feature_vector, feature_dim)
            VALUES (?, ?, ?, ?)
        ''', (image_id, self.model_name, feature_blob, self.feature_dim))
        
        conn.commit()
        conn.close()
    
    def load_features_from_db(self, image_id: int) -> Optional[np.ndarray]:
        """Load features from database"""
        conn = self.get_database_connection()
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT feature_vector FROM image_features 
            WHERE image_id = ? AND model_name = ?
        ''', (image_id, self.model_name))
        
        result = cursor.fetchone()
        conn.close()
        
        if result:
            return pickle.loads(result[0])
        return None
    
    def extract_all_features(self, batch_size: int = 32) -> Dict:
        """Extract features for all images in database"""
        print(f"🚀 Starting feature extraction with {self.model_name}")
        print(f"📊 Loading model...")
        
        # Load model
        self.load_model()
        
        # Get all images from database
        images = self.get_all_images_from_db()
        print(f"📸 Found {len(images)} images in database")
        
        # Process in batches
        total_processed = 0
        total_features = 0
        
        for i in range(0, len(images), batch_size):
            batch = images[i:i + batch_size]
            image_paths = [img['filepath'] for img in batch]
            
            print(f"\n🔄 Processing batch {i//batch_size + 1}/{(len(images) + batch_size - 1)//batch_size}")
            print(f"  📁 Images {i+1}-{min(i+batch_size, len(images))}")
            
            # Extract features for batch
            features = self.extract_batch_features(image_paths)
            
            # Save features to database
            for j, (img, feature) in enumerate(zip(batch, features)):
                if not np.all(feature == 0):  # Skip failed extractions
                    self.save_features_to_db(img['id'], feature)
                    total_features += 1
                total_processed += 1
            
            print(f"  ✅ Batch completed: {len([f for f in features if not np.all(f == 0)])}/{len(features)} successful")
        
        print(f"\n✅ Feature extraction completed!")
        print(f"📊 Total processed: {total_processed}")
        print(f"📊 Successful extractions: {total_features}")
        
        return {
            'total_processed': total_processed,
            'successful_extractions': total_features,
            'model_name': self.model_name,
            'feature_dim': self.feature_dim
        }
    
    def get_extraction_stats(self) -> Dict:
        """Get statistics about feature extraction"""
        conn = self.get_database_connection()
        cursor = conn.cursor()
        
        # Check if features table exists
        cursor.execute('''
            SELECT name FROM sqlite_master 
            WHERE type='table' AND name='image_features'
        ''')
        
        if not cursor.fetchone():
            conn.close()
            return {'extracted_features': 0, 'total_images': 0, 'coverage': 0.0}
        
        # Get total images
        cursor.execute('SELECT COUNT(*) FROM images')
        total_images = cursor.fetchone()[0]
        
        # Get extracted features for this model
        cursor.execute('''
            SELECT COUNT(*) FROM image_features 
            WHERE model_name = ?
        ''', (self.model_name,))
        
        extracted_features = cursor.fetchone()[0]
        
        conn.close()
        
        coverage = (extracted_features / total_images * 100) if total_images > 0 else 0
        
        return {
            'extracted_features': extracted_features,
            'total_images': total_images,
            'coverage': coverage,
            'model_name': self.model_name
        }
    
    def save_model_info(self, output_path: str):
        """Save model information and configuration"""
        model_info = {
            'model_name': self.model_name,
            'feature_dim': self.feature_dim,
            'extraction_date': datetime.now().isoformat(),
            'stats': self.get_extraction_stats()
        }
        
        with open(output_path, 'w') as f:
            json.dump(model_info, f, indent=2)
        
        print(f"📄 Model info saved to: {output_path}") 