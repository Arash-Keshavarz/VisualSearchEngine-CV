import os
import sys
import time
import numpy as np
import sqlite3
from pathlib import Path
import argparse
import json
from datetime import datetime

# Add the src directory to the path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from feature_extractor.vit_extractor import ViTFeatureExtractor
from feature_extractor.clip_extractor import CLIPFeatureExtractor

class ModelComparator:
    """
    Compare different feature extraction models (ViT vs CLIP)
    """
    
    def __init__(self, db_path: str = "visual_search_dataset.db"):
        self.db_path = db_path
        self.results = {}
        
    def run_extraction_comparison(self, test_images: int = 50):
        """Compare feature extraction performance between ViT and CLIP"""
        print("🔬 Starting Model Comparison: ViT vs CLIP")
        print("=" * 60)
        
        # Initialize extractors
        vit_extractor = ViTFeatureExtractor(db_path=self.db_path)
        clip_extractor = CLIPFeatureExtractor(db_path=self.db_path)
        
        # Get test images from database
        images = self._get_test_images(test_images)
        
        if not images:
            print("❌ No images found in database. Please run dataset_to_db.py first.")
            return
        
        print(f"📸 Testing with {len(images)} images")
        
        # Test ViT
        print("\n🧠 Testing ViT Feature Extractor...")
        vit_results = self._test_extractor(vit_extractor, images, "ViT")
        
        # Test CLIP
        print("\n🎯 Testing CLIP Feature Extractor...")
        clip_results = self._test_extractor(clip_extractor, images, "CLIP")
        
        # Compare results
        self._compare_results(vit_results, clip_results)
        
        return {
            'vit': vit_results,
            'clip': clip_results
        }
    
    def _get_test_images(self, limit: int) -> list:
        """Get test images from database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT i.id, i.filename, i.filepath, c.name as category
            FROM images i
            JOIN categories c ON i.category_id = c.id
            ORDER BY RANDOM()
            LIMIT ?
        ''', (limit,))
        
        images = []
        for row in cursor.fetchall():
            images.append({
                'id': row[0],
                'filename': row[1],
                'filepath': row[2],
                'category': row[3]
            })
        
        conn.close()
        return images
    
    def _test_extractor(self, extractor, images: list, model_name: str) -> dict:
        """Test a specific feature extractor"""
        results = {
            'model_name': model_name,
            'total_images': len(images),
            'successful_extractions': 0,
            'failed_extractions': 0,
            'extraction_times': [],
            'feature_vectors': [],
            'model_info': extractor.get_model_info()
        }
        
        # Load model
        start_time = time.time()
        extractor.load_model()
        model_load_time = time.time() - start_time
        results['model_load_time'] = model_load_time
        
        print(f"  ⏱️ Model load time: {model_load_time:.2f}s")
        
        # Extract features
        for i, img in enumerate(images):
            start_time = time.time()
            
            try:
                feature = extractor.extract_features(img['filepath'])
                extraction_time = time.time() - start_time
                
                if not np.all(feature == 0):  # Successful extraction
                    results['successful_extractions'] += 1
                    results['extraction_times'].append(extraction_time)
                    results['feature_vectors'].append({
                        'image_id': img['id'],
                        'filename': img['filename'],
                        'category': img['category'],
                        'feature': feature,
                        'extraction_time': extraction_time
                    })
                else:
                    results['failed_extractions'] += 1
                    
            except Exception as e:
                results['failed_extractions'] += 1
                print(f"    ⚠️ Error processing {img['filename']}: {e}")
            
            if (i + 1) % 10 == 0:
                print(f"    ✅ Processed {i + 1}/{len(images)} images")
        
        # Calculate statistics
        if results['extraction_times']:
            results['avg_extraction_time'] = np.mean(results['extraction_times'])
            results['std_extraction_time'] = np.std(results['extraction_times'])
            results['min_extraction_time'] = np.min(results['extraction_times'])
            results['max_extraction_time'] = np.max(results['extraction_times'])
        else:
            results['avg_extraction_time'] = 0
            results['std_extraction_time'] = 0
            results['min_extraction_time'] = 0
            results['max_extraction_time'] = 0
        
        results['success_rate'] = (results['successful_extractions'] / len(images)) * 100
        
        return results
    
    def _compare_results(self, vit_results: dict, clip_results: dict):
        """Compare and display results between ViT and CLIP"""
        print("\n" + "=" * 60)
        print("📊 COMPARISON RESULTS")
        print("=" * 60)
        
        # Performance comparison
        print("\n⚡ PERFORMANCE COMPARISON:")
        print(f"{'Metric':<25} {'ViT':<15} {'CLIP':<15} {'Winner':<10}")
        print("-" * 65)
        
        # Success rate
        vit_success = vit_results['success_rate']
        clip_success = clip_results['success_rate']
        success_winner = "ViT" if vit_success > clip_success else "CLIP" if clip_success > vit_success else "Tie"
        print(f"{'Success Rate (%)':<25} {vit_success:<15.2f} {clip_success:<15.2f} {success_winner:<10}")
        
        # Average extraction time
        vit_time = vit_results['avg_extraction_time']
        clip_time = clip_results['avg_extraction_time']
        time_winner = "CLIP" if clip_time < vit_time else "ViT" if vit_time < clip_time else "Tie"
        print(f"{'Avg Extraction Time (s)':<25} {vit_time:<15.4f} {clip_time:<15.4f} {time_winner:<10}")
        
        # Model load time
        vit_load = vit_results['model_load_time']
        clip_load = clip_results['model_load_time']
        load_winner = "CLIP" if clip_load < vit_load else "ViT" if vit_load < clip_load else "Tie"
        print(f"{'Model Load Time (s)':<25} {vit_load:<15.4f} {clip_load:<15.4f} {load_winner:<10}")
        
        # Feature dimension
        vit_dim = vit_results['model_info']['feature_dim']
        clip_dim = clip_results['model_info']['feature_dim']
        print(f"{'Feature Dimension':<25} {vit_dim:<15} {clip_dim:<15} {'N/A':<10}")
        
        # Detailed statistics
        print("\n📈 DETAILED STATISTICS:")
        print(f"\n🧠 ViT Results:")
        print(f"  - Successful extractions: {vit_results['successful_extractions']}/{vit_results['total_images']}")
        print(f"  - Success rate: {vit_results['success_rate']:.2f}%")
        print(f"  - Avg extraction time: {vit_results['avg_extraction_time']:.4f}s")
        print(f"  - Model load time: {vit_results['model_load_time']:.4f}s")
        print(f"  - Feature dimension: {vit_results['model_info']['feature_dim']}")
        
        print(f"\n🎯 CLIP Results:")
        print(f"  - Successful extractions: {clip_results['successful_extractions']}/{clip_results['total_images']}")
        print(f"  - Success rate: {clip_results['success_rate']:.2f}%")
        print(f"  - Avg extraction time: {clip_results['avg_extraction_time']:.4f}s")
        print(f"  - Model load time: {clip_results['model_load_time']:.4f}s")
        print(f"  - Feature dimension: {clip_results['model_info']['feature_dim']}")
        
        # Recommendations
        print("\n💡 RECOMMENDATIONS:")
        if vit_results['success_rate'] > clip_results['success_rate']:
            print("  ✅ ViT has better success rate")
        elif clip_results['success_rate'] > vit_results['success_rate']:
            print("  ✅ CLIP has better success rate")
        else:
            print("  ✅ Both models have similar success rates")
            
        if vit_results['avg_extraction_time'] < clip_results['avg_extraction_time']:
            print("  ⚡ ViT is faster for feature extraction")
        elif clip_results['avg_extraction_time'] < vit_results['avg_extraction_time']:
            print("  ⚡ CLIP is faster for feature extraction")
        else:
            print("  ⚡ Both models have similar extraction speeds")
        
        # Save results
        self._save_comparison_results(vit_results, clip_results)
    
    def _save_comparison_results(self, vit_results: dict, clip_results: dict):
        """Save comparison results to file"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"model_comparison_{timestamp}.json"
        
        # Clean results for JSON serialization
        def clean_for_json(obj):
            if isinstance(obj, dict):
                return {k: clean_for_json(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [clean_for_json(item) for item in obj]
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif hasattr(obj, 'dtype'):  # numpy scalar
                return obj.item()
            else:
                return obj
        
        results = {
            'comparison_date': datetime.now().isoformat(),
            'vit_results': clean_for_json(vit_results),
            'clip_results': clean_for_json(clip_results),
            'summary': {
                'vit_success_rate': vit_results['success_rate'],
                'clip_success_rate': clip_results['success_rate'],
                'vit_avg_time': vit_results['avg_extraction_time'],
                'clip_avg_time': clip_results['avg_extraction_time'],
                'vit_feature_dim': vit_results['model_info']['feature_dim'],
                'clip_feature_dim': clip_results['model_info']['feature_dim']
            }
        }
        
        with open(filename, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\n📄 Comparison results saved to: {filename}")
    
    def run_full_extraction_comparison(self):
        """Run full feature extraction for both models and compare"""
        print("🚀 Running Full Feature Extraction Comparison")
        print("=" * 60)
        
        # Initialize extractors
        vit_extractor = ViTFeatureExtractor(db_path=self.db_path)
        clip_extractor = CLIPFeatureExtractor(db_path=self.db_path)
        
        # Extract features for all images with ViT
        print("\n🧠 Running ViT feature extraction...")
        vit_stats = vit_extractor.extract_all_features()
        
        # Extract features for all images with CLIP
        print("\n🎯 Running CLIP feature extraction...")
        clip_stats = clip_extractor.extract_all_features()
        
        # Compare full extraction results
        print("\n📊 FULL EXTRACTION COMPARISON:")
        print(f"{'Metric':<25} {'ViT':<15} {'CLIP':<15}")
        print("-" * 55)
        print(f"{'Total Processed':<25} {vit_stats['total_processed']:<15} {clip_stats['total_processed']:<15}")
        print(f"{'Successful Extractions':<25} {vit_stats['successful_extractions']:<15} {clip_stats['successful_extractions']:<15}")
        print(f"{'Feature Dimension':<25} {vit_stats['feature_dim']:<15} {clip_stats['feature_dim']:<15}")
        
        # Save model info
        vit_extractor.save_model_info("vit_model_info.json")
        clip_extractor.save_model_info("clip_model_info.json")
        
        return {
            'vit_stats': vit_stats,
            'clip_stats': clip_stats
        }

def main():
    parser = argparse.ArgumentParser(description="Compare ViT and CLIP feature extractors")
    parser.add_argument('--db_path', default='visual_search_dataset.db',
                       help='Path to SQLite database file')
    parser.add_argument('--test_images', type=int, default=50,
                       help='Number of test images for comparison')
    parser.add_argument('--full_extraction', action='store_true',
                       help='Run full feature extraction for all images')
    
    args = parser.parse_args()
    
    comparator = ModelComparator(args.db_path)
    
    if args.full_extraction:
        comparator.run_full_extraction_comparison()
    else:
        comparator.run_extraction_comparison(args.test_images)

if __name__ == "__main__":
    main() 