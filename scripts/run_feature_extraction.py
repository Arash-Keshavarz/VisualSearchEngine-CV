#!/usr/bin/env python3
"""
Main script for running feature extraction with ViT and CLIP models.
This script demonstrates the complete pipeline from database to feature extraction.
"""

import os
import sys
import argparse
from pathlib import Path

# Add the parent directory to the path for imports
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from feature_extractor.vit_extractor import ViTFeatureExtractor
from feature_extractor.clip_extractor import CLIPFeatureExtractor
from feature_extractor.compare_models import ModelComparator

def check_dependencies():
    """Check if required dependencies are installed"""
    required_packages = ['torch', 'transformers', 'PIL', 'numpy']
    missing_packages = []
    
    for package in required_packages:
        try:
            if package == 'PIL':
                import PIL
            else:
                __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print("❌ Missing required packages:")
        for package in missing_packages:
            print(f"  - {package}")
        print("\n📦 Install them with:")
        print("  pip install torch transformers pillow numpy")
        return False
    
    print("✅ All required packages are installed")
    return True

def check_database(db_path: str):
    """Check if database exists and has images"""
    if not os.path.exists(db_path):
        print(f"❌ Database not found: {db_path}")
        print("💡 Please run dataset_to_db.py first to create the database")
        return False
    
    # Check if database has images
    import sqlite3
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        cursor.execute('SELECT COUNT(*) FROM images')
        image_count = cursor.fetchone()[0]
        
        cursor.execute('SELECT COUNT(*) FROM categories')
        category_count = cursor.fetchone()[0]
        
        conn.close()
        
        if image_count == 0:
            print("❌ No images found in database")
            print("💡 Please run dataset_to_db.py first to populate the database")
            return False
        
        print(f"✅ Database found with {image_count} images in {category_count} categories")
        return True
    except Exception as e:
        print(f"❌ Error accessing database: {e}")
        return False

def run_vit_extraction(db_path: str, batch_size: int = 32):
    """Run ViT feature extraction"""
    print("\n🧠 Running ViT Feature Extraction")
    print("=" * 50)
    
    vit_extractor = ViTFeatureExtractor(db_path=db_path)
    
    try:
        results = vit_extractor.extract_all_features(batch_size=batch_size)
        print(f"\n✅ ViT extraction completed!")
        print(f"📊 Results: {results}")
        
        # Save model info
        vit_extractor.save_model_info("vit_model_info.json")
        
        return results
    except Exception as e:
        print(f"❌ Error in ViT extraction: {e}")
        return None

def run_clip_extraction(db_path: str, batch_size: int = 32):
    """Run CLIP feature extraction"""
    print("\n🎯 Running CLIP Feature Extraction")
    print("=" * 50)
    
    clip_extractor = CLIPFeatureExtractor(db_path=db_path)
    
    try:
        results = clip_extractor.extract_all_features(batch_size=batch_size)
        print(f"\n✅ CLIP extraction completed!")
        print(f"📊 Results: {results}")
        
        # Save model info
        clip_extractor.save_model_info("clip_model_info.json")
        
        return results
    except Exception as e:
        print(f"❌ Error in CLIP extraction: {e}")
        return None

def run_comparison(db_path: str, test_images: int = 50):
    """Run model comparison"""
    print("\n🔬 Running Model Comparison")
    print("=" * 50)
    
    comparator = ModelComparator(db_path=db_path)
    results = comparator.run_extraction_comparison(test_images)
    
    return results

def main():
    parser = argparse.ArgumentParser(description="Run feature extraction with ViT and CLIP")
    parser.add_argument('--db_path', default='data/datasets/visual_search_dataset.db',
                       help='Path to SQLite database file')
    parser.add_argument('--model', choices=['vit', 'clip', 'both', 'compare'],
                       default='compare', help='Which model(s) to run')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size for feature extraction')
    parser.add_argument('--test_images', type=int, default=50,
                       help='Number of test images for comparison')
    parser.add_argument('--skip_checks', action='store_true',
                       help='Skip dependency and database checks')
    
    args = parser.parse_args()
    
    print("🚀 Visual Search Engine - Feature Extraction")
    print("=" * 60)
    
    # Check dependencies
    if not args.skip_checks:
        if not check_dependencies():
            return
    
    # Check database
    if not args.skip_checks:
        if not check_database(args.db_path):
            return
    
    # Run selected model(s)
    results = {}
    
    if args.model in ['vit', 'both']:
        vit_results = run_vit_extraction(args.db_path, args.batch_size)
        if vit_results:
            results['vit'] = vit_results
    
    if args.model in ['clip', 'both']:
        clip_results = run_clip_extraction(args.db_path, args.batch_size)
        if clip_results:
            results['clip'] = clip_results
    
    if args.model in ['compare', 'both']:
        comparison_results = run_comparison(args.db_path, args.test_images)
        if comparison_results:
            results['comparison'] = comparison_results
    
    # Summary
    print("\n" + "=" * 60)
    print("📋 EXECUTION SUMMARY")
    print("=" * 60)
    
    if 'vit' in results:
        print(f"✅ ViT extraction completed: {results['vit']['successful_extractions']} features")
    
    if 'clip' in results:
        print(f"✅ CLIP extraction completed: {results['clip']['successful_extractions']} features")
    
    if 'comparison' in results:
        print("✅ Model comparison completed")
    
    print(f"\n📄 Results saved to:")
    if 'vit' in results:
        print("  - vit_model_info.json")
    if 'clip' in results:
        print("  - clip_model_info.json")
    if 'comparison' in results:
        print("  - model_comparison_*.json")
    
    print("\n🎉 Feature extraction pipeline completed!")

if __name__ == "__main__":
    main() 