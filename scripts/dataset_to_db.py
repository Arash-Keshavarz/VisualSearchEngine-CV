import os
import sqlite3
import hashlib
from datetime import datetime
from pathlib import Path
import argparse
from typing import List, Dict, Tuple

class DatasetToDatabase:
    def __init__(self, db_path: str = "visual_search_dataset.db"):
        self.db_path = db_path
        self.conn = None
        self.cursor = None
        
    def connect(self):
        """Initialize database connection"""
        self.conn = sqlite3.connect(self.db_path)
        self.cursor = self.conn.cursor()
        self.create_tables()
        
    def create_tables(self):
        """Create the necessary tables with proper indexing"""
        # Categories table
        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS categories (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT UNIQUE NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Images table with foreign key to categories
        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS images (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                filename TEXT NOT NULL,
                filepath TEXT UNIQUE NOT NULL,
                category_id INTEGER NOT NULL,
                file_size INTEGER,
                width INTEGER,
                height INTEGER,
                file_hash TEXT,
                indexed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (category_id) REFERENCES categories (id)
            )
        ''')
        
        # Create indexes for efficient querying
        self.cursor.execute('CREATE INDEX IF NOT EXISTS idx_images_category ON images (category_id)')
        self.cursor.execute('CREATE INDEX IF NOT EXISTS idx_images_filename ON images (filename)')
        self.cursor.execute('CREATE INDEX IF NOT EXISTS idx_images_hash ON images (file_hash)')
        
        self.conn.commit()
        
    def get_image_info(self, filepath: str) -> Dict:
        """Extract metadata from an image file"""
        try:
            from PIL import Image
            with Image.open(filepath) as img:
                width, height = img.size
        except ImportError:
            # Fallback if PIL is not available
            width, height = None, None
        except Exception:
            width, height = None, None
            
        # Calculate file hash for duplicate detection
        try:
            with open(filepath, 'rb') as f:
                file_hash = hashlib.md5(f.read()).hexdigest()
        except Exception:
            file_hash = None
            
        return {
            'file_size': os.path.getsize(filepath),
            'width': width,
            'height': height,
            'file_hash': file_hash
        }
        
    def insert_category(self, category_name: str) -> int:
        """Insert or get category ID"""
        self.cursor.execute(
            'INSERT OR IGNORE INTO categories (name) VALUES (?)',
            (category_name,)
        )
        self.cursor.execute('SELECT id FROM categories WHERE name = ?', (category_name,))
        return self.cursor.fetchone()[0]
        
    def insert_image(self, filepath: str, category_id: int, image_info: Dict):
        """Insert image metadata into database"""
        filename = os.path.basename(filepath)
        
        self.cursor.execute('''
            INSERT OR REPLACE INTO images 
            (filename, filepath, category_id, file_size, width, height, file_hash)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (
            filename,
            filepath,
            category_id,
            image_info['file_size'],
            image_info['width'],
            image_info['height'],
            image_info['file_hash']
        ))
        
    def scan_dataset(self, dataset_path: str) -> Tuple[int, int]:
        """Scan the dataset and populate the database"""
        dataset_path = Path(dataset_path)
        total_images = 0
        total_categories = 0
        
        print(f"🔍 Scanning dataset at: {dataset_path}")
        
        # Get all category directories
        category_dirs = [d for d in dataset_path.iterdir() 
                       if d.is_dir() and not d.name.startswith('.')]
        
        for category_dir in category_dirs:
            category_name = category_dir.name
            print(f"\n📁 Processing category: {category_name}")
            
            # Insert category
            category_id = self.insert_category(category_name)
            total_categories += 1
            
            # Get all image files in this category
            image_extensions = {'.jpg', '.jpeg', '.png', '.gif', '.bmp'}
            image_files = [
                f for f in category_dir.iterdir()
                if f.is_file() and f.suffix.lower() in image_extensions
            ]
            
            category_count = 0
            for image_file in image_files:
                try:
                    image_info = self.get_image_info(str(image_file))
                    self.insert_image(str(image_file), category_id, image_info)
                    category_count += 1
                    total_images += 1
                    
                    if category_count % 10 == 0:
                        print(f"  ✅ Processed {category_count} images...")
                        
                except Exception as e:
                    print(f"  ⚠️ Error processing {image_file}: {e}")
                    
            print(f"  ✅ Category '{category_name}': {category_count} images")
            self.conn.commit()
            
        return total_images, total_categories
        
    def get_statistics(self) -> Dict:
        """Get database statistics"""
        self.cursor.execute('SELECT COUNT(*) FROM categories')
        category_count = self.cursor.fetchone()[0]
        
        self.cursor.execute('SELECT COUNT(*) FROM images')
        image_count = self.cursor.fetchone()[0]
        
        self.cursor.execute('''
            SELECT c.name, COUNT(i.id) as image_count
            FROM categories c
            LEFT JOIN images i ON c.id = i.category_id
            GROUP BY c.id, c.name
            ORDER BY image_count DESC
        ''')
        category_stats = self.cursor.fetchall()
        
        return {
            'total_categories': category_count,
            'total_images': image_count,
            'category_breakdown': category_stats
        }
        
    def close(self):
        """Close database connection"""
        if self.conn:
            self.conn.close()
            
    def query_images_by_category(self, category_name: str, limit: int = 10) -> List[Dict]:
        """Query images by category name"""
        self.cursor.execute('''
            SELECT i.filename, i.filepath, i.file_size, i.width, i.height
            FROM images i
            JOIN categories c ON i.category_id = c.id
            WHERE c.name = ?
            ORDER BY i.filename
            LIMIT ?
        ''', (category_name, limit))
        
        return [
            {
                'filename': row[0],
                'filepath': row[1],
                'file_size': row[2],
                'width': row[3],
                'height': row[4]
            }
            for row in self.cursor.fetchall()
        ]

def main():
    parser = argparse.ArgumentParser(description="Transfer dataset to SQLite database")
    parser.add_argument('--dataset_path', default='dataset', 
                       help='Path to dataset directory')
    parser.add_argument('--db_path', default='visual_search_dataset.db',
                       help='Path to SQLite database file')
    parser.add_argument('--stats', action='store_true',
                       help='Show database statistics after processing')
    
    args = parser.parse_args()
    
    # Initialize database
    db = DatasetToDatabase(args.db_path)
    db.connect()
    
    try:
        # Scan and populate database
        print("🚀 Starting dataset to database transfer...")
        total_images, total_categories = db.scan_dataset(args.dataset_path)
        
        print(f"\n✅ Transfer completed!")
        print(f"📊 Total categories: {total_categories}")
        print(f"📊 Total images: {total_images}")
        
        if args.stats:
            print("\n📈 Database Statistics:")
            stats = db.get_statistics()
            print(f"  Categories: {stats['total_categories']}")
            print(f"  Images: {stats['total_images']}")
            print("\n  Category Breakdown:")
            for category, count in stats['category_breakdown']:
                print(f"    {category}: {count} images")
                
    except Exception as e:
        print(f"❌ Error: {e}")
    finally:
        db.close()

if __name__ == "__main__":
    main() 