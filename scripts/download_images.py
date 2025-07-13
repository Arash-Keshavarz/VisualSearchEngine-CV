import os
import argparse
from icrawler.builtin import GoogleImageCrawler, BingImageCrawler, BaiduImageCrawler
from icrawler.downloader import ImageDownloader

# ---------- Custom Downloader to avoid file overwriting and ensure continuous indexing ----------
class UniqueNameDownloader(ImageDownloader):
    def get_filename(self, task, default_ext):
        # List all image files in the directory
        root_dir = self.storage.root_dir  # <-- FIXED
        existing_files = [
            f for f in os.listdir(root_dir)
            if f.lower().endswith(('.jpg', '.jpeg', '.png'))
        ]
        # Extract numeric part of filenames (e.g., 000123.jpg -> 123)
        indices = []
        for fname in existing_files:
            try:
                idx = int(os.path.splitext(fname)[0])
                indices.append(idx)
            except ValueError:
                continue
        next_idx = max(indices) + 1 if indices else 1
        # Ensure extension starts with a dot
        ext = default_ext if default_ext.startswith('.') else f'.{default_ext}'
        filename = '{:06d}{}'.format(next_idx, ext)
        return filename

# ---------- Utility Functions ----------
def ensure_dir(path):
    os.makedirs(path, exist_ok=True)

def get_existing_image_count(folder):
    return len([
        f for f in os.listdir(folder)
        if f.lower().endswith(('.jpg', '.jpeg', '.png')) and os.path.isfile(os.path.join(folder, f))
    ])

def download_from_source(keyword, max_images, output_folder, crawler_class):
    crawler = crawler_class(
        downloader_cls=UniqueNameDownloader,
        storage={'root_dir': output_folder}
    )
    try:
        crawler.crawl(keyword=keyword, max_num=max_images)
    except Exception as e:
        print(f"⚠️ Error using {crawler_class.__name__} for '{keyword}': {e}")

# ---------- Main Function ----------
def download_images_for_category(category, num_required):
    category_dir = os.path.join('dataset', category)
    ensure_dir(category_dir)

    # You can expand this list for better coverage
    subkeywords = [category]  # optionally add synonyms
    crawlers = [GoogleImageCrawler, BingImageCrawler, BaiduImageCrawler]

    for subkeyword in subkeywords:
        for crawler_class in crawlers:
            current_count = get_existing_image_count(category_dir)
            if current_count >= num_required:
                print(f"✅ Collected {current_count} images for '{category}'")
                return

            remaining = num_required - current_count
            print(f"🔄 Downloading {remaining} more for '{category}' using '{subkeyword}' via {crawler_class.__name__}")
            download_from_source(subkeyword, remaining, category_dir, crawler_class)

    final_count = get_existing_image_count(category_dir)
    print(f"✅ Final image count for '{category}': {final_count}")
    if final_count < num_required:
        print(f"⚠️ Warning: Only {final_count} images downloaded for '{category}'.")

# ---------- CLI Entry ----------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download shopping images per category using multiple sources.")
    parser.add_argument('--categories', nargs='+', required=True, help='Main categories (e.g., sneakers t-shirt jacket)')
    parser.add_argument('--num', type=int, default=100, help='Minimum number of images per category')

    args = parser.parse_args()

    for category in args.categories:
        print(f"\n📥 Starting download for: {category}")
        download_images_for_category(category, args.num)
