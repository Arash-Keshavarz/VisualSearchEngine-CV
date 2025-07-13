# 🔍 Visual Search Engine

A powerful visual search engine that uses deep learning models (CLIP and ViT) to find similar images in a large dataset. Built with Python, PyTorch, FAISS, and Gradio.

## 🎯 Features

- **Dual Model Support**: CLIP for semantic similarity, ViT for visual features
- **Fast Similarity Search**: FAISS indexing for efficient retrieval
- **Web Interface**: Beautiful Gradio UI for easy interaction
- **Scalable**: Handles large image datasets efficiently
- **Cross-Platform**: Works on macOS, Linux, and Windows

## 🏗️ Architecture

![Pipeline](pipeline.png)

The system consists of several key components:

1. **Image Collection**: Downloads images from multiple sources (Google, Bing, Baidu)
2. **Feature Extraction**: Extracts deep features using CLIP and ViT models
3. **Database Storage**: SQLite database for metadata and feature vectors
4. **FAISS Indexing**: High-performance similarity search indexing
5. **Web Interface**: Gradio-based UI for image search

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- Conda (recommended)
- Git

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/VisualSearchEngine-CV.git
   cd VisualSearchEngine-CV
   ```

2. **Create conda environment**
   ```bash
   conda create -n VisualEngine python=3.9
   conda activate VisualEngine
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set environment variable** (for macOS)
   ```bash
   export KMP_DUPLICATE_LIB_OK=TRUE
   ```

### Usage

#### Option 1: CLIP Model (Recommended for semantic similarity)
```bash
conda activate VisualEngine
export KMP_DUPLICATE_LIB_OK=TRUE
python gradio_app_clip_only.py
```

#### Option 2: ViT Model (For visual feature matching)
```bash
conda activate VisualEngine
export KMP_DUPLICATE_LIB_OK=TRUE
python gradio_app_vit_safe.py
```

Open your browser to `http://localhost:7860` to access the web interface.

## 📹 Demo

Watch the demo video to see the visual search engine in action:

[![Demo Video](Demo_Visual_engine.mp4)](Demo_Visual_engine.mp4)

## 🧠 Models

### CLIP (Contrastive Language-Image Pre-training)
- **Use Case**: Semantic similarity, understanding image content
- **Features**: 512-dimensional feature vectors
- **Strengths**: Better for understanding image meaning and context
- **App**: `gradio_app_clip_only.py`

### ViT (Vision Transformer)
- **Use Case**: Visual feature matching, detailed image analysis
- **Features**: 768-dimensional feature vectors
- **Strengths**: Better for visual pattern recognition
- **App**: `gradio_app_vit_safe.py`

## 📁 Project Structure

```
VisualSearchEngine-CV/
├── data/
│   └── datasets/
│       ├── dataset/          # Image dataset
│       └── visual_search_dataset.db  # SQLite database
├── feature_extractor/
│   ├── base_extractor.py     # Base feature extractor class
│   ├── clip_extractor.py     # CLIP feature extractor
│   └── vit_extractor.py      # ViT feature extractor
├── scripts/
│   ├── download_images.py    # Image downloader
│   ├── dataset_to_db.py      # Database creation
│   └── run_feature_extraction.py  # Feature extraction
├── utils/
│   └── database_utils.py     # Database utilities
├── gradio_app_clip_only.py   # CLIP-only web interface
├── gradio_app_vit_safe.py    # ViT-only web interface
├── visual_search_engine.py    # Core search engine
├── requirements.txt           # Python dependencies
└── README.md                 # This file
```

## 🔧 Configuration

### Database
The system uses SQLite for storing image metadata and feature vectors:
- **Images table**: Image metadata (filename, path, size, category)
- **Categories table**: Image categories
- **Image_features table**: Feature vectors for each model

### FAISS Index
- **Index Type**: Flat index with cosine similarity
- **Normalization**: L2 normalization for accurate similarity scores
- **Storage**: Pickled index files (`faiss_index_clip.pkl`, `faiss_index_vit.pkl`)

## 📊 Performance

- **Feature Extraction**: ~2-3 seconds per image
- **Search Speed**: <100ms for similarity search
- **Index Size**: ~933 images with both CLIP and ViT features
- **Memory Usage**: ~2GB for both models

## 🛠️ Development

### Adding New Models
1. Create a new feature extractor in `feature_extractor/`
2. Inherit from `BaseFeatureExtractor`
3. Implement `extract_features()` method
4. Add to the search engine

### Extending the Dataset
1. Add images to `data/datasets/dataset/`
2. Run `scripts/dataset_to_db.py` to update database
3. Run `scripts/run_feature_extraction.py` to extract features

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [CLIP](https://github.com/openai/CLIP) by OpenAI
- [ViT](https://github.com/google-research/vision_transformer) by Google Research
- [FAISS](https://github.com/facebookresearch/faiss) by Facebook Research
- [Gradio](https://github.com/gradio-app/gradio) for the web interface
- [Hugging Face Transformers](https://github.com/huggingface/transformers) for model loading

## 📞 Contact

For questions or support, please open an issue on GitHub or contact the maintainers.

---

**Note**: This project is designed to work with separate apps for each model to avoid memory conflicts and ensure stability. CLIP works best with MPS (Apple Silicon), while ViT is optimized for CPU usage.