# layout-duplicates-remover
```bash
python layout_detector.py "/Users/test/Downloads/classification-training-v1-Mar-2025-2"
open /Users/test/layout-detector/layout_analysis_report_20250813_110653.html
```

### Layout Similarity Detection

  - Extracts text blocks, bounding boxes, images, and page structure from PDFs using PyMuPDF
  - Processes images using OpenCV to detect regions and create layout signatures based on normalized layout elements
  - Compares spatial distribution using histograms
  - Analyzes document structure (page count, block count, aspect ratios)

### Modes

  - within_class: find duplicates only within same document type (same subfolder)
  - across_class: find similar layouts across different document types (different subfolder)
  - or both

### Features

  - Configurable similarity threshold (0-1)
  - HTML and JSON reporting with analysis results
  - Cleaned dataset creation with duplicate removal
