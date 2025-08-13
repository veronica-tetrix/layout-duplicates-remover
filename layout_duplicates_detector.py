import os
import hashlib
import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Set, Optional
from collections import defaultdict
import cv2
from PIL import Image
import fitz
from sklearn.metrics.pairwise import cosine_similarity
import shutil
import logging
from datetime import datetime

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DocumentLayoutAnalyzer:
    """
    Analyzes document layouts in a structured dataset where subfolders represent document types.
    Can detect duplicates within classes or across classes based on configuration.
    """
    
    def __init__(self, 
                 similarity_threshold: float = 0.95,
                 detect_mode: str = 'within_class'):
        """
        Initialize the analyzer.
        
        Args:
            similarity_threshold: Threshold for considering layouts as similar (0-1)
            detect_mode: 'within_class' - only detect duplicates within same document type
                        'across_class' - detect duplicates across all document types
                        'both' - report both types of duplicates
        """
        self.similarity_threshold = similarity_threshold
        self.detect_mode = detect_mode
        self.dataset_structure = {}
        self.layout_signatures = {}
        self.documents_to_keep = defaultdict(set)
        self.duplicates_within_class = defaultdict(lambda: defaultdict(list))
        self.duplicates_across_class = defaultdict(list)
        
    def extract_layout_features(self, file_path: str) -> Dict:
        """
        Extract layout features from a document.
        """
        features = {
            'text_blocks': [],
            'bbox_positions': [],
            'page_structure': [],
            'visual_hash': None,
            'layout_signature': None
        }
        
        try:
            if file_path.lower().endswith('.pdf'):
                features = self._extract_pdf_layout(file_path)
            elif file_path.lower().endswith(('.jpg', '.jpeg', '.png', '.tiff', '.bmp')):
                features = self._extract_image_layout(file_path)
            else:
                logger.warning(f"Unsupported file format: {file_path}")
        except Exception as e:
            logger.error(f"Error processing {file_path}: {str(e)}")
            
        return features
    
    def _extract_pdf_layout(self, pdf_path: str) -> Dict:
        """Extract layout features from PDF documents."""
        features = {
            'text_blocks': [],
            'bbox_positions': [],
            'page_structure': [],
            'visual_hash': [],
            'layout_signature': None
        }
        
        try:
            pdf_document = fitz.open(pdf_path)
            
            all_layout_elements = []
            
            for page_num, page in enumerate(pdf_document):
                blocks = page.get_text("blocks")
                
                page_rect = page.rect
                page_width = page_rect.width
                page_height = page_rect.height
                
                normalized_blocks = []
                for block in blocks:
                    if len(block) >= 5:
                        x0, y0, x1, y1 = block[:4]
                        norm_bbox = (
                            round(x0/page_width, 3), 
                            round(y0/page_height, 3),
                            round(x1/page_width, 3), 
                            round(y1/page_height, 3)
                        )
                        normalized_blocks.append({
                            'bbox': norm_bbox,
                            'type': 'text',
                            'size': len(block[4]) if len(block) > 4 else 0
                        })
                        all_layout_elements.append(norm_bbox)
                
                features['text_blocks'].extend(normalized_blocks)
                features['bbox_positions'].extend([b['bbox'] for b in normalized_blocks])
                
                try:
                    image_list = page.get_images()
                    for img in image_list:
                        # img is a tuple: (xref, smask, width, height, bpc, colorspace, alt. colorspace, name, filter, referencer)
                        xref = img[0]
                        
                        try:
                            img_rects = page.get_image_rects(xref)
                            if img_rects:
                                for img_rect in img_rects:
                                    # img_rect is a tuple: (rect, transform_matrix)
                                    rect = img_rect[0] if isinstance(img_rect, tuple) else img_rect
                                    norm_img_bbox = (
                                        round(rect.x0/page_width, 3),
                                        round(rect.y0/page_height, 3),
                                        round(rect.x1/page_width, 3),
                                        round(rect.y1/page_height, 3)
                                    )
                                    features['text_blocks'].append({
                                        'bbox': norm_img_bbox,
                                        'type': 'image',
                                        'size': 0
                                    })
                                    all_layout_elements.append(('img', norm_img_bbox))
                        except:
                            pass
                            
                except Exception as img_error:
                    logger.debug(f"Could not extract images from page {page_num}: {str(img_error)}")
                
                features['page_structure'].append({
                    'num_blocks': len(normalized_blocks),
                    'num_images': len(page.get_images()) if hasattr(page, 'get_images') else 0,
                    'page_num': page_num,
                    'aspect_ratio': round(page_width/page_height, 2)
                })
                
                try:
                    pix = page.get_pixmap(matrix=fitz.Matrix(0.3, 0.3))
                    img_data = pix.tobytes("png")
                    visual_hash = hashlib.md5(img_data).hexdigest()[:16]
                    features['visual_hash'].append(visual_hash)
                except:
                    features['visual_hash'].append(None)
            
            pdf_document.close()
            
            layout_str = json.dumps(all_layout_elements, sort_keys=True)
            features['layout_signature'] = hashlib.sha256(layout_str.encode()).hexdigest()[:32]
            
        except Exception as e:
            logger.error(f"Error extracting PDF layout from {pdf_path}: {str(e)}")
            
        return features
    
    def _extract_image_layout(self, image_path: str) -> Dict:
        """Extract layout from image documents using computer vision."""
        features = {
            'text_blocks': [],
            'bbox_positions': [],
            'visual_hash': None,
            'layout_signature': None
        }
        
        try:
            image = cv2.imread(image_path)
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            _, binary = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)
            
            contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            h, w = image.shape[:2]
            
            layout_elements = []
            
            for contour in contours:
                x, y, width, height = cv2.boundingRect(contour)
                if width > 10 and height > 10:
                    norm_bbox = (
                        round(x/w, 3), 
                        round(y/h, 3), 
                        round((x+width)/w, 3), 
                        round((y+height)/h, 3)
                    )
                    features['bbox_positions'].append(norm_bbox)
                    features['text_blocks'].append({
                        'bbox': norm_bbox,
                        'type': 'region',
                        'area': round((width/w) * (height/h), 4)
                    })
                    layout_elements.append(norm_bbox)
            
                    resized = cv2.resize(gray, (128, 128))
            features['visual_hash'] = hashlib.md5(resized.tobytes()).hexdigest()[:16]
            
            layout_str = json.dumps(sorted(layout_elements), sort_keys=True)
            features['layout_signature'] = hashlib.sha256(layout_str.encode()).hexdigest()[:32]
            
        except Exception as e:
            logger.error(f"Error extracting image layout: {str(e)}")
            
        return features
    
    def compute_layout_similarity(self, features1: Dict, features2: Dict) -> float:
        """
        Compute similarity between two document layouts.
        """
        if not features1['bbox_positions'] or not features2['bbox_positions']:
            return 0.0
        
        scores = []
        
        if features1.get('layout_signature') and features2.get('layout_signature'):
            if features1['layout_signature'] == features2['layout_signature']:
                return 1.0
        
        spatial_sim = self._spatial_histogram_similarity(
            features1['bbox_positions'], 
            features2['bbox_positions']
        )
        scores.append(('spatial', spatial_sim, 0.35))
        
        struct_sim = self._structure_similarity(features1, features2)
        scores.append(('structure', struct_sim, 0.35))
        
        if features1.get('visual_hash') and features2.get('visual_hash'):
            visual_sim = self._visual_hash_similarity(
                features1['visual_hash'], 
                features2['visual_hash']
            )
            scores.append(('visual', visual_sim, 0.3))
        
        total_weight = sum(score[2] for score in scores)
        if total_weight > 0:
            weighted_sum = sum(score[1] * score[2] for score in scores)
            return weighted_sum / total_weight
        
        return 0.0
    
    def _spatial_histogram_similarity(self, boxes1: List, boxes2: List, 
                                     grid_size: int = 5) -> float:
        """Compare spatial distribution of layout elements using grid histograms."""
        def create_spatial_histogram(boxes, grid_size):
            histogram = np.zeros((grid_size, grid_size))
            for box in boxes:
                x_center = (box[0] + box[2]) / 2
                y_center = (box[1] + box[3]) / 2
                
                x_bin = min(int(x_center * grid_size), grid_size - 1)
                y_bin = min(int(y_center * grid_size), grid_size - 1)
                
                histogram[y_bin, x_bin] += 1
            
            if histogram.sum() > 0:
                histogram = histogram / histogram.sum()
            return histogram.flatten()
        
        hist1 = create_spatial_histogram(boxes1, grid_size)
        hist2 = create_spatial_histogram(boxes2, grid_size)
        
        if np.any(hist1) and np.any(hist2):
            similarity = cosine_similarity([hist1], [hist2])[0][0]
            return max(0, similarity)
        return 0.0
    
    def _structure_similarity(self, features1: Dict, features2: Dict) -> float:
        """Compare structural properties of documents."""
        similarities = []
        
        num_blocks1 = len(features1['text_blocks'])
        num_blocks2 = len(features2['text_blocks'])
        if num_blocks1 > 0 or num_blocks2 > 0:
            block_sim = 1 - abs(num_blocks1 - num_blocks2) / max(num_blocks1, num_blocks2)
            similarities.append(block_sim)
        
        num_pages1 = len(features1.get('page_structure', [1]))
        num_pages2 = len(features2.get('page_structure', [1]))
        if num_pages1 > 0 or num_pages2 > 0:
            page_sim = 1 - abs(num_pages1 - num_pages2) / max(num_pages1, num_pages2)
            similarities.append(page_sim)
        
        if features1.get('page_structure') and features2.get('page_structure'):
            ar1 = features1['page_structure'][0].get('aspect_ratio', 1)
            ar2 = features2['page_structure'][0].get('aspect_ratio', 1)
            ar_sim = 1 - abs(ar1 - ar2) / max(ar1, ar2)
            similarities.append(ar_sim)
        
        return np.mean(similarities) if similarities else 0.0
    
    def _visual_hash_similarity(self, hash1, hash2) -> float:
        """Compare visual hashes."""
        if isinstance(hash1, list) and isinstance(hash2, list):
            common_hashes = set(hash1) & set(hash2)
            total_hashes = set(hash1) | set(hash2)
            return len(common_hashes) / len(total_hashes) if total_hashes else 0.0
        else:
            return 1.0 if hash1 == hash2 else 0.0
    
    def analyze_dataset(self, dataset_path: str) -> Tuple[Dict, Dict]:
        """
        Analyze the dataset respecting the folder structure.
        
        Args:
            dataset_path: Root path containing subfolders for each document type
            
        Returns:
            Tuple of (dataset_structure, analysis_results)
        """
        logger.info(f"Analyzing dataset at: {dataset_path}")
        
        self.dataset_structure = self._discover_structure(dataset_path)
        
        total_docs = sum(len(docs) for docs in self.dataset_structure.values())
        processed = 0
        
        for doc_class, doc_paths in self.dataset_structure.items():
            logger.info(f"\nProcessing document class: {doc_class}")
            logger.info(f"Found {len(doc_paths)} documents in this class")
            
            class_features = {}
            
            for doc_path in doc_paths:
                processed += 1
                logger.info(f"[{processed}/{total_docs}] Extracting features: {os.path.basename(doc_path)}")
                features = self.extract_layout_features(doc_path)
                class_features[doc_path] = features
                self.layout_signatures[doc_path] = {
                    'class': doc_class,
                    'features': features
                }
            
            if self.detect_mode in ['within_class', 'both']:
                self._find_duplicates_within_class(doc_class, class_features)
        
        if self.detect_mode in ['across_class', 'both']:
            self._find_duplicates_across_classes()
        
        results = self._compile_results()
        
        return self.dataset_structure, results
    
    def _discover_structure(self, dataset_path: str) -> Dict[str, List[str]]:
        """Discover the folder structure and document organization."""
        structure = defaultdict(list)
        supported_extensions = ('.pdf', '.jpg', '.jpeg', '.png', '.tiff', '.bmp')
        
        subdirs = [d for d in os.listdir(dataset_path) 
                  if os.path.isdir(os.path.join(dataset_path, d))]
        
        if subdirs:
            for subdir in subdirs:
                subdir_path = os.path.join(dataset_path, subdir)
                for root, _, files in os.walk(subdir_path):
                    for file in files:
                        if file.lower().endswith(supported_extensions):
                            structure[subdir].append(os.path.join(root, file))
        else:
            logger.warning("No subdirectories found. Treating all documents as single class.")
            for file in os.listdir(dataset_path):
                if file.lower().endswith(supported_extensions):
                    structure['all_documents'].append(os.path.join(dataset_path, file))
        
        return dict(structure)
    
    def _find_duplicates_within_class(self, doc_class: str, class_features: Dict):
        """Find duplicate layouts within a document class."""
        processed = set()
        docs_to_keep = []
        
        doc_list = list(class_features.keys())
        
        for i, doc1 in enumerate(doc_list):
            if doc1 in processed:
                continue
            
            similar_docs = []
            features1 = class_features[doc1]
            
            for doc2 in doc_list[i+1:]:
                if doc2 not in processed:
                    features2 = class_features[doc2]
                    similarity = self.compute_layout_similarity(features1, features2)
                    
                    if similarity >= self.similarity_threshold:
                        similar_docs.append((doc2, similarity))
            
            docs_to_keep.append(doc1)
            processed.add(doc1)
            
            if similar_docs:
                for dup_doc, sim_score in similar_docs:
                    self.duplicates_within_class[doc_class][doc1].append({
                        'document': dup_doc,
                        'similarity': round(sim_score, 3)
                    })
                    processed.add(dup_doc)
        
        self.documents_to_keep[doc_class] = set(docs_to_keep)
    
    def _find_duplicates_across_classes(self):
        """Find documents with similar layouts across different classes."""
        all_docs = []
        for doc_path, data in self.layout_signatures.items():
            all_docs.append((doc_path, data['class'], data['features']))
        
        for i, (doc1, class1, features1) in enumerate(all_docs):
            similar_across = []
            
            for doc2, class2, features2 in all_docs[i+1:]:
                if class1 != class2:
                    similarity = self.compute_layout_similarity(features1, features2)
                    
                    if similarity >= self.similarity_threshold:
                        similar_across.append({
                            'document': doc2,
                            'class': class2,
                            'similarity': round(similarity, 3)
                        })
            
            if similar_across:
                self.duplicates_across_class[doc1] = {
                    'class': class1,
                    'similar_in_other_classes': similar_across
                }
    
    def _compile_results(self) -> Dict:
        """Compile analysis results into a structured format."""
        results = {
            'summary': {
                'total_documents': sum(len(docs) for docs in self.dataset_structure.values()),
                'document_classes': len(self.dataset_structure),
                'detection_mode': self.detect_mode,
                'similarity_threshold': self.similarity_threshold
            },
            'by_class': {},
            'cross_class_similarities': []
        }
        
        for doc_class, doc_paths in self.dataset_structure.items():
            class_results = {
                'total_documents': len(doc_paths),
                'unique_layouts': len(self.documents_to_keep.get(doc_class, [])),
                'duplicate_groups': len(self.duplicates_within_class.get(doc_class, {})),
                'duplicates_found': sum(
                    len(dups) for dups in self.duplicates_within_class.get(doc_class, {}).values()
                )
            }
            results['by_class'][doc_class] = class_results
        
        if self.duplicates_across_class:
            results['cross_class_similarities'] = self.duplicates_across_class
            results['summary']['cross_class_similar_pairs'] = len(self.duplicates_across_class)
        
        return results
    
    def generate_report(self, output_path: str = None) -> str:
        """Generate a detailed HTML report of the analysis."""
        if output_path is None:
            output_path = f"layout_analysis_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
        
        html_content = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>Document Layout Analysis Report</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 20px; background-color: #f5f5f5; }
                .container { max-width: 1200px; margin: 0 auto; background-color: white; padding: 20px; }
                h1 { color: #333; border-bottom: 3px solid #4CAF50; padding-bottom: 10px; }
                h2 { color: #555; margin-top: 30px; }
                .summary { background-color: #e8f5e9; padding: 15px; border-radius: 5px; margin: 20px 0; }
                .class-section { background-color: #f9f9f9; padding: 15px; margin: 15px 0; border-left: 4px solid #2196F3; }
                .duplicate-group { background-color: #fff3e0; padding: 10px; margin: 10px 0; border-radius: 3px; }
                .warning { background-color: #ffebee; padding: 15px; border-left: 4px solid #f44336; margin: 20px 0; }
                .info { background-color: #e3f2fd; padding: 15px; border-left: 4px solid #2196F3; margin: 20px 0; }
                table { width: 100%; border-collapse: collapse; margin: 15px 0; }
                th, td { padding: 10px; text-align: left; border: 1px solid #ddd; }
                th { background-color: #4CAF50; color: white; }
                .metric { display: inline-block; margin: 10px 20px 10px 0; }
                .metric-label { font-weight: bold; color: #666; }
                .metric-value { font-size: 24px; color: #2196F3; }
                .recommendation { background-color: #f0f4f8; padding: 15px; border-radius: 5px; margin: 20px 0; }
            </style>
        </head>
        <body>
            <div class="container">
        """
        
        html_content += f"""
                <h1>Document Layout Analysis Report</h1>
                <p>Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
                
                <div class="summary">
                    <h2>Summary</h2>
                    <div class="metric">
                        <span class="metric-label">Total Documents:</span>
                        <span class="metric-value">{sum(len(docs) for docs in self.dataset_structure.values())}</span>
                    </div>
                    <div class="metric">
                        <span class="metric-label">Document Classes:</span>
                        <span class="metric-value">{len(self.dataset_structure)}</span>
                    </div>
                    <div class="metric">
                        <span class="metric-label">Detection Mode:</span>
                        <span class="metric-value">{self.detect_mode}</span>
                    </div>
                    <div class="metric">
                        <span class="metric-label">Similarity Threshold:</span>
                        <span class="metric-value">{self.similarity_threshold}</span>
                    </div>
                </div>
        """
        
        html_content += """
                <div class="recommendation">
                    <h2>Recommendations</h2>
                    <ul>
                        <li><strong>Within-Class Duplicates:</strong> These documents have very similar layouts within the same document type. 
                            Consider removing these to reduce training time without losing classification accuracy.</li>
                        <li><strong>Cross-Class Similarities:</strong> Documents with similar layouts across different types are actually 
                            valuable for training as they help the model learn that layout alone isn't sufficient for classification.</li>
                        <li><strong>Action:</strong> Use the 'copy_unique_documents' method to create a cleaned dataset with duplicates removed.</li>
                    </ul>
                </div>
        """
        
        html_content += "<h2>Within-Class Layout Analysis</h2>"
        
        for doc_class in self.dataset_structure:
            total = len(self.dataset_structure[doc_class])
            unique = len(self.documents_to_keep.get(doc_class, []))
            duplicates = total - unique
            
            html_content += f"""
                <div class="class-section">
                    <h3>{doc_class}</h3>
                    <table>
                        <tr>
                            <th>Metric</th>
                            <th>Value</th>
                            <th>Percentage</th>
                        </tr>
                        <tr>
                            <td>Total Documents</td>
                            <td>{total}</td>
                            <td>100%</td>
                        </tr>
                        <tr>
                            <td>Unique Layouts</td>
                            <td>{unique}</td>
                            <td>{unique/total*100:.1f}%</td>
                        </tr>
                        <tr>
                            <td>Duplicate Documents</td>
                            <td>{duplicates}</td>
                            <td>{duplicates/total*100:.1f}%</td>
                        </tr>
                    </table>
            """
            
            if doc_class in self.duplicates_within_class and self.duplicates_within_class[doc_class]:
                html_content += "<h4>Duplicate Groups:</h4>"
                for original, duplicates in self.duplicates_within_class[doc_class].items():
                    html_content += f"""
                        <div class="duplicate-group">
                            <strong>Original:</strong> {os.path.basename(original)}<br>
                            <strong>Duplicates ({len(duplicates)}):</strong>
                            <ul>
                    """
                    for dup in duplicates:
                        html_content += f"<li>{os.path.basename(dup['document'])} (similarity: {dup['similarity']})</li>"
                    html_content += "</ul></div>"
            
            html_content += "</div>"
        
        if self.duplicates_across_class:
            html_content += """
                <div class="warning">
                    <h2>Cross-Class Layout Similarities</h2>
                    <p>The following documents have similar layouts but belong to different document types. 
                    These are NOT duplicates and should typically be kept for training.</p>
                </div>
            """
            
            for doc1, info in list(self.duplicates_across_class.items())[:10]:  # Show first 10
                html_content += f"""
                    <div class="info">
                        <strong>{os.path.basename(doc1)}</strong> (Class: {info['class']})<br>
                        Similar to:
                        <ul>
                """
                for similar in info['similar_in_other_classes']:
                    html_content += f"<li>{os.path.basename(similar['document'])} (Class: {similar['class']}, Similarity: {similar['similarity']})</li>"
                html_content += "</ul></div>"
        
        html_content += """
            </div>
        </body>
        </html>
        """
        
        with open(output_path, 'w') as f:
            f.write(html_content)
        
        logger.info(f"HTML report saved to: {output_path}")
        return output_path
    
    def save_json_report(self, output_path: str = None) -> str:
        """Save analysis results as JSON."""
        if output_path is None:
            output_path = f"layout_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        report = {
            'metadata': {
                'generated': datetime.now().isoformat(),
                'dataset_path': None,
                'detection_mode': self.detect_mode,
                'similarity_threshold': self.similarity_threshold
            },
            'dataset_structure': {
                class_name: [os.path.basename(p) for p in paths]
                for class_name, paths in self.dataset_structure.items()
            },
            'documents_to_keep': {
                class_name: [os.path.basename(p) for p in paths]
                for class_name, paths in self.documents_to_keep.items()
            },
            'within_class_duplicates': {
                class_name: {
                    os.path.basename(orig): [
                        {
                            'document': os.path.basename(dup['document']),
                            'similarity': dup['similarity']
                        }
                        for dup in dups
                    ]
                    for orig, dups in class_dups.items()
                }
                for class_name, class_dups in self.duplicates_within_class.items()
            },
            'cross_class_similarities': {
                os.path.basename(doc): {
                    'class': info['class'],
                    'similar_to': [
                        {
                            'document': os.path.basename(s['document']),
                            'class': s['class'],
                            'similarity': s['similarity']
                        }
                        for s in info['similar_in_other_classes']
                    ]
                }
                for doc, info in self.duplicates_across_class.items()
            }
        }
        
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"JSON report saved to: {output_path}")
        return output_path
    
    def copy_unique_documents(self, output_dir: str, preserve_structure: bool = True):
        """
        Copy documents with unique layouts to a new directory.
        
        Args:
            output_dir: Directory to copy unique documents to
            preserve_structure: If True, maintain the folder structure
        """
        os.makedirs(output_dir, exist_ok=True)
        
        total_copied = 0
        
        for doc_class, docs_to_keep in self.documents_to_keep.items():
            if preserve_structure:
                class_output_dir = os.path.join(output_dir, doc_class)
                os.makedirs(class_output_dir, exist_ok=True)
            else:
                class_output_dir = output_dir
            
            for doc_path in docs_to_keep:
                filename = os.path.basename(doc_path)
                
                if preserve_structure:
                    dest_path = os.path.join(class_output_dir, filename)
                else:
                    dest_path = os.path.join(class_output_dir, f"{doc_class}_{filename}")
                
                if os.path.exists(dest_path):
                    base, ext = os.path.splitext(filename)
                    counter = 1
                    while os.path.exists(dest_path):
                        if preserve_structure:
                            dest_path = os.path.join(class_output_dir, f"{base}_{counter}{ext}")
                        else:
                            dest_path = os.path.join(class_output_dir, f"{doc_class}_{base}_{counter}{ext}")
                        counter += 1
                
                shutil.copy2(doc_path, dest_path)
                total_copied += 1
                logger.info(f"Copied: {filename} -> {dest_path}")
        
        logger.info(f"\nCopied {total_copied} unique documents to: {output_dir}")
        
        summary_path = os.path.join(output_dir, "dataset_summary.txt")
        with open(summary_path, 'w') as f:
            f.write(f"Dataset cleaned on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Detection mode: {self.detect_mode}\n")
            f.write(f"Similarity threshold: {self.similarity_threshold}\n\n")
            
            for doc_class in self.documents_to_keep:
                original_count = len(self.dataset_structure[doc_class])
                kept_count = len(self.documents_to_keep[doc_class])
                removed_count = original_count - kept_count
                
                f.write(f"{doc_class}:\n")
                f.write(f"  Original: {original_count} documents\n")
                f.write(f"  Kept: {kept_count} documents\n")
                f.write(f"  Removed: {removed_count} duplicates\n\n")


def main():
    """Main function to run the layout analysis."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Analyze document layouts in a structured dataset and detect duplicates'
    )
    parser.add_argument('dataset_path', 
                       help='Path to dataset root (containing subfolders for each document type)')
    parser.add_argument('--threshold', type=float, default=0.95,
                       help='Similarity threshold (0-1, default: 0.95)')
    parser.add_argument('--mode', choices=['within_class', 'across_class', 'both'],
                       default='within_class',
                       help='Detection mode (default: within_class)')
    parser.add_argument('--output-dir', default='cleaned_dataset',
                       help='Directory for cleaned dataset (default: cleaned_dataset)')
    parser.add_argument('--no-structure', action='store_true',
                       help='Do not preserve folder structure in output')
    
    args = parser.parse_args()
    
    analyzer = DocumentLayoutAnalyzer(
        similarity_threshold=args.threshold,
        detect_mode=args.mode
    )
    
    print("\nDocument Layout Analysis")
    print(f"Dataset: {args.dataset_path}")
    print(f"Mode: {args.mode}")
    print(f"Threshold: {args.threshold}")
    
    dataset_structure, _ = analyzer.analyze_dataset(args.dataset_path)
    
    html_report = analyzer.generate_report()
    json_report = analyzer.save_json_report()
    
    print("\nAnalysis Complete:")
    
    for doc_class in dataset_structure:
        total = len(dataset_structure[doc_class])
        unique = len(analyzer.documents_to_keep.get(doc_class, []))
        print(f"\n{doc_class}:")
        print(f"  Total documents: {total}")
        print(f"  Unique layouts: {unique}")
        print(f"  Duplicates to remove: {total - unique}")
    
    if analyzer.duplicates_across_class:
        print(f"\nFound {len(analyzer.duplicates_across_class)} documents with")
        print(f"   similar layouts across different classes.")
        print(f"   These are typically NOT duplicates and should be kept.")
    
    print(f"\nReports generated:")
    print(f"   HTML: {html_report}")
    print(f"   JSON: {json_report}")
    
    response = input(f"\nCreate cleaned dataset in '{args.output_dir}'? (y/n): ")
    if response.lower() == 'y':
        analyzer.copy_unique_documents(
            args.output_dir, 
            preserve_structure=not args.no_structure
        )
        print(f"\nCleaned dataset created in: {args.output_dir}")
    
    print("\nProcess complete.")


if __name__ == "__main__":
    main()