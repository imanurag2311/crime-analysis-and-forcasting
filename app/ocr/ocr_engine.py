"""
OCR Engine for Mumbai Police Crime Reports
============================================
Processes scanned PDFs using PyMuPDF → OpenCV → EasyOCR pipeline.
Extracts structured table data with bounding box row reconstruction.
"""

import re
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

import numpy as np

try:
    import fitz  # PyMuPDF
except (ImportError, OSError):
    fitz = None

try:
    import cv2
except (ImportError, OSError):
    cv2 = None

try:
    import easyocr
except (ImportError, OSError):
    easyocr = None

from app.core.config import (
    CLAHE_CLIP_LIMIT, CLAHE_TILE_GRID, SHARPENING_KERNEL,
    GAUSSIAN_BLUR_KERNEL, DESKEW_MIN_ANGLE, DESKEW_MAX_ANGLE,
    PDF_RENDER_DPI, OCR_ROW_Y_TOLERANCE, OCR_CONFIDENCE_THRESHOLD,
    OCR_LANGUAGES, DATE_PATTERNS, MONTH_NAME_MAP,
    SUMMARY_ROW_KEYWORDS, MEASUREMENT_KEYWORDS, SUB_BREAKDOWN_KEYWORDS
)

logger = logging.getLogger(__name__)


class OCREngine:
    """
    Computer Vision + OCR engine for extracting structured crime data
    from scanned Mumbai Police PDF reports.
    
    Pipeline:
        PDF → PyMuPDF rendering → OpenCV preprocessing → EasyOCR →
        Bounding box extraction → Horizontal row parser → 
        Registered (R) column extraction → Filtering → Validation
    """

    def __init__(self):
        """Initialize the OCR engine with EasyOCR reader."""
        self._reader = None
        self._check_dependencies()

    def _check_dependencies(self):
        """Check that required libraries are available."""
        missing = []
        if fitz is None:
            missing.append("PyMuPDF (pip install PyMuPDF)")
        if cv2 is None:
            missing.append("OpenCV (pip install opencv-python)")
        if easyocr is None:
            missing.append("EasyOCR (pip install easyocr)")
        if missing:
            logger.warning(f"Missing OCR dependencies: {', '.join(missing)}")

    @property
    def reader(self):
        """Lazy-load EasyOCR reader."""
        if self._reader is None and easyocr is not None:
            logger.info("Initializing EasyOCR reader...")
            self._reader = easyocr.Reader(OCR_LANGUAGES, gpu=False)
            logger.info("EasyOCR reader initialized")
        return self._reader

    # ================================================================
    # PDF RENDERING
    # ================================================================

    def render_pdf_page(self, pdf_path: str, page_number: int,
                        dpi: int = PDF_RENDER_DPI) -> Optional[np.ndarray]:
        """
        Render a single PDF page to a high-DPI image using PyMuPDF.
        
        Args:
            pdf_path: Path to the PDF file
            page_number: Zero-indexed page number
            dpi: Rendering DPI (default 300)
            
        Returns:
            numpy array (BGR image) or None on failure
        """
        if fitz is None:
            logger.error("PyMuPDF not available")
            return None

        try:
            doc = fitz.open(pdf_path)
            if page_number >= len(doc):
                logger.error(
                    f"Page {page_number} out of range (PDF has {len(doc)} pages)"
                )
                doc.close()
                return None

            page = doc[page_number]
            zoom = dpi / 72.0
            matrix = fitz.Matrix(zoom, zoom)
            pix = page.get_pixmap(matrix=matrix)

            # Convert to numpy array
            img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(
                pix.height, pix.width, pix.n
            )

            # Convert RGB to BGR for OpenCV
            if pix.n == 4:  # RGBA
                img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
            elif pix.n == 3:  # RGB
                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

            doc.close()
            logger.info(
                f"Rendered page {page_number} of {pdf_path} at {dpi} DPI "
                f"({img.shape[1]}x{img.shape[0]})"
            )
            return img

        except Exception as e:
            logger.error(f"Error rendering PDF page {page_number}: {e}")
            return None

    def get_pdf_page_count(self, pdf_path: str) -> int:
        """Get total number of pages in a PDF."""
        if fitz is None:
            return 0
        try:
            doc = fitz.open(pdf_path)
            count = len(doc)
            doc.close()
            return count
        except Exception as e:
            logger.error(f"Error reading PDF: {e}")
            return 0

    def extract_native_text(self, pdf_path: str, page_number: int) -> str:
        """
        Attempt native/vector text extraction from PDF page.
        Falls back to OCR if native text is empty or unsuitable.
        """
        if fitz is None:
            return ""
        try:
            doc = fitz.open(pdf_path)
            if page_number >= len(doc):
                doc.close()
                return ""
            page = doc[page_number]
            text = page.get_text("text")
            doc.close()
            return text.strip()
        except Exception as e:
            logger.error(f"Error extracting native text: {e}")
            return ""

    # ================================================================
    # OPENCV IMAGE PREPROCESSING
    # ================================================================

    def preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """
        Apply full OpenCV preprocessing pipeline for OCR.
        
        Steps:
            1. Convert to grayscale
            2. Rescaling (if needed)
            3. CLAHE contrast enhancement
            4. Gaussian blur denoising
            5. Sharpening
            6. Otsu binary thresholding
            7. Deskewing
            
        Args:
            image: BGR image as numpy array
            
        Returns:
            Preprocessed binary image
        """
        if cv2 is None:
            logger.error("OpenCV not available")
            return image

        # Step 1: Convert to grayscale
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()

        # Step 2: Rescaling for low-DPI scans
        gray = self._rescale_if_needed(gray)

        # Step 3: CLAHE contrast enhancement
        gray = self._apply_clahe(gray)

        # Step 4: Gaussian blur denoising
        gray = cv2.GaussianBlur(gray, GAUSSIAN_BLUR_KERNEL, 0)

        # Step 5: Sharpening
        gray = self._apply_sharpening(gray)

        # Step 6: Otsu binary thresholding
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # Step 7: Deskewing
        binary = self._deskew(binary)

        logger.info("Image preprocessing completed")
        return binary

    def _rescale_if_needed(self, gray: np.ndarray,
                            min_height: int = 2000) -> np.ndarray:
        """Rescale image using bilinear interpolation if too small."""
        h, w = gray.shape[:2]
        if h < min_height:
            scale = min_height / h
            new_w = int(w * scale)
            new_h = int(h * scale)
            gray = cv2.resize(gray, (new_w, new_h),
                              interpolation=cv2.INTER_LINEAR)
            logger.info(f"Rescaled image from {w}x{h} to {new_w}x{new_h}")
        return gray

    def _apply_clahe(self, gray: np.ndarray) -> np.ndarray:
        """Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)."""
        clahe = cv2.createCLAHE(
            clipLimit=CLAHE_CLIP_LIMIT,
            tileGridSize=CLAHE_TILE_GRID
        )
        return clahe.apply(gray)

    def _apply_sharpening(self, gray: np.ndarray) -> np.ndarray:
        """Apply sharpening kernel."""
        kernel = np.array(SHARPENING_KERNEL, dtype=np.float32)
        return cv2.filter2D(gray, -1, kernel)

    def _deskew(self, binary: np.ndarray) -> np.ndarray:
        """
        Deskew image by estimating page tilt from minimum bounding rectangle.
        Corrects angles within -15° to +15°.
        """
        try:
            coords = np.column_stack(np.where(binary < 128))
            if len(coords) < 100:
                return binary

            rect = cv2.minAreaRect(coords)
            angle = rect[-1]

            # Normalize angle
            if angle < -45:
                angle = -(90 + angle)
            else:
                angle = -angle

            # Only correct within limits
            if DESKEW_MIN_ANGLE <= angle <= DESKEW_MAX_ANGLE and abs(angle) > 0.5:
                h, w = binary.shape[:2]
                center = (w // 2, h // 2)
                M = cv2.getRotationMatrix2D(center, angle, 1.0)
                binary = cv2.warpAffine(
                    binary, M, (w, h),
                    flags=cv2.INTER_CUBIC,
                    borderMode=cv2.BORDER_REPLICATE
                )
                logger.info(f"Deskewed image by {angle:.2f}°")

        except Exception as e:
            logger.warning(f"Deskew failed: {e}")

        return binary

    # ================================================================
    # EASYOCR TEXT EXTRACTION
    # ================================================================

    def extract_text_with_boxes(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """
        Run EasyOCR on a preprocessed image.
        
        Returns list of dicts with:
            - text: detected text
            - bbox: bounding box coordinates
            - x_min, y_min, x_max, y_max, y_center
            - confidence: OCR confidence score
        """
        if self.reader is None:
            logger.error("EasyOCR reader not available")
            return []

        try:
            results = self.reader.readtext(image)
            ocr_boxes = []

            for (bbox, text, confidence) in results:
                # bbox is [[x1,y1],[x2,y2],[x3,y3],[x4,y4]]
                xs = [point[0] for point in bbox]
                ys = [point[1] for point in bbox]

                x_min = min(xs)
                y_min = min(ys)
                x_max = max(xs)
                y_max = max(ys)
                y_center = (y_min + y_max) / 2

                ocr_boxes.append({
                    'text': text.strip(),
                    'bbox': bbox,
                    'x_min': x_min,
                    'y_min': y_min,
                    'x_max': x_max,
                    'y_max': y_max,
                    'y_center': y_center,
                    'confidence': confidence
                })

            logger.info(f"EasyOCR extracted {len(ocr_boxes)} text regions")
            return ocr_boxes

        except Exception as e:
            logger.error(f"EasyOCR extraction failed: {e}")
            return []

    # ================================================================
    # HORIZONTAL ROW PARSER
    # ================================================================

    def group_into_rows(self, ocr_boxes: List[Dict[str, Any]],
                        y_tolerance: float = OCR_ROW_Y_TOLERANCE) -> List[List[Dict]]:
        """
        Group OCR bounding boxes into horizontal table rows using y-center proximity.
        
        Algorithm:
            1. Sort boxes by y_center
            2. Group boxes where y_center difference < y_tolerance
            3. Sort each row by x_min (left to right)
            
        Args:
            ocr_boxes: List of OCR box dicts with y_center
            y_tolerance: Maximum y-center difference for same-row grouping
            
        Returns:
            List of rows, each row is a list of box dicts sorted left-to-right
        """
        if not ocr_boxes:
            return []

        # Sort by y_center
        sorted_boxes = sorted(ocr_boxes, key=lambda b: b['y_center'])

        rows = []
        current_row = [sorted_boxes[0]]

        for box in sorted_boxes[1:]:
            if abs(box['y_center'] - current_row[-1]['y_center']) <= y_tolerance:
                current_row.append(box)
            else:
                # Sort current row by x_min (left to right)
                current_row.sort(key=lambda b: b['x_min'])
                rows.append(current_row)
                current_row = [box]

        # Don't forget the last row
        current_row.sort(key=lambda b: b['x_min'])
        rows.append(current_row)

        logger.info(f"Grouped {len(ocr_boxes)} boxes into {len(rows)} rows")
        return rows

    # ================================================================
    # REGISTERED COLUMN R EXTRACTION
    # ================================================================

    def extract_registered_cases(self, rows: List[List[Dict]]) -> List[Dict[str, Any]]:
        """
        Extract crime category and Registered (R) case counts from table rows.
        
        CRITICAL: Only uses the Registered (R) column.
        NEVER uses Detected (D) column.
        NEVER combines R + D.
        
        Assumes the first numeric value in each row corresponds to
        the Registered Cases column.
        
        Returns list of extracted records with:
            - crime_category
            - case_count (Registered R only)
            - confidence (average of boxes)
            - raw_text
            - status: 'accepted', 'rejected', 'flagged'
            - rejection_reason (if rejected)
        """
        extracted = []

        for row in rows:
            text_boxes = []
            num_boxes = []
            total_confidence = 0

            for box in row:
                total_confidence += box.get('confidence', 0)
                text = box['text'].strip()

                # Try to parse as number
                cleaned_num = re.sub(r'[,\s]', '', text)
                try:
                    val = int(cleaned_num)
                    num_boxes.append({
                        'value': val,
                        'text': text,
                        'confidence': box.get('confidence', 0)
                    })
                except ValueError:
                    if text:
                        text_boxes.append({
                            'text': text,
                            'confidence': box.get('confidence', 0)
                        })

            if not text_boxes or not num_boxes:
                continue

            # Combine text boxes to form category name
            category_text = ' '.join([tb['text'] for tb in text_boxes])
            avg_confidence = total_confidence / len(row) if row else 0
            raw_text = ' | '.join([b['text'] for b in row])

            # Check for rejection
            rejection = self._check_rejection(category_text, raw_text)
            if rejection:
                extracted.append({
                    'crime_category': category_text,
                    'case_count': num_boxes[0]['value'],
                    'confidence': avg_confidence,
                    'raw_text': raw_text,
                    'status': 'rejected',
                    'rejection_reason': rejection
                })
                continue

            # Use FIRST numeric value = Registered (R) column
            registered_count = num_boxes[0]['value']

            # Flag low confidence
            status = 'accepted'
            if avg_confidence < OCR_CONFIDENCE_THRESHOLD:
                status = 'flagged'

            extracted.append({
                'crime_category': category_text,
                'case_count': registered_count,
                'confidence': avg_confidence,
                'raw_text': raw_text,
                'status': status,
                'rejection_reason': ''
            })

        accepted = sum(1 for r in extracted if r['status'] == 'accepted')
        flagged = sum(1 for r in extracted if r['status'] == 'flagged')
        rejected = sum(1 for r in extracted if r['status'] == 'rejected')
        logger.info(
            f"Extracted records: {accepted} accepted, "
            f"{flagged} flagged, {rejected} rejected"
        )

        return extracted

    def _check_rejection(self, category: str, raw_text: str) -> Optional[str]:
        """
        Check if a row should be rejected.
        
        Excludes: summary rows, total rows, sub-breakdowns,
        POCSO/PITA sub-clauses, measurement rows.
        """
        lower_cat = category.lower()
        lower_raw = raw_text.lower()

        # Check summary/total rows
        for keyword in SUMMARY_ROW_KEYWORDS:
            if keyword in lower_cat or keyword in lower_raw:
                return f"Summary row (contains '{keyword}')"

        # Check measurement rows
        for keyword in MEASUREMENT_KEYWORDS:
            if keyword in lower_cat or keyword in lower_raw:
                return f"Measurement/sub-breakdown row (contains '{keyword}')"

        # Check sub-breakdown keywords
        for keyword in SUB_BREAKDOWN_KEYWORDS:
            # Only reject if it looks like a sub-clause, not a main crime category
            if keyword in lower_cat and len(category.split()) <= 3:
                return f"Sub-breakdown row (contains '{keyword}')"

        return None

    # ================================================================
    # DATE EXTRACTION FROM HEADERS
    # ================================================================

    def extract_date_from_text(self, text: str) -> Optional[Tuple[int, int]]:
        """
        Extract month and year from report header text.
        
        Supports patterns:
            - DD/MM/YYYY, DD.MM.YYYY
            - Month Name YYYY
            - Abbreviated Month YYYY
            
        Returns (month, year) tuple or None.
        """
        if not text:
            return None

        # Pattern 1: DD/MM/YYYY or DD.MM.YYYY
        match = re.search(r'(\d{2})[\/\.](\d{2})[\/\.](\d{4})', text)
        if match:
            month = int(match.group(2))
            year = int(match.group(3))
            if 1 <= month <= 12 and 1900 <= year <= 2100:
                return (month, year)

        # Pattern 2: Full month name + year
        text_lower = text.lower()
        for month_name, month_num in MONTH_NAME_MAP.items():
            pattern = rf'\b{month_name}\b\s*[\-,]?\s*(\d{{4}})'
            match = re.search(pattern, text_lower)
            if match:
                year = int(match.group(1))
                if 1900 <= year <= 2100:
                    return (month_num, year)

        return None

    # ================================================================
    # FULL PAGE PROCESSING
    # ================================================================

    def process_pdf_page(self, pdf_path: str, page_number: int) -> Dict[str, Any]:
        """
        Process a single PDF page through the full OCR pipeline.
        
        Pipeline:
            1. Attempt native text extraction
            2. If native text insufficient, fall back to OCR:
               a. Render page at high DPI
               b. OpenCV preprocessing
               c. EasyOCR extraction
               d. Horizontal row parsing
               e. Registered column extraction
            3. Extract date from header
            4. Return structured results
            
        Returns dict with extracted records, date info, and processing metadata.
        """
        result = {
            'page_number': page_number,
            'source_file': str(pdf_path),
            'records': [],
            'month': None,
            'year': None,
            'total_boxes': 0,
            'total_rows': 0,
            'accepted': 0,
            'flagged': 0,
            'rejected': 0,
            'error': None,
            'method': 'ocr'
        }

        try:
            # Step 1: Try native text extraction
            native_text = self.extract_native_text(pdf_path, page_number)

            use_ocr = True
            if native_text and len(native_text) > 100:
                # Check if native text has enough structured content
                lines = [l.strip() for l in native_text.split('\n') if l.strip()]
                has_numbers = any(re.search(r'\d+', line) for line in lines)
                if has_numbers and len(lines) > 5:
                    # Native text might be usable, but still prefer OCR for tables
                    logger.info("Native text available but using OCR for table extraction")

                # Try to extract date from native text
                date_info = self.extract_date_from_text(native_text)
                if date_info:
                    result['month'], result['year'] = date_info

            # Step 2: Full OCR pipeline
            if use_ocr:
                result['method'] = 'ocr'

                # Render page
                image = self.render_pdf_page(pdf_path, page_number)
                if image is None:
                    result['error'] = f"Failed to render page {page_number}"
                    return result

                # Preprocess
                processed = self.preprocess_image(image)

                # Extract text with bounding boxes
                # Use original image for EasyOCR (it handles its own preprocessing)
                ocr_boxes = self.extract_text_with_boxes(image)
                result['total_boxes'] = len(ocr_boxes)

                if not ocr_boxes:
                    result['error'] = "No text detected by OCR"
                    return result

                # Try to extract date from OCR text if not found in native text
                if result['month'] is None:
                    all_text = ' '.join([b['text'] for b in ocr_boxes])
                    date_info = self.extract_date_from_text(all_text)
                    if date_info:
                        result['month'], result['year'] = date_info

                # Group into rows
                rows = self.group_into_rows(ocr_boxes)
                result['total_rows'] = len(rows)

                # Extract registered cases
                records = self.extract_registered_cases(rows)
                result['records'] = records
                result['accepted'] = sum(
                    1 for r in records if r['status'] == 'accepted'
                )
                result['flagged'] = sum(
                    1 for r in records if r['status'] == 'flagged'
                )
                result['rejected'] = sum(
                    1 for r in records if r['status'] == 'rejected'
                )

        except Exception as e:
            result['error'] = str(e)
            logger.error(f"Error processing page {page_number}: {e}")

        return result

    def process_full_pdf(self, pdf_path: str) -> Dict[str, Any]:
        """
        Process an entire PDF document through the OCR pipeline.
        
        Returns comprehensive results with per-page and aggregate data.
        """
        total_pages = self.get_pdf_page_count(pdf_path)
        if total_pages == 0:
            return {
                'source_file': pdf_path,
                'total_pages': 0,
                'error': 'Could not read PDF',
                'pages': []
            }

        logger.info(f"Processing PDF: {pdf_path} ({total_pages} pages)")

        pages_results = []
        total_accepted = 0
        total_flagged = 0
        total_rejected = 0

        for page_num in range(total_pages):
            logger.info(f"Processing page {page_num + 1}/{total_pages}")
            try:
                page_result = self.process_pdf_page(pdf_path, page_num)
                pages_results.append(page_result)
                total_accepted += page_result.get('accepted', 0)
                total_flagged += page_result.get('flagged', 0)
                total_rejected += page_result.get('rejected', 0)
            except Exception as e:
                logger.error(f"Error on page {page_num}: {e}")
                pages_results.append({
                    'page_number': page_num,
                    'error': str(e),
                    'records': []
                })

        return {
            'source_file': pdf_path,
            'total_pages': total_pages,
            'pages': pages_results,
            'total_accepted': total_accepted,
            'total_flagged': total_flagged,
            'total_rejected': total_rejected,
            'total_records': total_accepted + total_flagged
        }

    def process_image_file(self, image_path: str) -> Dict[str, Any]:
        """
        Process a single image file (PNG, JPG, JPEG, TIFF, BMP, WEBP) through OCR.
        
        Returns single-page OCR results dictionary.
        """
        result = {
            'page_number': 0,
            'source_file': str(image_path),
            'records': [],
            'month': None,
            'year': None,
            'total_boxes': 0,
            'total_rows': 0,
            'accepted': 0,
            'flagged': 0,
            'rejected': 0,
            'error': None,
            'method': 'ocr'
        }

        try:
            if cv2 is None:
                result['error'] = "OpenCV not available for image processing"
                return result

            image = cv2.imread(str(image_path))
            if image is None:
                result['error'] = f"Failed to load image file from {image_path}"
                return result

            logger.info(f"Loaded image: {image_path} ({image.shape[1]}x{image.shape[0]})")

            # Extract text with bounding boxes
            ocr_boxes = self.extract_text_with_boxes(image)
            result['total_boxes'] = len(ocr_boxes)

            if not ocr_boxes:
                result['error'] = "No text detected by OCR in image"
                return result

            all_text = ' '.join([b['text'] for b in ocr_boxes])
            date_info = self.extract_date_from_text(all_text)
            if date_info:
                result['month'], result['year'] = date_info

            rows = self.group_into_rows(ocr_boxes)
            result['total_rows'] = len(rows)

            records = self.extract_registered_cases(rows)
            result['records'] = records
            result['accepted'] = sum(1 for r in records if r['status'] == 'accepted')
            result['flagged'] = sum(1 for r in records if r['status'] == 'flagged')
            result['rejected'] = sum(1 for r in records if r['status'] == 'rejected')

        except Exception as e:
            result['error'] = str(e)
            logger.error(f"Error processing image {image_path}: {e}")

        return result

    def process_file(self, file_path: str) -> Dict[str, Any]:
        """
        Process any document or image file (PDF, PNG, JPG, JPEG, TIFF, BMP, WEBP).
        """
        ext = Path(file_path).suffix.lower()
        if ext in ('.png', '.jpg', '.jpeg', '.tiff', '.bmp', '.webp'):
            img_res = self.process_image_file(file_path)
            return {
                'source_file': str(file_path),
                'total_pages': 1,
                'pages': [img_res],
                'total_accepted': img_res.get('accepted', 0),
                'total_flagged': img_res.get('flagged', 0),
                'total_rejected': img_res.get('rejected', 0),
                'total_records': img_res.get('accepted', 0) + img_res.get('flagged', 0)
            }
        else:
            return self.process_full_pdf(file_path)
