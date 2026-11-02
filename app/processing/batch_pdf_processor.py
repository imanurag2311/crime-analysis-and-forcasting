"""
Batch PDF Processor
====================
Processes multiple Mumbai Police PDF reports through the OCR pipeline.
Handles multi-page/multi-month PDFs, deduplication, and database insertion.
"""

import logging
from pathlib import Path
from typing import List, Dict, Any, Optional

import pandas as pd

from app.core.config import PRIMARY_COLUMNS, RAW_DATA_DIR
from app.ocr.ocr_engine import OCREngine
from app.processing.ipc_bns_crosswalk import IPCBNSCrosswalk
from app.processing.data_validator import DataValidator
from app.processing.data_processor import DataProcessor
from app.database.database import CrimeDatabase

logger = logging.getLogger(__name__)


class BatchPDFProcessor:
    """
    Processes multiple PDF files through the complete OCR pipeline.
    
    Responsibilities:
        - Process multiple PDF files
        - Process multiple pages
        - Detect month/year
        - Extract registered cases
        - Filter invalid rows
        - Apply crosswalk
        - Validate data
        - Deduplicate
        - Append valid records to master dataset
    """

    def __init__(self, db: Optional[CrimeDatabase] = None):
        """Initialize with OCR engine, crosswalk, validator, and database."""
        self.ocr_engine = OCREngine()
        self.crosswalk = IPCBNSCrosswalk()
        self.validator = DataValidator()
        self.processor = DataProcessor()
        self.db = db or CrimeDatabase()

    def process_single_pdf(self, pdf_path: str,
                            default_month: Optional[int] = None,
                            default_year: Optional[int] = None) -> Dict[str, Any]:
        """
        Process a single PDF through the complete pipeline.
        
        Args:
            pdf_path: Path to PDF file
            default_month: Fallback month if not detected from header
            default_year: Fallback year if not detected from header
            
        Returns:
            Processing result with extracted records and metadata
        """
        logger.info(f"Processing PDF: {pdf_path}")

        result = {
            'source_file': pdf_path,
            'total_pages': 0,
            'total_extracted': 0,
            'accepted_records': [],
            'flagged_records': [],
            'rejected_records': [],
            'errors': [],
            'processing_logs': []
        }

        # Run OCR pipeline (supports both PDFs and images)
        ocr_result = self.ocr_engine.process_file(pdf_path)

        if 'error' in ocr_result and ocr_result.get('total_pages', 0) == 0:
            result['errors'].append(ocr_result['error'])
            logger.error(f"Failed to process file: {ocr_result.get('error')}")
            return result

        result['total_pages'] = ocr_result.get('total_pages', 0)

        # Process each page's results
        for page_data in ocr_result.get('pages', []):
            page_num = page_data.get('page_number', 0)

            if page_data.get('error'):
                result['errors'].append(
                    f"Page {page_num}: {page_data['error']}"
                )
                self.db.log_processing(
                    source_file=pdf_path,
                    page_number=page_num,
                    rows_detected=0, rows_accepted=0, rows_rejected=0,
                    processing_type='ocr',
                    status='error',
                    error_message=page_data['error']
                )
                continue

            # Determine month/year for this page
            page_month = page_data.get('month') or default_month
            page_year = page_data.get('year') or default_year

            records = page_data.get('records', [])
            page_accepted = 0
            page_rejected = 0
            rejection_reasons = []

            for record in records:
                # Apply crosswalk and standardization
                category = self.crosswalk.standardize_category(
                    record.get('crime_category', '')
                )
                legal_section = record.get('legal_section', '')
                crosswalk_data = self.crosswalk.apply_crosswalk(legal_section)

                processed_record = {
                    'crime_category': category,
                    'month': page_month,
                    'year': page_year,
                    'case_count': record.get('case_count', 0),
                    'legal_section': legal_section,
                    'page': page_num,
                    'source_file': pdf_path,
                    'standard_group': crosswalk_data.get('standard_group', ''),
                    'domain': crosswalk_data.get('domain', ''),
                    'legal_section_bns': crosswalk_data.get('legal_section_bns', ''),
                    'ocr_confidence': record.get('confidence', 0),
                    'raw_text': record.get('raw_text', ''),
                    'validation_status': record.get('status', 'pending')
                }

                status = record.get('status', 'pending')

                if status == 'rejected':
                    result['rejected_records'].append(processed_record)
                    page_rejected += 1
                    reason = record.get('rejection_reason', 'Unknown')
                    rejection_reasons.append(reason)
                elif status == 'flagged':
                    result['flagged_records'].append(processed_record)
                    page_accepted += 1  # Flagged are still counted
                else:
                    result['accepted_records'].append(processed_record)
                    page_accepted += 1

            # Log processing
            self.db.log_processing(
                source_file=pdf_path,
                page_number=page_num,
                rows_detected=len(records),
                rows_accepted=page_accepted,
                rows_rejected=page_rejected,
                rejection_reasons='; '.join(rejection_reasons),
                processing_type='ocr',
                status='completed'
            )

            result['processing_logs'].append({
                'page': page_num,
                'month': page_month,
                'year': page_year,
                'rows_detected': len(records),
                'accepted': page_accepted,
                'rejected': page_rejected
            })

        result['total_extracted'] = (
            len(result['accepted_records']) +
            len(result['flagged_records'])
        )

        logger.info(
            f"PDF processing complete: {result['total_extracted']} extracted, "
            f"{len(result['rejected_records'])} rejected"
        )

        return result

    def process_multiple_pdfs(self, pdf_paths: List[str],
                               default_month: Optional[int] = None,
                               default_year: Optional[int] = None) -> Dict[str, Any]:
        """
        Process multiple PDF files.
        
        Returns aggregated results across all PDFs.
        """
        all_results = {
            'total_files': len(pdf_paths),
            'total_pages': 0,
            'total_extracted': 0,
            'total_rejected': 0,
            'accepted_records': [],
            'flagged_records': [],
            'rejected_records': [],
            'file_results': [],
            'errors': []
        }

        for pdf_path in pdf_paths:
            try:
                result = self.process_single_pdf(
                    pdf_path, default_month, default_year
                )
                all_results['file_results'].append(result)
                all_results['total_pages'] += result.get('total_pages', 0)
                all_results['total_extracted'] += result.get('total_extracted', 0)
                all_results['total_rejected'] += len(result.get('rejected_records', []))
                all_results['accepted_records'].extend(result.get('accepted_records', []))
                all_results['flagged_records'].extend(result.get('flagged_records', []))
                all_results['rejected_records'].extend(result.get('rejected_records', []))
                all_results['errors'].extend(result.get('errors', []))
            except Exception as e:
                logger.error(f"Error processing {pdf_path}: {e}")
                all_results['errors'].append(f"{pdf_path}: {str(e)}")

        return all_results

    def save_accepted_records(self, records: List[Dict],
                               data_source: str = "ocr") -> Dict[str, int]:
        """
        Save accepted records to the database.
        
        Validates and deduplicates before insertion.
        """
        if not records:
            return {"inserted": 0, "duplicates": 0, "errors": 0}

        df = pd.DataFrame(records)

        # Ensure all primary columns
        for col in PRIMARY_COLUMNS:
            if col not in df.columns:
                df[col] = ''

        # Validate
        validation = self.validator.validate_dataframe(df)
        valid_df = validation.get('valid_df', pd.DataFrame())

        if valid_df.empty:
            logger.warning("No valid records to save after validation")
            return {"inserted": 0, "duplicates": 0, "errors": len(records)}

        # Create new version
        version = self.db.create_version(
            update_type='ocr_import',
            action_summary=f"OCR import: {len(valid_df)} records"
        )

        # Insert
        result = self.db.insert_records(valid_df, data_source=data_source,
                                         dataset_version=version)

        # Save OCR audit records
        ocr_audit = []
        for _, row in df.iterrows():
            ocr_audit.append({
                'source_file': row.get('source_file', ''),
                'page_number': row.get('page'),
                'raw_text': row.get('raw_text', ''),
                'parsed_category': row.get('crime_category', ''),
                'parsed_count': row.get('case_count', 0),
                'confidence': row.get('ocr_confidence'),
                'status': row.get('validation_status', 'accepted')
            })
        self.db.save_ocr_records(ocr_audit)

        return result

    def scan_directory_for_pdfs(self, directory: Optional[str] = None) -> List[str]:
        """Scan a directory for PDF files."""
        scan_dir = Path(directory) if directory else RAW_DATA_DIR
        pdf_files = sorted(scan_dir.glob("*.pdf"))
        logger.info(f"Found {len(pdf_files)} PDF files in {scan_dir}")
        return [str(f) for f in pdf_files]
