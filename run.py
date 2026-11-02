"""
Crime Analysis & Forecasting System — CLI Orchestrator
======================================================
Entry point for all system operations.

Usage:
    python run.py --dashboard          Launch Streamlit dashboard
    python run.py --pipeline           Run full OCR → SARIMA pipeline
    python run.py --import-excel FILE  Import initial Excel dataset
    python run.py --test               Run pytest test suite
    python run.py --train              Train SARIMA models
    python run.py --status             Show system status
"""

import argparse
import sys
import os
import logging
from pathlib import Path
from datetime import datetime

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from app.core.config import (
    LOG_DIR, LOG_FORMAT, LOG_DATE_FORMAT, LOG_FILE,
    DASHBOARD_HOST, DASHBOARD_PORT, RAW_DATA_DIR
)


def setup_logging():
    """Configure application logging."""
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format=LOG_FORMAT,
        datefmt=LOG_DATE_FORMAT,
        handlers=[
            logging.FileHandler(LOG_FILE, encoding='utf-8'),
            logging.StreamHandler(sys.stdout)
        ]
    )


def run_dashboard():
    """Launch the Streamlit dashboard."""
    logger = logging.getLogger(__name__)
    logger.info("Launching Streamlit dashboard...")

    dashboard_path = PROJECT_ROOT / "app" / "dashboard" / "dashboard.py"

    os.system(
        f'streamlit run "{dashboard_path}" '
        f'--server.address {DASHBOARD_HOST} '
        f'--server.port {DASHBOARD_PORT} '
        f'--theme.base dark '
        f'--browser.gatherUsageStats false'
    )


def run_pipeline():
    """Run complete OCR → cleaning → SARIMA pipeline."""
    logger = logging.getLogger(__name__)
    logger.info("=" * 60)
    logger.info("STARTING FULL PIPELINE")
    logger.info("=" * 60)

    from app.database.database import CrimeDatabase
    from app.processing.batch_pdf_processor import BatchPDFProcessor
    from app.models.sarima_model import SARIMAModel

    db = CrimeDatabase()
    batch_processor = BatchPDFProcessor(db=db)

    # Step 1: Scan for PDFs
    pdf_files = batch_processor.scan_directory_for_pdfs()

    if pdf_files:
        logger.info(f"Found {len(pdf_files)} PDF files to process")

        # Step 2: Process PDFs
        results = batch_processor.process_multiple_pdfs(pdf_files)
        logger.info(
            f"PDF processing: {results['total_extracted']} extracted, "
            f"{results['total_rejected']} rejected"
        )

        # Step 3: Save accepted records
        all_accepted = results.get('accepted_records', []) + results.get('flagged_records', [])
        if all_accepted:
            save_result = batch_processor.save_accepted_records(all_accepted)
            logger.info(f"Saved: {save_result}")
    else:
        logger.info("No PDF files found in data/raw/")

    # Step 4: Check if database has data
    if db.get_record_count() == 0:
        logger.warning("No data in database. Import data first using --import-excel")
        return

    # Step 5: Train SARIMA models
    logger.info("Training SARIMA models...")
    sarima = SARIMAModel(db=db)
    training_results = sarima.train_all_categories()

    logger.info(
        f"Training complete: {training_results['successful']} successful, "
        f"{training_results['failed']} failed"
    )

    logger.info("=" * 60)
    logger.info("PIPELINE COMPLETE")
    logger.info("=" * 60)


def import_excel(filepath: str, auto_confirm: bool = False):
    """Import initial Excel or CSV dataset."""
    logger = logging.getLogger(__name__)
    logger.info(f"Importing dataset: {filepath}")

    if not os.path.exists(filepath):
        logger.error(f"File not found: {filepath}")
        print(f"ERROR: File not found: {filepath}")
        return

    ext = Path(filepath).suffix.lower()
    if ext not in ('.xlsx', '.xls', '.csv'):
        logger.error(f"Invalid file type: {ext}. Expected .xlsx, .xls, or .csv")
        print(f"ERROR: Invalid file type. Expected .xlsx, .xls, or .csv")
        return

    from app.database.database import CrimeDatabase
    from app.processing.data_processor import DataProcessor
    from app.processing.data_validator import DataValidator

    db = CrimeDatabase()
    processor = DataProcessor()

    # Process Excel/CSV
    result = processor.process_excel_import(filepath)

    if not result['success']:
        logger.error(f"Import failed: {result.get('error')}")
        print(f"ERROR: {result.get('error')}")
        return

    print(f"\n{'=' * 50}")
    print(f"Dataset Import Summary")
    print(f"{'=' * 50}")
    print(f"File: {filepath}")
    print(f"Total rows: {result['total_rows']}")
    print(f"Valid rows: {result['valid_rows']}")
    print(f"Invalid rows: {result['invalid_rows']}")
    print(f"Warnings: {len(result['warnings'])}")

    # Show column mapping
    col_analysis = result.get('column_analysis', {})
    if col_analysis.get('column_mapping'):
        print(f"\nColumn Mapping:")
        for excel_col, req_col in col_analysis['column_mapping'].items():
            print(f"  {excel_col} -> {req_col}")

    if col_analysis.get('unmapped_required'):
        print(f"\nUnmapped required columns: {col_analysis['unmapped_required']}")

    # Show warnings
    if result['warnings']:
        print(f"\nWarnings:")
        for w in result['warnings'][:10]:
            print(f"  [Warning] {w}")

    # Show errors
    if result['errors']:
        print(f"\nErrors:")
        for e in result['errors'][:10]:
            print(f"  [Error] {e}")

    # Confirm import
    valid_df = result.get('valid_df')
    if valid_df is not None and not valid_df.empty:
        print(f"\n{len(valid_df)} valid records ready to import.")
        print(f"\nPreview (first 5 rows):")
        preview_cols = ['crime_category', 'month', 'year', 'case_count',
                        'legal_section', 'domain']
        avail = [c for c in preview_cols if c in valid_df.columns]
        print(valid_df[avail].head().to_string(index=False))

        if auto_confirm:
            response = 'y'
            print("\nAuto-confirming import (-y/--yes specified).")
        else:
            response = input("\nProceed with import? (y/n): ").strip().lower()

        if response == 'y':
            version = db.create_version(
                update_type='excel_import',
                action_summary=f"Initial import from {Path(filepath).name}: {len(valid_df)} records"
            )
            insert_result = db.insert_records(
                valid_df, data_source='excel_import',
                dataset_version=version
            )
            print(f"\nImport complete!")
            print(f"  Inserted: {insert_result['inserted']}")
            print(f"  Duplicates: {insert_result['duplicates']}")
            print(f"  Errors: {insert_result['errors']}")
            print(f"  Dataset Version: v{version}")
            logger.info(f"Import complete: {insert_result}")
        else:
            print("Import cancelled.")
    else:
        print("No valid records to import.")


def run_tests():
    """Run the pytest test suite."""
    import subprocess
    result = subprocess.run(
        [sys.executable, '-m', 'pytest', 'tests/', '-v', '--tb=short'],
        cwd=str(PROJECT_ROOT)
    )
    sys.exit(result.returncode)


def train_sarima():
    """Train SARIMA models for all categories."""
    logger = logging.getLogger(__name__)
    logger.info("Training SARIMA models...")

    from app.database.database import CrimeDatabase
    from app.models.sarima_model import SARIMAModel

    db = CrimeDatabase()

    if db.get_record_count() == 0:
        print("No data in database. Import data first.")
        return

    sarima = SARIMAModel(db=db)
    results = sarima.train_all_categories()

    print(f"\n{'=' * 50}")
    print(f"SARIMA Training Results")
    print(f"{'=' * 50}")
    print(f"Total categories: {results['total_categories']}")
    print(f"Successful: {results['successful']}")
    print(f"Failed: {results['failed']}")
    print(f"Insufficient data: {results['insufficient_data']}")

    for cat, res in results['category_results'].items():
        if res['success']:
            m = res.get('metrics', {})
            rmse_val = m.get('rmse')
            r2_val = m.get('r_squared')
            rmse_str = f"{rmse_val:.2f}" if isinstance(rmse_val, (int, float)) else str(rmse_val)
            r2_str = f"{r2_val:.4f}" if isinstance(r2_val, (int, float)) else str(r2_val)
            print(f"  [OK] {cat}: RMSE={rmse_str}, R2={r2_str}")
        else:
            print(f"  [FAIL] {cat}: {res.get('error', 'Unknown error')}")


def show_status():
    """Show system status."""
    from app.database.database import CrimeDatabase
    from app.models.sarima_model import SARIMAModel

    db = CrimeDatabase()
    stats = db.get_database_stats()
    sarima = SARIMAModel(db=db)
    training_status = sarima.get_training_status()

    print(f"\n{'=' * 50}")
    print(f"Crime Analysis & Forecasting System - Status")
    print(f"{'=' * 50}")
    print(f"\n[Dataset]")
    print(f"  Total Records: {stats['total_records']:,}")
    print(f"  Categories: {stats['total_categories']}")
    print(f"  Year Range: {stats['min_year']} - {stats['max_year']}")
    print(f"  Latest Month: {stats['latest_month']}/{stats['latest_year']}")
    print(f"  Dataset Version: v{stats['current_version']}")
    print(f"  Total Cases: {stats['total_cases']:,}")

    print(f"\n[SARIMA]")
    print(f"  Trained Models: {training_status['total_models']}")
    print(f"  Last Training: {training_status['last_trained']}")

    if training_status['untrained_categories']:
        print(f"  Untrained: {', '.join(training_status['untrained_categories'][:5])}")


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Crime Analysis & Forecasting System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run.py --dashboard                Launch the Streamlit dashboard
  python run.py --import-excel data.xlsx   Import initial Excel dataset
  python run.py --pipeline                 Run full OCR → SARIMA pipeline
  python run.py --train                    Train SARIMA models
  python run.py --test                     Run test suite
  python run.py --status                   Show system status
        """
    )

    parser.add_argument('--dashboard', action='store_true',
                        help='Launch Streamlit dashboard')
    parser.add_argument('--pipeline', action='store_true',
                        help='Run complete pipeline (OCR → SARIMA)')
    parser.add_argument('--import-excel', type=str, metavar='FILE',
                        help='Import initial Excel/CSV dataset')
    parser.add_argument('-y', '--yes', action='store_true',
                        help='Auto-confirm prompts (non-interactive mode)')
    parser.add_argument('--test', action='store_true',
                        help='Run pytest test suite')
    parser.add_argument('--train', action='store_true',
                        help='Train SARIMA models')
    parser.add_argument('--status', action='store_true',
                        help='Show system status')

    args = parser.parse_args()

    setup_logging()

    if args.dashboard:
        run_dashboard()
    elif args.pipeline:
        run_pipeline()
    elif args.import_excel:
        import_excel(args.import_excel, auto_confirm=args.yes)
    elif args.test:
        run_tests()
    elif args.train:
        train_sarima()
    elif args.status:
        show_status()
    else:
        parser.print_help()


if __name__ == '__main__':
    main()
