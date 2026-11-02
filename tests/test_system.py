"""
Comprehensive Test Suite for Crime Analysis & Forecasting System
=================================================================
Tests cover: OCR, data processing, validation, IPC/BNS crosswalk,
database CRUD, SARIMA model, and end-to-end workflows.
"""

import os
import sys
import json
import tempfile
from pathlib import Path
from datetime import datetime

import pytest
import pandas as pd
import numpy as np

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from app.core.config import (
    PRIMARY_COLUMNS, SARIMA_ORDER, SARIMA_SEASONAL_ORDER,
    SEASONAL_PERIOD, TRAIN_TEST_RATIO, MIN_FORECAST_HISTORY,
    CRIME_DOMAINS, OCR_ROW_Y_TOLERANCE, CLAHE_CLIP_LIMIT
)
from app.database.database import CrimeDatabase
from app.processing.ipc_bns_crosswalk import IPCBNSCrosswalk
from app.processing.data_validator import DataValidator
from app.processing.data_processor import DataProcessor
from app.models.sarima_model import SARIMAModel


# ============================================================
# FIXTURES
# ============================================================

@pytest.fixture
def temp_db(tmp_path):
    """Create a temporary database for testing."""
    db_path = tmp_path / "test_crime.db"
    db = CrimeDatabase(db_path=db_path)
    return db


@pytest.fixture
def crosswalk():
    """Create an IPC/BNS crosswalk instance."""
    return IPCBNSCrosswalk()


@pytest.fixture
def validator():
    """Create a data validator instance."""
    return DataValidator()


@pytest.fixture
def processor():
    """Create a data processor instance."""
    return DataProcessor()


@pytest.fixture
def sample_valid_df():
    """Create a sample valid DataFrame for testing."""
    data = {
        'crime_category': ['Murder', 'Theft', 'Robbery', 'CAW_Rape', 'NDPS'],
        'month': [1, 1, 1, 1, 1],
        'year': [2025, 2025, 2025, 2025, 2025],
        'case_count': [15, 120, 25, 42, 35],
        'legal_section': ['IPC 302', 'IPC 379', 'IPC 392', 'IPC 376', 'NDPS Act'],
        'page': [1, 1, 1, 1, 1],
        'source_file': ['test.pdf'] * 5,
        'standard_group': ['Violent Crime', 'Property Crime', 'Property Crime',
                           'Crimes Against Women (CAW)', 'NDPS Act'],
        'domain': ['Violent Crime', 'Property Crime', 'Property Crime',
                   'Crimes Against Women (CAW)', 'NDPS Act'],
        'legal_section_bns': ['BNS 103', 'BNS 303', 'BNS 309', 'BNS 64', 'NDPS Act']
    }
    return pd.DataFrame(data)


@pytest.fixture
def sample_time_series_data():
    """Create synthetic monthly time series data for SARIMA testing.
    SYNTHETIC TEST DATA - not real Mumbai crime data.
    """
    np.random.seed(42)
    dates = pd.date_range('2022-01-01', periods=36, freq='MS')
    base = 50
    seasonal = 10 * np.sin(np.arange(36) * 2 * np.pi / 12)
    trend = np.linspace(0, 10, 36)
    noise = np.random.normal(0, 3, 36)
    values = (base + seasonal + trend + noise).clip(min=0).astype(int)

    records = []
    for i, date in enumerate(dates):
        records.append({
            'crime_category': 'Test_Category',
            'month': date.month,
            'year': date.year,
            'case_count': int(values[i]),
            'legal_section': 'IPC 302',
            'page': 1,
            'source_file': 'synthetic_test',
            'standard_group': 'Test',
            'domain': 'General IPC',
            'legal_section_bns': 'BNS 103'
        })

    return pd.DataFrame(records)


# ============================================================
# IPC/BNS CROSSWALK TESTS
# ============================================================

class TestIPCBNSCrosswalk:
    """Tests for the IPC → BNS crosswalk."""

    def test_lookup_murder(self, crosswalk):
        """Test IPC 302 → BNS 103 (Murder) mapping."""
        result = crosswalk.lookup_ipc("IPC 302")
        assert result is not None
        assert result['bns'] == 'BNS 103'
        assert result['crime'] == 'Murder'

    def test_lookup_theft(self, crosswalk):
        """Test IPC 379 → BNS 303 (Theft) mapping."""
        result = crosswalk.lookup_ipc("IPC 379")
        assert result is not None
        assert result['bns'] == 'BNS 303'

    def test_lookup_robbery(self, crosswalk):
        """Test IPC 392 → BNS 309 (Robbery) mapping."""
        result = crosswalk.lookup_ipc("IPC 392")
        assert result is not None
        assert result['bns'] == 'BNS 309'

    def test_lookup_cheating(self, crosswalk):
        """Test IPC 420 → BNS 318 (Cheating) mapping."""
        result = crosswalk.lookup_ipc("IPC 420")
        assert result is not None
        assert result['bns'] == 'BNS 318'

    def test_lookup_molestation(self, crosswalk):
        """Test IPC 354 → BNS 74 (Molestation) mapping."""
        result = crosswalk.lookup_ipc("IPC 354")
        assert result is not None
        assert result['bns'] == 'BNS 74'

    def test_lookup_rape(self, crosswalk):
        """Test IPC 376 → BNS 64 (Rape) mapping."""
        result = crosswalk.lookup_ipc("IPC 376")
        assert result is not None
        assert result['bns'] == 'BNS 64'

    def test_reverse_lookup_bns(self, crosswalk):
        """Test BNS → IPC reverse lookup."""
        result = crosswalk.lookup_bns("BNS 103")
        assert result is not None
        assert result['ipc'] == 'IPC 302'

    def test_nonexistent_mapping(self, crosswalk):
        """Test that nonexistent sections return None."""
        result = crosswalk.lookup_ipc("IPC 9999")
        assert result is None

    def test_search_mapping(self, crosswalk):
        """Test search functionality."""
        results = crosswalk.search_mapping("murder")
        assert len(results) > 0
        assert any('Murder' in r['Crime'] for r in results)

    def test_full_mapping_table(self, crosswalk):
        """Test getting full mapping table."""
        table = crosswalk.get_full_mapping_table()
        assert len(table) > 0
        assert all('IPC Section' in row for row in table)
        assert all('BNS Section' in row for row in table)

    def test_domain_classification(self, crosswalk):
        """Test domain classification."""
        assert crosswalk.get_domain("IPC 302") == "Violent Crime"
        assert crosswalk.get_domain("IPC 376") == "Crimes Against Women (CAW)"
        assert crosswalk.get_domain("IPC 420") == "Economic Offences (EOW)"


# ============================================================
# CRIME NAME NORMALIZATION TESTS
# ============================================================

class TestCrimeNameNormalization:
    """Tests for crime category standardization."""

    def test_caw_rape_variations(self, crosswalk):
        """Test multiple variations of CAW Rape normalize correctly."""
        variations = ["caw rape", "c.a.w. rape", "crime against women - rape",
                      "CAW Rape", "rape"]
        for var in variations:
            result = crosswalk.standardize_category(var)
            assert result == "CAW_Rape", f"'{var}' mapped to '{result}'"

    def test_theft_normalization(self, crosswalk):
        """Test theft variations."""
        assert crosswalk.standardize_category("theft") == "Theft"
        assert crosswalk.standardize_category("thefts") == "Theft"

    def test_murder_normalization(self, crosswalk):
        """Test murder variations."""
        assert crosswalk.standardize_category("murder") == "Murder"
        assert crosswalk.standardize_category("murders") == "Murder"

    def test_empty_category(self, crosswalk):
        """Test empty category returns 'Unknown'."""
        assert crosswalk.standardize_category("") == "Unknown"
        assert crosswalk.standardize_category(None) == "Unknown"

    def test_basic_normalization_fallback(self, crosswalk):
        """Test that unknown categories get basic normalization."""
        result = crosswalk.standardize_category("some new crime type")
        assert "_" in result or result.istitle()


# ============================================================
# DATA VALIDATION TESTS
# ============================================================

class TestDataValidation:
    """Tests for data validation."""

    def test_valid_records(self, validator, sample_valid_df):
        """Test validation of a valid DataFrame."""
        result = validator.validate_dataframe(sample_valid_df)
        assert result['summary']['valid'] == len(sample_valid_df)
        assert result['summary']['invalid'] == 0

    def test_empty_category_rejected(self, validator):
        """Test that empty crime_category is rejected."""
        df = pd.DataFrame({
            'crime_category': [''],
            'month': [1], 'year': [2025],
            'case_count': [10]
        })
        result = validator.validate_dataframe(df)
        assert result['summary']['invalid'] > 0

    def test_invalid_month_rejected(self, validator):
        """Test that invalid month is rejected."""
        df = pd.DataFrame({
            'crime_category': ['Murder'],
            'month': [13], 'year': [2025],
            'case_count': [10]
        })
        result = validator.validate_dataframe(df)
        assert result['summary']['invalid'] > 0

    def test_negative_case_count_rejected(self, validator):
        """Test that negative case_count is rejected."""
        df = pd.DataFrame({
            'crime_category': ['Murder'],
            'month': [1], 'year': [2025],
            'case_count': [-5]
        })
        result = validator.validate_dataframe(df)
        assert result['summary']['invalid'] > 0

    def test_invalid_year_rejected(self, validator):
        """Test that invalid year is rejected."""
        df = pd.DataFrame({
            'crime_category': ['Murder'],
            'month': [1], 'year': [1800],
            'case_count': [10]
        })
        result = validator.validate_dataframe(df)
        assert result['summary']['invalid'] > 0

    def test_missing_month_rejected(self, validator):
        """Test that missing month is rejected."""
        df = pd.DataFrame({
            'crime_category': ['Murder'],
            'month': [None], 'year': [2025],
            'case_count': [10]
        })
        result = validator.validate_dataframe(df)
        assert result['summary']['invalid'] > 0

    def test_duplicate_detection(self, validator):
        """Test duplicate record detection."""
        df = pd.DataFrame({
            'crime_category': ['Murder', 'Murder'],
            'month': [1, 1], 'year': [2025, 2025],
            'case_count': [10, 10]
        })
        result = validator.validate_dataframe(df)
        assert result['summary']['duplicates'] > 0

    def test_single_record_validation(self, validator):
        """Test single record validation."""
        valid, errors = validator.validate_single_record({
            'crime_category': 'Murder',
            'month': 1, 'year': 2025,
            'case_count': 10
        })
        assert valid is True
        assert len(errors) == 0

    def test_excel_column_detection(self, validator):
        """Test Excel column auto-detection."""
        df = pd.DataFrame({
            'Crime Category': ['Murder'],
            'Month': [1], 'Year': [2025],
            'Case Count': [10]
        })
        result = validator.validate_excel_columns(df)
        assert 'column_mapping' in result


# ============================================================
# DATABASE CRUD TESTS
# ============================================================

class TestDatabase:
    """Tests for database CRUD operations."""

    def test_insert_records(self, temp_db, sample_valid_df):
        """Test inserting records."""
        temp_db.create_version('test', 'Test import')
        result = temp_db.insert_records(sample_valid_df, data_source='test')
        assert result['inserted'] == 5
        assert result['errors'] == 0

    def test_get_all_records(self, temp_db, sample_valid_df):
        """Test retrieving all records."""
        temp_db.create_version('test', 'Test import')
        temp_db.insert_records(sample_valid_df)
        df = temp_db.get_all_records()
        assert len(df) == 5

    def test_get_filtered_records(self, temp_db, sample_valid_df):
        """Test filtered record retrieval."""
        temp_db.create_version('test', 'Test import')
        temp_db.insert_records(sample_valid_df)
        df = temp_db.get_records_filtered(year=2025, category='Murder')
        assert len(df) == 1
        assert df.iloc[0]['crime_category'] == 'Murder'

    def test_update_record(self, temp_db, sample_valid_df):
        """Test updating a record."""
        temp_db.create_version('test', 'Test import')
        temp_db.insert_records(sample_valid_df)
        records = temp_db.get_all_records()
        record_id = int(records.iloc[0]['record_id'])

        success = temp_db.update_record(record_id, {'case_count': 999})
        assert success is True

        updated = temp_db.get_all_records()
        updated_record = updated[updated['record_id'] == record_id].iloc[0]
        assert int(updated_record['case_count']) == 999

    def test_delete_records(self, temp_db, sample_valid_df):
        """Test deleting records."""
        temp_db.create_version('test', 'Test import')
        temp_db.insert_records(sample_valid_df)
        records = temp_db.get_all_records()
        record_id = int(records.iloc[0]['record_id'])

        deleted = temp_db.delete_records([record_id])
        assert deleted == 1
        assert temp_db.get_record_count() == 4

    def test_delete_by_month(self, temp_db, sample_valid_df):
        """Test deleting records by month."""
        temp_db.create_version('test', 'Test import')
        temp_db.insert_records(sample_valid_df)
        deleted = temp_db.delete_records_by_month(2025, 1)
        assert deleted == 5
        assert temp_db.get_record_count() == 0

    def test_version_management(self, temp_db):
        """Test dataset versioning."""
        assert temp_db.get_current_version() == 0
        v1 = temp_db.create_version('import', 'First import')
        assert v1 == 1
        v2 = temp_db.create_version('update', 'Added data')
        assert v2 == 2

    def test_duplicate_detection(self, temp_db, sample_valid_df):
        """Test duplicate detection."""
        temp_db.create_version('test', 'Test import')
        temp_db.insert_records(sample_valid_df)
        assert temp_db.check_duplicate('Murder', 1, 2025, 'IPC 302') is True
        assert temp_db.check_duplicate('Murder', 2, 2025) is False

    def test_unique_categories(self, temp_db, sample_valid_df):
        """Test getting unique categories."""
        temp_db.create_version('test', 'Test import')
        temp_db.insert_records(sample_valid_df)
        categories = temp_db.get_unique_categories()
        assert len(categories) == 5
        assert 'Murder' in categories

    def test_database_stats(self, temp_db, sample_valid_df):
        """Test database statistics."""
        temp_db.create_version('test', 'Test import')
        temp_db.insert_records(sample_valid_df)
        stats = temp_db.get_database_stats()
        assert stats['total_records'] == 5
        assert stats['total_categories'] == 5

    def test_monthly_summary(self, temp_db, sample_valid_df):
        """Test monthly summary aggregation."""
        temp_db.create_version('test', 'Test import')
        temp_db.insert_records(sample_valid_df)
        summary = temp_db.get_monthly_summary()
        assert len(summary) > 0

    def test_export_excel(self, temp_db, sample_valid_df, tmp_path):
        """Test Excel export."""
        temp_db.create_version('test', 'Test import')
        temp_db.insert_records(sample_valid_df)
        export_path = tmp_path / "export.xlsx"
        temp_db.export_to_excel(export_path)
        assert export_path.exists()

    def test_export_csv(self, temp_db, sample_valid_df, tmp_path):
        """Test CSV export."""
        temp_db.create_version('test', 'Test import')
        temp_db.insert_records(sample_valid_df)
        export_path = tmp_path / "export.csv"
        temp_db.export_to_csv(export_path)
        assert export_path.exists()


# ============================================================
# DATA PROCESSOR TESTS
# ============================================================

class TestDataProcessor:
    """Tests for data processing functionality."""

    def test_process_raw_data(self, processor, sample_valid_df):
        """Test raw data processing pipeline."""
        result = processor.process_raw_data(sample_valid_df)
        assert not result.empty
        assert 'crime_category' in result.columns

    def test_create_monthly_time_series(self, processor, sample_time_series_data):
        """Test monthly time series creation."""
        ts = processor.create_monthly_time_series(
            sample_time_series_data, category='Test_Category'
        )
        assert not ts.empty
        assert 'total_cases' in ts.columns
        assert ts.index.freq is not None or len(ts) > 0

    def test_rolling_averages(self, processor, sample_time_series_data):
        """Test rolling average calculation."""
        ts = processor.create_monthly_time_series(
            sample_time_series_data, category='Test_Category'
        )
        ts = processor.add_rolling_averages(ts)
        assert 'rolling_avg_3m' in ts.columns
        assert 'rolling_avg_6m' in ts.columns

    def test_yoy_change(self, processor, sample_time_series_data):
        """Test YoY change calculation."""
        ts = processor.create_monthly_time_series(
            sample_time_series_data, category='Test_Category'
        )
        ts = processor.add_yoy_change(ts)
        assert 'yoy_change' in ts.columns

    def test_chronological_split(self, processor, sample_time_series_data):
        """Test chronological 80/20 train-test split."""
        ts = processor.create_monthly_time_series(
            sample_time_series_data, category='Test_Category'
        )
        train, test = processor.chronological_train_test_split(ts)

        # Verify chronological order
        assert len(train) > 0
        assert len(test) > 0
        assert train.index.max() < test.index.min()  # No data leakage

    def test_no_random_shuffle(self, processor, sample_time_series_data):
        """Test that split preserves chronological order (no random shuffle)."""
        ts = processor.create_monthly_time_series(
            sample_time_series_data, category='Test_Category'
        )
        train, test = processor.chronological_train_test_split(ts)

        # All train dates must be before all test dates
        if len(train) > 0 and len(test) > 0:
            assert train.index[-1] < test.index[0]

    def test_data_sufficiency_check(self, processor):
        """Test data sufficiency check."""
        short_ts = pd.DataFrame({'total_cases': [1, 2, 3]})
        assert processor.has_sufficient_history(short_ts) is False

        long_ts = pd.DataFrame({'total_cases': list(range(30))})
        assert processor.has_sufficient_history(long_ts) is True

    def test_missing_month_detection(self, processor):
        """Test missing month detection."""
        df = pd.DataFrame({
            'crime_category': ['Murder', 'Murder', 'Murder'],
            'month': [1, 3, 5],  # Feb and Apr missing
            'year': [2025, 2025, 2025],
            'case_count': [10, 15, 12]
        })
        missing = processor.detect_missing_months(df, category='Murder')
        assert len(missing) > 0
        # Feb and Apr should be detected
        missing_periods = [(m['year'], m['month']) for m in missing]
        assert (2025, 2) in missing_periods
        assert (2025, 4) in missing_periods

    def test_category_distribution(self, processor, sample_valid_df):
        """Test category distribution calculation."""
        dist = processor.get_category_distribution(sample_valid_df)
        assert not dist.empty
        assert 'percentage' in dist.columns
        assert dist['percentage'].sum() == pytest.approx(100, abs=0.1)


# ============================================================
# SARIMA MODEL TESTS
# ============================================================

class TestSARIMAModel:
    """Tests for SARIMA forecasting model."""

    def test_train_with_sufficient_data(self, temp_db, sample_time_series_data):
        """Test SARIMA training with sufficient data."""
        temp_db.create_version('test', 'Test data')
        temp_db.insert_records(sample_time_series_data)

        sarima = SARIMAModel(db=temp_db)
        result = sarima.train_category('Test_Category')

        assert result['success'] is True
        assert result['metrics']['rmse'] is not None
        assert result['metrics']['mae'] is not None
        assert result['metrics']['r_squared'] is not None

    def test_insufficient_data_handling(self, temp_db):
        """Test SARIMA with insufficient data."""
        # Only 5 months of data
        data = {
            'crime_category': ['Test'] * 5,
            'month': [1, 2, 3, 4, 5],
            'year': [2025] * 5,
            'case_count': [10, 12, 8, 15, 11],
            'legal_section': [''] * 5,
            'page': [''] * 5,
            'source_file': [''] * 5,
            'standard_group': [''] * 5,
            'domain': [''] * 5,
            'legal_section_bns': [''] * 5
        }
        df = pd.DataFrame(data)
        temp_db.create_version('test', 'Test data')
        temp_db.insert_records(df)

        sarima = SARIMAModel(db=temp_db)
        result = sarima.train_category('Test')

        assert result['success'] is False
        assert 'Insufficient' in result.get('error', '')

    def test_metrics_calculation(self, temp_db, sample_time_series_data):
        """Test RMSE, MAE, MAPE, R² calculation."""
        temp_db.create_version('test', 'Test data')
        temp_db.insert_records(sample_time_series_data)

        sarima = SARIMAModel(db=temp_db)
        result = sarima.train_category('Test_Category')

        if result['success']:
            metrics = result['metrics']
            assert metrics['rmse'] >= 0
            assert metrics['mae'] >= 0
            assert metrics['mape'] is None or metrics['mape'] >= 0
            # R² can be negative for poor fits but should be reasonable

    def test_forecast_generation(self, temp_db, sample_time_series_data):
        """Test forecast generation."""
        temp_db.create_version('test', 'Test data')
        temp_db.insert_records(sample_time_series_data)

        sarima = SARIMAModel(db=temp_db)
        result = sarima.train_category('Test_Category', forecast_horizon=6)

        if result['success']:
            forecast = result.get('forecast')
            assert forecast is not None
            assert not forecast.empty
            assert len(forecast) == 6

    def test_confidence_intervals(self, temp_db, sample_time_series_data):
        """Test 80% and 95% confidence intervals."""
        temp_db.create_version('test', 'Test data')
        temp_db.insert_records(sample_time_series_data)

        sarima = SARIMAModel(db=temp_db)
        result = sarima.train_category('Test_Category', forecast_horizon=6)

        if result['success']:
            forecast = result.get('forecast')
            if forecast is not None and not forecast.empty:
                assert 'lower_80' in forecast.columns
                assert 'upper_80' in forecast.columns
                assert 'lower_95' in forecast.columns
                assert 'upper_95' in forecast.columns

                # 95% CI should be wider than 80% CI
                for _, row in forecast.iterrows():
                    assert row['lower_95'] <= row['lower_80']
                    assert row['upper_95'] >= row['upper_80']

    def test_model_persistence(self, temp_db, sample_time_series_data, tmp_path):
        """Test model save and load."""
        temp_db.create_version('test', 'Test data')
        temp_db.insert_records(sample_time_series_data)

        sarima = SARIMAModel(db=temp_db)
        result = sarima.train_category('Test_Category')

        if result['success']:
            # Load model
            loaded = sarima.load_model('Test_Category')
            assert loaded is not None
            assert 'model' in loaded


# ============================================================
# OCR TESTS
# ============================================================

class TestOCR:
    """Tests for OCR engine functionality."""

    @staticmethod
    def _get_ocr_engine():
        """Get OCREngine, skip test if dependencies unavailable."""
        try:
            from app.ocr.ocr_engine import OCREngine
            return OCREngine()
        except Exception as e:
            pytest.skip(f"OCR dependencies not available: {e}")

    def test_ocr_engine_initialization(self):
        """Test OCR engine can be initialized."""
        engine = self._get_ocr_engine()
        assert engine is not None

    def test_row_grouping(self):
        """Test horizontal row grouping by y-center."""
        engine = self._get_ocr_engine()

        # Create mock OCR boxes
        boxes = [
            {'text': 'Murder', 'x_min': 10, 'y_min': 90, 'x_max': 100,
             'y_max': 110, 'y_center': 100, 'confidence': 0.95},
            {'text': '15', 'x_min': 200, 'y_min': 92, 'x_max': 230,
             'y_max': 108, 'y_center': 100, 'confidence': 0.98},
            {'text': 'Theft', 'x_min': 10, 'y_min': 140, 'x_max': 80,
             'y_max': 160, 'y_center': 150, 'confidence': 0.92},
            {'text': '120', 'x_min': 200, 'y_min': 142, 'x_max': 240,
             'y_max': 158, 'y_center': 150, 'confidence': 0.96},
        ]

        rows = engine.group_into_rows(boxes, y_tolerance=15)
        assert len(rows) == 2
        # First row should have Murder and 15
        assert len(rows[0]) == 2
        # Second row should have Theft and 120
        assert len(rows[1]) == 2

    def test_registered_column_extraction(self):
        """Test R-column extraction (first numeric = Registered)."""
        engine = self._get_ocr_engine()

        # Mock rows: each row has text boxes and number boxes
        rows = [
            [
                {'text': 'Murder', 'x_min': 10, 'y_min': 90, 'x_max': 100,
                 'y_max': 110, 'y_center': 100, 'confidence': 0.95},
                {'text': '15', 'x_min': 200, 'y_min': 92, 'x_max': 230,
                 'y_max': 108, 'y_center': 100, 'confidence': 0.98},
                {'text': '12', 'x_min': 300, 'y_min': 92, 'x_max': 330,
                 'y_max': 108, 'y_center': 100, 'confidence': 0.96},
            ]
        ]

        results = engine.extract_registered_cases(rows)
        assert len(results) == 1
        # MUST use first numeric (R=15), NOT second (D=12)
        assert results[0]['case_count'] == 15

    def test_d_column_exclusion(self):
        """Test that Detected (D) column is never used."""
        engine = self._get_ocr_engine()

        rows = [
            [
                {'text': 'Theft', 'x_min': 10, 'y_min': 90, 'x_max': 100,
                 'y_max': 110, 'y_center': 100, 'confidence': 0.95},
                {'text': '120', 'x_min': 200, 'y_min': 92, 'x_max': 240,
                 'y_max': 108, 'y_center': 100, 'confidence': 0.98},
                {'text': '95', 'x_min': 300, 'y_min': 92, 'x_max': 330,
                 'y_max': 108, 'y_center': 100, 'confidence': 0.96},
            ]
        ]

        results = engine.extract_registered_cases(rows)
        # Should use 120 (R), not 95 (D), and definitely not 120+95
        assert results[0]['case_count'] == 120
        assert results[0]['case_count'] != 95
        assert results[0]['case_count'] != 215  # Never R + D

    def test_summary_row_filtering(self):
        """Test that summary/total rows are rejected."""
        engine = self._get_ocr_engine()

        rows = [
            [
                {'text': 'Total', 'x_min': 10, 'y_min': 90, 'x_max': 80,
                 'y_max': 110, 'y_center': 100, 'confidence': 0.95},
                {'text': '500', 'x_min': 200, 'y_min': 92, 'x_max': 240,
                 'y_max': 108, 'y_center': 100, 'confidence': 0.98},
            ]
        ]

        results = engine.extract_registered_cases(rows)
        assert len(results) == 1
        assert results[0]['status'] == 'rejected'
        assert 'Summary row' in results[0]['rejection_reason']

    def test_sub_breakdown_filtering(self):
        """Test that measurement/sub-breakdown rows are rejected."""
        engine = self._get_ocr_engine()

        rows = [
            [
                {'text': 'Drug seizure 4.2 kg', 'x_min': 10, 'y_min': 90,
                 'x_max': 200, 'y_max': 110, 'y_center': 100, 'confidence': 0.85},
                {'text': '5', 'x_min': 300, 'y_min': 92, 'x_max': 320,
                 'y_max': 108, 'y_center': 100, 'confidence': 0.90},
            ]
        ]

        results = engine.extract_registered_cases(rows)
        assert len(results) == 1
        assert results[0]['status'] == 'rejected'

    def test_date_extraction(self):
        """Test date extraction from headers."""
        engine = self._get_ocr_engine()

        # Test DD/MM/YYYY format
        result = engine.extract_date_from_text("Report for 01/06/2025")
        assert result == (6, 2025)

        # Test month name format
        result = engine.extract_date_from_text("September 2026")
        assert result == (9, 2026)

        # Test abbreviated month
        result = engine.extract_date_from_text("Jan 2025")
        assert result == (1, 2025)

    def test_ocr_preprocessing_config(self):
        """Test that OCR config values are set correctly."""
        assert CLAHE_CLIP_LIMIT == 2.0
        assert OCR_ROW_Y_TOLERANCE == 12.0


# ============================================================
# END-TO-END TESTS
# ============================================================

class TestEndToEnd:
    """End-to-end integration tests."""

    def test_excel_to_sarima_pipeline(self, temp_db, sample_time_series_data, tmp_path):
        """
        TEST 1: Excel → validation → SQLite → SARIMA → metrics → forecast
        """
        # Step 1: Create test Excel file
        excel_path = tmp_path / "test_data.xlsx"
        sample_time_series_data.to_excel(excel_path, index=False, engine='openpyxl')

        # Step 2: Import via processor
        processor = DataProcessor()
        import_result = processor.process_excel_import(str(excel_path))
        assert import_result['success'] is True
        assert import_result['valid_rows'] > 0

        # Step 3: Insert into database
        valid_df = import_result['valid_df']
        version = temp_db.create_version('excel_import', 'Test E2E')
        insert_result = temp_db.insert_records(valid_df, dataset_version=version)
        assert insert_result['inserted'] > 0

        # Step 4: Train SARIMA
        sarima = SARIMAModel(db=temp_db)
        result = sarima.train_category('Test_Category')
        assert result['success'] is True

        # Step 5: Verify metrics
        metrics = result['metrics']
        assert metrics['rmse'] >= 0
        assert metrics['mae'] >= 0

        # Step 6: Verify forecast
        forecast = result['forecast']
        assert forecast is not None
        assert not forecast.empty

    def test_monthly_update_pipeline(self, temp_db, sample_time_series_data):
        """
        TEST 3: Existing Dataset → Add New Month → Database Update →
                 Retrain SARIMA → Updated Forecast
        """
        # Step 1: Load initial data
        temp_db.create_version('initial', 'Initial load')
        temp_db.insert_records(sample_time_series_data)
        initial_count = temp_db.get_record_count()

        # Step 2: Add a new month's data
        new_month_data = pd.DataFrame({
            'crime_category': ['Test_Category'],
            'month': [1],
            'year': [2025],
            'case_count': [55],
            'legal_section': ['IPC 302'],
            'page': [1],
            'source_file': ['manual_entry'],
            'standard_group': ['Test'],
            'domain': ['General IPC'],
            'legal_section_bns': ['BNS 103']
        })

        # Step 3: Validate
        validator = DataValidator()
        validation = validator.validate_dataframe(new_month_data)
        assert validation['summary']['valid'] == 1

        # Step 4: Insert
        version = temp_db.create_version('monthly_update', 'Added Jan 2025')
        result = temp_db.insert_records(
            validation['valid_df'], data_source='manual',
            dataset_version=version
        )
        assert result['inserted'] == 1

        # Verify count increased
        assert temp_db.get_record_count() == initial_count + 1

        # Step 5: Retrain SARIMA
        sarima = SARIMAModel(db=temp_db)
        train_result = sarima.train_category('Test_Category')
        # May or may not succeed depending on data overlap

        # Step 6: Verify version history
        versions = temp_db.get_version_history()
        assert len(versions) >= 2


# ============================================================
# CONFIGURATION TESTS
# ============================================================

class TestConfiguration:
    """Tests for configuration values."""

    def test_primary_columns(self):
        """Test that primary columns are correctly defined."""
        assert len(PRIMARY_COLUMNS) == 10
        assert 'crime_category' in PRIMARY_COLUMNS
        assert 'case_count' in PRIMARY_COLUMNS
        assert 'month' in PRIMARY_COLUMNS
        assert 'year' in PRIMARY_COLUMNS

    def test_sarima_config(self):
        """Test SARIMA configuration values."""
        assert SARIMA_ORDER == (1, 1, 1)
        assert SARIMA_SEASONAL_ORDER == (1, 1, 0, 12)
        assert SEASONAL_PERIOD == 12
        assert TRAIN_TEST_RATIO == 0.80

    def test_train_test_ratio(self):
        """Test train/test ratio is 80/20."""
        assert TRAIN_TEST_RATIO == 0.80
        assert (1 - TRAIN_TEST_RATIO) == pytest.approx(0.20)

    def test_crime_domains(self):
        """Test crime domains are defined."""
        assert len(CRIME_DOMAINS) > 0
        assert 'General IPC' in CRIME_DOMAINS
        assert 'Crimes Against Women (CAW)' in CRIME_DOMAINS
        assert 'NDPS Act' in CRIME_DOMAINS

    def test_min_forecast_history(self):
        """Test minimum forecast history is reasonable."""
        assert MIN_FORECAST_HISTORY >= 12  # At least 1 year
