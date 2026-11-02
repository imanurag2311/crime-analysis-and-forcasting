"""
Data Processor
===============
Handles data cleaning, standardization, time series preparation,
rolling averages, YoY calculations, and train/test splitting.
"""

import logging
from typing import Optional, List, Dict, Any, Tuple

import pandas as pd
import numpy as np

from app.core.config import (
    PRIMARY_COLUMNS, ROLLING_WINDOW_3, ROLLING_WINDOW_6,
    TRAIN_TEST_RATIO, MIN_FORECAST_HISTORY
)
from app.processing.ipc_bns_crosswalk import IPCBNSCrosswalk
from app.processing.data_validator import DataValidator

logger = logging.getLogger(__name__)


class DataProcessor:
    """
    Processes and prepares crime data for analysis and forecasting.
    
    Responsibilities:
        1. Standardize dates
        2. Standardize crime names
        3. Detect missing months
        4. Create continuous monthly time series
        5. Handle unusual values
        6. Fill missing months using interpolation
        7. Create rolling averages (3-month, 6-month)
        8. Calculate YoY changes
        9. Aggregate monthly registered cases
        10. Create chronological 80/20 train-test split
        11. Prevent data leakage
    """

    def __init__(self):
        """Initialize with crosswalk and validator."""
        self.crosswalk = IPCBNSCrosswalk()
        self.validator = DataValidator()

    def process_raw_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Full processing pipeline for raw crime data.
        
        Steps:
            1. Clean column names
            2. Standardize dates
            3. Standardize crime categories
            4. Apply IPC→BNS crosswalk
            5. Validate records
            6. Remove duplicates
            7. Sort chronologically
        """
        logger.info(f"Processing {len(df)} raw records")

        # Step 1: Clean column names
        df = self._clean_column_names(df)

        # Step 2: Standardize dates
        df = self._standardize_dates(df)

        # Step 3: Standardize crime categories
        df = self._standardize_categories(df)

        # Step 4: Apply IPC→BNS crosswalk
        df = self._apply_crosswalk(df)

        # Step 5: Ensure numeric case_count
        df = self._ensure_numeric_counts(df)

        # Step 6: Sort chronologically
        df = df.sort_values(['year', 'month', 'crime_category']).reset_index(drop=True)

        logger.info(f"Processing complete: {len(df)} records")
        return df

    def _clean_column_names(self, df: pd.DataFrame) -> pd.DataFrame:
        """Normalize column names to lowercase with underscores."""
        df.columns = [
            col.strip().lower().replace(' ', '_').replace('-', '_')
            for col in df.columns
        ]
        return df

    def _standardize_dates(self, df: pd.DataFrame) -> pd.DataFrame:
        """Ensure month and year columns are proper integers."""
        if 'month' in df.columns:
            df['month'] = pd.to_numeric(df['month'], errors='coerce').astype('Int64')
        if 'year' in df.columns:
            df['year'] = pd.to_numeric(df['year'], errors='coerce').astype('Int64')
        return df

    def _standardize_categories(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply crime category standardization using crosswalk aliases."""
        if 'crime_category' in df.columns:
            df['crime_category'] = df['crime_category'].apply(
                lambda x: self.crosswalk.standardize_category(str(x)) if pd.notna(x) else 'Unknown'
            )
        return df

    def _apply_crosswalk(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply IPC→BNS crosswalk to populate BNS section and domain."""
        if 'legal_section' not in df.columns:
            df['legal_section'] = ''
        if 'legal_section_bns' not in df.columns:
            df['legal_section_bns'] = ''
        if 'domain' not in df.columns:
            df['domain'] = ''
        if 'standard_group' not in df.columns:
            df['standard_group'] = ''

        for idx, row in df.iterrows():
            legal = str(row.get('legal_section', '')).strip()
            cat = str(row.get('crime_category', '')).strip()

            if legal and legal != 'nan':
                crosswalk_result = self.crosswalk.apply_crosswalk(legal)
                if crosswalk_result['legal_section_bns']:
                    df.at[idx, 'legal_section_bns'] = crosswalk_result['legal_section_bns']
                if crosswalk_result['domain'] and crosswalk_result['domain'] != 'Other':
                    df.at[idx, 'domain'] = crosswalk_result['domain']
                if crosswalk_result['standard_group']:
                    df.at[idx, 'standard_group'] = crosswalk_result['standard_group']
            elif cat:
                # Fallback to category legal mapping
                cat_info = self.crosswalk.get_category_legal_info(cat)
                if cat_info.get('ipc') and not row.get('legal_section'):
                    df.at[idx, 'legal_section'] = cat_info['ipc']
                if cat_info.get('bns'):
                    df.at[idx, 'legal_section_bns'] = cat_info['bns']
                if cat_info.get('domain'):
                    df.at[idx, 'domain'] = cat_info['domain']
                    df.at[idx, 'standard_group'] = cat_info['domain']

        return df

    def _ensure_numeric_counts(self, df: pd.DataFrame) -> pd.DataFrame:
        """Ensure case_count is numeric and non-negative."""
        if 'case_count' in df.columns:
            df['case_count'] = pd.to_numeric(df['case_count'], errors='coerce').fillna(0).astype(int)
            df.loc[df['case_count'] < 0, 'case_count'] = 0
        return df

    # ================================================================
    # MONTHLY TIME SERIES CREATION
    # ================================================================

    def create_monthly_time_series(self, df: pd.DataFrame,
                                    category: Optional[str] = None) -> pd.DataFrame:
        """
        Create a continuous monthly time series for a crime category.
        
        - Aggregates monthly registered case counts
        - Creates DatetimeIndex with YYYY-MM-01
        - Ensures continuous monthly frequency
        - Handles missing periods via interpolation
        """
        if df.empty:
            return pd.DataFrame()

        # Filter by category if specified (and not Total Cases aggregate)
        if category and category not in ("Total Cases (All Crimes)", "TOTAL CRIME CASES", "Total Cases"):
            df = df[df['crime_category'] == category].copy()

        if df.empty:
            return pd.DataFrame()

        # Aggregate monthly counts
        monthly = df.groupby(['year', 'month']).agg(
            total_cases=('case_count', 'sum')
        ).reset_index()

        # Create datetime index
        monthly['date'] = pd.to_datetime(
            monthly['year'].astype(str) + '-' +
            monthly['month'].astype(str).str.zfill(2) + '-01'
        )
        monthly = monthly.set_index('date').sort_index()

        # Ensure continuous monthly frequency
        if len(monthly) > 1:
            full_range = pd.date_range(
                start=monthly.index.min(),
                end=monthly.index.max(),
                freq='MS'
            )
            monthly = monthly.reindex(full_range)
            monthly.index.name = 'date'

            # Fill missing months
            missing_count = monthly['total_cases'].isna().sum()
            if missing_count > 0:
                logger.info(f"Filling {missing_count} missing months via interpolation")
                monthly['total_cases'] = monthly['total_cases'].interpolate(
                    method='linear'
                ).round(0)
                # Fill year and month from index
                monthly['year'] = monthly.index.year
                monthly['month'] = monthly.index.month
            else:
                monthly['year'] = monthly.index.year
                monthly['month'] = monthly.index.month
        else:
            monthly['year'] = monthly.index.year
            monthly['month'] = monthly.index.month

        monthly['total_cases'] = monthly['total_cases'].fillna(0).astype(int)

        return monthly

    def detect_missing_months(self, df: pd.DataFrame,
                               category: Optional[str] = None) -> List[Dict]:
        """Detect gaps in monthly data coverage."""
        if df.empty:
            return []

        if category:
            df = df[df['crime_category'] == category]

        if df.empty:
            return []

        existing = set()
        for _, row in df.iterrows():
            existing.add((int(row['year']), int(row['month'])))

        if not existing:
            return []

        min_year = min(y for y, m in existing)
        max_year = max(y for y, m in existing)
        min_month = min(m for y, m in existing if y == min_year)
        max_month = max(m for y, m in existing if y == max_year)

        missing = []
        for year in range(min_year, max_year + 1):
            start_m = min_month if year == min_year else 1
            end_m = max_month if year == max_year else 12
            for month in range(start_m, end_m + 1):
                if (year, month) not in existing:
                    missing.append({'year': year, 'month': month})

        return missing

    # ================================================================
    # ROLLING AVERAGES & YoY
    # ================================================================

    def add_rolling_averages(self, ts: pd.DataFrame) -> pd.DataFrame:
        """Add 3-month and 6-month rolling averages to a time series."""
        if 'total_cases' in ts.columns:
            ts[f'rolling_avg_{ROLLING_WINDOW_3}m'] = (
                ts['total_cases'].rolling(window=ROLLING_WINDOW_3, min_periods=1).mean().round(1)
            )
            ts[f'rolling_avg_{ROLLING_WINDOW_6}m'] = (
                ts['total_cases'].rolling(window=ROLLING_WINDOW_6, min_periods=1).mean().round(1)
            )
        return ts

    def add_yoy_change(self, ts: pd.DataFrame) -> pd.DataFrame:
        """Calculate Year-over-Year percentage change."""
        if 'total_cases' in ts.columns:
            ts['yoy_change'] = ts['total_cases'].pct_change(periods=12) * 100
            ts['yoy_change'] = ts['yoy_change'].round(2)
        return ts

    # ================================================================
    # TRAIN / TEST SPLIT
    # ================================================================

    def chronological_train_test_split(
        self, ts: pd.DataFrame,
        train_ratio: float = TRAIN_TEST_RATIO
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Chronological 80/20 train-test split.
        
        NEVER randomly shuffles time-series data.
        Prevents data leakage by using only earlier observations for training.
        
        Args:
            ts: Time series DataFrame (must be sorted chronologically)
            train_ratio: Proportion for training (default 0.80)
            
        Returns:
            (train_df, test_df)
        """
        if ts.empty:
            return ts, pd.DataFrame()

        n = len(ts)
        split_idx = int(n * train_ratio)

        # Ensure at least 1 observation in test set
        if split_idx >= n:
            split_idx = n - 1

        train = ts.iloc[:split_idx].copy()
        test = ts.iloc[split_idx:].copy()

        logger.info(
            f"Train/test split: {len(train)} train ({train_ratio:.0%}), "
            f"{len(test)} test ({1 - train_ratio:.0%})"
        )

        return train, test

    def has_sufficient_history(self, ts: pd.DataFrame,
                                min_observations: int = MIN_FORECAST_HISTORY) -> bool:
        """Check if there are enough observations for SARIMA."""
        return len(ts) >= min_observations

    # ================================================================
    # EXCEL IMPORT PROCESSING
    # ================================================================

    def process_excel_import(self, filepath: str,
                              column_mapping: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
        """
        Process an Excel file for import into the system.
        
        Steps:
            1. Read Excel file
            2. Detect/apply column mapping
            3. Validate
            4. Process data
            5. Return results with preview
        """
        from pathlib import Path
        try:
            ext = Path(filepath).suffix.lower()
            if ext == '.csv':
                df = pd.read_csv(filepath)
            else:
                df = pd.read_excel(filepath, engine='openpyxl')
            logger.info(f"Read file: {filepath} ({len(df)} rows, {len(df.columns)} cols)")
        except Exception as e:
            return {
                'success': False,
                'error': f"Failed to read file: {e}",
                'df': pd.DataFrame()
            }

        if df.empty:
            return {
                'success': False,
                'error': "Excel file is empty",
                'df': pd.DataFrame()
            }

        # Detect columns
        col_analysis = self.validator.validate_excel_columns(df)

        # Apply column mapping
        if column_mapping:
            df = df.rename(columns=column_mapping)
        elif col_analysis['column_mapping']:
            df = df.rename(columns=col_analysis['column_mapping'])

        # Ensure required columns exist (with empty defaults)
        for col in PRIMARY_COLUMNS:
            if col not in df.columns:
                df[col] = ''

        # Process the data
        df = self.process_raw_data(df)

        # Validate
        validation = self.validator.validate_dataframe(df)

        return {
            'success': True,
            'df': df,
            'valid_df': validation['valid_df'],
            'invalid_df': validation['invalid_df'],
            'column_analysis': col_analysis,
            'validation': validation,
            'total_rows': len(df),
            'valid_rows': validation['summary']['valid'],
            'invalid_rows': validation['summary']['invalid'],
            'warnings': validation['warnings'],
            'errors': validation['errors']
        }

    # ================================================================
    # AGGREGATE STATISTICS
    # ================================================================

    def get_annual_summary(self, df: pd.DataFrame) -> pd.DataFrame:
        """Get annual aggregated crime counts by category."""
        if df.empty:
            return pd.DataFrame()

        annual = df.groupby(['year', 'crime_category']).agg(
            total_cases=('case_count', 'sum')
        ).reset_index()

        return annual.sort_values(['year', 'crime_category'])

    def get_category_distribution(self, df: pd.DataFrame) -> pd.DataFrame:
        """Get total case distribution by crime category."""
        if df.empty:
            return pd.DataFrame()

        dist = df.groupby('crime_category').agg(
            total_cases=('case_count', 'sum')
        ).reset_index()

        dist = dist.sort_values('total_cases', ascending=False)
        dist['percentage'] = (
            dist['total_cases'] / dist['total_cases'].sum() * 100
        ).round(2)

        return dist

    def get_domain_distribution(self, df: pd.DataFrame) -> pd.DataFrame:
        """Get total case distribution by domain."""
        if df.empty:
            return pd.DataFrame()

        dist = df.groupby('domain').agg(
            total_cases=('case_count', 'sum')
        ).reset_index()

        dist = dist.sort_values('total_cases', ascending=False)
        return dist
