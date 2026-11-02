
import logging
import pickle
import json
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any, Tuple, List

import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import warnings

from app.core.config import (
    SARIMA_ORDER, SARIMA_SEASONAL_ORDER, SEASONAL_PERIOD,
    TRAIN_TEST_RATIO, MIN_FORECAST_HISTORY, DEFAULT_FORECAST_HORIZON,
    SARIMA_MODEL_DIR, FORECAST_OUTPUT_DIR, METRICS_OUTPUT_DIR,
    CONFIDENCE_LEVELS
)
from app.processing.data_processor import DataProcessor
from app.database.database import CrimeDatabase

logger = logging.getLogger(__name__)

# Suppress convergence warnings during fitting
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)


class SARIMAModel:
    """
    SARIMA / SARIMAX forecasting engine for Mumbai crime data.
    
    For each crime category:
        1. Aggregate monthly registered case counts
        2. Create DatetimeIndex (YYYY-MM-01)
        3. Ensure continuous monthly frequency
        4. Handle missing periods
        5. Perform chronological 80/20 split
        6. Train SARIMA
        7. Generate validation predictions
        8. Calculate RMSE, MAE, MAPE, R²
        9. Refine parameters if necessary
        10. Retrain on full historical data
        11. Generate 3–6 month forecasts
        12. Generate 80% and 95% confidence intervals
        13. Save model
        14. Save forecast
        15. Save metrics
    """

    def __init__(self, db: Optional[CrimeDatabase] = None):
        """Initialize SARIMA engine."""
        self.db = db or CrimeDatabase()
        self.processor = DataProcessor()
        self.models: Dict[str, Any] = {}
        self.metrics: Dict[str, Dict] = {}
        self.forecasts: Dict[str, pd.DataFrame] = {}

    # ================================================================
    # MODEL TRAINING
    # ================================================================

    def train_category(self, category: str,
                       order: Tuple = SARIMA_ORDER,
                       seasonal_order: Tuple = SARIMA_SEASONAL_ORDER,
                       forecast_horizon: int = DEFAULT_FORECAST_HORIZON) -> Dict[str, Any]:
        """
        Train SARIMA model for a specific crime category.
        
        Returns dict with training results, metrics, and forecasts.
        """
        logger.info(f"Training SARIMA for: {category}")

        result = {
            'category': category,
            'success': False,
            'error': None,
            'metrics': {},
            'forecast': None,
            'model_info': {}
        }

        try:
            # Step 1: Get data from database
            df = self.db.get_all_records()
            if df.empty:
                result['error'] = "No data available in database"
                return result

            # Step 2: Create monthly time series
            ts = self.processor.create_monthly_time_series(df, category=category)
            if ts.empty or 'total_cases' not in ts.columns:
                result['error'] = f"No data found for category: {category}"
                return result

            series = ts['total_cases']

            # Step 3: Check data sufficiency
            if not self.processor.has_sufficient_history(ts):
                result['error'] = (
                    f"Insufficient historical data for SARIMA forecasting. "
                    f"Have {len(ts)} observations, need at least {MIN_FORECAST_HISTORY}."
                )
                logger.warning(result['error'])
                return result

            # Step 4: Chronological 80/20 split
            split_idx = int(len(series) * TRAIN_TEST_RATIO)
            train = series.iloc[:split_idx]
            test = series.iloc[split_idx:]

            if len(test) == 0:
                test = series.iloc[-1:]
                train = series.iloc[:-1]

            logger.info(f"Split: {len(train)} train, {len(test)} test")

            # Step 5: Train SARIMA on training data
            model_fit = self._fit_sarima(train, order, seasonal_order)
            if model_fit is None:
                # Try with simpler parameters
                logger.info("Trying simplified SARIMA parameters")
                model_fit = self._fit_sarima(
                    train, (1, 1, 0), (1, 0, 0, SEASONAL_PERIOD)
                )
                if model_fit is None:
                    # Try even simpler
                    model_fit = self._fit_sarima(
                        train, (1, 0, 0), (0, 0, 0, SEASONAL_PERIOD)
                    )
                    if model_fit is None:
                        result['error'] = "SARIMA convergence failure"
                        return result
                    order = (1, 0, 0)
                    seasonal_order = (0, 0, 0, SEASONAL_PERIOD)
                else:
                    order = (1, 1, 0)
                    seasonal_order = (1, 0, 0, SEASONAL_PERIOD)

            # Step 6: Generate validation predictions
            predictions = model_fit.forecast(steps=len(test))
            predictions = predictions.clip(lower=0)  # No negative predictions

            # Step 7: Calculate metrics
            metrics = self._calculate_metrics(
                test.values, predictions.values,
                order, seasonal_order,
                train, test
            )
            result['metrics'] = metrics

            # Step 8: Retrain on full dataset
            full_model_fit = self._fit_sarima(series, order, seasonal_order)
            if full_model_fit is None:
                full_model_fit = model_fit  # Fall back to train-only model

            # Step 9: Generate forecasts with confidence intervals
            forecast_df = self._generate_forecast(
                full_model_fit, series, forecast_horizon
            )
            result['forecast'] = forecast_df

            # Step 10: Save everything
            self._save_model(category, full_model_fit, order, seasonal_order)
            self._save_metrics(category, metrics)
            self._save_forecast(category, forecast_df)

            # Store in memory
            self.models[category] = full_model_fit
            self.metrics[category] = metrics
            self.forecasts[category] = forecast_df

            # Save to database
            dataset_version = self.db.get_current_version()
            self.db.save_metrics(metrics, category, dataset_version)
            if forecast_df is not None and not forecast_df.empty:
                self.db.save_forecasts(forecast_df, category, dataset_version)

            result['success'] = True
            result['model_info'] = {
                'order': str(order),
                'seasonal_order': str(seasonal_order),
                'training_start': str(series.index.min()),
                'training_end': str(series.index.max()),
                'train_size': len(train),
                'test_size': len(test),
                'total_observations': len(series),
                'dataset_version': dataset_version,
                'trained_at': datetime.now().isoformat()
            }

            logger.info(
                f"SARIMA training complete for {category}: "
                f"RMSE={metrics.get('rmse', 'N/A'):.2f}, "
                f"R²={metrics.get('r_squared', 'N/A'):.4f}"
            )

        except Exception as e:
            result['error'] = str(e)
            logger.error(f"SARIMA training failed for {category}: {e}")

        return result

    def train_all_categories(self, forecast_horizon: int = DEFAULT_FORECAST_HORIZON) -> Dict[str, Any]:
        """
        Train SARIMA models for all available crime categories plus Total Cases.
        
        Returns summary of training results.
        """
        categories = ["Total Cases (All Crimes)"] + self.db.get_unique_categories()
        logger.info(f"Training SARIMA for {len(categories)} categories")

        results = {
            'total_categories': len(categories),
            'successful': 0,
            'failed': 0,
            'insufficient_data': 0,
            'category_results': {}
        }

        for category in categories:
            cat_result = self.train_category(
                category, forecast_horizon=forecast_horizon
            )
            results['category_results'][category] = cat_result

            if cat_result['success']:
                results['successful'] += 1
            elif 'Insufficient' in str(cat_result.get('error', '')):
                results['insufficient_data'] += 1
            else:
                results['failed'] += 1

        logger.info(
            f"Training complete: {results['successful']} successful, "
            f"{results['failed']} failed, "
            f"{results['insufficient_data']} insufficient data"
        )

        return results

    # ================================================================
    # SARIMA FITTING
    # ================================================================

    def _fit_sarima(self, series: pd.Series,
                    order: Tuple, seasonal_order: Tuple) -> Optional[Any]:
        """
        Fit a SARIMAX model to the time series.
        
        Returns fitted model or None on failure.
        """
        try:
            model = SARIMAX(
                series,
                order=order,
                seasonal_order=seasonal_order,
                enforce_stationarity=False,
                enforce_invertibility=False
            )
            fit_result = model.fit(disp=False, maxiter=200)
            logger.info(f"SARIMA({order}x{seasonal_order}) fitted successfully")
            return fit_result
        except Exception as e:
            logger.warning(f"SARIMA fitting failed with {order}x{seasonal_order}: {e}")
            return None

    # ================================================================
    # METRICS CALCULATION
    # ================================================================

    def _calculate_metrics(self, actual: np.ndarray, predicted: np.ndarray,
                           order: Tuple, seasonal_order: Tuple,
                           train: pd.Series, test: pd.Series) -> Dict[str, Any]:
        """
        Calculate RMSE, MAE, MAPE, R² for validation.
        
        Handles zero actual values safely for MAPE.
        """
        # RMSE
        rmse = np.sqrt(mean_squared_error(actual, predicted))

        # MAE
        mae = mean_absolute_error(actual, predicted)

        # MAPE - handle zero values safely
        non_zero_mask = actual != 0
        if non_zero_mask.any():
            mape = np.mean(
                np.abs((actual[non_zero_mask] - predicted[non_zero_mask]) /
                       actual[non_zero_mask])
            ) * 100
        else:
            mape = float('inf')

        # R²
        r_squared = r2_score(actual, predicted)

        metrics = {
            'rmse': round(float(rmse), 4),
            'mae': round(float(mae), 4),
            'mape': round(float(mape), 4) if mape != float('inf') else None,
            'r_squared': round(float(r_squared), 4),
            'sarima_order': str(order),
            'sarima_seasonal_order': str(seasonal_order),
            'training_start': str(train.index.min()),
            'training_end': str(train.index.max()),
            'train_size': len(train),
            'test_size': len(test)
        }

        return metrics

    # ================================================================
    # FORECAST GENERATION
    # ================================================================

    def _generate_forecast(self, model_fit, series: pd.Series,
                           horizon: int) -> pd.DataFrame:
        """
        Generate future forecasts with 80% and 95% confidence intervals.
        """
        try:
            # Forecast
            forecast_result = model_fit.get_forecast(steps=horizon)
            forecast_values = forecast_result.predicted_mean.clip(lower=0)

            # Confidence intervals
            ci_80 = forecast_result.conf_int(alpha=0.20)
            ci_95 = forecast_result.conf_int(alpha=0.05)

            # Create forecast DataFrame
            forecast_dates = pd.date_range(
                start=series.index.max() + pd.DateOffset(months=1),
                periods=horizon,
                freq='MS'
            )

            forecast_df = pd.DataFrame({
                'forecast_date': forecast_dates,
                'forecast_value': forecast_values.values,
                'lower_80': ci_80.iloc[:, 0].clip(lower=0).values,
                'upper_80': ci_80.iloc[:, 1].values,
                'lower_95': ci_95.iloc[:, 0].clip(lower=0).values,
                'upper_95': ci_95.iloc[:, 1].values
            })

            # Round values
            for col in ['forecast_value', 'lower_80', 'upper_80', 'lower_95', 'upper_95']:
                forecast_df[col] = forecast_df[col].round(0).astype(int)

            logger.info(f"Generated {horizon}-month forecast")
            return forecast_df

        except Exception as e:
            logger.error(f"Forecast generation failed: {e}")
            return pd.DataFrame()

    # ================================================================
    # MODEL PERSISTENCE
    # ================================================================

    def _safe_filename(self, category: str) -> str:
        """Sanitize category string for safe filesystem usage."""
        import re
        return re.sub(r'[^\w\-]', '_', category)

    def _save_model(self, category: str, model_fit,
                    order: Tuple, seasonal_order: Tuple):
        """Save trained SARIMA model to disk."""
        filename = self._safe_filename(category)
        model_path = SARIMA_MODEL_DIR / f"{filename}.pkl"
        try:
            with open(model_path, 'wb') as f:
                pickle.dump({
                    'model': model_fit,
                    'order': order,
                    'seasonal_order': seasonal_order,
                    'category': category,
                    'trained_at': datetime.now().isoformat(),
                    'dataset_version': self.db.get_current_version()
                }, f)
            logger.info(f"Model saved: {model_path}")
        except Exception as e:
            logger.error(f"Failed to save model for {category}: {e}")

    def _save_metrics(self, category: str, metrics: Dict):
        """Save model metrics to disk."""
        filename = self._safe_filename(category)
        metrics_path = METRICS_OUTPUT_DIR / f"{filename}_metrics.json"
        try:
            with open(metrics_path, 'w') as f:
                json.dump(metrics, f, indent=2, default=str)
            logger.info(f"Metrics saved: {metrics_path}")
        except Exception as e:
            logger.error(f"Failed to save metrics for {category}: {e}")

    def _save_forecast(self, category: str, forecast_df: pd.DataFrame):
        """Save forecast results to disk."""
        if forecast_df.empty:
            return
        filename = self._safe_filename(category)
        forecast_path = FORECAST_OUTPUT_DIR / f"{filename}_forecast.csv"
        try:
            forecast_df.to_csv(forecast_path, index=False)
            logger.info(f"Forecast saved: {forecast_path}")
        except Exception as e:
            logger.error(f"Failed to save forecast for {category}: {e}")

    def load_model(self, category: str) -> Optional[Dict]:
        """Load a previously trained model from disk."""
        filename = self._safe_filename(category)
        model_path = SARIMA_MODEL_DIR / f"{filename}.pkl"
        if not model_path.exists():
            logger.warning(f"No saved model found for: {category}")
            return None

        try:
            with open(model_path, 'rb') as f:
                data = pickle.load(f)
            self.models[category] = data['model']
            logger.info(f"Model loaded: {category}")
            return data
        except Exception as e:
            logger.error(f"Failed to load model for {category}: {e}")
            return None

    def load_all_models(self) -> int:
        """Load all saved models. Returns count of loaded models."""
        loaded = 0
        for model_file in SARIMA_MODEL_DIR.glob("*.pkl"):
            category = model_file.stem
            if self.load_model(category):
                loaded += 1
        logger.info(f"Loaded {loaded} models")
        return loaded

    # ================================================================
    # FORECAST RETRIEVAL
    # ================================================================

    def get_forecast(self, category: str,
                     horizon: Optional[int] = None) -> Optional[pd.DataFrame]:
        """Get forecast for a specific category."""
        # Try from memory
        if category in self.forecasts:
            forecast = self.forecasts[category]
            if horizon and len(forecast) > horizon:
                return forecast.head(horizon)
            return forecast

        # Try from database
        forecast = self.db.get_forecasts(category)
        if not forecast.empty:
            return forecast

        # Try from file
        filename = self._safe_filename(category)
        forecast_path = FORECAST_OUTPUT_DIR / f"{filename}_forecast.csv"
        if forecast_path.exists():
            return pd.read_csv(forecast_path)

        return None

    def get_metrics(self, category: str) -> Optional[Dict]:
        """Get metrics for a specific category."""
        if category in self.metrics:
            return self.metrics[category]

        # Try from database
        metrics_df = self.db.get_metrics(category)
        if not metrics_df.empty:
            return metrics_df.iloc[0].to_dict()

        # Try from file
        filename = self._safe_filename(category)
        metrics_path = METRICS_OUTPUT_DIR / f"{filename}_metrics.json"
        if metrics_path.exists():
            with open(metrics_path) as f:
                return json.load(f)

        return None

    def get_historical_with_predictions(self, category: str) -> Optional[Dict[str, pd.DataFrame]]:
        """
        Get historical data, train/test split, and predictions for visualization.
        """
        df = self.db.get_all_records()
        if df.empty:
            return None

        ts = self.processor.create_monthly_time_series(df, category=category)
        if ts.empty:
            return None

        series = ts['total_cases']

        # Split
        split_idx = int(len(series) * TRAIN_TEST_RATIO)
        train = series.iloc[:split_idx]
        test = series.iloc[split_idx:]

        # Get model predictions on test set
        predictions = None
        if category in self.models:
            try:
                preds = self.models[category].forecast(steps=len(test))
                predictions = preds.clip(lower=0)
            except Exception:
                pass

        # Get forecast
        forecast = self.get_forecast(category)

        return {
            'full_series': series,
            'train': train,
            'test': test,
            'predictions': predictions,
            'forecast': forecast
        }

    def get_training_status(self) -> Dict[str, Any]:
        """Get overall training status."""
        model_files = list(SARIMA_MODEL_DIR.glob("*.pkl"))
        categories = self.db.get_unique_categories()

        last_trained = None
        if model_files:
            latest = max(model_files, key=lambda f: f.stat().st_mtime)
            last_trained = datetime.fromtimestamp(
                latest.stat().st_mtime
            ).isoformat()

        return {
            'total_models': len(model_files),
            'total_categories': len(categories),
            'trained_categories': [f.stem for f in model_files],
            'untrained_categories': [
                c for c in categories
                if c not in [f.stem for f in model_files]
            ],
            'last_trained': last_trained,
            'needs_retraining': len(model_files) < len(categories) if categories else False
        }
