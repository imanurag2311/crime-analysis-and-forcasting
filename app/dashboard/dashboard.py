"""
Mumbai Crime Analysis & Forecasting System — Streamlit Dashboard
=================================================================
Professional dark glassmorphism-style interactive dashboard with 5 main tabs:
    1. Crime Analytics & Trends
    2. SARIMA Forecasting Engine
    3. IPC ↔ BNS Crosswalk
    4. Monthly Dataset Manager
    5. Raw Data OCR & Processing Hub
"""

import os
import sys
import io
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from app.core.config import (
    DASHBOARD_TITLE, PRIMARY_COLUMNS, CRIME_DOMAINS,
    ALLOWED_UPLOAD_EXTENSIONS, SARIMA_MODEL_DIR, DATABASE_PATH,
    DEFAULT_FORECAST_HORIZON, MIN_FORECAST_HORIZON, MAX_FORECAST_HORIZON,
    PROCESSED_DATA_DIR, UPLOADS_DIR
)
from app.database.database import CrimeDatabase
from app.analysis.eda_analyzer import EDAAnalyzer
from app.models.sarima_model import SARIMAModel
from app.processing.data_processor import DataProcessor
from app.processing.data_validator import DataValidator
from app.processing.ipc_bns_crosswalk import IPCBNSCrosswalk

logger = logging.getLogger(__name__)

# ============================================================
# PAGE CONFIG & STYLING
# ============================================================

st.set_page_config(
    page_title=DASHBOARD_TITLE,
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for dark glassmorphism styling
st.markdown("""
<style>
    /* Global dark theme enhancements */
    .stApp {
        background: linear-gradient(135deg, #0f0f23 0%, #1a1a3e 50%, #0f0f23 100%);
    }
    
    /* Glassmorphism card */
    .glass-card {
        background: rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(10px);
        border-radius: 16px;
        border: 1px solid rgba(255, 255, 255, 0.1);
        padding: 20px;
        margin: 10px 0;
    }
    
    /* KPI cards */
    .kpi-card {
        background: rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(10px);
        border-radius: 12px;
        border: 1px solid rgba(255, 255, 255, 0.1);
        padding: 16px 20px;
        text-align: center;
        transition: transform 0.2s;
        min-height: 110px;
        height: 110px;
        display: flex;
        flex-direction: column;
        justify-content: center;
        align-items: center;
    }
    .kpi-card:hover {
        transform: translateY(-2px);
    }
    .kpi-value {
        font-size: 1.8rem;
        font-weight: 700;
        background: linear-gradient(90deg, #6366f1, #ec4899);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin: 4px 0;
        line-height: 1.2;
    }
    .kpi-label {
        font-size: 0.85rem;
        color: rgba(255, 255, 255, 0.6);
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    /* Status badges */
    .status-ready {
        background: linear-gradient(90deg, #22c55e33, #14b8a633);
        border: 1px solid #22c55e;
        color: #22c55e;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: 600;
    }
    .status-warning {
        background: linear-gradient(90deg, #f59e0b33, #eab30833);
        border: 1px solid #f59e0b;
        color: #f59e0b;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: 600;
    }
    
    /* Header styling */
    .main-header {
        text-align: center;
        padding: 10px 0 20px 0;
    }
    .main-header h1 {
        background: linear-gradient(90deg, #6366f1, #ec4899, #14b8a6);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 2.2rem;
        font-weight: 800;
        margin-bottom: 4px;
    }
    .main-header p {
        color: rgba(255, 255, 255, 0.5);
        font-size: 0.9rem;
    }
    
    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        background: rgba(255, 255, 255, 0.05);
        border-radius: 8px;
        border: 1px solid rgba(255, 255, 255, 0.1);
        padding: 8px 16px;
    }
    
    /* Metric overrides */
    [data-testid="stMetricValue"] {
        font-size: 1.8rem;
    }
    
    /* Divider */
    .divider {
        height: 1px;
        background: linear-gradient(90deg, transparent, rgba(255,255,255,0.2), transparent);
        margin: 20px 0;
    }
</style>
""", unsafe_allow_html=True)


# ============================================================
# INITIALIZE SERVICES (cached)
# ============================================================

@st.cache_resource
def get_database():
    """Get cached database instance."""
    return CrimeDatabase()


@st.cache_resource
def get_eda_analyzer():
    """Get cached EDA analyzer."""
    return EDAAnalyzer(db=get_database())


@st.cache_resource
def get_sarima_model():
    """Get cached SARIMA model."""
    model = SARIMAModel(db=get_database())
    model.load_all_models()
    return model


@st.cache_resource
def get_data_processor():
    """Get cached data processor."""
    return DataProcessor()


@st.cache_resource
def get_data_validator():
    """Get cached data validator."""
    return DataValidator()


@st.cache_resource
def get_crosswalk():
    """Get cached crosswalk."""
    return IPCBNSCrosswalk()


# ============================================================
# HEADER
# ============================================================

st.markdown("""
<div class="main-header">
    <h1>🔍 Crime Analysis & Forecasting System</h1>
    <p>Crime Trend Analysis · SARIMA Forecasting · IPC ↔ BNS Crosswalk</p>
</div>
<div class="divider"></div>
""", unsafe_allow_html=True)


# ============================================================
# SIDEBAR — SYSTEM STATUS
# ============================================================

with st.sidebar:
    st.markdown("### 📊 System Status")

    db = get_database()
    stats = db.get_database_stats()

    # Dataset status
    if stats['total_records'] > 0:
        st.markdown(
            '<span class="status-ready">● Dataset Ready</span>',
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            '<span class="status-warning">● No Data Loaded</span>',
            unsafe_allow_html=True
        )

    st.markdown(f"**Dataset Version:** v{stats['current_version']}")
    st.markdown(f"**Total Records:** {stats['total_records']:,}")

    if stats['latest_month'] and stats['latest_year']:
        month_names = [
            '', 'Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
            'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'
        ]
        latest_m = month_names[stats['latest_month']] if stats['latest_month'] else '—'
        st.markdown(f"**Latest Month:** {latest_m} {stats['latest_year']}")

    st.markdown("---")

    # SARIMA status
    sarima = get_sarima_model()
    training_status = sarima.get_training_status()

    if training_status['total_models'] > 0:
        st.markdown(
            '<span class="status-ready">● SARIMA Trained</span>',
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            '<span class="status-warning">● SARIMA Not Trained</span>',
            unsafe_allow_html=True
        )

    st.markdown(f"**Trained Models:** {training_status['total_models']}")
    if training_status['last_trained']:
        st.markdown(f"**Last Training:** {training_status['last_trained'][:19]}")

    st.markdown("---")

    # Quick actions
    st.markdown("### ⚡ Quick Actions")

    if st.button("🔄 Refresh Data", key="sidebar_refresh", use_container_width=True):
        st.cache_resource.clear()
        st.rerun()

    if stats['total_records'] > 0:
        # Export buttons
        if st.button("📥 Export Dataset (Excel)", key="sidebar_export_excel",
                      use_container_width=True):
            export_path = PROCESSED_DATA_DIR / f"crime_dataset_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx"
            db.export_to_excel(export_path)
            st.success(f"Exported to {export_path.name}")

        if st.button("📥 Export Dataset (CSV)", key="sidebar_export_csv",
                      use_container_width=True):
            export_path = PROCESSED_DATA_DIR / f"crime_dataset_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
            db.export_to_csv(export_path)
            st.success(f"Exported to {export_path.name}")


# ============================================================
# MAIN TABS
# ============================================================

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📈 Crime Analytics & Trends",
    "🔮 SARIMA Forecasting",
    "⚖️ IPC ↔ BNS Crosswalk",
    "📋 Monthly Dataset Manager",
    "🔬 OCR Processing Hub"
])


# ============================================================
# TAB 1 — CRIME ANALYTICS & TRENDS
# ============================================================

with tab1:
    st.markdown("## 📈 Crime Analytics & Trends")

    if stats['total_records'] == 0:
        st.info("No data loaded yet. Use **Tab 4 (Monthly Dataset Manager)** to import data.")
    else:
        eda = get_eda_analyzer()

        # KPI Cards
        crosswalk = get_crosswalk()
        top_head_display = crosswalk.get_display_name(stats['top_crime_name']) if stats['top_crime_name'] else 'N/A'
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown(f"""
            <div class="kpi-card">
                <div class="kpi-label">Total Cases</div>
                <div class="kpi-value">{stats['total_cases']:,}</div>
            </div>
            """, unsafe_allow_html=True)
        with col2:
            st.markdown(f"""
            <div class="kpi-card">
                <div class="kpi-label">Top Crime Head</div>
                <div class="kpi-value">{top_head_display}</div>
            </div>
            """, unsafe_allow_html=True)
        with col3:
            st.markdown(f"""
            <div class="kpi-card">
                <div class="kpi-label">Crime Categories</div>
                <div class="kpi-value">{stats['total_categories']}</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

        # Filters
        filter_col1, filter_col2, filter_col3, filter_col4 = st.columns(4)
        crosswalk = get_crosswalk()

        with filter_col1:
            years = ['All'] + [str(y) for y in db.get_unique_years()]
            selected_year = st.selectbox("📅 Year", years, key="eda_year")
            year_filter = int(selected_year) if selected_year != 'All' else None

        with filter_col2:
            categories = ['All'] + db.get_unique_categories()
            selected_cat = st.selectbox(
                "📂 Category", categories, key="eda_cat",
                format_func=lambda c: "All Categories" if c == 'All' else crosswalk.get_display_name(c)
            )
            cat_filter = selected_cat if selected_cat != 'All' else None

        with filter_col3:
            domains = ['All'] + db.get_unique_domains()
            selected_domain = st.selectbox("🏷️ Domain", domains, key="eda_domain")

        with filter_col4:
            chart_type = st.selectbox("📊 Chart Focus", [
                "Monthly Trend", "Annual Trend", "Category Distribution",
                "Seasonal Patterns", "YoY Analysis", "Spike Detection",
                "Rolling Averages", "Domain Distribution"
            ], key="chart_type")

        st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

        # Chart display
        if chart_type == "Monthly Trend":
            fig = eda.plot_monthly_trend(category=cat_filter, year=year_filter)
            st.plotly_chart(fig, use_container_width=True)

        elif chart_type == "Annual Trend":
            fig = eda.plot_annual_trend(category=cat_filter)
            st.plotly_chart(fig, use_container_width=True)

        elif chart_type == "Category Distribution":
            fig = eda.plot_category_distribution(year=year_filter)
            st.plotly_chart(fig, use_container_width=True)

        elif chart_type == "Seasonal Patterns":
            fig = eda.plot_seasonal_patterns(category=cat_filter)
            st.plotly_chart(fig, use_container_width=True)

        elif chart_type == "YoY Analysis":
            fig = eda.plot_yoy_analysis(category=cat_filter)
            st.plotly_chart(fig, use_container_width=True)

        elif chart_type == "Spike Detection":
            fig = eda.plot_spike_detection(category=cat_filter)
            st.plotly_chart(fig, use_container_width=True)

            spikes = eda.detect_spikes(category=cat_filter)
            if not spikes.empty:
                st.markdown("#### ⚠️ Detected Spikes")
                st.dataframe(spikes[['total_cases', 'deviation']].reset_index(),
                             use_container_width=True)

        elif chart_type == "Rolling Averages":
            if cat_filter:
                fig = eda.plot_rolling_averages(cat_filter)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Select a specific crime category to view rolling averages.")

        elif chart_type == "Domain Distribution":
            fig = eda.plot_domain_distribution(year=year_filter)
            st.plotly_chart(fig, use_container_width=True)

        # Multi-category comparison
        st.markdown("---")
        st.markdown("### 📊 Category Comparison")
        all_cats = db.get_unique_categories()
        if len(all_cats) >= 2:
            compare_cats = st.multiselect(
                "Select categories to compare",
                all_cats, default=all_cats[:min(3, len(all_cats))],
                key="compare_cats",
                format_func=crosswalk.get_display_name
            )
            if compare_cats:
                fig = eda.plot_category_comparison(compare_cats, year=year_filter)
                st.plotly_chart(fig, use_container_width=True)


# ============================================================
# TAB 2 — SARIMA FORECASTING ENGINE
# ============================================================

with tab2:
    st.markdown("## 🔮 SARIMA Forecasting Engine")

    if stats['total_records'] == 0:
        st.info("No data loaded. Import data first to use forecasting.")
    else:
        sarima = get_sarima_model()
        processor = get_data_processor()
        crosswalk = get_crosswalk()

        # Controls
        ctrl_col1, ctrl_col2, ctrl_col3 = st.columns([2, 1, 1])
        with ctrl_col1:
            raw_cats = db.get_unique_categories()
            forecast_categories = ["Total Cases (All Crimes)"] + raw_cats if raw_cats else ["Total Cases (All Crimes)"]
            selected_forecast_cat = st.selectbox(
                "🎯 Crime Category",
                forecast_categories,
                key="forecast_cat",
                format_func=crosswalk.get_display_name
            )
        with ctrl_col2:
            forecast_horizon = st.selectbox(
                "📏 Forecast Horizon (months)",
                [3, 4, 5, 6], index=3, key="forecast_horizon"
            )
        with ctrl_col3:
            st.markdown("<br>", unsafe_allow_html=True)
            train_btn = st.button(
                "🚀 Train / Retrain SARIMA",
                key="train_sarima",
                use_container_width=True
            )

        st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

        # Train button handler
        if train_btn and selected_forecast_cat:
            with st.spinner(f"Training SARIMA for {selected_forecast_cat}..."):
                result = sarima.train_category(
                    selected_forecast_cat,
                    forecast_horizon=forecast_horizon
                )

                if result['success']:
                    st.success(
                        f"✅ SARIMA trained successfully for {selected_forecast_cat}"
                    )
                    st.cache_resource.clear()
                    st.rerun()
                else:
                    st.error(f"❌ Training failed: {result.get('error', 'Unknown error')}")

        # Train all button
        if st.button("🔄 Train All Categories", key="train_all_sarima"):
            with st.spinner("Training SARIMA for all categories..."):
                results = sarima.train_all_categories(forecast_horizon=forecast_horizon)
                st.success(
                    f"✅ Training complete: {results['successful']} successful, "
                    f"{results['failed']} failed, "
                    f"{results['insufficient_data']} insufficient data"
                )
                st.cache_resource.clear()
                st.rerun()

        st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

        # Display results
        if selected_forecast_cat:
            metrics = sarima.get_metrics(selected_forecast_cat)
            forecast = sarima.get_forecast(selected_forecast_cat)
            hist_data = sarima.get_historical_with_predictions(selected_forecast_cat)

            if metrics:
                # Metric cards
                m_col1, m_col2, m_col3, m_col4 = st.columns(4)
                with m_col1:
                    rmse_val = metrics.get('rmse', 'N/A')
                    st.markdown(f"""
                    <div class="kpi-card">
                        <div class="kpi-label">RMSE</div>
                        <div class="kpi-value">{rmse_val if isinstance(rmse_val, str) else f'{rmse_val:.2f}'}</div>
                    </div>
                    """, unsafe_allow_html=True)
                with m_col2:
                    mae_val = metrics.get('mae', 'N/A')
                    st.markdown(f"""
                    <div class="kpi-card">
                        <div class="kpi-label">MAE</div>
                        <div class="kpi-value">{mae_val if isinstance(mae_val, str) else f'{mae_val:.2f}'}</div>
                    </div>
                    """, unsafe_allow_html=True)
                with m_col3:
                    mape_val = metrics.get('mape', 'N/A')
                    st.markdown(f"""
                    <div class="kpi-card">
                        <div class="kpi-label">MAPE (%)</div>
                        <div class="kpi-value">{mape_val if isinstance(mape_val, str) or mape_val is None else f'{mape_val:.2f}'}</div>
                    </div>
                    """, unsafe_allow_html=True)
                with m_col4:
                    r2_val = metrics.get('r_squared', 'N/A')
                    st.markdown(f"""
                    <div class="kpi-card">
                        <div class="kpi-label">R²</div>
                        <div class="kpi-value">{r2_val if isinstance(r2_val, str) else f'{r2_val:.4f}'}</div>
                    </div>
                    """, unsafe_allow_html=True)



            # Forecast visualization
            if hist_data:
                fig = go.Figure()

                # Historical data
                full_series = hist_data.get('full_series')
                train = hist_data.get('train')
                test = hist_data.get('test')
                predictions = hist_data.get('predictions')

                if full_series is not None:
                    fig.add_trace(go.Scatter(
                        x=full_series.index, y=full_series.values,
                        mode='lines+markers', name='Historical Actual',
                        line=dict(color='#6366f1', width=2),
                        marker=dict(size=3),
                        hovertemplate='%{x|%b %Y}<br>Actual: %{y:.0f}<extra></extra>'
                    ))

                # Train/test split line
                if train is not None and test is not None and len(test) > 0:
                    fig.add_vline(
                        x=test.index[0], line_dash="dash",
                        line_color="rgba(255,255,255,0.3)",
                        annotation_text="Train/Test Split"
                    )

                # Validation predictions
                if predictions is not None and len(predictions) > 0:
                    fig.add_trace(go.Scatter(
                        x=test.index, y=predictions,
                        mode='lines+markers', name='Validation Predictions',
                        line=dict(color='#f59e0b', width=2, dash='dot'),
                        marker=dict(size=4),
                        hovertemplate='%{x|%b %Y}<br>Predicted: %{y:.0f}<extra></extra>'
                    ))

                # Forecast
                forecast_df = hist_data.get('forecast')
                if forecast_df is not None and not forecast_df.empty:
                    if 'forecast_date' in forecast_df.columns:
                        f_dates = pd.to_datetime(forecast_df['forecast_date'])
                    else:
                        f_dates = forecast_df.index

                    f_values = forecast_df['forecast_value']

                    fig.add_trace(go.Scatter(
                        x=f_dates, y=f_values,
                        mode='lines+markers', name='Forecast',
                        line=dict(color='#22c55e', width=3),
                        marker=dict(size=6, symbol='diamond'),
                        hovertemplate='%{x|%b %Y}<br>Forecast: %{y:.0f}<extra></extra>'
                    ))

                    # 80% CI
                    if 'lower_80' in forecast_df.columns and 'upper_80' in forecast_df.columns:
                        fig.add_trace(go.Scatter(
                            x=f_dates, y=forecast_df['upper_80'],
                            mode='lines', name='80% CI Upper',
                            line=dict(width=0), showlegend=False
                        ))
                        fig.add_trace(go.Scatter(
                            x=f_dates, y=forecast_df['lower_80'],
                            mode='lines', name='80% CI',
                            line=dict(width=0),
                            fill='tonexty',
                            fillcolor='rgba(34, 197, 94, 0.2)',
                            hovertemplate='80% CI: %{y:.0f}<extra></extra>'
                        ))

                    # 95% CI
                    if 'lower_95' in forecast_df.columns and 'upper_95' in forecast_df.columns:
                        fig.add_trace(go.Scatter(
                            x=f_dates, y=forecast_df['upper_95'],
                            mode='lines', name='95% CI Upper',
                            line=dict(width=0), showlegend=False
                        ))
                        fig.add_trace(go.Scatter(
                            x=f_dates, y=forecast_df['lower_95'],
                            mode='lines', name='95% CI',
                            line=dict(width=0),
                            fill='tonexty',
                            fillcolor='rgba(34, 197, 94, 0.1)',
                            hovertemplate='95% CI: %{y:.0f}<extra></extra>'
                        ))

                fig.update_layout(
                    title=f'SARIMA Forecast — {selected_forecast_cat}',
                    xaxis_title='Date', yaxis_title='Registered Cases',
                    template='plotly_dark',
                    hovermode='x unified',
                    legend=dict(orientation='h', yanchor='bottom', y=1.02)
                )

                st.plotly_chart(fig, use_container_width=True)

                # Forecast table
                if forecast_df is not None and not forecast_df.empty:
                    st.markdown("#### 📋 Forecast Details")
                    st.dataframe(forecast_df, use_container_width=True)

                    # Export forecast
                    csv_data = forecast_df.to_csv(index=False)
                    st.download_button(
                        "📥 Download Forecast (CSV)",
                        data=csv_data,
                        file_name=f"{selected_forecast_cat}_forecast.csv",
                        mime="text/csv"
                    )
            else:
                if not metrics:
                    st.info(
                        f"No SARIMA model trained for **{selected_forecast_cat}**. "
                        f"Click 'Train / Retrain SARIMA' to get started."
                    )


# ============================================================
# TAB 3 — IPC ↔ BNS CROSSWALK
# ============================================================

with tab3:
    st.markdown("## ⚖️ IPC ↔ BNS Crosswalk")
    st.markdown(
        "Search and browse the Indian Penal Code (IPC) 1860 to "
        "Bharatiya Nyaya Sanhita (BNS) 2023 statutory mapping."
    )

    crosswalk = get_crosswalk()

    # Search
    search_col1, search_col2 = st.columns([3, 1])
    with search_col1:
        search_query = st.text_input(
            "🔍 Search (IPC section, BNS section, crime name, or domain)",
            placeholder="e.g., 302, Murder, BNS 103, Robbery...",
            key="crosswalk_search"
        )
    with search_col2:
        domain_filter = st.selectbox(
            "🏷️ Filter by Domain",
            ['All'] + CRIME_DOMAINS,
            key="crosswalk_domain"
        )

    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

    # Get mapping data
    if search_query:
        results = crosswalk.search_mapping(search_query)
    else:
        results = crosswalk.get_full_mapping_table()

    # Apply domain filter
    if domain_filter != 'All':
        results = [r for r in results if r.get('Domain') == domain_filter]

    if results:
        results_df = pd.DataFrame(results)
        st.markdown(f"**Showing {len(results)} mappings**")
        st.dataframe(
            results_df,
            use_container_width=True,
            height=500,
            column_config={
                "IPC Section": st.column_config.TextColumn("IPC Section", width="medium"),
                "Crime": st.column_config.TextColumn("Crime", width="large"),
                "BNS Section": st.column_config.TextColumn("BNS Section", width="medium"),
                "Domain": st.column_config.TextColumn("Domain", width="medium"),
            }
        )

        # Download
        csv_data = results_df.to_csv(index=False)
        st.download_button(
            "📥 Download Mapping (CSV)",
            data=csv_data,
            file_name="ipc_bns_crosswalk.csv",
            mime="text/csv"
        )
    else:
        st.info("No matching records found.")

    # Quick lookup
    st.markdown("---")
    st.markdown("### 🔎 Quick Lookup")
    lookup_col1, lookup_col2 = st.columns(2)

    with lookup_col1:
        st.markdown("#### IPC → BNS")
        ipc_input = st.text_input("Enter IPC Section", placeholder="e.g., 302, IPC 376",
                                   key="ipc_lookup")
        if ipc_input:
            result = crosswalk.lookup_ipc(ipc_input)
            if result:
                st.success(f"**BNS Section:** {result['bns']}")
                st.markdown(f"**Crime:** {result['crime']}")
                st.markdown(f"**Domain:** {result['domain']}")
            else:
                st.warning("No mapping found for this IPC section.")

    with lookup_col2:
        st.markdown("#### BNS → IPC")
        bns_input = st.text_input("Enter BNS Section", placeholder="e.g., 103, BNS 64",
                                   key="bns_lookup")
        if bns_input:
            result = crosswalk.lookup_bns(bns_input)
            if result:
                st.success(f"**IPC Section:** {result['ipc']}")
                st.markdown(f"**Crime:** {result['crime']}")
                st.markdown(f"**Domain:** {result['domain']}")
            else:
                st.warning("No mapping found for this BNS section.")


# ============================================================
# TAB 4 — MONTHLY DATASET MANAGER
# ============================================================

with tab4:
    st.markdown("## 📋 Monthly Dataset Manager")

    db = get_database()
    processor = get_data_processor()
    validator = get_data_validator()

    # Current dataset info
    info_col1, info_col2, info_col3, info_col4 = st.columns(4)
    with info_col1:
        st.metric("Dataset Version", f"v{stats['current_version']}")
    with info_col2:
        st.metric("Total Records", f"{stats['total_records']:,}")
    with info_col3:
        latest_str = "N/A"
        if stats['latest_month'] and stats['latest_year']:
            month_names_full = [
                '', 'Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'
            ]
            latest_str = f"{month_names_full[stats['latest_month']]} {stats['latest_year']}"
        st.metric("Latest Month", latest_str)
    with info_col4:
        st.metric("Categories", stats['total_categories'])

    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

    # Sub-sections
    manager_action = st.radio(
        "Select Action",
        ["📊 View Dataset", "➕ Add Monthly Data", "📤 Upload Excel",
         "✏️ Edit Records", "🗑️ Delete Records", "📜 Version History"],
        horizontal=True, key="manager_action"
    )

    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

    # ---- VIEW DATASET ----
    if manager_action == "📊 View Dataset":
        st.markdown("### View Current Dataset")

        view_col1, view_col2, view_col3 = st.columns(3)
        with view_col1:
            v_years = ['All'] + [str(y) for y in db.get_unique_years()]
            v_year = st.selectbox("Year", v_years, key="view_year")
        with view_col2:
            v_months = ['All'] + [str(m) for m in range(1, 13)]
            v_month = st.selectbox("Month", v_months, key="view_month")
        with view_col3:
            v_cats = ['All'] + db.get_unique_categories()
            v_cat = st.selectbox("Category", v_cats, key="view_cat")

        view_df = db.get_records_filtered(
            year=int(v_year) if v_year != 'All' else None,
            month=int(v_month) if v_month != 'All' else None,
            category=v_cat if v_cat != 'All' else None
        )

        if not view_df.empty:
            st.markdown(f"**Showing {len(view_df)} records**")
            display_cols = ['record_id', 'crime_category', 'month', 'year',
                            'case_count', 'legal_section', 'domain',
                            'legal_section_bns', 'data_source']
            available_cols = [c for c in display_cols if c in view_df.columns]
            st.dataframe(view_df[available_cols], use_container_width=True, height=400)

            # Export
            csv_data = view_df.to_csv(index=False)
            st.download_button(
                "📥 Download Current View (CSV)",
                data=csv_data,
                file_name="crime_data_view.csv",
                mime="text/csv"
            )
        else:
            st.info("No records found matching filters.")

    # ---- ADD MONTHLY DATA ----
    elif manager_action == "➕ Add Monthly Data":
        st.markdown("### Add Monthly Crime Data")
        st.markdown("Enter crime data for a specific month manually.")

        add_col1, add_col2 = st.columns(2)
        with add_col1:
            add_year = st.number_input("Year", min_value=2000, max_value=2100,
                                        value=2026, key="add_year")
        with add_col2:
            month_names_full = [
                'January', 'February', 'March', 'April', 'May', 'June',
                'July', 'August', 'September', 'October', 'November', 'December'
            ]
            add_month_name = st.selectbox("Month", month_names_full, key="add_month")
            add_month = month_names_full.index(add_month_name) + 1

        # Check existing records
        existing = db.get_records_filtered(year=add_year, month=add_month)
        if not existing.empty:
            st.warning(
                f"⚠️ {len(existing)} records already exist for "
                f"{add_month_name} {add_year}. "
                f"Adding more will not overwrite existing records."
            )

        st.markdown("#### Enter Crime Records")
        st.markdown("Add rows below. Use the '+' button to add more rows.")

        # Dynamic entry using st.data_editor
        crosswalk = get_crosswalk()
        categories = db.get_unique_categories()
        if not categories:
            categories = [
                'Murder', 'Attempt_to_Murder', 'Robbery', 'Theft',
                'CAW_Rape', 'CAW_Molestation', 'Cheating_Fraud',
                'Kidnapping', 'NDPS', 'Cyber_Crime'
            ]

        default_data = pd.DataFrame({
            'crime_category': [''] * 3,
            'case_count': [0] * 3,
            'legal_section': [''] * 3,
            'domain': [''] * 3,
            'legal_section_bns': [''] * 3
        })

        edited_df = st.data_editor(
            default_data,
            num_rows="dynamic",
            use_container_width=True,
            column_config={
                "crime_category": st.column_config.SelectboxColumn(
                    "Crime Category",
                    options=categories + ['Other (type below)'],
                    required=True
                ),
                "case_count": st.column_config.NumberColumn(
                    "Case Count", min_value=0, required=True
                ),
                "legal_section": st.column_config.TextColumn("Legal Section (IPC)"),
                "domain": st.column_config.SelectboxColumn("Domain", options=CRIME_DOMAINS),
                "legal_section_bns": st.column_config.TextColumn("BNS Section"),
            },
            key="add_data_editor"
        )

        # Validate and save
        validate_col, save_col = st.columns(2)

        with validate_col:
            if st.button("✅ Validate", key="validate_add", use_container_width=True):
                # Filter out empty rows
                valid_entries = edited_df[
                    (edited_df['crime_category'].str.strip() != '') &
                    (edited_df['case_count'] > 0)
                ].copy()

                if valid_entries.empty:
                    st.error("No valid entries to validate.")
                else:
                    valid_entries['month'] = add_month
                    valid_entries['year'] = add_year
                    valid_entries['page'] = ''
                    valid_entries['source_file'] = 'manual_entry'
                    valid_entries['standard_group'] = ''

                    validation = validator.validate_dataframe(valid_entries)
                    summary = validation['summary']

                    st.markdown(f"""
                    | Metric | Count |
                    |--------|-------|
                    | Records Detected | {summary['total']} |
                    | Valid | {summary['valid']} |
                    | Invalid | {summary['invalid']} |
                    | Warnings | {summary['warnings']} |
                    | Duplicates | {summary['duplicates']} |
                    """)

                    if validation['warnings']:
                        for w in validation['warnings']:
                            st.warning(w)
                    if validation['errors']:
                        for e in validation['errors'][:10]:
                            st.error(e)

                    st.session_state['validated_add_data'] = valid_entries

        with save_col:
            if st.button("💾 Confirm & Add to Dataset", key="save_add",
                          use_container_width=True):
                if 'validated_add_data' in st.session_state:
                    data_to_save = st.session_state['validated_add_data']
                    version = db.create_version(
                        update_type='manual_add',
                        action_summary=f"Added {len(data_to_save)} records for {add_month_name} {add_year}"
                    )
                    result = db.insert_records(
                        data_to_save, data_source='manual',
                        dataset_version=version
                    )
                    st.success(
                        f"✅ Inserted {result['inserted']} records. "
                        f"Duplicates: {result['duplicates']}. "
                        f"Errors: {result['errors']}."
                    )
                    del st.session_state['validated_add_data']
                    st.cache_resource.clear()
                    st.rerun()
                else:
                    st.warning("Please validate data first.")

    # ---- UPLOAD EXCEL ----
    elif manager_action == "📤 Upload Excel":
        st.markdown("### Upload Excel Monthly Data")
        st.markdown("Upload an Excel file containing crime data for one or more months.")

        uploaded_file = st.file_uploader(
            "Upload Excel file (.xlsx)",
            type=['xlsx', 'xls'],
            key="upload_excel"
        )

        if uploaded_file:
            # Save uploaded file
            save_path = UPLOADS_DIR / uploaded_file.name
            with open(save_path, 'wb') as f:
                f.write(uploaded_file.getvalue())

            # Process
            import_result = processor.process_excel_import(str(save_path))

            if import_result['success']:
                st.success(f"✅ Excel file read successfully: {import_result['total_rows']} rows")

                # Show column mapping
                col_analysis = import_result.get('column_analysis', {})
                if col_analysis.get('column_mapping'):
                    with st.expander("📋 Column Mapping"):
                        for excel_col, req_col in col_analysis['column_mapping'].items():
                            st.markdown(f"- `{excel_col}` → `{req_col}`")

                    if col_analysis.get('unmapped_required'):
                        st.warning(
                            f"⚠️ Unmapped required columns: "
                            f"{', '.join(col_analysis['unmapped_required'])}"
                        )

                # Validation summary
                st.markdown(f"""
                | Metric | Count |
                |--------|-------|
                | Total Rows | {import_result['total_rows']} |
                | Valid | {import_result['valid_rows']} |
                | Invalid | {import_result['invalid_rows']} |
                | Warnings | {len(import_result['warnings'])} |
                """)

                # Show warnings
                for w in import_result['warnings'][:10]:
                    st.warning(w)

                # Preview
                valid_df = import_result.get('valid_df', pd.DataFrame())
                if not valid_df.empty:
                    st.markdown("#### Preview")
                    preview_cols = [c for c in PRIMARY_COLUMNS if c in valid_df.columns]
                    st.dataframe(valid_df[preview_cols].head(50),
                                 use_container_width=True)

                    # Confirm import
                    if st.button("💾 Confirm & Import to Database",
                                  key="confirm_excel_import",
                                  use_container_width=True):
                        version = db.create_version(
                            update_type='excel_import',
                            action_summary=f"Imported {len(valid_df)} records from {uploaded_file.name}"
                        )
                        result = db.insert_records(
                            valid_df, data_source='excel_import',
                            dataset_version=version
                        )
                        st.success(
                            f"✅ Imported {result['inserted']} records. "
                            f"Duplicates: {result['duplicates']}. "
                            f"Errors: {result['errors']}."
                        )
                        st.cache_resource.clear()
                        st.rerun()
            else:
                st.error(f"❌ Failed to process Excel: {import_result.get('error')}")

    # ---- EDIT RECORDS ----
    elif manager_action == "✏️ Edit Records":
        st.markdown("### Edit Existing Records")

        edit_col1, edit_col2 = st.columns(2)
        with edit_col1:
            edit_years = db.get_unique_years()
            if edit_years:
                edit_year = st.selectbox("Year", edit_years, key="edit_year")
            else:
                st.info("No data available.")
                edit_year = None
        with edit_col2:
            if edit_year:
                edit_month = st.selectbox("Month", list(range(1, 13)), key="edit_month")
            else:
                edit_month = None

        if edit_year and edit_month:
            edit_df = db.get_records_filtered(year=edit_year, month=edit_month)

            if not edit_df.empty:
                st.markdown(f"**{len(edit_df)} records for {edit_month}/{edit_year}**")

                edit_cols = ['record_id', 'crime_category', 'case_count',
                             'legal_section', 'domain', 'legal_section_bns']
                available_edit_cols = [c for c in edit_cols if c in edit_df.columns]

                edited_records = st.data_editor(
                    edit_df[available_edit_cols],
                    use_container_width=True,
                    disabled=['record_id'],
                    key="edit_records_editor"
                )

                if st.button("💾 Save Changes", key="save_edits",
                              use_container_width=True):
                    updated_count = 0
                    for idx, row in edited_records.iterrows():
                        original = edit_df.loc[idx]
                        changes = {}
                        for col in ['crime_category', 'case_count', 'legal_section',
                                     'domain', 'legal_section_bns']:
                            if col in row and col in original:
                                if str(row[col]) != str(original[col]):
                                    changes[col] = row[col]

                        if changes:
                            db.update_record(int(original['record_id']), changes)
                            updated_count += 1

                    if updated_count > 0:
                        db.create_version(
                            update_type='edit',
                            action_summary=f"Edited {updated_count} records for {edit_month}/{edit_year}"
                        )
                        st.success(f"✅ Updated {updated_count} records.")
                        st.cache_resource.clear()
                        st.rerun()
                    else:
                        st.info("No changes detected.")
            else:
                st.info(f"No records found for {edit_month}/{edit_year}.")

    # ---- DELETE RECORDS ----
    elif manager_action == "🗑️ Delete Records":
        st.markdown("### Delete Records")
        st.warning("⚠️ Deletion is permanent. Proceed with caution.")

        del_col1, del_col2 = st.columns(2)
        with del_col1:
            del_years = db.get_unique_years()
            if del_years:
                del_year = st.selectbox("Year", del_years, key="del_year")
            else:
                del_year = None
        with del_col2:
            if del_year:
                del_month = st.selectbox("Month", list(range(1, 13)), key="del_month")
            else:
                del_month = None

        if del_year and del_month:
            del_df = db.get_records_filtered(year=del_year, month=del_month)

            if not del_df.empty:
                st.markdown(f"**{len(del_df)} records for {del_month}/{del_year}**")
                display_cols = ['record_id', 'crime_category', 'case_count',
                                'legal_section', 'domain']
                available_cols = [c for c in display_cols if c in del_df.columns]
                st.dataframe(del_df[available_cols], use_container_width=True)

                # Select records to delete
                del_option = st.radio(
                    "Delete option",
                    ["Delete ALL records for this month",
                     "Delete selected records by ID"],
                    key="del_option"
                )

                if del_option == "Delete ALL records for this month":
                    st.error(
                        f"⚠️ You are about to delete {len(del_df)} records "
                        f"for {del_month}/{del_year}."
                    )
                    if st.button("🗑️ Confirm Delete All", key="confirm_del_all",
                                  type="primary"):
                        deleted = db.delete_records_by_month(del_year, del_month)
                        db.create_version(
                            update_type='delete',
                            action_summary=f"Deleted {deleted} records for {del_month}/{del_year}"
                        )
                        st.success(f"✅ Deleted {deleted} records.")
                        st.cache_resource.clear()
                        st.rerun()
                else:
                    record_ids = st.multiselect(
                        "Select Record IDs to delete",
                        del_df['record_id'].tolist(),
                        key="del_record_ids"
                    )
                    if record_ids:
                        st.error(f"⚠️ You are about to delete {len(record_ids)} records.")
                        if st.button("🗑️ Confirm Delete Selected", key="confirm_del_selected"):
                            deleted = db.delete_records(record_ids)
                            db.create_version(
                                update_type='delete',
                                action_summary=f"Deleted {deleted} selected records"
                            )
                            st.success(f"✅ Deleted {deleted} records.")
                            st.cache_resource.clear()
                            st.rerun()
            else:
                st.info(f"No records found for {del_month}/{del_year}.")

    # ---- VERSION HISTORY ----
    elif manager_action == "📜 Version History":
        st.markdown("### Dataset Version History")
        versions = db.get_version_history()
        if not versions.empty:
            st.dataframe(versions, use_container_width=True)
        else:
            st.info("No version history available.")


# ============================================================
# TAB 5 — RAW DATA OCR & PROCESSING HUB
# ============================================================

with tab5:
    st.markdown("## 🔬 Raw Data OCR & Processing Hub")
    st.markdown("Process scanned crime report PDFs or document images through the OCR pipeline.")

    # 3-step workflow
    ocr_step = st.radio(
        "Workflow Step",
        ["Step 1: Extract Data from PDF / Image",
         "Step 2: Review & Add to Clean Dataset",
         "Step 3: Retrain SARIMA & Update System"],
        key="ocr_step"
    )

    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

    # ---- STEP 1: EXTRACT DATA ----
    if ocr_step == "Step 1: Extract Data from PDF / Image":
        st.markdown("### Step 1 — Extract Data from PDF / Image")
        st.markdown(
            "Upload a crime report PDF or document image (PNG, JPG, JPEG, TIFF, BMP, WEBP). "
            "The system will run: File → OpenCV preprocessing → EasyOCR → Row parsing"
        )

        pdf_file = st.file_uploader(
            "Upload PDF or Image",
            type=['pdf', 'png', 'jpg', 'jpeg', 'tiff', 'bmp', 'webp'],
            key="ocr_pdf_upload"
        )

        ocr_col1, ocr_col2 = st.columns(2)
        with ocr_col1:
            ocr_default_month = st.selectbox(
                "Default Month (if not detected)",
                list(range(1, 13)), key="ocr_default_month"
            )
        with ocr_col2:
            ocr_default_year = st.number_input(
                "Default Year (if not detected)",
                min_value=2000, max_value=2100, value=2026,
                key="ocr_default_year"
            )

        if pdf_file and st.button("🔍 Run OCR Pipeline", key="run_ocr",
                                    use_container_width=True):
            # Save uploaded PDF
            pdf_save_path = UPLOADS_DIR / pdf_file.name
            with open(pdf_save_path, 'wb') as f:
                f.write(pdf_file.getvalue())

            with st.spinner("Running OCR pipeline... This may take a moment."):
                try:
                    from app.processing.batch_pdf_processor import BatchPDFProcessor
                    batch_processor = BatchPDFProcessor(db=db)
                    result = batch_processor.process_single_pdf(
                        str(pdf_save_path),
                        default_month=ocr_default_month,
                        default_year=ocr_default_year
                    )

                    st.success(f"✅ OCR complete! Pages: {result.get('total_pages', 0)}")

                    # Show results
                    accepted = result.get('accepted_records', [])
                    flagged = result.get('flagged_records', [])
                    rejected = result.get('rejected_records', [])

                    res_col1, res_col2, res_col3 = st.columns(3)
                    with res_col1:
                        st.metric("Accepted", len(accepted))
                    with res_col2:
                        st.metric("Flagged", len(flagged))
                    with res_col3:
                        st.metric("Rejected", len(rejected))

                    # Accepted records
                    if accepted:
                        st.markdown("#### ✅ Accepted Records")
                        acc_df = pd.DataFrame(accepted)
                        display = ['crime_category', 'case_count', 'month', 'year',
                                   'ocr_confidence']
                        avail = [c for c in display if c in acc_df.columns]
                        st.dataframe(acc_df[avail], use_container_width=True)

                    # Flagged records
                    if flagged:
                        st.markdown("#### ⚠️ Flagged Records (Low Confidence)")
                        flag_df = pd.DataFrame(flagged)
                        display = ['crime_category', 'case_count', 'month', 'year',
                                   'ocr_confidence', 'raw_text']
                        avail = [c for c in display if c in flag_df.columns]
                        st.dataframe(flag_df[avail], use_container_width=True)

                    # Rejected records
                    if rejected:
                        with st.expander("🚫 Rejected Records"):
                            rej_df = pd.DataFrame(rejected)
                            display = ['crime_category', 'raw_text', 'rejection_reason']
                            avail = [c for c in display if c in rej_df.columns]
                            st.dataframe(rej_df[avail], use_container_width=True)

                    # Store results for step 2
                    st.session_state['ocr_accepted'] = accepted
                    st.session_state['ocr_flagged'] = flagged

                    if result.get('errors'):
                        with st.expander("⚠️ Processing Errors"):
                            for err in result['errors']:
                                st.error(err)

                except ImportError as e:
                    st.error(
                        f"OCR dependencies not fully installed: {e}. "
                        f"Install with: pip install PyMuPDF opencv-python easyocr"
                    )
                except Exception as e:
                    st.error(f"OCR processing error: {e}")

    # ---- STEP 2: ADD TO DATASET ----
    elif ocr_step == "Step 2: Review & Add to Clean Dataset":
        st.markdown("### Step 2 — Review & Add to Clean Dataset")

        if 'ocr_accepted' in st.session_state:
            accepted = st.session_state['ocr_accepted']
            flagged = st.session_state.get('ocr_flagged', [])

            all_records = accepted + flagged
            st.markdown(f"**{len(all_records)} records ready for review**")

            if all_records:
                review_df = pd.DataFrame(all_records)
                display_cols = ['crime_category', 'case_count', 'month', 'year',
                                'domain', 'legal_section', 'legal_section_bns',
                                'ocr_confidence', 'validation_status']
                avail = [c for c in display_cols if c in review_df.columns]
                st.dataframe(review_df[avail], use_container_width=True)

                if st.button("💾 Confirm & Save to Database", key="ocr_save",
                              use_container_width=True):
                    from app.processing.batch_pdf_processor import BatchPDFProcessor
                    batch_proc = BatchPDFProcessor(db=db)
                    result = batch_proc.save_accepted_records(all_records, data_source='ocr')
                    st.success(
                        f"✅ Saved {result['inserted']} records. "
                        f"Duplicates: {result['duplicates']}."
                    )
                    del st.session_state['ocr_accepted']
                    if 'ocr_flagged' in st.session_state:
                        del st.session_state['ocr_flagged']
                    st.cache_resource.clear()
                    st.rerun()
        else:
            st.info("No OCR results available. Run Step 1 first.")

    # ---- STEP 3: RETRAIN ----
    elif ocr_step == "Step 3: Retrain SARIMA & Update System":
        st.markdown("### Step 3 — Retrain SARIMA & Update System")

        if stats['total_records'] == 0:
            st.info("No data available. Import data first.")
        else:
            st.markdown(
                "Click below to retrain all SARIMA models with the updated dataset."
            )

            retrain_horizon = st.selectbox(
                "Forecast Horizon",
                [3, 4, 5, 6], index=3, key="retrain_horizon"
            )

            if st.button("🚀 Retrain All SARIMA Models", key="ocr_retrain",
                          use_container_width=True, type="primary"):
                sarima = get_sarima_model()
                with st.spinner("Retraining SARIMA models..."):
                    results = sarima.train_all_categories(
                        forecast_horizon=retrain_horizon
                    )

                st.success(
                    f"✅ Retraining complete!\n\n"
                    f"- Successful: {results['successful']}\n"
                    f"- Failed: {results['failed']}\n"
                    f"- Insufficient data: {results['insufficient_data']}"
                )

                # Show per-category results
                with st.expander("📋 Detailed Results"):
                    for cat, res in results['category_results'].items():
                        if res['success']:
                            m = res.get('metrics', {})
                            st.markdown(
                                f"✅ **{cat}**: RMSE={m.get('rmse', 'N/A')}, "
                                f"R²={m.get('r_squared', 'N/A')}"
                            )
                        else:
                            st.markdown(
                                f"❌ **{cat}**: {res.get('error', 'Unknown error')}"
                            )

                st.cache_resource.clear()
                st.rerun()

            st.markdown("---")

            # Processing logs
            st.markdown("### 📜 Processing Logs")
            logs = db.get_processing_logs()
            if not logs.empty:
                st.dataframe(logs.head(50), use_container_width=True)
            else:
                st.info("No processing logs yet.")
