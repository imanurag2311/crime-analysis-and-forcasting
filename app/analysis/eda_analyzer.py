"""
Exploratory Data Analysis (EDA) Analyzer
==========================================
Generates interactive Plotly charts for crime trend analysis.
Supports monthly/annual trends, category distributions, rolling averages,
seasonal patterns, and YoY comparisons.
"""

import logging
from typing import Optional, List, Dict, Any

import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from app.core.config import ROLLING_WINDOW_3, ROLLING_WINDOW_6, MONTH_NAME_MAP
from app.processing.data_processor import DataProcessor
from app.processing.ipc_bns_crosswalk import IPCBNSCrosswalk
from app.database.database import CrimeDatabase

logger = logging.getLogger(__name__)

# Month names for display
MONTH_NAMES = [
    '', 'January', 'February', 'March', 'April', 'May', 'June',
    'July', 'August', 'September', 'October', 'November', 'December'
]

# Professional color palette
COLORS = [
    '#6366f1', '#ec4899', '#14b8a6', '#f59e0b', '#ef4444',
    '#8b5cf6', '#06b6d4', '#84cc16', '#f97316', '#d946ef',
    '#0ea5e9', '#22c55e', '#e11d48', '#a855f7', '#eab308'
]


class EDAAnalyzer:
    """
    Generates exploratory data analysis visualizations.
    """

    def __init__(self, db: Optional[CrimeDatabase] = None):
        """Initialize with database connection."""
        self.db = db or CrimeDatabase()
        self.processor = DataProcessor()
        self.crosswalk = IPCBNSCrosswalk()

    def _get_data(self, year: Optional[int] = None,
                  category: Optional[str] = None,
                  domain: Optional[str] = None) -> pd.DataFrame:
        """Get filtered data from database."""
        df = self.db.get_records_filtered(year=year, category=category, domain=domain)
        return df

    # ================================================================
    # MONTHLY CRIME TREND
    # ================================================================

    def plot_monthly_trend(self, category: Optional[str] = None,
                           year: Optional[int] = None) -> go.Figure:
        """Create monthly crime trend line chart."""
        df = self._get_data(year=year, category=category)
        if df.empty:
            return self._empty_figure("No data available for monthly trend")

        if category:
            ts = self.processor.create_monthly_time_series(df, category=category)
            if ts.empty:
                return self._empty_figure(f"No data for {category}")

            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=ts.index, y=ts['total_cases'],
                mode='lines+markers',
                name=category,
                line=dict(color=COLORS[0], width=2),
                marker=dict(size=4),
                hovertemplate='%{x|%b %Y}<br>Cases: %{y}<extra></extra>'
            ))

            # Add rolling averages
            ts = self.processor.add_rolling_averages(ts)
            if f'rolling_avg_{ROLLING_WINDOW_3}m' in ts.columns:
                fig.add_trace(go.Scatter(
                    x=ts.index, y=ts[f'rolling_avg_{ROLLING_WINDOW_3}m'],
                    mode='lines', name=f'{ROLLING_WINDOW_3}-month MA',
                    line=dict(color=COLORS[1], width=1.5, dash='dash'),
                    hovertemplate='%{x|%b %Y}<br>3m Avg: %{y:.1f}<extra></extra>'
                ))
            if f'rolling_avg_{ROLLING_WINDOW_6}m' in ts.columns:
                fig.add_trace(go.Scatter(
                    x=ts.index, y=ts[f'rolling_avg_{ROLLING_WINDOW_6}m'],
                    mode='lines', name=f'{ROLLING_WINDOW_6}-month MA',
                    line=dict(color=COLORS[2], width=1.5, dash='dot'),
                    hovertemplate='%{x|%b %Y}<br>6m Avg: %{y:.1f}<extra></extra>'
                ))

            fig.update_layout(
                title=f'Monthly Crime Trend — {category}',
                xaxis_title='Date', yaxis_title='Registered Cases',
                template='plotly_dark',
                hovermode='x unified'
            )
        else:
            # All categories aggregated
            monthly = df.groupby(['year', 'month']).agg(
                total=('case_count', 'sum')
            ).reset_index()
            monthly['date'] = pd.to_datetime(
                monthly['year'].astype(str) + '-' +
                monthly['month'].astype(str).str.zfill(2) + '-01'
            )
            monthly = monthly.sort_values('date')

            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=monthly['date'], y=monthly['total'],
                mode='lines+markers',
                name='Total Cases',
                line=dict(color=COLORS[0], width=2),
                fill='tozeroy',
                fillcolor='rgba(99, 102, 241, 0.1)',
                hovertemplate='%{x|%b %Y}<br>Total Cases: %{y}<extra></extra>'
            ))

            fig.update_layout(
                title='Monthly Crime Trend — All Categories',
                xaxis_title='Date', yaxis_title='Total Registered Cases',
                template='plotly_dark',
                hovermode='x unified'
            )

        return fig

    # ================================================================
    # ANNUAL TREND
    # ================================================================

    def plot_annual_trend(self, category: Optional[str] = None) -> go.Figure:
        """Create annual crime trend bar chart."""
        df = self._get_data(category=category)
        if df.empty:
            return self._empty_figure("No data available for annual trend")

        annual = df.groupby('year').agg(total=('case_count', 'sum')).reset_index()

        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=annual['year'], y=annual['total'],
            marker_color=COLORS[0],
            text=annual['total'],
            textposition='auto',
            hovertemplate='Year: %{x}<br>Total Cases: %{y}<extra></extra>'
        ))

        fig.update_layout(
            title=f'Annual Crime Trend{" — " + category if category else ""}',
            xaxis_title='Year', yaxis_title='Total Registered Cases',
            template='plotly_dark'
        )

        return fig

    # ================================================================
    # CATEGORY DISTRIBUTION
    # ================================================================

    def plot_category_distribution(self, year: Optional[int] = None,
                                    top_n: int = 15) -> go.Figure:
        """Create crime category distribution chart."""
        df = self._get_data(year=year)
        if df.empty:
            return self._empty_figure("No data available")

        dist = self.processor.get_category_distribution(df)
        if len(dist) > top_n:
            dist = dist.head(top_n)

        display_categories = [self.crosswalk.get_display_name(c) for c in dist['crime_category']]

        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=dist['total_cases'],
            y=display_categories,
            orientation='h',
            marker_color=COLORS[:len(dist)],
            text=[f"{v:,} ({p:.1f}%)" for v, p in
                  zip(dist['total_cases'], dist['percentage'])],
            textposition='auto',
            hovertemplate='%{y}<br>Cases: %{x:,}<extra></extra>'
        ))

        fig.update_layout(
            title=f'Crime Category Distribution (Top {top_n})'
                  f'{" — " + str(year) if year else ""}',
            xaxis_title='Total Registered Cases',
            yaxis_title='',
            template='plotly_dark',
            height=max(400, len(dist) * 35),
            yaxis=dict(autorange='reversed')
        )

        return fig

    # ================================================================
    # SEASONAL PATTERNS
    # ================================================================

    def plot_seasonal_patterns(self, category: Optional[str] = None) -> go.Figure:
        """Create seasonal pattern heatmap / box plot."""
        df = self._get_data(category=category)
        if df.empty:
            return self._empty_figure("No data available")

        monthly_avg = df.groupby('month').agg(
            avg_cases=('case_count', 'mean'),
            total_cases=('case_count', 'sum'),
            count=('case_count', 'count')
        ).reset_index()

        monthly_avg['month_name'] = monthly_avg['month'].apply(
            lambda m: MONTH_NAMES[int(m)] if 1 <= int(m) <= 12 else str(m)
        )

        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=monthly_avg['month_name'],
            y=monthly_avg['avg_cases'],
            marker_color=[COLORS[i % len(COLORS)] for i in range(len(monthly_avg))],
            text=monthly_avg['avg_cases'].round(1),
            textposition='auto',
            hovertemplate='%{x}<br>Avg Cases: %{y:.1f}<extra></extra>'
        ))

        fig.update_layout(
            title=f'Seasonal Pattern — Average Monthly Cases'
                  f'{" — " + category if category else ""}',
            xaxis_title='Month', yaxis_title='Average Registered Cases',
            template='plotly_dark'
        )

        return fig

    # ================================================================
    # YEAR-OVER-YEAR ANALYSIS
    # ================================================================

    def plot_yoy_analysis(self, category: Optional[str] = None) -> go.Figure:
        """Create Year-over-Year comparison chart."""
        df = self._get_data(category=category)
        if df.empty:
            return self._empty_figure("No data available")

        annual = df.groupby('year').agg(total=('case_count', 'sum')).reset_index()
        annual = annual.sort_values('year')
        annual['yoy_change'] = annual['total'].pct_change() * 100

        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=('Annual Cases', 'Year-over-Year Change (%)'),
            vertical_spacing=0.15
        )

        fig.add_trace(go.Bar(
            x=annual['year'], y=annual['total'],
            name='Total Cases',
            marker_color=COLORS[0],
            hovertemplate='Year: %{x}<br>Cases: %{y:,}<extra></extra>'
        ), row=1, col=1)

        colors_yoy = ['#22c55e' if v >= 0 else '#ef4444'
                       for v in annual['yoy_change'].fillna(0)]
        fig.add_trace(go.Bar(
            x=annual['year'], y=annual['yoy_change'],
            name='YoY Change %',
            marker_color=colors_yoy,
            text=annual['yoy_change'].apply(
                lambda x: f"{x:+.1f}%" if pd.notna(x) else "N/A"
            ),
            textposition='outside',
            hovertemplate='Year: %{x}<br>YoY Change: %{y:+.1f}%<extra></extra>'
        ), row=2, col=1)

        fig.update_layout(
            title=f'Year-over-Year Analysis'
                  f'{" — " + category if category else ""}',
            template='plotly_dark',
            showlegend=False,
            height=600
        )

        return fig

    # ================================================================
    # CATEGORY COMPARISON
    # ================================================================

    def plot_category_comparison(self, categories: List[str],
                                  year: Optional[int] = None) -> go.Figure:
        """Compare multiple crime categories on the same chart."""
        fig = go.Figure()

        for i, cat in enumerate(categories):
            df = self._get_data(category=cat, year=year)
            if df.empty:
                continue

            ts = self.processor.create_monthly_time_series(df, category=cat)
            if ts.empty:
                continue

            fig.add_trace(go.Scatter(
                x=ts.index, y=ts['total_cases'],
                mode='lines+markers',
                name=cat,
                line=dict(color=COLORS[i % len(COLORS)], width=2),
                marker=dict(size=4),
                hovertemplate='%{x|%b %Y}<br>' + cat + ': %{y}<extra></extra>'
            ))

        fig.update_layout(
            title='Crime Category Comparison',
            xaxis_title='Date', yaxis_title='Registered Cases',
            template='plotly_dark',
            hovermode='x unified',
            legend=dict(orientation='h', yanchor='bottom', y=1.02)
        )

        return fig

    # ================================================================
    # DOMAIN DISTRIBUTION
    # ================================================================

    def plot_domain_distribution(self, year: Optional[int] = None) -> go.Figure:
        """Create domain-wise crime distribution pie chart."""
        df = self._get_data(year=year)
        if df.empty:
            return self._empty_figure("No data available")

        domain_dist = self.processor.get_domain_distribution(df)
        domain_dist = domain_dist[domain_dist['domain'].notna() & (domain_dist['domain'] != '')]

        if domain_dist.empty:
            return self._empty_figure("No domain data available")

        fig = go.Figure(data=[go.Pie(
            labels=domain_dist['domain'],
            values=domain_dist['total_cases'],
            hole=0.4,
            marker_colors=COLORS[:len(domain_dist)],
            textinfo='label+percent',
            hovertemplate='%{label}<br>Cases: %{value:,}<br>Share: %{percent}<extra></extra>'
        )])

        fig.update_layout(
            title=f'Crime Distribution by Domain'
                  f'{" — " + str(year) if year else ""}',
            template='plotly_dark'
        )

        return fig

    # ================================================================
    # CRIME SPIKE DETECTION
    # ================================================================

    def detect_spikes(self, category: Optional[str] = None,
                      threshold: float = 2.0) -> pd.DataFrame:
        """
        Detect unusual crime spikes (values > mean + threshold*std).
        
        Returns DataFrame of spike observations.
        """
        df = self._get_data(category=category)
        if df.empty:
            return pd.DataFrame()

        ts = self.processor.create_monthly_time_series(df, category=category)
        if ts.empty or 'total_cases' not in ts.columns:
            return pd.DataFrame()

        mean_val = ts['total_cases'].mean()
        std_val = ts['total_cases'].std()

        if std_val == 0:
            return pd.DataFrame()

        spike_threshold = mean_val + threshold * std_val
        spikes = ts[ts['total_cases'] > spike_threshold].copy()
        spikes['deviation'] = (
            (spikes['total_cases'] - mean_val) / std_val
        ).round(2)

        return spikes

    def plot_spike_detection(self, category: Optional[str] = None) -> go.Figure:
        """Visualize crime spikes on the time series."""
        df = self._get_data(category=category)
        if df.empty:
            return self._empty_figure("No data available")

        ts = self.processor.create_monthly_time_series(df, category=category)
        if ts.empty:
            return self._empty_figure("No time series data")

        mean_val = ts['total_cases'].mean()
        std_val = ts['total_cases'].std()
        threshold = mean_val + 2 * std_val

        spikes = ts[ts['total_cases'] > threshold]

        fig = go.Figure()

        fig.add_trace(go.Scatter(
            x=ts.index, y=ts['total_cases'],
            mode='lines',
            name='Monthly Cases',
            line=dict(color=COLORS[0], width=2)
        ))

        # Mean line
        fig.add_hline(y=mean_val, line_dash="dash",
                       line_color="gray",
                       annotation_text=f"Mean: {mean_val:.0f}")

        # Threshold line
        fig.add_hline(y=threshold, line_dash="dot",
                       line_color="#ef4444",
                       annotation_text=f"Spike Threshold: {threshold:.0f}")

        # Spike markers
        if not spikes.empty:
            fig.add_trace(go.Scatter(
                x=spikes.index, y=spikes['total_cases'],
                mode='markers',
                name='Spikes',
                marker=dict(color='#ef4444', size=10, symbol='star'),
                hovertemplate='%{x|%b %Y}<br>Cases: %{y}<br>⚠️ SPIKE<extra></extra>'
            ))

        fig.update_layout(
            title=f'Crime Spike Detection'
                  f'{" — " + category if category else ""}',
            xaxis_title='Date', yaxis_title='Registered Cases',
            template='plotly_dark'
        )

        return fig

    # ================================================================
    # ROLLING AVERAGES
    # ================================================================

    def plot_rolling_averages(self, category: str) -> go.Figure:
        """Plot time series with 3-month and 6-month rolling averages."""
        df = self._get_data(category=category)
        if df.empty:
            return self._empty_figure(f"No data for {category}")

        ts = self.processor.create_monthly_time_series(df, category=category)
        if ts.empty:
            return self._empty_figure(f"No time series for {category}")

        ts = self.processor.add_rolling_averages(ts)

        fig = go.Figure()

        fig.add_trace(go.Scatter(
            x=ts.index, y=ts['total_cases'],
            mode='lines+markers', name='Monthly Cases',
            line=dict(color=COLORS[0], width=1),
            marker=dict(size=3),
            opacity=0.6
        ))

        if f'rolling_avg_{ROLLING_WINDOW_3}m' in ts.columns:
            fig.add_trace(go.Scatter(
                x=ts.index, y=ts[f'rolling_avg_{ROLLING_WINDOW_3}m'],
                mode='lines', name='3-Month MA',
                line=dict(color=COLORS[1], width=2)
            ))

        if f'rolling_avg_{ROLLING_WINDOW_6}m' in ts.columns:
            fig.add_trace(go.Scatter(
                x=ts.index, y=ts[f'rolling_avg_{ROLLING_WINDOW_6}m'],
                mode='lines', name='6-Month MA',
                line=dict(color=COLORS[2], width=2.5)
            ))

        fig.update_layout(
            title=f'Rolling Averages — {category}',
            xaxis_title='Date', yaxis_title='Cases',
            template='plotly_dark',
            hovermode='x unified'
        )

        return fig

    # ================================================================
    # HELPER
    # ================================================================

    def _empty_figure(self, message: str) -> go.Figure:
        """Create an empty figure with a message."""
        fig = go.Figure()
        fig.add_annotation(
            text=message,
            xref="paper", yref="paper",
            x=0.5, y=0.5,
            showarrow=False,
            font=dict(size=16, color="gray")
        )
        fig.update_layout(
            template='plotly_dark',
            xaxis=dict(showgrid=False, zeroline=False, visible=False),
            yaxis=dict(showgrid=False, zeroline=False, visible=False)
        )
        return fig
