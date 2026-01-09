"""
BiasGuard - Interactive Visualization Engine
Microsoft Imagine Cup (2026 Season)

Purpose: Generate research-grade, explainable, and decision-ready
interactive HTML visualizations using Plotly.

Outputs:
- Comprehensive fairness dashboards
- Radar charts for metric comparison
- Bias detection visualizations
"""

import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from config import Config


class BiasVisualizer:
    """
    Generate interactive HTML visualizations for bias analysis.

    Uses Plotly for production-quality charts that are:
    - Interactive (hover, zoom, pan)
    - Exportable (PNG, SVG)
    - Accessible (screen reader compatible)
    - Professional (Microsoft design language)
    """

    def __init__(self):
        """Initialize visualizer with Microsoft-inspired color scheme."""
        print("\n" + "=" * 80)
        print("VISUALIZATION ENGINE INITIALIZED")
        print("=" * 80)
        self.colors = Config.COLORS

    def create_comprehensive_dashboard(self, data, baseline_metrics,
                                       fair_metrics, output_path):
        """
        Create comprehensive 6-chart fairness dashboard.

        Charts included:
        1. Hiring rate by gender (outcome fairness)
        2. Interview score distribution (input consistency)
        3. Experience vs hiring rate (merit alignment)
        4. Department-level disparity (organizational patterns)
        5. Education impact on hiring (credential influence)
        6. Selection rate comparison (baseline vs BiasGuard)

        Args:
            data: Full hiring dataset (DataFrame)
            baseline_metrics: Fairness metrics before mitigation
            fair_metrics: Fairness metrics after mitigation
            output_path: Path for output HTML file
        """

        print("\nGenerating comprehensive dashboard...")

        # ============================================================
        # Create 3x2 Subplot Grid
        # ============================================================
        fig = make_subplots(
            rows=3, cols=2,
            subplot_titles=(
                'Hiring Rate by Gender<br><sub>Outcome-level fairness</sub>',
                'Interview Score Distribution<br><sub>Input signal consistency</sub>',
                'Experience vs Hiring Rate<br><sub>Merit vs outcome alignment</sub>',
                'Department-Level Disparity<br><sub>Organizational bias patterns</sub>',
                'Education Impact on Hiring<br><sub>Credential influence</sub>',
                'Selection Rate Comparison<br><sub>Baseline vs BiasGuard</sub>'
            ),
            specs=[
                [{'type': 'bar'}, {'type': 'box'}],
                [{'type': 'scatter'}, {'type': 'bar'}],
                [{'type': 'bar'}, {'type': 'bar'}]
            ],
            vertical_spacing=0.12,
            horizontal_spacing=0.15
        )

        # ============================================================
        # Chart 1: Hiring Rate by Gender
        # Shows overall outcome disparity
        # ============================================================
        gender_stats = data.groupby('gender')['hired'].agg(
            ['mean', 'count']
        ).reset_index()

        fig.add_trace(
            go.Bar(
                x=gender_stats['gender'],
                y=gender_stats['mean'] * 100,
                text=[f"{v:.1f}%" for v in gender_stats['mean'] * 100],
                textposition='outside',
                marker_color=[self.colors['male'], self.colors['female']],
                name='Hiring Rate',
                showlegend=False
            ),
            row=1, col=1
        )

        # ============================================================
        # Chart 2: Interview Score Distribution
        # Shows if input data is fair (should be similar distributions)
        # ============================================================
        for gender in ['male', 'female']:
            fig.add_trace(
                go.Box(
                    y=data[data['gender'] == gender]['interview_score'],
                    name=gender.capitalize(),
                    marker_color=self.colors[gender],
                    boxmean='sd',  # Show mean and standard deviation
                    showlegend=False
                ),
                row=1, col=2
            )

        # ============================================================
        # Chart 3: Experience vs Hiring Rate
        # Shows if experience translates fairly to hiring outcomes
        # ============================================================
        # Bin experience into 5 groups for clarity
        exp_bins = pd.cut(data['years_experience'], bins=5)
        exp_stats = data.groupby(
            [exp_bins, 'gender']
        )['hired'].mean().reset_index()

        for gender in ['male', 'female']:
            gender_data = exp_stats[exp_stats['gender'] == gender]
            fig.add_trace(
                go.Scatter(
                    x=[str(b) for b in gender_data['years_experience']],
                    y=gender_data['hired'] * 100,
                    mode='lines+markers',
                    name=gender.capitalize(),
                    line=dict(color=self.colors[gender], width=3),
                    marker=dict(size=10),
                    showlegend=False
                ),
                row=2, col=1
            )

        # ============================================================
        # Chart 4: Department-Level Disparity
        # Shows if bias exists across all departments
        # ============================================================
        dept_stats = data.groupby(
            ['department', 'gender']
        )['hired'].mean().unstack()

        for gender in ['male', 'female']:
            fig.add_trace(
                go.Bar(
                    x=dept_stats.index,
                    y=dept_stats[gender] * 100,
                    name=gender.capitalize(),
                    marker_color=self.colors[gender],
                    showlegend=False
                ),
                row=2, col=2
            )

        # ============================================================
        # Chart 5: Education Impact on Hiring
        # Shows if credentials affect genders differently
        # ============================================================
        edu_stats = data.groupby(
            ['education_level', 'gender']
        )['hired'].mean().unstack()
        edu_labels = ['High School', 'Bachelor', 'Master', 'PhD']

        for gender in ['male', 'female']:
            fig.add_trace(
                go.Bar(
                    x=edu_labels,
                    y=edu_stats[gender] * 100,
                    name=gender.capitalize(),
                    marker_color=self.colors[gender],
                    showlegend=True  # Show legend only once
                ),
                row=3, col=1
            )

        # ============================================================
        # Chart 6: Model Comparison (Baseline vs BiasGuard)
        # Shows improvement after bias mitigation
        # ============================================================
        for gender in ['Male', 'Female']:
            gender_key = f'{gender.lower()}_selection_rate'

            fig.add_trace(
                go.Bar(
                    x=['Baseline', 'BiasGuard'],
                    y=[
                        baseline_metrics[gender_key] * 100,
                        fair_metrics[gender_key] * 100
                    ],
                    name=gender,
                    marker_color=self.colors[gender.lower()],
                    text=[
                        f"{baseline_metrics[gender_key] * 100:.1f}%",
                        f"{fair_metrics[gender_key] * 100:.1f}%"
                    ],
                    textposition='outside',
                    showlegend=False
                ),
                row=3, col=2
            )

        # ============================================================
        # Global Layout Settings
        # ============================================================
        fig.update_layout(
            height=1400,
            width=1400,
            title_text=(
                "<b>BiasGuard Hiring Fairness Dashboard</b><br>"
                "<sub>Transparent, explainable, and responsible AI insights</sub>"
            ),
            title_font_size=26,
            showlegend=True,
            font=dict(
                family=Config.CHART_CONFIG['font_family'],
                size=12
            ),
            # Add footer attribution
            annotations=[dict(
                text="BiasGuard | Microsoft Imagine Cup (2026 Season)",
                showarrow=False,
                xref="paper",
                yref="paper",
                x=0.5,
                y=-0.05,
                xanchor='center',
                font=dict(size=10, color='gray')
            )]
        )

        # Save to HTML file
        fig.write_html(output_path)
        print(f"✓ Dashboard saved: {output_path}")

    def create_fairness_radar(self, baseline_metrics, fair_metrics, output_path):
        """
        Create fairness metrics radar chart for visual comparison.

        Compares baseline vs BiasGuard across multiple fairness dimensions.
        Higher scores = better fairness.

        Args:
            baseline_metrics: Fairness metrics before mitigation
            fair_metrics: Fairness metrics after mitigation
            output_path: Path for output HTML file
        """

        print("\nGenerating fairness radar chart...")

        # ============================================================
        # Define Fairness Dimensions
        # ============================================================
        categories = [
            'Demographic Parity',
            'EEOC Compliance',
            'Selection Balance',
            'Overall Fairness'
        ]

        # ============================================================
        # Calculate Baseline Fairness Scores (0-100 scale)
        # FIX: Use only metrics that actually exist
        # ============================================================
        baseline_scores = [
            # Demographic parity: penalize large differences
            max(0, 100 - abs(
                baseline_metrics['demographic_parity_difference']
            ) * 500),

            # EEOC compliance: direct percentage
            baseline_metrics['eeoc_80_percent_ratio'] * 100,

            # Selection balance: penalize rate differences
            max(0, 100 - abs(
                baseline_metrics['male_selection_rate'] -
                baseline_metrics['female_selection_rate']
            ) * 500),

            # Overall fairness: weighted EEOC score
            baseline_metrics['eeoc_80_percent_ratio'] * 75
        ]

        # ============================================================
        # Calculate BiasGuard Fairness Scores
        # ============================================================
        fair_scores = [
            max(0, 100 - abs(
                fair_metrics['demographic_parity_difference']
            ) * 500),

            fair_metrics['eeoc_80_percent_ratio'] * 100,

            max(0, 100 - abs(
                fair_metrics['male_selection_rate'] -
                fair_metrics['female_selection_rate']
            ) * 500),

            fair_metrics['eeoc_80_percent_ratio'] * 75
        ]

        # ============================================================
        # Create Radar Chart
        # ============================================================
        fig = go.Figure()

        # Baseline model trace (red, showing problems)
        fig.add_trace(go.Scatterpolar(
            r=baseline_scores,
            theta=categories,
            fill='toself',
            name='Baseline Model',
            line=dict(color=self.colors['danger'], width=3),
            fillcolor='rgba(209,52,56,0.2)'
        ))

        # BiasGuard model trace (green, showing improvement)
        fig.add_trace(go.Scatterpolar(
            r=fair_scores,
            theta=categories,
            fill='toself',
            name='BiasGuard Model',
            line=dict(color=self.colors['success'], width=3),
            fillcolor='rgba(16,124,16,0.2)'
        ))

        # ============================================================
        # Layout Configuration
        # ============================================================
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 100],
                    ticksuffix='%'
                )
            ),
            title="<b>Fairness Improvement Overview</b><br><sub>Higher scores indicate better fairness</sub>",
            font=dict(
                family=Config.CHART_CONFIG['font_family'],
                size=14
            ),
            height=700,
            width=900,
            annotations=[dict(
                text="BiasGuard | Microsoft Imagine Cup (2026 Season)",
                showarrow=False,
                xref="paper",
                yref="paper",
                x=0.5,
                y=-0.1,
                xanchor='center',
                font=dict(size=10, color='gray')
            )]
        )

        fig.write_html(output_path)
        print(f"✓ Fairness radar saved: {output_path}")

    def create_language_bias_chart(self, language_analysis, output_path):
        """
        Create job description bias analysis visualization.

        Shows results from Azure AI Language Service analysis:
        - Bias scores per job
        - Distribution of bias levels
        - Category breakdown
        - Most frequent biased terms

        Args:
            language_analysis: Results from Azure AI Language analysis
            output_path: Path for output HTML file
        """

        # Validate input data
        if not language_analysis or 'results' not in language_analysis:
            print("  ⚠️  No language analysis data to visualize")
            return

        print("\nGenerating language bias analysis...")

        results = language_analysis['results']

        # ============================================================
        # Create 2x2 Subplot Grid
        # ============================================================
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                'Bias Score by Job<br><sub>Language risk intensity</sub>',
                'Bias Level Distribution<br><sub>Overall language health</sub>',
                'Bias Categories<br><sub>Types of exclusion</sub>',
                'Top Biased Terms<br><sub>Most frequent risks</sub>'
            ),
            specs=[
                [{'type': 'bar'}, {'type': 'pie'}],
                [{'type': 'bar'}, {'type': 'bar'}]
            ]
        )

        # ============================================================
        # Chart 1: Bias Score by Job
        # ============================================================
        job_titles = [r['job_title'][:20] for r in results]  # Truncate long titles
        bias_scores = [r['bias_score'] for r in results]

        # Color-code bars by severity
        bar_colors = [
            self.colors['danger'] if s >= 50 else
            self.colors['warning'] if s >= 25 else
            self.colors['success']
            for s in bias_scores
        ]

        fig.add_trace(
            go.Bar(
                x=job_titles,
                y=bias_scores,
                marker_color=bar_colors,
                text=[f"{s}/100" for s in bias_scores],
                textposition='outside',
                showlegend=False
            ),
            row=1, col=1
        )

        # ============================================================
        # Chart 2: Bias Level Distribution (Pie Chart)
        # FIX: Compute summary from results (not from non-existent key)
        # ============================================================
        bias_levels = {}
        for r in results:
            level = r['bias_level']
            bias_levels[level] = bias_levels.get(level, 0) + 1

        fig.add_trace(
            go.Pie(
                labels=list(bias_levels.keys()),
                values=list(bias_levels.values()),
                marker=dict(
                    colors=[
                        self.colors['danger'],
                        self.colors['warning'],
                        self.colors['success']
                    ]
                ),
                showlegend=True
            ),
            row=1, col=2
        )

        # ============================================================
        # Chart 3: Bias Category Breakdown
        # Shows which types of bias are most common
        # ============================================================
        category_counts = {}
        for r in results:
            for term in r['detected_terms']:
                cat = term['category']
                category_counts[cat] = category_counts.get(cat, 0) + 1

        if category_counts:
            fig.add_trace(
                go.Bar(
                    x=list(category_counts.keys()),
                    y=list(category_counts.values()),
                    marker_color=self.colors['primary'],
                    text=list(category_counts.values()),
                    textposition='outside',
                    showlegend=False
                ),
                row=2, col=1
            )

        # ============================================================
        # Chart 4: Most Frequent Biased Terms
        # ============================================================
        term_counts = {}
        for r in results:
            for term in r['detected_terms']:
                t = term['term']
                term_counts[t] = term_counts.get(t, 0) + 1

        if term_counts:
            # Get top 10 most frequent terms
            top_terms = sorted(
                term_counts.items(),
                key=lambda x: x[1],
                reverse=True
            )[:10]

            fig.add_trace(
                go.Bar(
                    x=[t[0] for t in top_terms],
                    y=[t[1] for t in top_terms],
                    marker_color=self.colors['danger'],
                    text=[t[1] for t in top_terms],
                    textposition='outside',
                    showlegend=False
                ),
                row=2, col=2
            )

        # ============================================================
        # Global Layout
        # ============================================================
        fig.update_layout(
            title_text="<b>Job Description Language Bias Analysis</b><br><sub>Powered by Azure AI Language Service</sub>",
            font=dict(
                family=Config.CHART_CONFIG['font_family'],
                size=12
            ),
            height=1000,
            width=1200,
            annotations=[dict(
                text="BiasGuard | Microsoft Imagine Cup (2026 Season)",
                showarrow=False,
                xref="paper",
                yref="paper",
                x=0.5,
                y=-0.05,
                xanchor='center',
                font=dict(size=10, color='gray')
            )]
        )

        fig.write_html(output_path)
        print(f"✓ Language bias chart saved: {output_path}")

    def generate_complete_report(self, data, baseline_metrics, fair_metrics,
                                 language_analysis, output_dir):
        """
        Generate all visualizations in one call.

        Creates:
        1. Comprehensive dashboard (6 charts)
        2. Fairness radar comparison
        3. Language bias analysis

        Args:
            data: Full hiring dataset
            baseline_metrics: Pre-mitigation fairness metrics
            fair_metrics: Post-mitigation fairness metrics
            language_analysis: Azure AI Language results
            output_dir: Directory for output HTML files
        """

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Generate all visualizations
        self.create_comprehensive_dashboard(
            data, baseline_metrics, fair_metrics,
            output_dir / '01_comprehensive_dashboard.html'
        )

        self.create_fairness_radar(
            baseline_metrics, fair_metrics,
            output_dir / '02_fairness_radar.html'
        )

        if language_analysis:
            self.create_language_bias_chart(
                language_analysis,
                output_dir / '03_language_bias.html'  # FIX: Correct numbering
            )

        print("\n" + "=" * 80)
        print("✅ ALL VISUALIZATIONS GENERATED SUCCESSFULLY")
        print("=" * 80)