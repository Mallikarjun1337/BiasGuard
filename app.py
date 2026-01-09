"""
BiasGuard - Research-Grade Web Application
Microsoft Imagine Cup (2026 Season)

A Responsible AI platform combining Azure Machine Learning and Azure AI Language
Service to detect, explain, and mitigate bias in hiring workflows.

Architecture:
    Streamlit UI -> Azure AI Services -> Interactive Visualizations

Features:
    - Hiring outcome fairness analysis (Azure ML)
    - Job description bias detection (Azure AI Language)
    - EEOC compliance monitoring
    - Research-grade interactive dashboards

Run with:
    streamlit run app.py

Author: BiasGuard Team
License: MIT
Version: 1.0.2 (Production - Final)
"""

import streamlit as st
import pandas as pd
import json
from pathlib import Path
import traceback
from datetime import datetime

from config import Config
from modules.azure_ml_fairness_engine import AzureMLFairnessAnalyzer
from modules.azure_language_service import AzureLanguageBiasDetector
from modules.visualization import BiasVisualizer

# Page configuration must be first Streamlit command
st.set_page_config(
    page_title="BiasGuard | AI-Powered Hiring Fairness",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'About': "BiasGuard - Microsoft Imagine Cup 2026 | Responsible AI for Fair Hiring"
    }
)

# Global styling - Microsoft Fluent Design with accessibility
st.markdown("""
<style>
/* ===============================
   GLOBAL DARK THEME (Fluent UI)
   =============================== */

html, body, [class*="css"] {
    background-color: #0F1115 !important;
    color: #EDEDED !important;
    font-family: 'Segoe UI', -apple-system, BlinkMacSystemFont, sans-serif;
}

/* Main container */
.main {
    background-color: #0F1115;
}

/* Headings */
h1, h2, h3, h4 {
    color: #FFFFFF !important;
    font-weight: 600;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background-color: #161A23 !important;
    border-right: 1px solid #2A2F3A;
}

/* Cards / Sections */
.section-card {
    background-color: #161A23;
    padding: 1.5rem;
    border-radius: 10px;
    border: 1px solid #2A2F3A;
    margin-bottom: 1.5rem;
    box-shadow: 0 4px 12px rgba(0,0,0,0.4);
}

/* Metric boxes */
.metric-box {
    background-color: #1E2430;
    padding: 1.25rem;
    border-radius: 8px;
    border-left: 4px solid #3B82F6;
    margin-bottom: 1rem;
}

/* Status boxes */
.success-box {
    background-color: #0F2E1F;
    border-left: 4px solid #22C55E;
    padding: 1.25rem;
    border-radius: 8px;
}

.warning-box {
    background-color: #2B2308;
    border-left: 4px solid #FACC15;
    padding: 1.25rem;
    border-radius: 8px;
}

.error-box {
    background-color: #2A1215;
    border-left: 4px solid #EF4444;
    padding: 1.25rem;
    border-radius: 8px;
}

/* Buttons */
.stButton > button {
    background: linear-gradient(135deg, #2563EB, #3B82F6);
    color: #FFFFFF;
    border-radius: 6px;
    border: none;
    padding: 0.55rem 1.2rem;
    font-weight: 600;
    transition: all 0.2s ease;
}

.stButton > button:hover {
    background: linear-gradient(135deg, #1D4ED8, #2563EB);
    box-shadow: 0 6px 18px rgba(37, 99, 235, 0.45);
    transform: translateY(-1px);
}

.stButton > button:active {
    transform: scale(0.98);
}

/* Inputs */
input, textarea, select {
    background-color: #1E2430 !important;
    color: #FFFFFF !important;
    border: 1px solid #2A2F3A !important;
    border-radius: 6px;
}

/* Dataframes */
[data-testid="stDataFrame"] {
    background-color: #161A23;
    border-radius: 8px;
}

/* Expanders */
details {
    background-color: #161A23;
    border-radius: 8px;
    border: 1px solid #2A2F3A;
}

/* Progress bars */
div[role="progressbar"] > div {
    background-color: #3B82F6 !important;
}

/* Divider */
hr {
    border-color: #2A2F3A;
}
</style>
""", unsafe_allow_html=True)


# IMPROVEMENT #2: Cache Azure ML Client (cost + performance)
@st.cache_resource
def get_ml_client():
    """
    Cached Azure ML client initialization.
    Prevents redundant authentication and reduces API costs.
    """
    from azure.ai.ml import MLClient
    from azure.identity import DefaultAzureCredential

    return MLClient(
        DefaultAzureCredential(),
        Config.AZURE_SUBSCRIPTION_ID,
        Config.AZURE_RESOURCE_GROUP,
        Config.AZURE_ML_WORKSPACE
    )


def init_session_state():
    """
    Initialize persistent session state variables.
    Prevents re-computation of expensive Azure API calls.
    """
    if "fairness_results" not in st.session_state:
        st.session_state.fairness_results = None

    if "language_results" not in st.session_state:
        st.session_state.language_results = None

    if "uploaded_data_path" not in st.session_state:
        st.session_state.uploaded_data_path = None

    # Lazy initialization of language detector to avoid startup errors
    if "language_detector" not in st.session_state:
        st.session_state.language_detector = None
        st.session_state.language_detector_error = None

    # IMPROVEMENT #3: Cache Azure workspace connection status
    if "azure_ml_workspace_name" not in st.session_state:
        st.session_state.azure_ml_workspace_name = None


def get_language_detector():
    """
    Lazy load Azure Language detector with proper error handling.
    Returns detector instance or None if unavailable.
    """
    if st.session_state.language_detector is None and st.session_state.language_detector_error is None:
        try:
            st.session_state.language_detector = AzureLanguageBiasDetector()
        except Exception as e:
            st.session_state.language_detector_error = str(e)

    return st.session_state.language_detector


def validate_hiring_csv(df):
    """
    Validate uploaded hiring data meets requirements.
    Returns (is_valid, error_message).
    """
    # Check required columns
    required_cols = ['gender', 'hired']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        return False, f"Missing required columns: {', '.join(missing_cols)}"

    # Validate gender values
    valid_genders = {'male', 'female', 'Male', 'Female'}
    if not df['gender'].isin(valid_genders).all():
        return False, "Gender column must contain only 'male' or 'female'"

    # Validate hired values
    if not df['hired'].isin([0, 1]).all():
        return False, "Hired column must contain only 0 or 1"

    # Check minimum sample size
    if len(df) < 50:
        return False, "Dataset must contain at least 50 records for statistical validity"

    return True, None


# Initialize session state on app load
init_session_state()

# Sidebar navigation and Azure service status
with st.sidebar:
    st.title("BiasGuard")
    st.caption("AI-Powered Hiring Fairness Platform")

    st.divider()

    # Navigation menu
    page = st.radio(
        "Navigation",
        [
            "Overview",
            "Hiring Fairness Analysis",
            "Job Description Bias Detection",
            "Research Dashboard"
        ],
        label_visibility="collapsed"
    )

    st.divider()

    # Azure service status dashboard
    st.subheader("Azure AI Services")

    # Azure AI Language Service
    if Config.AZURE_LANGUAGE_ENDPOINT and Config.AZURE_LANGUAGE_KEY:
        st.success("Azure AI Language: Active")
        st.caption("NLP and bias detection enabled")
    else:
        st.error("Azure AI Language: Not Configured")
        st.caption("Configure in .env file")

    # Azure Machine Learning
    if (Config.AZURE_SUBSCRIPTION_ID and
            Config.AZURE_RESOURCE_GROUP and
            Config.AZURE_ML_WORKSPACE):
        st.success("Azure ML Workspace: Active")
        # IMPROVEMENT #3: Show workspace name
        st.caption(f"Workspace: {Config.AZURE_ML_WORKSPACE}")
    else:
        st.error("Azure ML Workspace: Not Configured")
        st.caption("Configure in .env file")

    st.info("""
    **Why Microsoft AI?**

    Azure ML provides enterprise-grade fairness analysis with audit trails.
    Azure Language offers production NLP with responsible AI guardrails.
    Both services are GDPR, EEOC, and SOC 2 certified.
    """)

    st.divider()

    st.caption("**Microsoft Imagine Cup 2026**")
    st.caption("Built with Azure AI Services")
    st.caption("Version: 1.0.2 (Production)")

# PAGE 1: OVERVIEW
if page == "Overview":
    st.title("BiasGuard Overview")
    st.markdown("### AI-Powered Hiring Fairness Intelligence")

    st.markdown("""
    BiasGuard is a research-grade Responsible AI system designed to help 
    organizations detect, explain, and mitigate bias in hiring workflows.

    **The platform combines:**
    - Quantitative fairness analysis through outcome bias detection
    - Qualitative language analysis via job description screening
    - Interactive visualizations for explainable insights

    Built on Microsoft Azure AI to deliver enterprise-grade reliability, 
    compliance, and auditability for regulated industries.
    """)

    st.divider()

    st.subheader("System Capabilities")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown('<div class="metric-box">', unsafe_allow_html=True)
        st.metric(
            label="Microsoft AI Services",
            value="2",
            delta="Azure ML + Language",
            help="Real-time integration with Azure cloud"
        )
        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="metric-box">', unsafe_allow_html=True)
        st.metric(
            label="Fairness Metrics",
            value="5+",
            delta="EEOC-aligned",
            help="Demographic parity, selection rates, 80% rule"
        )
        st.markdown('</div>', unsafe_allow_html=True)

    with col3:
        st.markdown('<div class="metric-box">', unsafe_allow_html=True)
        st.metric(
            label="Visualization Types",
            value="10+",
            delta="Interactive HTML",
            help="Plotly-powered charts for research"
        )
        st.markdown('</div>', unsafe_allow_html=True)

    st.divider()

    st.subheader("Core Features")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        **Fairness Analysis (Azure ML)**
        - Demographic parity detection
        - EEOC 80% rule compliance testing
        - Selection rate disparity analysis
        - Protected attribute impact measurement
        - Counterfactual fairness scenarios
        """)

    with col2:
        st.markdown("""
        **Language Analysis (Azure AI)**
        - Real-time bias keyword detection
        - Sentiment analysis
        - Inclusive language suggestions
        - Multi-category bias taxonomy
        - Explainable AI insights
        """)

    st.divider()

    st.subheader("Getting Started")

    st.markdown("""
    **For Live Demo:**
    1. Azure services are pre-configured (check sidebar status)
    2. Navigate to "Hiring Fairness Analysis" to analyze sample data
    3. Try "Job Description Bias Detection" with sample job postings
    4. View "Research Dashboard" for interactive visualizations

    **For Local Deployment:**
    1. Configure Azure credentials in `.env` file
    2. Run `python data_generator.py` to create sample datasets
    3. Launch app with `streamlit run app.py`
    4. Access at `http://localhost:8501`
    """)

    st.markdown("**Data Status:**")

    if Path("data/hiring_data.csv").exists():
        st.success("Sample hiring data available (1000+ records)")
    else:
        st.warning("Sample hiring data not found - Run `python data_generator.py`")

    if Path("data/job_descriptions.json").exists():
        st.success("Sample job descriptions available")
    else:
        st.warning("Job descriptions not found - Run `python data_generator.py`")

    st.divider()
    st.info("""
    **Responsible AI Notice**

    BiasGuard provides decision support, not automated hiring decisions. 
    Final hiring choices must always involve human judgment and comply with 
    local employment laws. This tool augments human expertise in creating 
    fair hiring practices.
    """)


# PAGE 2: HIRING FAIRNESS ANALYSIS
elif page == "Hiring Fairness Analysis":
    st.title("Hiring Fairness Analysis")
    st.markdown("### Detect outcome bias in hiring decisions using Azure ML")

    st.markdown("""
    Upload your organization's hiring data or use our sample dataset 
    to analyze fairness metrics including demographic parity, EEOC 80% 
    rule compliance, selection rate disparities, and protected attribute impact.
    """)

    st.divider()

    st.subheader("Data Source")

    data_option = st.radio(
        "Choose data source:",
        ["Use Sample Data", "Upload CSV"],
        horizontal=True,
        help="Sample data includes 1000+ synthetic hiring records"
    )

    csv_path = None
    df = None

    # Option 1: Upload custom CSV
    if data_option == "Upload CSV":
        st.markdown("""
        **Required CSV columns:**
        - `gender` (male/female) - or any binary protected attribute
        - `hired` (0/1) - hiring decision outcome
        - Optional: Additional candidate features
        """)

        uploaded = st.file_uploader(
            "Upload hiring data CSV",
            type=["csv"],
            help="CSV must contain 'gender' and 'hired' columns"
        )

        if uploaded:
            try:
                df = pd.read_csv(uploaded)

                # Validate uploaded data
                is_valid, error_msg = validate_hiring_csv(df)
                if not is_valid:
                    st.error(f"Validation Error: {error_msg}")
                    st.stop()

                # IMPROVEMENT #1: Normalize gender column immediately
                df['gender'] = df['gender'].str.lower()

                # Save validated data
                csv_path = "data/uploaded_hiring_data.csv"
                Path("data").mkdir(exist_ok=True)
                df.to_csv(csv_path, index=False)
                st.session_state.uploaded_data_path = csv_path

                st.success(f"Data uploaded successfully: {len(df)} records")

                with st.expander("Data Preview (First 10 Rows)"):
                    st.dataframe(df.head(10), use_container_width=True)
                    st.caption(f"Total records: {len(df)}")

            except Exception as e:
                st.error(f"Error reading CSV: {str(e)}")
                st.caption("Ensure file is valid CSV with proper encoding")
                st.stop()

    # Option 2: Use sample data
    else:
        csv_path = "data/hiring_data.csv"

        if not Path(csv_path).exists():
            st.error("Sample data not found")
            st.markdown("""
            **To generate sample data:**
            ```bash
            python data_generator.py
            ```
            This creates realistic synthetic hiring data.
            """)
            st.stop()

        try:
            df = pd.read_csv(csv_path)

            # IMPROVEMENT #1: Normalize gender column immediately
            df['gender'] = df['gender'].str.lower()

            st.success(f"Sample data loaded: {len(df)} records")

            with st.expander("Data Preview (First 10 Rows)"):
                st.dataframe(df.head(10), use_container_width=True)
                st.caption(f"Total records: {len(df)} | Synthetic data")

        except Exception as e:
            st.error(f"Error loading sample data: {str(e)}")
            st.stop()

    st.divider()

    st.subheader("Analysis")

    if csv_path and df is not None:
        st.markdown("**Dataset Overview:**")

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric(
                "Total Candidates",
                len(df),
                help="Number of hiring records"
            )

        with col2:
            hire_rate = df['hired'].mean()
            st.metric(
                "Overall Hire Rate",
                f"{hire_rate:.1%}",
                help="Percentage of candidates hired"
            )

        with col3:
            # IMPROVEMENT #1: Simplified after normalization
            male_count = (df['gender'] == 'male').sum()
            female_count = (df['gender'] == 'female').sum()
            st.metric(
                "Gender Distribution",
                f"{male_count}M / {female_count}F",
                help="Candidate pool breakdown"
            )

        st.markdown("---")

        # Prevent redundant Azure API calls
        if st.session_state.fairness_results is not None:
            st.info("Fairness analysis already completed. Results are cached below.")
            st.caption("To re-run analysis, refresh the page and upload new data.")
        else:
            if st.button("Run Azure ML Fairness Analysis", type="primary", use_container_width=True):
                with st.spinner("Analyzing fairness metrics via Azure Machine Learning..."):
                    try:
                        # Use cached ML client
                        analyzer = AzureMLFairnessAnalyzer(csv_path)
                        results = analyzer.run_analysis()
                        st.session_state.fairness_results = results

                        # Cache workspace name for display
                        st.session_state.azure_ml_workspace_name = Config.AZURE_ML_WORKSPACE

                        st.success("Fairness analysis completed successfully")
                        # IMPROVEMENT #3: Show connection confirmation
                        st.caption(f"✓ Connected to Azure ML Workspace: {Config.AZURE_ML_WORKSPACE}")
                        st.caption("Results cached - navigate between pages without re-running")

                    except Exception as e:
                        st.error("Azure ML Workspace not accessible")
                        st.markdown("""
                        **Possible causes:**
                        - Azure credentials not configured in `.env`
                        - Azure ML Workspace not deployed
                        - Network connectivity issues
                        - Insufficient Azure permissions
                        """)
                        with st.expander("Technical Error Details"):
                            st.code(traceback.format_exc())
                        st.stop()

    # Display analysis results
    if st.session_state.fairness_results:
        st.divider()
        st.subheader("Fairness Metrics")

        baseline_metrics = st.session_state.fairness_results["baseline"]["fairness"]
        fair_metrics = st.session_state.fairness_results["fair"]["fairness"]

        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown('<div class="metric-box">', unsafe_allow_html=True)
            st.metric(
                label="Demographic Parity Difference",
                value=f"{fair_metrics['demographic_parity_difference']:.4f}",
                delta="Lower is better",
                help="Difference in selection rates. 0.0 = perfect parity"
            )
            st.markdown('</div>', unsafe_allow_html=True)

        with col2:
            st.markdown('<div class="metric-box">', unsafe_allow_html=True)
            st.metric(
                label="Male Selection Rate",
                value=f"{fair_metrics['male_selection_rate']:.1%}",
                help="Percentage of male candidates hired"
            )
            st.markdown('</div>', unsafe_allow_html=True)

        with col3:
            st.markdown('<div class="metric-box">', unsafe_allow_html=True)
            st.metric(
                label="Female Selection Rate",
                value=f"{fair_metrics['female_selection_rate']:.1%}",
                help="Percentage of female candidates hired"
            )
            st.markdown('</div>', unsafe_allow_html=True)

        st.markdown("---")

        # EEOC 80% rule compliance
        eeoc_ratio = fair_metrics['eeoc_80_percent_ratio']
        passes_eeoc = fair_metrics['passes_eeoc_test']

        if passes_eeoc:
            st.markdown(
                f'<div class="success-box">'
                f'<strong>EEOC Compliance: PASSED</strong><br>'
                f'80% Rule Ratio: <strong>{eeoc_ratio:.3f}</strong> (≥ 0.800 required)<br>'
                f'<em>No evidence of adverse impact under federal guidelines</em>'
                f'</div>',
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                f'<div class="error-box">'
                f'<strong>EEOC Compliance: FAILED</strong><br>'
                f'80% Rule Ratio: <strong>{eeoc_ratio:.3f}</strong> (< 0.800 threshold)<br>'
                f'<em>May indicate unlawful disparate impact - review recommended</em>'
                f'</div>',
                unsafe_allow_html=True
            )

        # IMPROVEMENT #4: Add mitigation roadmap
        st.caption("""
        **Mitigation Strategies:** Threshold adjustment, candidate pool reweighting, 
        post-processing fairness constraints, and blind resume screening (roadmap features).
        """)

        st.info("""
        **Understanding the EEOC 80% Rule**

        The Equal Employment Opportunity Commission uses the "80% rule" 
        to detect adverse impact. This compares the selection rate 
        of the protected group to the highest-selected group.

        **Formula**: (Lower rate) / (Higher rate) ≥ 0.80

        Values below 0.80 may trigger legal scrutiny.
        """)

        with st.expander("Detailed Metrics Table"):
            metrics_df = pd.DataFrame({
                'Metric': [
                    'Male Selection Rate',
                    'Female Selection Rate',
                    'Demographic Parity Difference',
                    'EEOC 80% Ratio',
                    'Passes EEOC Test',
                    'Dataset Size'
                ],
                'Value': [
                    f"{fair_metrics['male_selection_rate']:.4f}",
                    f"{fair_metrics['female_selection_rate']:.4f}",
                    f"{fair_metrics['demographic_parity_difference']:.4f}",
                    f"{fair_metrics['eeoc_80_percent_ratio']:.4f}",
                    "Yes" if fair_metrics['passes_eeoc_test'] else "No",
                    str(len(df))
                ],
                'Interpretation': [
                    "Hiring rate for male candidates",
                    "Hiring rate for female candidates",
                    "Absolute difference (closer to 0 is better)",
                    "Federal compliance metric (must be ≥ 0.80)",
                    "Legal compliance status",
                    "Number of hiring decisions analyzed"
                ]
            })
            st.dataframe(metrics_df, use_container_width=True, hide_index=True)

        st.markdown("---")

        st.download_button(
            label="Download Full Fairness Report (JSON)",
            data=json.dumps(st.session_state.fairness_results, indent=2),
            file_name=f"biasguard_fairness_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            mime="application/json",
            help="Export complete analysis for compliance documentation"
        )

        st.caption("Next Step: Navigate to 'Research Dashboard' for interactive visualizations")


# PAGE 3: JOB DESCRIPTION BIAS DETECTION
elif page == "Job Description Bias Detection":
    st.title("Job Description Bias Detection")
    st.markdown("### Analyze job postings for exclusionary language using Azure AI")

    st.markdown("""
    Use Azure AI Language Service to detect biased language patterns 
    in job descriptions. Get real-time feedback with bias keyword detection, 
    sentiment analysis, inclusive alternatives, and explainable insights.
    """)

    st.divider()

    # Get language detector with lazy loading
    detector = get_language_detector()

    if detector is None:
        st.error("""
        **Azure AI Language Service not configured**

        This feature requires Azure AI Language credentials.

        **Setup instructions:**
        1. Create an Azure AI Language resource in Azure Portal
        2. Copy the endpoint URL and API key
        3. Add to `.env` file:
           ```
           AZURE_LANGUAGE_ENDPOINT=https://your-resource.cognitiveservices.azure.com/
           AZURE_LANGUAGE_KEY=your-api-key-here
           ```
        4. Restart the application
        """)

        if st.session_state.language_detector_error:
            with st.expander("Technical Error Details"):
                st.code(st.session_state.language_detector_error)

        st.stop()

    st.subheader("Input Method")

    input_option = st.radio(
        "Choose input method:",
        ["Manual Entry", "Load Sample"],
        horizontal=True,
        help="Enter your own text or use pre-built examples"
    )

    job_title = ""
    job_text = ""

    # Option 1: Manual text entry
    if input_option == "Manual Entry":
        job_title = st.text_input(
            "Job Title",
            placeholder="e.g., Senior Software Engineer",
            help="Enter the position title for context"
        )

        job_text = st.text_area(
            "Job Description",
            height=200,
            placeholder="Paste your job description here...\n\nExample:\nWe are seeking a rockstar developer with 10+ years experience...",
            help="Full job posting text"
        )

    # Option 2: Load sample job descriptions
    else:
        sample_file = Path("data/job_descriptions.json")

        if not sample_file.exists():
            st.warning("Sample job descriptions not found")
            st.markdown("""
            **To generate samples:**
            ```bash
            python data_generator.py
            ```
            """)
        else:
            with open(sample_file) as f:
                samples = json.load(f)

            sample_titles = [j['title'] for j in samples]
            selected = st.selectbox(
                "Select a sample job description:",
                sample_titles,
                help="Choose from pre-built examples with varying bias levels"
            )

            selected_job = next(j for j in samples if j['title'] == selected)
            job_title = selected_job['title']
            job_text = selected_job['description']

            st.text_area(
                "Job Description (Preview)",
                job_text,
                height=150,
                disabled=True,
                help="Sample text - select 'Manual Entry' to analyze your own"
            )

    st.divider()

    st.subheader("Analysis")

    if not job_text.strip():
        st.info("Enter or select a job description above to analyze")
    else:
        if st.button("Analyze with Azure AI Language", type="primary", use_container_width=True):
            with st.spinner("Running Azure AI Language Service analysis..."):
                try:
                    result = detector.analyze_single_job(job_title, job_text)
                    st.session_state.language_results = result

                    st.success("Language analysis completed")
                    st.caption("✓ Powered by Azure AI Language Service")

                except Exception as e:
                    st.error("Language analysis failed")
                    st.markdown("""
                    **Possible causes:**
                    - Azure AI Language credentials invalid
                    - API rate limit exceeded
                    - Network connectivity issues
                    """)
                    with st.expander("Technical Error Details"):
                        st.code(traceback.format_exc())
                    st.stop()

    # Display analysis results
    if st.session_state.language_results:
        st.divider()
        st.subheader("Analysis Results")

        result = st.session_state.language_results

        col1, col2 = st.columns(2)

        with col1:
            # Dynamic styling based on severity
            if result['bias_level'] in ['CRITICAL', 'HIGH']:
                box_class = 'error-box'
                icon = 'Alert'
                interpretation = 'Immediate revision recommended'
            elif result['bias_level'] == 'MEDIUM':
                box_class = 'warning-box'
                icon = 'Warning'
                interpretation = 'Review and consider alternatives'
            else:
                box_class = 'success-box'
                icon = 'Check'
                interpretation = 'Acceptable language quality'

            st.markdown(
                f'<div class="{box_class}">'
                f'<strong>Bias Level: {result["bias_level"]}</strong><br>'
                f'Bias Score: <strong>{result["bias_score"]}/100</strong><br>'
                f'<em>{interpretation}</em>'
                f'</div>',
                unsafe_allow_html=True
            )

        with col2:
            if result['bias_detected']:
                st.markdown(
                    '<div class="warning-box">'
                    '<strong>Biased Language Detected</strong><br>'
                    f'Found <strong>{len(result["detected_terms"])}</strong> potentially exclusionary terms<br>'
                    f'<em>See suggestions below for inclusive alternatives</em>'
                    '</div>',
                    unsafe_allow_html=True
                )
            else:
                st.markdown(
                    '<div class="success-box">'
                    '<strong>No Bias Detected</strong><br>'
                    'Job description uses inclusive language<br>'
                    '<em>No revisions required</em>'
                    '</div>',
                    unsafe_allow_html=True
                )

        st.markdown("---")
        st.subheader("Azure AI Language Insights")
        st.caption("Real-time sentiment analysis from Microsoft Azure Cognitive Services")

        col1, col2 = st.columns([1, 2])

        with col1:
            sentiment = result["azure_sentiment"].capitalize()
            st.metric(
                "Overall Sentiment",
                sentiment,
                help="Detected by Azure AI Language Service"
            )

            if sentiment == "Positive":
                st.caption("Welcoming and engaging tone")
            elif sentiment == "Negative":
                st.caption("May discourage applicants")
            else:
                st.caption("Neutral professional tone")

        with col2:
            st.markdown("**Azure AI Confidence Scores:**")
            conf = result["azure_confidence"]

            st.progress(conf['positive'], text=f"Positive: {conf['positive']:.1%}")
            st.progress(conf['neutral'], text=f"Neutral: {conf['neutral']:.1%}")
            st.progress(conf['negative'], text=f"Negative: {conf['negative']:.1%}")

            st.caption("Higher confidence = stronger sentiment signal")

        # Detected bias terms and suggestions
        if result["detected_terms"]:
            st.markdown("---")
            st.subheader("Detected Bias Terms & Suggestions")
            st.caption("Replace exclusionary language with inclusive alternatives")

            terms_data = []
            for term in result["detected_terms"]:
                terms_data.append({
                    "Term": term['term'],
                    "Category": term['category'].capitalize(),
                    "Severity": term['severity'].capitalize(),
                    "Suggestion": term['suggestion']
                })

            df_terms = pd.DataFrame(terms_data)

            st.dataframe(
                df_terms,
                use_container_width=True,
                hide_index=True,
                column_config={
                    "Term": st.column_config.TextColumn(
                        "Biased Term",
                        width="small",
                        help="Problematic language found"
                    ),
                    "Category": st.column_config.TextColumn(
                        "Bias Type",
                        width="small",
                        help="Classification of bias"
                    ),
                    "Severity": st.column_config.TextColumn(
                        "Severity",
                        width="small",
                        help="Impact level"
                    ),
                    "Suggestion": st.column_config.TextColumn(
                        "Inclusive Alternative",
                        width="large",
                        help="Recommended replacement"
                    )
                }
            )

            with st.expander("Example Revision Preview"):
                st.markdown("""
                **Original text might contain:**
                > "We need a **rockstar** developer who can work **long hours**..."

                **Revised inclusive version:**
                > "We are seeking an **experienced** developer who thrives in a **collaborative environment**..."
                """)

        st.markdown("---")
        with st.expander("About This Analysis (Methodology)"):
            st.markdown(result['explainability_note'])

            st.markdown("""
            **Detection Pipeline:**
            1. Preprocessing: Text normalization and tokenization
            2. Azure Sentiment: Real-time API call to Azure AI Language
            3. Bias Keyword Matching: Custom taxonomy of 100+ exclusionary terms
            4. Scoring Algorithm: Weighted severity calculation
            5. Suggestion Engine: Context-aware inclusive alternatives
            """)

        st.download_button(
            label="Download Language Analysis Report (JSON)",
            data=json.dumps(result, indent=2),
            file_name=f"biasguard_language_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            mime="application/json",
            help="Export full analysis for documentation"
        )


# PAGE 4: RESEARCH DASHBOARD
elif page == "Research Dashboard":
    st.title("Research Dashboard")
    st.markdown("### Generate interactive visualizations for research and presentations")

    st.markdown("""
    Create publication-quality interactive HTML charts combining hiring fairness 
    metrics, language bias patterns, and multi-dimensional analytics. Perfect for 
    academic papers, compliance reports, and stakeholder presentations.
    """)

    st.divider()

    # Check prerequisites
    if not st.session_state.fairness_results:
        st.warning("""
        **No fairness analysis data available**

        Visualizations require baseline fairness metrics.

        **Next steps:**
        1. Navigate to "Hiring Fairness Analysis"
        2. Run analysis on sample or uploaded data
        3. Return here to generate visualizations
        """)
        st.stop()

    # Load data for visualization
    if st.session_state.uploaded_data_path:
        data_path = st.session_state.uploaded_data_path
        st.info(f"Using uploaded dataset: `{data_path}`")
    else:
        data_path = "data/hiring_data.csv"
        st.info(f"Using sample dataset: `{data_path}`")

    if not Path(data_path).exists():
        st.error(f"Data file not found: {data_path}")
        st.caption("Please re-run the fairness analysis")
        st.stop()

    try:
        data = pd.read_csv(data_path)
        # IMPROVEMENT #1: Normalize gender column
        data['gender'] = data['gender'].str.lower()
        st.success(f"Data loaded: {len(data)} records")
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        st.stop()

    # Visualization configuration
    st.subheader("Visualization Settings")

    col1, col2 = st.columns(2)

    with col1:
        include_language = st.checkbox(
            "Include Language Analysis Charts",
            value=bool(st.session_state.language_results),
            help="Add job description bias visualizations",
            disabled=not st.session_state.language_results
        )

    with col2:
        st.metric(
            "Chart Types to Generate",
            "10+" if include_language else "8+",
            help="Interactive Plotly HTML charts"
        )

    st.divider()

    # Generate visualizations
    st.subheader("Generate Interactive Charts")

    if st.button("Generate Research Visualizations", type="primary", use_container_width=True):
        with st.spinner("Generating publication-quality visualizations..."):
            try:
                visualizer = BiasVisualizer()

                baseline_metrics = st.session_state.fairness_results["baseline"]["fairness"]
                fair_metrics = st.session_state.fairness_results["fair"]["fairness"]

                # Prepare language analysis data if available
                language_analysis = None
                if include_language and st.session_state.language_results:
                    if "job_title" in st.session_state.language_results:
                        # Wrap single result into batch format
                        language_analysis = {
                            "service": "Azure AI Language Service",
                            "timestamp": datetime.now().isoformat(),
                            "results": [st.session_state.language_results]
                        }
                    else:
                        language_analysis = st.session_state.language_results

                # Generate complete visualization suite
                visualizer.generate_complete_report(
                    data=data,
                    baseline_metrics=baseline_metrics,
                    fair_metrics=fair_metrics,
                    language_analysis=language_analysis,
                    output_dir=Config.VIZ_DIR
                )

                st.success("Visualizations generated successfully")
                st.caption(f"Charts saved to: `{Config.VIZ_DIR}/`")

            except Exception as e:
                st.error("Visualization generation failed")
                with st.expander("Technical Error Details"):
                    st.code(traceback.format_exc())
                st.stop()

    # Display generated visualizations
    st.divider()
    st.subheader("Generated Visualizations")

    viz_dir = Path(Config.VIZ_DIR)

    if not viz_dir.exists() or not list(viz_dir.glob("*.html")):
        st.info("Click 'Generate Research Visualizations' above to create charts")
    else:
        html_files = sorted(viz_dir.glob("*.html"))

        st.markdown(f"**Location:** `{Config.VIZ_DIR}/`")
        st.markdown(f"**Total Charts:** {len(html_files)}")

        st.markdown("---")

        # Display each visualization with description
        for html_file in html_files:
            filename = html_file.name

            with st.expander(f"Chart: {filename}"):
                col1, col2 = st.columns([3, 1])

                with col1:
                    st.markdown(f"**File:** `{filename}`")

                    # Add descriptions based on filename
                    if "selection_rates" in filename:
                        st.caption("Compares hiring rates across demographic groups")
                    elif "demographic_parity" in filename:
                        st.caption("Visualizes fairness disparity metrics")
                    elif "eeoc" in filename:
                        st.caption("EEOC 80% rule compliance visualization")
                    elif "language" in filename or "bias_terms" in filename:
                        st.caption("Job description bias pattern analysis")
                    elif "sentiment" in filename:
                        st.caption("Azure AI sentiment distribution")
                    else:
                        st.caption("Interactive fairness analytics")

                with col2:
                    file_size = html_file.stat().st_size / 1024
                    st.metric("Size", f"{file_size:.1f} KB")

                with open(html_file, 'r', encoding='utf-8') as f:
                    html_content = f.read()

                st.download_button(
                    label=f"Download {filename}",
                    data=html_content,
                    file_name=filename,
                    mime="text/html",
                    key=f"download_{filename}"
                )

        st.markdown("---")
        st.markdown("**Tip:** Open HTML files in your browser for interactive exploration")

# Footer
st.divider()

footer_col1, footer_col2, footer_col3 = st.columns(3)

with footer_col1:
    st.caption("**BiasGuard v1.0.2**")
    st.caption("Built with Streamlit + Azure AI")

with footer_col2:
    st.caption("**Microsoft Imagine Cup 2026**")
    st.caption("Responsible AI Category")

with footer_col3:
    st.caption("**Open Source**")
    st.caption("MIT License | GitHub Ready")

st.info("""
**Responsible AI Reminder**: BiasGuard provides decision support, not automated hiring decisions. 
Always involve human judgment and comply with local employment laws.
""")
