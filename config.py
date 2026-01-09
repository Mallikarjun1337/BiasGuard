"""
BiasGuard Configuration
Microsoft Imagine Cup (2026 Season)

Centralized configuration for:
- Azure AI services (Language + Machine Learning)
- Fairness analysis thresholds (EEOC-compliant)
- Visualization standards (Microsoft design system)
- Responsible AI defaults
"""



import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()


class Config:
    """
    Central configuration hub for BiasGuard platform.

    All Azure credentials, analysis parameters, and design
    standards are managed here for security and maintainability.
    """

    # ================================================================
    # Project Metadata (Competition-ready branding)
    # ================================================================
    PROJECT_NAME = "BiasGuard"
    PROJECT_TAGLINE = "Building Fair, Explainable, and Responsible AI"
    COMPETITION = "Microsoft Imagine Cup (2026 Season)"

    # ================================================================
    # Project Directory Structure
    # ================================================================
    BASE_DIR = Path(__file__).parent
    DATA_DIR = BASE_DIR / 'data'
    OUTPUT_DIR = BASE_DIR / 'output'
    REPORTS_DIR = OUTPUT_DIR / 'reports'
    VIZ_DIR = OUTPUT_DIR / 'visualizations'

    # ================================================================
    # Azure AI Language Service (Text Analytics)
    # Microsoft AI Service #2
    # ================================================================
    AZURE_LANGUAGE_ENDPOINT = os.getenv('AZURE_LANGUAGE_ENDPOINT')
    AZURE_LANGUAGE_KEY = os.getenv('AZURE_LANGUAGE_KEY')

    # ================================================================
    # Azure Machine Learning
    # Microsoft AI Service #1
    # ================================================================
    AZURE_SUBSCRIPTION_ID = os.getenv('AZURE_SUBSCRIPTION_ID')
    AZURE_RESOURCE_GROUP = os.getenv('AZURE_RESOURCE_GROUP')
    AZURE_ML_WORKSPACE = os.getenv('AZURE_ML_WORKSPACE')
    # ================================================================
    # Demo Mode (Imagine Cup / Public Deployment)
    # ================================================================
    # When TRUE:
    # - Azure ML is simulated
    # - Fairness logic still runs
    # - No Azure authentication required
    #
    # When FALSE:
    # - Full Azure ML pipeline is used
    #
    DEMO_MODE = os.getenv("DEMO_MODE", "true").lower() == "true"

    # ================================================================
    # Fairness Thresholds (Industry Standards)
    # Based on EEOC guidelines and academic research
    # ================================================================
    BIAS_THRESHOLDS = {
        'demographic_parity': 0.2,  # Max acceptable rate difference
        'equalized_odds': 0.2,  # Max TPR/FPR difference
        'eeoc_80_percent': 0.8,  # EEOC 4/5ths rule (80% minimum)
        'selection_rate_diff': 0.15  # Max selection rate gap
    }

    # ================================================================
    # Visualization Design System (Microsoft-inspired)
    # ================================================================
    COLORS = {
        'primary': '#0078D4',  # Microsoft Blue
        'success': '#107C10',  # Success Green
        'warning': '#FFB900',  # Warning Amber
        'danger': '#D13438',  # Error Red
        'neutral': '#605E5C',  # Neutral Gray
        'accent': '#8764B8',  # Purple Accent
        'male': '#0078D4',  # Gender-neutral blue
        'female': '#D13438',  # Distinct red
        'background': '#F3F2F1'  # Subtle gray background
    }

    # ================================================================
    # Chart Styling Standards
    # ================================================================
    CHART_CONFIG = {
        'font_family': 'Segoe UI, sans-serif',  # Microsoft design language
        'title_font_size': 20,
        'axis_font_size': 12,
        'legend_font_size': 11,
        'default_height': 600,
        'default_width': 1000
    }

    # ================================================================
    # Analysis Parameters (Reproducible ML)
    # ================================================================
    ANALYSIS_CONFIG = {
        'test_size': 0.3,  # 70/30 train-test split
        'random_state': 42,  # Reproducibility seed
        'cv_folds': 5,  # Cross-validation folds
        'max_depth': 5,  # Decision tree depth limit
        'fairness_epsilon': 0.01  # Fairness constraint tolerance
    }

    # ================================================================
    # Job Description Bias Keywords (Research-based Taxonomy)
    # Sources: Gender Decoder, academic bias research
    # ================================================================
    BIAS_KEYWORDS = {
        'age': {
            'terms': [
                'young', 'old', 'energetic', 'mature', 'recent graduate',
                'digital native', 'experienced', 'senior', 'junior',
                'fresh', 'seasoned'
            ],
            'severity': 'high',
            'description': 'Age-related discriminatory language'
        },
        'gender': {
            'terms': [
                'rockstar', 'ninja', 'guru', 'aggressive', 'dominant',
                'competitive', 'strong', 'ambitious', 'assertive', 'chairman'
            ],
            'severity': 'high',
            'description': 'Gender-coded language (masculine-coded terms)'
        },
        'origin': {
            'terms': [
                'native', 'local', 'domestic', 'foreign', 'immigrant'
            ],
            'severity': 'medium',
            'description': 'National origin bias'
        },
        'physical': {
            'terms': [
                'fit', 'athletic', 'healthy', 'active', 'able-bodied'
            ],
            'severity': 'medium',
            'description': 'Physical ability bias (ableist language)'
        },
        'cultural': {
            'terms': [
                'culture fit', 'team player', 'work hard play hard',
                'family-like', 'fast-paced'
            ],
            'severity': 'low',
            'description': 'Cultural fit bias (can exclude diverse candidates)'
        }
    }

    # ================================================================
    # Neutral Language Replacements
    # Suggestions for more inclusive alternatives
    # ================================================================
    REPLACEMENTS = {
        'young': 'adaptable',
        'old': 'experienced',
        'energetic': 'motivated',
        'rockstar': 'skilled professional',
        'ninja': 'expert',
        'guru': 'specialist',
        'aggressive': 'proactive',
        'dominant': 'confident',
        'competitive': 'driven',
        'native': 'fluent',
        'recent graduate': 'early-career professional',
        'digital native': 'technology-proficient',
        'culture fit': 'values alignment'
    }

    # ================================================================
    # Utility Methods
    # ================================================================

    @classmethod
    def create_directories(cls):
        """
        Create all necessary output directories.

        Safe to call multiple times (idempotent).
        """
        for directory in [
            cls.DATA_DIR,
            cls.OUTPUT_DIR,
            cls.REPORTS_DIR,
            cls.VIZ_DIR
        ]:
            directory.mkdir(parents=True, exist_ok=True)

    @classmethod
    def validate_azure_credentials(cls):
        """
        Validate Azure AI Language Service credentials.

        Returns:
            bool: True if credentials are present

        Raises:
            ValueError: If credentials are missing
        """
        if not cls.AZURE_LANGUAGE_ENDPOINT or not cls.AZURE_LANGUAGE_KEY:
            raise ValueError(
                "Azure Language Service credentials not found.\n"
                "Set AZURE_LANGUAGE_ENDPOINT and AZURE_LANGUAGE_KEY in .env"
            )
        return True

    @classmethod
    def validate_azure_ml_credentials(cls):
        """
        Validate Azure Machine Learning credentials.

        In DEMO_MODE, validation is skipped (Imagine Cup safe).
        """
        if cls.DEMO_MODE:
            return True

        if not (
                cls.AZURE_SUBSCRIPTION_ID and
                cls.AZURE_RESOURCE_GROUP and
                cls.AZURE_ML_WORKSPACE
        ):
            raise ValueError(
                "Azure ML credentials not found.\n"
                "Set AZURE_SUBSCRIPTION_ID, AZURE_RESOURCE_GROUP, "
                "and AZURE_ML_WORKSPACE in .env"
            )
        return True

    @classmethod
    def get_status(cls):
        """
        Get current configuration and environment status.

        Returns:
            dict: Status of directories and Azure services
        """
        return {
            'directories': {
                'data': cls.DATA_DIR.exists(),
                'output': cls.OUTPUT_DIR.exists(),
                'reports': cls.REPORTS_DIR.exists(),
                'visualizations': cls.VIZ_DIR.exists()
            },
            'azure_language_configured': bool(
                cls.AZURE_LANGUAGE_ENDPOINT and cls.AZURE_LANGUAGE_KEY
            ),
            'azure_ml_configured': {
                'configured': True if cls.DEMO_MODE else bool(
                    cls.AZURE_SUBSCRIPTION_ID and
                    cls.AZURE_RESOURCE_GROUP and
                    cls.AZURE_ML_WORKSPACE
                ),
                'mode': 'DEMO' if cls.DEMO_MODE else 'LIVE',
                'workspace': cls.AZURE_ML_WORKSPACE
            }

        }


# Auto-create directories on import (safe and intentional)
Config.create_directories()
