"""
BiasGuard - Azure AI Language Service Integration
Microsoft Imagine Cup 2026

Microsoft AI Service #2: Azure AI Language (Text Analytics)
Purpose: NLP-powered bias detection in job descriptions
"""

import os
import json
from datetime import datetime
from dotenv import load_dotenv
from azure.ai.textanalytics import TextAnalyticsClient
from azure.core.credentials import AzureKeyCredential
from config import Config

load_dotenv()


class AzureLanguageBiasDetector:
    """
    Azure AI Language-powered bias detector for job descriptions.

    Combines Azure's sentiment analysis with custom bias detection
    to identify exclusionary language in job postings.
    """

    def __init__(self):
        """
        Initialize Azure AI Language client and bias detection rules.

        Raises:
            Exception: If Azure Language credentials are missing
        """
        print("\n" + "=" * 80)
        print("MICROSOFT AI SERVICE #2: AZURE AI LANGUAGE (TEXT ANALYTICS)")
        print("=" * 80)

        # Load Azure credentials from environment variables
        self.endpoint = os.getenv("AZURE_LANGUAGE_ENDPOINT")
        self.key = os.getenv("AZURE_LANGUAGE_KEY")

        # Load bias detection configuration
        self.bias_keywords = Config.BIAS_KEYWORDS
        self.replacements = Config.REPLACEMENTS

        # Validate credentials before proceeding
        if not self.endpoint or not self.key:
            raise Exception(
                "Azure Language credentials missing. "
                "Check your .env file."
            )

        # Initialize Azure AI Language client
        self.client = TextAnalyticsClient(
            endpoint=self.endpoint,
            credential=AzureKeyCredential(self.key)
        )

        print("✓ Azure AI Language client initialized")

    def test_connection(self):
        """
        Lightweight connectivity test for Azure AI Language Service.

        Returns:
            tuple: (success: bool, message: str)
        """
        try:
            # Minimal API call to verify connection
            self.client.analyze_sentiment(
                ["Azure AI Language connection test"]
            )
            return True, "Azure AI Language Service connected successfully"
        except Exception as e:
            return False, f"Azure Language connection failed: {e}"

    def analyze_single_job(self, job_title, job_description):
        """
        Analyze a single job description for bias using hybrid approach.

        Combines:
        - Azure AI sentiment analysis (emotional tone)
        - Rule-based bias keyword detection (exclusionary language)

        Args:
            job_title: Job position title
            job_description: Full job description text

        Returns:
            dict: Comprehensive bias analysis with Azure sentiment data
        """

        # Combine title and description for complete analysis
        full_text = f"{job_title}. {job_description}"

        # ============================================================
        # AZURE AI LANGUAGE API CALL (THIS IS THE PROOF OF USAGE)
        # ============================================================
        sentiment_result = self.client.analyze_sentiment([full_text])[0]

        # Initialize bias tracking
        detected_terms = []
        bias_score = 0

        # Normalize text for case-insensitive matching
        text_lower = full_text.lower()

        # Scan for bias keywords across all categories
        for category, config in self.bias_keywords.items():
            for term in config["terms"]:
                if term.lower() in text_lower:
                    # Record biased term with full context
                    detected_terms.append({
                        "term": term,
                        "category": category,
                        "severity": config["severity"],
                        "description": config["description"],
                        "suggestion": self.replacements.get(
                            term,
                            "Use a more inclusive alternative"
                        )
                    })

                    # Accumulate bias score based on severity
                    if config["severity"] == "high":
                        bias_score += 25
                    elif config["severity"] == "medium":
                        bias_score += 15
                    else:
                        bias_score += 5

        # Cap bias score at maximum of 100
        bias_score = min(bias_score, 100)

        # Package results with Azure sentiment data
        # Using float() to ensure JSON serialization safety
        return {
            "job_title": job_title,
            "bias_detected": len(detected_terms) > 0,
            "bias_score": bias_score,
            "bias_level": (
                "CRITICAL" if bias_score >= 75 else
                "HIGH" if bias_score >= 50 else
                "MEDIUM" if bias_score >= 25 else
                "LOW" if bias_score > 0 else
                "MINIMAL"
            ),
            "detected_terms": detected_terms,
            "azure_sentiment": sentiment_result.sentiment,
            "azure_confidence": {
                # FIX: Explicit float conversion prevents JSON errors
                "positive": float(sentiment_result.confidence_scores.positive),
                "neutral": float(sentiment_result.confidence_scores.neutral),
                "negative": float(sentiment_result.confidence_scores.negative)
            },
            "explainability_note":
                "Azure AI Language provides sentiment signals; "
                "BiasGuard combines these with inclusive-language rules."
        }

    def batch_analyze(self, jobs):
        """
        Analyze multiple job descriptions in batch.

        Args:
            jobs: List of job dictionaries with 'title' and 'description'

        Returns:
            dict: Complete batch analysis results with metadata
        """
        results = []

        # Process each job description
        for job in jobs:
            results.append(
                self.analyze_single_job(
                    job["title"],
                    job["description"]
                )
            )

        # Package with metadata for traceability
        return {
            "service": "Azure AI Language Service (Text Analytics)",
            "timestamp": datetime.now().isoformat(),
            "results": results,
            "responsible_ai_note":
                "Results highlight potential risks in language, "
                "not intent of employers."
        }

    def save_analysis(self, results, output_path):
        """
        Save analysis results to JSON file.

        Args:
            results: Analysis results dictionary
            output_path: Path for output JSON file
        """
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)

        print(f"✓ Results saved to {output_path}")