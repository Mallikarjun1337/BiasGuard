"""
BiasGuard - Universal Data Generator
Microsoft Imagine Cup (2026 Season)

Purpose: Generate realistic, explainable hiring data for demonstration.

This synthetic data INTENTIONALLY contains bias so BiasGuard can:
- Detect unfair patterns
- Visualize disparities
- Demonstrate mitigation strategies

IMPORTANT: Can be replaced with organization's real CSV data.
"""

import pandas as pd
import numpy as np
import json
from datetime import datetime, timedelta
from pathlib import Path


def generate_hiring_data(n_samples=1000, output_dir='data'):
    """
    Generate universal hiring dataset with intentional bias.

    CSV Format (UNIVERSAL - works with any hiring system):
    - candidate_id: Unique identifier
    - gender: male/female (sensitive attribute for fairness testing)
    - age: 22-60 (realistic working age range)
    - department: Engineering/Sales/HR/Marketing/Finance
    - education_level: 1=High School, 2=Bachelor, 3=Master, 4=PhD
    - years_experience: 0-25 years
    - interview_score: 0-100 (performance metric)
    - previous_companies: 0-8 (experience diversity)
    - referral: 0=No, 1=Yes (hiring advantage)
    - hired: 0=Rejected, 1=Hired (TARGET VARIABLE)
    - apply_date: Application submission date

    Args:
        n_samples: Number of candidate records to generate
        output_dir: Directory path for output CSV

    Returns:
        pd.DataFrame: Generated hiring dataset
    """

    # Set random seed for reproducibility (critical for demos)
    np.random.seed(42)

    print("\n" + "=" * 80)
    print("GENERATING UNIVERSAL HIRING DATASET")
    print("=" * 80)

    # ================================================================
    # Generate Base Candidate Attributes
    # ================================================================
    data = {
        'candidate_id': range(1, n_samples + 1),
        'gender': np.random.choice(['male', 'female'], n_samples),
        'age': np.random.randint(22, 61, n_samples),
        'department': np.random.choice(
            ['Engineering', 'Sales', 'HR', 'Marketing', 'Finance'],
            n_samples
        ),
        # Education distribution: 10% HS, 50% Bachelor, 30% Master, 10% PhD
        'education_level': np.random.choice(
            [1, 2, 3, 4],
            n_samples,
            p=[0.1, 0.5, 0.3, 0.1]
        ),
        'years_experience': np.random.randint(0, 26, n_samples),
        'interview_score': np.random.randint(50, 101, n_samples),
        'previous_companies': np.random.randint(0, 9, n_samples),
        # Referral distribution: 70% no referral, 30% referral
        'referral': np.random.choice([0, 1], n_samples, p=[0.7, 0.3])
    }

    df = pd.DataFrame(data)

    # ================================================================
    # Generate Application Dates (Last 6 Months)
    # ================================================================
    start_date = datetime.now() - timedelta(days=180)
    df['apply_date'] = [
        start_date + timedelta(days=np.random.randint(0, 180))
        for _ in range(n_samples)
    ]

    # ================================================================
    # Hiring Decision Logic with INTENTIONAL BIAS
    # ================================================================

    # Base hiring probability from legitimate factors
    hire_probability = (
            0.30 * (df['interview_score'] / 100) +  # 30% weight: performance
            0.20 * (df['education_level'] / 4) +  # 20% weight: education
            0.15 * (df['years_experience'] / 25) +  # 15% weight: experience
            0.15 * df['referral'] +  # 15% weight: referral
            0.10 * (df['previous_companies'] / 8)  # 10% weight: diversity
    )

    # ============================================================
    # INTENTIONAL GENDER BIAS (FOR FAIRNESS TESTING ONLY)
    # Male candidates receive a 15% probability boost
    # This simulates real-world systemic bias patterns
    # ============================================================
    # NOTE: Bias factor is intentionally injected for fairness testing only
    bias_factor = np.where(df['gender'] == 'male', 1.15, 1.0)
    hire_probability *= bias_factor

    # Apply threshold with randomness to generate binary outcomes
    df['hired'] = (
            hire_probability > np.random.uniform(0.4, 0.7, n_samples)
    ).astype(int)

    # ================================================================
    # Save Dataset to CSV
    # ================================================================
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    csv_path = output_path / 'hiring_data.csv'
    df.to_csv(csv_path, index=False)

    print(f"\n✓ Generated {n_samples} candidate records")
    print(f"✓ Saved to: {csv_path}")

    # ================================================================
    # Display Dataset Statistics
    # ================================================================
    print("\n" + "-" * 80)
    print("DATASET STATISTICS")
    print("-" * 80)

    male_rate = df[df['gender'] == 'male']['hired'].mean()
    female_rate = df[df['gender'] == 'female']['hired'].mean()

    # FIX: Safe bias gap calculation (handles edge cases)
    bias_gap = (male_rate - female_rate) if (
            pd.notna(male_rate) and pd.notna(female_rate)
    ) else 0.0

    print(f"Total candidates: {len(df)}")
    print(f"Male candidates: {(df['gender'] == 'male').mean():.1%}")
    print(f"Female candidates: {(df['gender'] == 'female').mean():.1%}")
    print("\nHiring rates:")
    print(f"  Male: {male_rate:.1%}")
    print(f"  Female: {female_rate:.1%}")
    print(f"  Overall: {df['hired'].mean():.1%}")

    print(f"\n⚠️  Gender bias gap: {bias_gap:.1%} (intentional)")
    print("   This bias exists so BiasGuard can detect and mitigate it.")

    # ================================================================
    # Dataset Metadata (Responsible AI Transparency)
    # ================================================================
    print("\nDataset Metadata:")
    print("  • Synthetic data for demo & testing")
    print("  • Contains intentional bias for fairness analysis")
    print("  • No real personal data used")
    print("  • Safe for public demonstration")
    print("  • Complies with ethical AI guidelines")

    return df


def generate_job_descriptions(output_dir='data'):
    """
    Generate sample job descriptions with varying bias levels.

    Creates realistic job postings that range from highly biased
    to inclusive, allowing Azure AI Language to demonstrate
    its detection capabilities.

    Args:
        output_dir: Directory path for output JSON

    Returns:
        list: Job descriptions with intentional bias patterns
    """

    jobs = [
        {
            "job_id": "JOB001",
            "title": "Senior Software Engineer",
            "description":
                "We're looking for a rockstar developer with aggressive "
                "problem-solving skills. Must be young and energetic with "
                "a competitive mindset. Native English speaker preferred."
        },
        {
            "job_id": "JOB002",
            "title": "Marketing Manager",
            "description":
                "Seeking an ambitious go-getter who can dominate the market. "
                "Strong leadership and assertive communication required."
        },
        {
            "job_id": "JOB003",
            "title": "Data Analyst",
            "description":
                "Looking for a skilled professional with strong analytical "
                "abilities. Experience with Python and SQL required. "
                "Collaborative team environment."
        },
        {
            "job_id": "JOB004",
            "title": "HR Specialist",
            "description":
                "Recent graduate wanted for entry-level HR position. "
                "Must be fit and able-bodied to handle fast-paced environment."
        },
        {
            "job_id": "JOB005",
            "title": "Product Manager",
            "description":
                "Experienced professional needed to lead product strategy. "
                "Strong communication skills and adaptability required."
        }
    ]

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    json_path = output_path / 'job_descriptions.json'
    with open(json_path, 'w') as f:
        json.dump(jobs, f, indent=2)

    print(f"\n✓ Generated {len(jobs)} job descriptions")
    print(f"✓ Saved to: {json_path}")

    return jobs


def main():
    """
    Generate all sample data for BiasGuard demonstration.

    Creates:
    - Hiring dataset with intentional bias
    - Job descriptions with varying bias levels
    """

    print("\n" + "=" * 80)
    print("BIASGUARD DATA GENERATOR")
    print("Microsoft Imagine Cup (2026 Season)")
    print("=" * 80)

    # Generate synthetic hiring data
    generate_hiring_data(n_samples=1000)

    # Generate biased job descriptions
    generate_job_descriptions()

    print("\n" + "=" * 80)
    print("✅ DATA GENERATION COMPLETE")
    print("=" * 80)
    print("\nNext step:")
    print("  → Run: python main.py")
    print("\nNOTE FOR PRODUCTION USE:")
    print("  Replace hiring_data.csv with your organization's real data")
    print("  (same column format, no code changes required)")


if __name__ == '__main__':
    main()