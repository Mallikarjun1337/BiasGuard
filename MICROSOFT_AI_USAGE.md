# Microsoft AI Services Usage – BiasGuard

## 1. Azure AI Language Service
Used to analyze job descriptions for biased or exclusionary language.
The service provides sentiment analysis and linguistic signals that are later
used for bias assessment.

Integration:
- Azure Cognitive Services (Text Analytics)
- Endpoint + API Key
- File: modules/azure_language_service.py

## 2. Azure Machine Learning (Fairlearn)
Used to evaluate algorithmic fairness in hiring decisions.

Capabilities:
- Demographic parity
- Equalized odds
- EEOC 80% rule
- Fairness-constrained learning

Integration:
- Azure ML Workspace
- Microsoft Fairlearn library
- File: modules/azure_ml_fairness_engine.py

## Why both are required
Language Service analyzes unstructured text,
while Azure ML governs fairness across protected groups.
BiasGuard cannot operate without either service.
