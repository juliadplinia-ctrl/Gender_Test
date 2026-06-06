# Cross-Lingual Bias Benchmarking in LLMs
An automated testing pipeline designed to evaluate and compare inherent biases in Large Language Models (LLMs) when prompted in different languages.

This project specifically tests whether the Llama 3 (8B) model exhibits different levels of bias when prompted in English versus direct Russian translations, utilizing an advanced "LLM-as-a-Judge" methodology.

## Project Objectives
Quantify Cross-Lingual Discrepancies: Determine if the language of the prompt alters the severity or manifestation of bias.
Multi-Dimensional Testing: Evaluate the model across three specific bias vectors:
 1. Cultural Bias (e.g., assuming Western/American norms as the default)
 2. Occupational Gender Bias (e.g., assigning traditional genders to specific roles)
 3. Age Bias (e.g., attributing generational stereotypes)
 4. Automated Scoring: Utilize a secondary, zero-temperature LLM judge to evaluate the primary model's outputs and provide binary scoring with justifications.
# Tech Stack
Language: Python 3.x
AI Engine: Ollama (Local inference, 100% free and private)
Models: llama3:8b (Used for both Subject Generation and Judge Evaluation)
Data Processing: Pandas
Visualization: Matplotlib, Seaborn

# Getting Started

### Environment Setup
Clone this repository and open it in your terminal. Create and activate a Python virtual environment:

Windows (PowerShell):

py -m venv env
.\env\Scripts\activate