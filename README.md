# Codemaverick Fine Tuning - LLM Response Classification

This project focuses on fine-tuning a machine learning model to classify which of two Large Language Model (LLM) responses is better, based on a given prompt. The project utilizes a dataset from Kaggle and implements a Logistic Regression pipeline with TF-IDF vectorization.

## Project Overview

The goal is to predict the "winner" between two models (Model A or Model B) or if it's a tie, based on their responses to a prompt. This is a multi-class classification problem.

## Dataset

The dataset is sourced from the [Kaggle LLM - Detect AI Generated Text](https://www.kaggle.com/competitions/llm-detect-ai-generated-text) (or similar competition/dataset context based on the notebook content).
*   **Train Data**: Contains prompts, responses from two models, and the winner label.
*   **Test Data**: Contains prompts and responses for inference.

## Prerequisites

To run this notebook, you need the following Python libraries:

*   `pandas`
*   `numpy`
*   `scikit-learn`
*   `regex`

You can install them using pip:

```bash
pip install pandas numpy scikit-learn
```

## Usage

1.  Clone the repository or download the `Codemaverick Fine Tuning.ipynb` file.
2.  Ensure you have the dataset available (update paths in the notebook if necessary).
3.  Open the notebook in Jupyter Notebook or JupyterLab.
4.  Run the cells sequentially to train the model and generate predictions.

## Methodology

The project follows these steps:

1.  **Data Loading**: Loads training and testing data from CSV files.
2.  **Target Construction**: Converts the winner columns (`winner_model_a`, `winner_model_b`, `winner_tie`) into a single target variable `y` (0, 1, 2).
3.  **Text Preprocessing**: Cleans the text data (prompts and responses) by removing special characters and formatting.
4.  **Feature Engineering**: Combines prompt, response A, and response B into a single text string with separators (`[SEP_A]`, `[SEP_B]`).
5.  **Vectorization**: Uses `TfidfVectorizer` to convert text data into numerical features (max features: 150,000, n-grams: 1-2).
6.  **Model Training**: Trains a `LogisticRegression` model (multinomial) on the TF-IDF features.
7.  **Evaluation**: Validates the model using a hold-out validation set and calculates Log Loss.
8.  **Inference**: Retrains the model on the full dataset and generates predictions for the test set.
9.  **Submission**: Creates a `submission.csv` file with prediction probabilities.

## Results

*   **Validation Log Loss**: ~1.10313

## Files

*   `Codemaverick Fine Tuning.ipynb`: The main Jupyter Notebook containing the code.
*   `README.md`: This file.
