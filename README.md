# Clickbait Headline Classification

This project implements a machine learning model to classify news headlines as clickbait or non-clickbait. It uses various NLP techniques including:
- Bag of Words (BOW)
- TF-IDF
- Sentence Transformers
- Logistic Regression and Naive Bayes classifiers

## Dataset
The dataset consists of 32,000 headlines:
- Clickbait headlines from BuzzFeed, Upworthy, ViralNova
- Non-clickbait headlines from WikiNews, NYT, The Guardian, and The Hindu

## Features
- Text preprocessing and feature extraction
- Model training and evaluation
- Ablation experiments
- Sentence embedding experiments using transformers
- Interactive Gradio web interface for real-time classification

## Requirements
- Python 3.x
- PyTorch
- scikit-learn
- sentence-transformers
- gradio
- seaborn
- pandas
- numpy

## Usage
1. Run the Jupyter notebook `Group3_YutongHe_Clickbait_Headline_Classification.ipynb` to train models
2. Use the Gradio interface to classify headlines in real-time

## Results
The best performing model (BOW + Logistic Regression) achieved:
- F1 Score: 0.9727
- Accuracy: 0.9727

## Acknowledgments
Based on the dataset from:
Chakraborty, A., Paranjape, B., Kakarla, S., & Ganguly, N. (2016). Stop Clickbait: Detecting and Preventing Clickbaits in Online News Media.
