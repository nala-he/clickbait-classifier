
import pandas as pd
from ablation_runner import run_ablation_experiment

# Ablation Batch Runner
# This module runs a series of ablation experiments with different configurations
# to evaluate the impact of various preprocessing steps on model performance.
# E.g., vectorization method, stopword removal, and input truncation.
def run_all_ablation_experiments(data):
    configs = []
    for vectorizer in ['tfidf', 'bow']:
        for stopwords in [True, False]:
            for truncate in [False, True]:
                result = run_ablation_experiment(
                    data=data,
                    vectorizer_type=vectorizer,
                    use_stopwords=stopwords,
                    truncate=truncate
                )
                configs.append(result)
    return pd.DataFrame(configs)
