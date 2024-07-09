# Import necessary libraries
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import auc, precision_recall_curve, roc_auc_score, roc_curve
from sklearn.model_selection import StratifiedKFold, cross_val_score, RandomizedSearchCV
from scipy.stats import zscore
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Data transformation and standardization

# Human proteins
new_all_ppi_hump = np.log10(
    all_ppi_hump.loc[all_ppi_hump.sum(axis=1)[all_ppi_hump.sum(axis=1) > 0].index].T
    + 1e-9
).apply(zscore)

def run_RF(study, new_mpa3):
    """
    Trains a Random Forest classifier on the input data and calculates AUC scores
    for the test data and cross-validation scores for the training data.
    
    Parameters:
    study: The study name used to split the training and test data
    new_mpa3: The dataset containing the samples
    
    Returns:
    metrics_all_df: A DataFrame containing AUC and AUPRC scores for the test data and cross-validation scores
    """
    # Split the data into training and testing sets based on the study name
    train_meta = meta[meta.study_name == study].set_index("new_sample_id")
    test_meta = meta[meta.study_name != study].set_index("new_sample_id")

    # Convert study conditions to binary labels
    train_meta.loc[train_meta.study_condition != "HC", "study_condition"] = 1
    train_meta.loc[train_meta.study_condition == "HC", "study_condition"] = 0
    test_meta.loc[test_meta.study_condition != "HC", "study_condition"] = 1
    test_meta.loc[test_meta.study_condition == "HC", "study_condition"] = 0

    # Extract features and labels for training and testing
    train_x = new_mpa3.loc[train_meta.index]
    train_y = train_meta.study_condition.tolist()
    test_x = new_mpa3.loc[test_meta.index]
    test_y = test_meta.study_condition.tolist()

    # Hyperparameter optimization using RandomizedSearchCV
    param_dist = {
        "n_estimators": [500, 1000, 2000, 3000, 4000, 5000],
        "max_depth": [None] + list(np.arange(10, 110, 10)),
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4],
        "bootstrap": [True, False],
        "class_weight": ["balanced", "balanced_subsample", None],
    }

    n_iter_search = 100
    random_search = RandomizedSearchCV(
        estimator=RandomForestClassifier(random_state=42, n_jobs=10),
        param_distributions=param_dist,
        n_iter=n_iter_search,
        scoring='roc_auc',
        cv=StratifiedKFold(n_splits=5, shuffle=True),
        random_state=42,
        n_jobs=10,
        refit=True,
        verbose=1,
    )
    
    # Fit the random search model
    random_search.fit(train_x, train_y)

    # Use the best estimator from the RandomizedSearchCV for further evaluation
    classifier = random_search.best_estimator_
    
    # Calculate test AUC and AUPRC for other studies
    metrics_all = []
    y_scores = classifier.predict_proba(test_x)[:, 1]
    test_score = pd.DataFrame([test_meta.study_name.tolist(), test_y, y_scores]).T

    for x in test_score[0].unique():
        for i in range(10):
            test_sub = test_score[test_score[0] == x]
            sub_num = test_sub[1].value_counts().min()
            
            test_sub_all = pd.concat([
                test_sub[test_sub[1] == 1].sample(sub_num),
                test_sub[test_sub[1] == 0].sample(sub_num),
            ])

            # AUC calculation
            y_true = test_sub_all[1].tolist()
            y_scores = test_sub_all[2].tolist()
            roc_auc = roc_auc_score(y_true, y_scores)

            # Precision-Recall Curve and AUPRC calculation
            precision, recall, _ = precision_recall_curve(y_true, y_scores)
            pr_auc = auc(recall, precision)

            metrics_all.append([study, x, roc_auc, pr_auc])

    metrics_all_df = pd.DataFrame(metrics_all, columns=["train", "test", "auc", "pr_auc"])

    # Manual 5-fold cross-validation for both AUC and AUPRC
    for i in range(5):
        strat_k_fold = StratifiedKFold(n_splits=5, shuffle=True, random_state=i)
        for train_index, test_index in strat_k_fold.split(train_x, train_y):
            X_train_fold, X_test_fold = train_x.iloc[train_index], train_x.iloc[test_index]
            y_train_fold, y_test_fold = [train_y[i] for i in train_index], [train_y[i] for i in test_index]

            classifier.fit(X_train_fold, y_train_fold)
            y_pred_proba = classifier.predict_proba(X_test_fold)[:, 1]

            # Calculate ROC AUC and AUPRC for each fold
            roc_auc = roc_auc_score(y_test_fold, y_pred_proba)
            precision, recall, _ = precision_recall_curve(y_test_fold, y_pred_proba)
            pr_auc = auc(recall, precision)

            metrics_all_df = metrics_all_df.append({"train": study, "test": study, "auc": roc_auc, "pr_auc": pr_auc}, ignore_index=True)

    return metrics_all_df

# Run the Random Forest classifier for each study and collect results
all_metrics_all = []
for x in tqdm(all_study):
    metrics_df = run_RF(x, new_all_ppi_hump)
    all_metrics_all.append(metrics_df)
    print(x, ":done")

# Concatenating all metrics
results_ppi_all = pd.concat(all_metrics_all)

# Saving the combined results
results_ppi_all.to_pickle(path_all + "combined_pred/pred_results_ppi_all.pkl")

# Aggregating and pivoting AUC results
results_ppi_auc = (
    results_ppi_all[['train', 'test', 'auc']]
    .groupby(['train', 'test'])
    .median()
    .reset_index()
    .pivot(index='train', columns='test', values='auc')
)

# Aggregating and pivoting AUPRC results
results_ppi_prc = (
    results_ppi_all[['train', 'test', 'pr_auc']]
    .groupby(['train', 'test'])
    .median()
    .reset_index()
    .pivot(index='train', columns='test', values='pr_auc')
)

# Plot the AUROC results using seaborn's clustermap
akws = {"ha": "center", "va": "center", "fontsize": 8}
cg = sns.clustermap(
    results_ppi_auc.fillna(0),
    annot=True,
    linewidths=0,
    cmap="Blues",
    fmt=".2f",
    cbar_kws=dict(use_gridspec=False),
    annot_kws=akws,
    linecolor="black",
    xticklabels=True,
    figsize=(6, 6),
    row_cluster=False,
    col_cluster=False,
    vmin=0.5,
    vmax=1,
    cbar=True,
)
cg.ax_col_dendrogram.set_visible(False)

plt.ylabel("")
plt.xlabel("")
plt.show()
