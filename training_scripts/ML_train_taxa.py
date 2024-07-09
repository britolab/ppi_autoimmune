# Import necessary libraries
import os
import random
import subprocess
from glob import glob

import matplotlib as mpl
import matplotlib.font_manager
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import animation
from scipy import stats
from tqdm import tqdm
import matplotlib.font_manager as fm
from scipy.stats import zscore
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import auc, precision_recall_curve, roc_auc_score, roc_curve
from sklearn.model_selection import StratifiedKFold, cross_val_score, RandomizedSearchCV

# Set up custom font for matplotlib
fe = fm.FontEntry(fname="/workdir/users/hz424/font/Helvetica.ttc", name="helvetica_new")
fm.fontManager.ttflist.insert(0, fe)
mpl.rcParams["font.family"] = "helvetica_new"
plt.rc("axes", unicode_minus=False)

# Load combined MP3 data
path_all = "../"
mpa3 = pd.read_pickle(path_all + "combined_mp3/combined_filtered.mpa3.pkl")

# Create MP3 with taxonomic levels
def create_taxonomic_level_mpa3(mpa3, level):
    return (
        pd.concat(
            [
                mpa3.reset_index(drop=True),
                pd.DataFrame(mpa3.index)[0].str.split("|", expand=True),
            ],
            axis=1,
        )
        .groupby(level)
        .sum()
    )

l1_mpas3 = create_taxonomic_level_mpa3(mpa3, 1)
l2_mpas3 = create_taxonomic_level_mpa3(mpa3, 2)
l3_mpas3 = create_taxonomic_level_mpa3(mpa3, 3)
l4_mpas3 = create_taxonomic_level_mpa3(mpa3, 4)
l5_mpas3 = create_taxonomic_level_mpa3(mpa3, 5)
l6_mpas3 = create_taxonomic_level_mpa3(mpa3, 6)

# Function to transform and standardize data
def transform(x):
    x = x.loc[x.sum(axis=1)[x.sum(axis=1) > 0].index.tolist()]
    x = np.log10(x.T + 1e-5).apply(zscore)
    return x

# Apply transformation to all taxonomic levels
l6_mpas3_t, l5_mpas3_t, l4_mpas3_t, l3_mpas3_t, l2_mpas3_t, l1_mpas3_t = list(
    map(transform, [l6_mpas3, l5_mpas3, l4_mpas3, l3_mpas3, l2_mpas3, l1_mpas3])
)

# Combine all transformed taxonomic levels
new_mpa3 = pd.concat(
    [l6_mpas3_t, l5_mpas3_t, l4_mpas3_t, l3_mpas3_t, l2_mpas3_t, l1_mpas3_t], axis=1
)

# Function to run Random Forest classifier with cross-validation
def run_RF(study, new_mpa3, model_save_path):
    # Prepare training and testing datasets
    train_meta = meta[meta.study_name == study].set_index("new_sample_id")
    test_meta = meta[meta.study_name != study].set_index("new_sample_id")

    # Convert study conditions to binary labels
    train_meta.loc[train_meta.study_condition != "HC", "study_condition"] = 1
    train_meta.loc[train_meta.study_condition == "HC", "study_condition"] = 0
    test_meta.loc[test_meta.study_condition != "HC", "study_condition"] = 1
    test_meta.loc[test_meta.study_condition == "HC", "study_condition"] = 0

    # Extract features and labels
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
    
    random_search.fit(train_x, train_y)

    # Use the best estimator from the RandomizedSearchCV for further evaluation
    classifier = random_search.best_estimator_
    
    # Save the trained model
    model_filename = os.path.join(model_save_path, f"{study}_taxa_RF_model.pkl")
    joblib.dump(classifier, model_filename)
    
    y_scores = classifier.predict_proba(test_x)[:, 1]
    test_score = pd.DataFrame([test_meta.study_name.tolist(), test_y, y_scores]).T

    # Calculate test AUC and AUPRC for other studies
    aucs_all = []
    prc_all = []
    for x in test_score[0].unique():
        for i in range(10):
            test_sub = test_score[test_score[0] == x]
            sub_num = test_sub[1].value_counts().min()
            # Balance test sets
            test_sub_all = pd.concat(
                [
                    test_sub[test_sub[1] == 1].sample(sub_num),
                    test_sub[test_sub[1] == 0].sample(sub_num),
                ]
            )

            aucs_all.append(
                [
                    study,
                    x,
                    roc_auc_score(test_sub_all[1].tolist(), test_sub_all[2].tolist()),
                ]
            )
            
            # Precision-Recall Curve and AUPRC calculation
            precision, recall, _ = precision_recall_curve(test_sub_all[1].tolist(), test_sub_all[2].tolist())
            auprc = auc(recall, precision)
            prc_all.append([study, x, auprc])

    # Convert lists to DataFrames
    aucs_all_df = pd.DataFrame(aucs_all, columns=["train", "test", "auc"])
    prc_all_df = pd.DataFrame(prc_all, columns=["train", "test", "pr_auc"])

    # Manual 5-fold cross-validation for both AUC and AUPRC
    cv_scores_auc = []
    cv_scores_prc = []
    strat_k_fold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    for train_index, test_index in strat_k_fold.split(train_x, train_y):
        X_train_fold, X_test_fold = train_x.iloc[train_index], train_x.iloc[test_index]
        y_train_fold, y_test_fold = [train_y[i] for i in train_index], [train_y[i] for i in test_index]

        classifier.fit(X_train_fold, y_train_fold)
        y_pred_proba = classifier.predict_proba(X_test_fold)[:, 1]

        # Calculate ROC AUC
        roc_auc = roc_auc_score(y_test_fold, y_pred_proba)
        cv_scores_auc.append([study, study, roc_auc])
        
        # Calculate Precision-Recall Curve and AUPRC
        precision, recall, _ = precision_recall_curve(y_test_fold, y_pred_proba)
        pr_auc = auc(recall, precision)
        cv_scores_prc.append([study, study, pr_auc])

    # Convert lists to DataFrames before concatenation
    cv_scores_auc_df = pd.DataFrame(cv_scores_auc, columns=["train", "test", "auc"])
    cv_scores_prc_df = pd.DataFrame(cv_scores_prc, columns=["train", "test", "pr_auc"])

    # Concatenate DataFrames
    aucs_all_combined = pd.concat([aucs_all_df, cv_scores_auc_df])
    prc_all_combined = pd.concat([prc_all_df, cv_scores_prc_df])

    return aucs_all_combined, prc_all_combined

# Run the Random Forest classifier for each study and collect results
all_aucs_all = []
all_prc_all = []

model_save_path = "trained_models/"  # Replace with your path

# Ensure the directory exists
os.makedirs(model_save_path, exist_ok=True)

for x in tqdm(all_study):
    auc_res, prc_res = run_RF(x, new_mpa3, model_save_path)
    all_aucs_all.append(auc_res)
    all_prc_all.append(prc_res)
    print(x, ":done")

# Combine results from all studies
results_mpa3_auc = pd.concat(all_aucs_all)
results_mpa3_prc = pd.concat(all_prc_all)

# Save Precision-Recall results
results_mpa3.to_pickle(path_all + "combined_pred/pred_results_mpa3.pkl")
results_mpa3_prc.to_pickle(path_all + "combined_pred/pred_results_mpa3_prc.pkl")

# Process AUC results for visualization
results_mpa3_auc = (
    results_mpa3_auc.groupby(["train", "test"])
    .median()
    .reset_index()
    .pivot(index="train", columns="test", values="auc")
)

# Process Precision-Recall results for visualization
results_mpa3_prc = (
    results_mpa3_prc.groupby(["train", "test"])
    .median()
    .reset_index()
    .pivot(index="train", columns="test", values="pr_auc")
)

# Plot the AUROC results using seaborn's clustermap
akws = {"ha": "center", "va": "center", "fontsize": 8}
cg = sns.clustermap(
    results_mpa3_auc.loc[all_study][all_study].fillna(0),
    annot=True,
    linewidths=0,
    cmap="Blues",
    fmt=".2f",
    cbar_kws=dict(use_gridspec=False, label="Effect size (q < 0.1)"),
    annot_kws=akws,
    linecolor="black",
    xticklabels=True,
    figsize=(7, 7),
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

# Plot the AUPRC results using seaborn's clustermap
cg_prc = sns.clustermap(
    results_mpa3_prc.loc[all_study][all_study].fillna(0),
    annot=True,
    linewidths=0,
    cmap="Reds",
    fmt=".2f",
    cbar_kws=dict(use_gridspec=False, label="Precision-Recall AUC"),
    annot_kws=akws,
    linecolor="black",
    xticklabels=True,
    figsize=(7, 7),
    row_cluster=False,
    col_cluster=False,
    vmin=0.5,
    vmax=1,
    cbar=True,
)
cg_prc.ax_col_dendrogram.set_visible(False)

plt.ylabel("")
plt.xlabel("")
plt.show()

   
