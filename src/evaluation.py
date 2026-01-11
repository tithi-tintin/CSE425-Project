from sklearn.metrics import (silhouette_score, calinski_harabasz_score, 
                             davies_bouldin_score, adjusted_rand_score, 
                             normalized_mutual_info_score, confusion_matrix)
import numpy as np

def purity_score(y_true, y_pred):
    """
    Calculate cluster purity.
    Purity = (1/N) * sum(max(intersection(cluster_k, class_j)))
    """
    # confusion matrix (contingency table)
    contingency_matrix = confusion_matrix(y_true, y_pred)
    # sum of max values in each column / total samples
    return np.sum(np.amax(contingency_matrix, axis=0)) / np.sum(contingency_matrix)

def evaluate_clustering(X, labels_pred, labels_true=None):
    # single cluster case
    if len(set(labels_pred)) < 2:
         # Standard metrics fail with < 2 clusters
         metrics = {
            "silhouette": -1,
            "calinski": 0,
            "davies_bouldin": 100
        }
    else:
        metrics = {
            "silhouette": silhouette_score(X, labels_pred),
            "calinski": calinski_harabasz_score(X, labels_pred),
            "davies_bouldin": davies_bouldin_score(X, labels_pred)
        }
    
    # Supervised Metrics (Hard Task Requirements)
    if labels_true is not None:
        metrics["ARI"] = adjusted_rand_score(labels_true, labels_pred)
        metrics["NMI"] = normalized_mutual_info_score(labels_true, labels_pred)
        metrics["Purity"] = purity_score(labels_true, labels_pred)
        
    return metrics
