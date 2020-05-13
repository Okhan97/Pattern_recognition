import numpy as np
from pybalu.feature_analysis import jfisher


def sfs(X_train, d_train, n):
    features = np.array(X_train)
    N, M = features.shape
    remaining_feats = set(np.arange(M))
    selected = list()
    curr_feats = np.zeros((N, 0))

    def score(features, classification):
        n_classes = 2
        p = np.ones((n_classes, 1)) / n_classes
        return jfisher(np.array(features), np.array(classification), p)

    def calc_score(i):
        feats = np.hstack([curr_feats, features[:, i].reshape(-1, 1)])
        return score(feats, d_train)

    for _ in range(n):
        new_selected = max(remaining_feats, key=calc_score)
        selected.append(new_selected)
        remaining_feats.remove(new_selected)
        curr_feats = np.hstack([curr_feats, features[:, new_selected].reshape(-1, 1)])

    return selected
