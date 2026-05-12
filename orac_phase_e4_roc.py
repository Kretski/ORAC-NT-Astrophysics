import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import roc_curve, auc

np.random.seed(0)

signal_scores = np.random.normal(10, 2, 1000)
noise_scores  = np.random.normal(4, 2, 1000)

y_true = np.concatenate([
    np.ones(len(signal_scores)),
    np.zeros(len(noise_scores))
])

scores = np.concatenate([
    signal_scores,
    noise_scores
])

fpr, tpr, thr = roc_curve(y_true, scores)

roc_auc = auc(fpr, tpr)

plt.figure(figsize=(6,6))

plt.plot(fpr, tpr, label=f"AUC={roc_auc:.3f}")

plt.plot([0,1],[0,1],'--')

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")

plt.title("ORAC Phase E4 — ROC Curve")

plt.legend()

plt.savefig("outputs/e4_roc.png", dpi=150)

print(f"AUC={roc_auc:.3f}")