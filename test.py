import matplotlib.pyplot as plt
import matplotlib
import numpy as np
from sklearn.metrics import recall_score, classification_report, accuracy_score

np.random.seed(0)

true = np.random.rand(5000).round()
acc=0.875
error_chance = 0.2
pred_mask = np.random.rand(5000) < acc
error_mask = np.random.rand(5000) < error_chance
print(sum(true))
print(sum(pred_mask)/5000)
pred = true == pred_mask
print(sum(pred))

accuracy = np.mean(pred == true)  
print(accuracy)

pred_err = pred.copy()
pred_err = pred_err & ~error_mask

accuracy_err = np.mean(pred_err == true)
print(acc, accuracy_err)

print(recall_score(true, pred))
print(recall_score(true, pred_err))
print(accuracy_score(true, pred_err))
print(classification_report(true, pred_err))