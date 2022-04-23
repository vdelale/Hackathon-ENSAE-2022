#%%
from database import df
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.model_selection import RandomizedSearchCV
import time

# %% XGBOOST
y = df["rating"]             
X = df.drop(['rating'], axis=1)

# Break off validation set from training data
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, test_size=0.2)


params = {
    'n_estimators':[100],
    'min_child_weight':[4,5], 
    'gamma':[i/10.0 for i in range(3,6)],  
    'subsample':[i/10.0 for i in range(6,11)],
    'colsample_bytree':[i/10.0 for i in range(6,11)], 
    'max_depth': [2,3,4,6,7],
    'eta': [i/10.0 for i in range(3,6)],
}

clf = XGBClassifier()

# run randomized search
n_iter_search = 20
random_search = RandomizedSearchCV(clf, 
                                   param_distributions=params,
                                   n_iter=n_iter_search, 
                                   cv=5,
                                   verbose=3,
                                   scoring="f1_weighted",
                                   n_jobs=-1)

start = time.time()
random_search.fit(X_train, y_train)
print("RandomizedSearchCV took %.2f seconds for %d candidates"
      " parameter settings." % ((time.time() - start), n_iter_search))

# %%
model = random_search.best_estimator_

#def predict_class(self, X):
#    out = self.predict(X)
#    return np.clip(np.round(out), 0, 3)
    
#model.predict_class = predict_class.__get__(model)

#model.predict_class(X_test)
# %%
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
predictions = model.predict(X_test)
cm = confusion_matrix(y_test, predictions)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot()
plt.show()
# %%
