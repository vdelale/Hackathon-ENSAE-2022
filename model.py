#%%
from database import main
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
import time
import pickle

"""
Bonne configuration après GridSearch

{'subsample': 1.0,
 'n_estimators': 250,
 'min_child_weight': 4,
 'max_depth': 4,
 'gamma': 0.5,
 'eta': 0.3,
 'colsample_bytree': 0.6}

"""

# %% XGBOOST
df = main()

y = df["rating"]             
X = df.drop(['rating'], axis=1)

# Break off test set from the training data
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, test_size=0.2)

# For the GridSearchCV/RandomizedSearchCV
params = {
    'n_estimators':[250],
    'min_child_weight':[4,5], 
    'gamma':[i/10.0 for i in range(4,6)],  
    'subsample':[i/10.0 for i in range(7,11)],
    'colsample_bytree':[i/10.0 for i in range(6,11)], 
    'max_depth': [4,6,7],
    'eta': [i/10.0 for i in range(3,6)],
}

# With fine-tuned parameters
clf = XGBClassifier(subsample=1.0,
                    n_estimators=250,
                    min_child_weight=4,
                    max_depth=4,
                    gamma=0.5,
                    eta=0.3,
                    colsample_bytree=0.6,
                    objective="multi:softprob")

# In comments: parameters for RandomizedSearchCV
#n_iter_search = 80 
#random_search = RandomizedSearchCV(clf, 
#                                   param_distributions=params,
#                                   n_iter=n_iter_search, 
#                                   cv=5,
#                                   verbose=2,
#                                   scoring="accuracy",
#                                   n_jobs=-1,
#                                   random_state=0
#                                   )

start = time.time()
clf.fit(X_train, y_train)
print("RandomizedSearchCV took %.2f seconds"
      " parameter settings." % ((time.time() - start)))


# %%

#def predict_class(self, X):
#    out = self.predict(X)
#    return np.clip(np.round(out), 0, 3)
    
#model.predict_class = predict_class.__get__(model)

#model.predict_class(X_test)
# %%
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
predictions = clf.predict(X_test)
cm = confusion_matrix(y_test, predictions)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot()
plt.show()
# %%

print(classification_report(y_test, clf.predict(X_test)))
# %%
f_i = clf.feature_importances_
features = X.columns


f_i, features = zip(*sorted(zip(f_i, features), reverse=True))