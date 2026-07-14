import pandas as pd
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
df=pd.read_csv("telecom_kpi_preprocessed.csv")
X=df.drop(columns=["Cell_ID","Timestamp","Network_Status"])
y=df["Network_Status"]
logistic_model=LogisticRegression(max_iter=1000,random_state=42)
decision_tree_classifier=DecisionTreeClassifier(random_state=42)
random_forest_classifier=RandomForestClassifier(n_estimators=100,random_state=42)
lr_scores=cross_val_score(logistic_model,X,y,cv=5,scoring="accuracy")
dt_scores=cross_val_score(decision_tree_classifier,X,y,cv=5,scoring="accuracy")
rf_scores=cross_val_score(random_forest_classifier,X,y,cv=5,scoring="accuracy")
print("Logistic Regression")
print("Scores:",lr_scores)
print("Mean Accuracy:",lr_scores.mean())
print("Standard Deviation:",lr_scores.std())
print("\nDecision Tree")
print("Scores:",dt_scores)
print("Mean Accuracy:",dt_scores.mean())
print("Standard Deviation:",dt_scores.std())
print("\nRandom Forest")
print("Scores:",rf_scores)
print("Mean Accuracy:",rf_scores.mean())
print("Standard Deviation:",rf_scores.std())