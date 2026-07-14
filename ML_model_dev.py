import pandas as pd
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler 
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
df=pd.read_csv("telecom_kpi_preprocessed.csv")
X=df.drop(columns=["Cell_ID","Timestamp","Network_Status"])
y=df["Network_Status"]
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.20,random_state=42,stratify=y)
scaler=StandardScaler()
X_train=scaler.fit_transform(X_train)
X_test=scaler.transform(X_test)
## Logistic Regression
logistic_model=LogisticRegression(random_state=42,max_iter=1000)
logistic_model.fit(X_train,y_train)
logistic_predictions=logistic_model.predict(X_test)
## Decision Tree Classifier
decision_tree=DecisionTreeClassifier(random_state=42)
decision_tree.fit(X_train,y_train)
decision_tree_predictions=decision_tree.predict(X_test)
## Random Forest Classifier
random_forest=RandomForestClassifier(n_estimators=100,random_state=42)
random_forest.fit(X_train,y_train)
random_forest_predictions=random_forest.predict(X_test)
print(logistic_predictions[:10])
print(decision_tree_predictions[:10])
print(random_forest_predictions[:10])