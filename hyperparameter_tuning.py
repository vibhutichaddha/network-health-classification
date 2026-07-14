import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
df=pd.read_csv("telecom_kpi_preprocessed.csv")
X=df.drop(columns=["Cell_ID","Timestamp","Network_Status"])
y=df["Network_Status"]
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.20,random_state=42,stratify=y)
decision_tree=DecisionTreeClassifier(random_state=42)
dt_parameters={"max_depth":[3,5,10,None],"min_samples_split":[2,5,10],"criterion":["gini","entropy"]}
dt_grid=GridSearchCV(estimator=decision_tree,param_grid=dt_parameters,cv=5,scoring="accuracy",n_jobs=-1)
dt_grid.fit(X_train,y_train)
print("Best Prameters:")
print(dt_grid.best_params_)
random_forest=RandomForestClassifier(random_state=42)
rf_parameters={"n_estimators":[50,100,200],"max_depth":[5,10,None],"min_samples_split":[2,5,10]}
rf_grid=GridSearchCV(estimator=random_forest,param_grid=rf_parameters,cv=5,scoring="accuracy",n_jobs=-1)
rf_grid.fit(X_train,y_train)
print(rf_grid.best_params_)
print(rf_grid.best_score_)