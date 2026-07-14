import pandas as pd 
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
df=pd.read_csv("telecom_kpi_preprocessed.csv")
X=df.drop(columns=["Cell_ID","Timestamp","Network_Status"])
y=df["Network_Status"]
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.20,random_state=42,stratify=y)
rf=RandomForestClassifier(n_estimators=100,random_state=42)
rf.fit(X_train,y_train)
importance=rf.feature_importances_
feature_importance=pd.DataFrame({"Feature":X.columns,"Importance":importance})
feature_importance=feature_importance.sort_values(by="Importance",ascending=False)
print(feature_importance)
top5=feature_importance.head(5)
print(top5)
plt.figure(figsize=(8,5))
plt.bar(feature_importance["Feature"],feature_importance["Importance"])
plt.xticks(rotation=45)
plt.xlabel("Features")
plt.ylabel("Importance Score")
plt.title("Random Forest Feature Importance")
plt.tight_layout()
plt.savefig("feature_importance.png",dpi=300)
plt.show()