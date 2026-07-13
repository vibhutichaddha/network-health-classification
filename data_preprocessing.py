import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler 
df=pd.read_csv("telecom_kpi_feature_engineered.csv")
print(df.isnull().sum())
numeric_columns=df.select_dtypes(include=["number"]).columns
df[numeric_columns]=df[numeric_columns].fillna(df[numeric_columns].mean())
print("Duplicate Records:",df.duplicated().sum())
df=df.drop_duplicates()
def classify_network(row):
    if (row["RSRP"]>-90 and row["SINR"]>20 and row["Latency"]<20):
        return "Healthy"
    elif (row["RSRP"]<-105 or row["SINR"]<10 or row["Latency"]>3):
        return "Congested"
    else:
        return "Degraded"
df["Network_Status"]=df.apply(classify_network,axis=1)
encoder=LabelEncoder()
df["Network_Status"]=encoder.fit_transform(df["Network_Status"])
print(dict(zip(encoder.classes_,encoder.transform(encoder.classes_))))
X=df.drop(columns=["Cell_ID","Timestamp","Network_Status"])
y=df["Network_Status"]
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.20,random_state=42,stratify=y)
scaler=StandardScaler()
X_train=scaler.fit_transform(X_train)
X_test=scaler.transform(X_test)
print("Training Features:",X_train.shape)
print("Testing Features:", X_test.shape)
print("Training Labels:",y_train.shape)
print("Testing Labels:",y_test.shape)
df.to_csv("telecom_kpi_preprocessed.csv",index=False)