import pandas as pd
df=pd.read_csv("telecom_kpi.csv")
df["Signal_Quality"]=(0.6*df["SINR"]+0.4*df["RSRP"]+120)
df[["RSRP","SINR","Signal_Quality"]].head()
Maximum_Users=df["Connected_Users"].max()
df["Network_Load"]=df["Connected_Users"]/Maximum_Users
df["Network_Health_Index"]=((df["Throughput"]/df["SINR"].replace(0,0.1))-df["Latency"]-df["Packet_Loss"])
print(df[["Signal_Quality","Network_Load","Network_Health_Index"]].head())
print(df.info())
df.to_csv("telecom_kpi_feature_engineered.csv",index=False)