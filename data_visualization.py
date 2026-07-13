import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
df=pd.read_csv("telecom_kpi.csv")
def classify_network(row):
    if row["RSRP"]>-90 and row["SINR"]>20 and row["Latency"]<20:
        return "Healthy"
    elif row["RSRP"]<-105 and row["SINR"]<10 and row["Latency"]>3:
        return "Congested"
    else:
        return "Degraded"
df["Network_Status"]=df.apply(classify_network,axis=1)
plt.figure(figsize=(6,5))
sns.countplot(x="Network_Status",data=df,order=["Healthy","Congested","Degraded"],palette="Set2")
plt.title("Network Status Distribution")
plt.xlabel("Network Status")
plt.ylabel("Number of cells")
plt.savefig("network_status_distribution.png",dpi=300,bbox_inches="tight")
plt.show()
plt.figure(figsize=(8,6))
sns.scatterplot(data=df,x="SINR",y="Throughput",hue="Network_Status",palette="Set1")
plt.title("SINR vs Throughput")
plt.xlabel("SINR(dB)")
plt.ylabel("Throughput(Mbps)")
plt.savefig("sinr_vs_throughput.png",dpi=300,bbox_inches="tight")
plt.show()
plt.figure(figsize=(8,5))
sns.histplot(df["RSRP"],bins=20,kde=True,color="skyblue")
plt.title("RSRP Distribution")
plt.xlabel("RSRP (dBm)")
plt.ylabel("Frequency")
plt.savefig("rsrp_distribution.png",dpi=300,bbox_inches="tight")
plt.show()
plt.figure(figsize=(8,6))
numeric_df=df.select_dtypes(include=["number"])
sns.heatmap(numeric_df.corr(), annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Heatmap")
plt.savefig("correlation_heatmap.png",dpi=300,bbox_inches="tight")
plt.show()