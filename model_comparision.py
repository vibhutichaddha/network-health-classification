import pandas as pd
comparision=pd.DataFrame({"Model":["LOGISTIC REGRESSION","DECISION TREE","RANDOM FOREST"],
                          "Accuracy":[0.980,0.960,0.995],
                          "Precision":[0.981127,0.960000,0.995074],
                          "Recall":[0.980,0.960,0.995],
                          "F1 Score":[0.979977,0.960000,0.994999]})
print(comparision)