import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# 'Mean Absolout Error'
df = pd.read_csv(f'csv\\result_final.csv')
print("df   :   ", df)
df = df.drop(["DeviceId", "Predicted Water Consumtion"], axis=1)
#result1 = df.groupby('Computation Range').agg({'Mean Absolout Error': ['min']}).plot()
result1 = df.groupby('Computation Range').agg({'Mean Absolout Error': ['min']})
result2 = df.groupby("Computation Range").agg({'Mean Absolout Error': ['min'], 'What Hour': 'sum'})
result3 = df.sort_values(by='Computation Range').drop_duplicates(subset='What Hour')
# result4 = df.loc[df.groupby("item")["diff"].idxmin()]
result4 = df.loc[df.groupby("Computation Range")["Mean Absolout Error"].idxmin()]
print("df : ", df)
print("result1 : ", result1)
print("result2  :   ", result2)
# print("result3  :   ", result3)
print("result4     :    ", result4)

computation_range  = result4 ['Computation Range'].tolist()
what_hour = result4 ['What Hour'].tolist()
mean_absolout_error = result4 ['Mean Absolout Error'].tolist()
plt.plot(computation_range, what_hour,   label = 'What Hour',  marker='o', linewidth=2)
plt.plot(computation_range, mean_absolout_error, label = 'Mean Absolout Error', marker='o', linewidth=2)
plt.xlabel('Computation Range')
plt.legend()
plt.xticks(computation_range)
# plt.yticks(np.arange(0, len(what_hour) + 1, 1))
plt.title('Compiotation Range And MAE Just For One ID')
plt.show()

