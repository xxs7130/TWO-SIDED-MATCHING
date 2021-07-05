import pandas as pd
from pandas import DataFrame
import numpy as np


A = [0.619,0.620,0.400,1.000,0.750,0.680,0.611,0.754,0.561,0.756,0.480,1.000,
     0.314,0.692,0.650,0.496,0.494,1.000,0.415,0.342,0.568,0.568,0.618,0.448]

B = [0.445,0.361,0.490,0.726,0.720,0.655,0.716,0.606,0.507,0.558,0.481,0.573,
     0.481,0.637,0.633,0.556,0.322,0.373,0.525,0.831,0.485,0.629,0.627,0.574,0]

# In[]
import matplotlib.pyplot as plt
import matplotlib
csv_file = "27solution.csv"
csv_data = pd.read_csv(csv_file, low_memory = False)#防止弹出警告
df = pd.DataFrame(csv_data)


df = pd.DataFrame(csv_data)
df['F1'] = df.apply(lambda x: np.dot(np.array(x),np.array(A)), axis=1)
# print(df['A'])
df['F2'] = df.apply(lambda x: np.dot(np.array(x),np.array(B)), axis=1)
# print(df['B'])
plt.xlabel('F1')
plt.ylabel('F2')
plt.scatter(df['F1'],df['F2'])
plt.savefig('pareto.pdf')
plt.show()