import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('orderDiffs.csv')
lengths = list( range(1, len(df) + 1))
df.index = lengths

df.plot()
plt.xlabel('Experiment days')
plt.ylabel('Sales diff percent')
plt.title('How long should we run experiments')
plt.show()
