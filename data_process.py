
#  %%
 
import pandas as pd
import dask
import dask.dataframe as dd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt

dd_df = dd.read_csv('out/*.csv')  
df = dd_df.compute();df
df["CRLB"] = np.sqrt(1/df["Fisher Information"])
sns.scatterplot(x="r",y="CRLB",data=df)

plt.ylim(0,1)
# %%
