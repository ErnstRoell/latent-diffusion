%load_ext autoreload    
%autoreload 2
from IPython.display import display

#|%%--%%| <02GfimQmvW|hgUlWfpvTB>

"""
Plot the training metrics.
"""

import pandas as pd
import seaborn as sns
import glob

files = glob.glob("results/mnist/metrics_**")
df_list = []
for f in files:
    df_list.append(pd.read_json(f,orient="records",lines=True))
df = pd.concat(df_list)



#|%%--%%| <hgUlWfpvTB|P9uZhHzFFO>
r"""°°°
Plot loss versus.
°°°"""
#|%%--%%| <P9uZhHzFFO|Va3IYWGuCX>

sns.relplot(data=df, x="epoch", y="loss", hue="model",kind="line",col="split");

#|%%--%%| <Va3IYWGuCX|9nEBY26PAP>
sns.relplot(data=df, x="epoch", y="loss", hue="split",kind="line",col="model");
#|%%--%%| <9nEBY26PAP|dLOZTb8k2i>
display(df[["split","model","loss"]].groupby(by=["model","split"]).agg("min").unstack())


