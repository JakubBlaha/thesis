# %%

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# %%
# CNN

plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman', 'DejaVu Serif', 'Serif']

conf_matrix = np.array([
    [41, 3, 40],
    [0, 59, 23],
    [14, 20, 88]
])

labels = ['GAD', 'SAD', 'Control']

sns.set(font_scale=1.2)
sns.set_style("white", {"font.family": "serif"})
plt.figure(figsize=(8, 6))

sns.heatmap(conf_matrix,
            annot=True,
            fmt='d',
            cmap='Blues',
            xticklabels=labels,
            yticklabels=labels,
            cbar=True,
            vmin=0,
            vmax=88)

plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')

plt.show()
