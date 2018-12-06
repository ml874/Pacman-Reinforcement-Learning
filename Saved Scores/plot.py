import sys
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pickle

args = (sys.argv[1:])

with open(args[0], 'rb') as f:
    results = pickle.load(f)

scores = results['scores']

sns.set(style="ticks")

fig, (ax_box, ax_hist) = plt.subplots(2, sharex=True,
                                    gridspec_kw={"height_ratios": (.15, .85)})

plt.suptitle("{}: {} Episodes".format(args[1],len(scores)), y = 0.95, fontsize=18)

sns.boxplot(scores, ax=ax_box)
sns.distplot(scores, ax=ax_hist, bins = np.arange(np.amax(scores), step = 50), norm_hist = False)

ax_box.set(yticks=[])
sns.despine(ax=ax_hist)
sns.despine(ax=ax_box, left=True)

plt.show()
