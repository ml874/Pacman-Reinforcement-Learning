import sys
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pickle

args = (sys.argv[1:])
model_name = args[0]

xs = []
mean_ys = []
yerr = []
min_ys = []
max_ys = []

if (model_name == "First AWS Model"):
    xs = np.arange(1000,10001, 1000)
    mean_ys = [240.73, 60., 60., 183.70, 72.98, 347.63, 218.22, 101.22, 60.36, 379.66]
    yerr = [51.8, 0., 0., 114.55, 15.36, 203.69, 3.83, 41.96, 4.63, 149.13]
    median_ys = [250, 60, 60, 160, 70, 310, 220, 100, 60, 370]
    min_ys = [140, 60, 60, 110, 70, 80, 210, 60, 120, 160]
    max_ys = [1010, 60, 60, 1720, 230, 2100, 220, 260, 60, 1010]

sns.set(style="ticks")

fig,ax = plt.subplots()

avg_se_line = ax.errorbar(x=xs,y=mean_ys,yerr=yerr,fmt='o', label = "Average Score w. SE", capsize = 2)
ax.plot(xs, median_ys, label = "Median Score")
ax.fill_between(xs, min_ys, max_ys, alpha = 0.2)

legend = ax.legend(loc='upper right')
plt.xlabel("# Epochs Trained")
plt.ylabel("Score")
plt.title("Score Data: {}".format(model_name))
plt.show()
