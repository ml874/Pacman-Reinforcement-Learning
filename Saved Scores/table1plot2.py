import sys
import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pickle

# python3 -i plot2.py "../performance_tests/data/first_aws_model/" "First AWS Model"
args = (sys.argv[1:])

dir = os.listdir(args[0])
logfile_names = [f for f in dir if f[:7] == "logfile"]
logfile_names = sorted(logfile_names, key = lambda f : float(f.split('--')[-1].split('.')[0]))

model_name = args[1]

results = {}
scores = {}
for logfile in logfile_names:
    with open(args[0] + logfile, 'rb') as f:
        results[logfile] = pickle.load(f)
        scores[logfile] = results[logfile]['scores']

min_epochs = int(min(logfile_names, key = lambda f : float(f.split('--')[-1].split('.')[0])).split('--')[-1].split('.')[0])
max_epochs = int(max(logfile_names, key = lambda f : float(f.split('--')[-1].split('.')[0])).split('--')[-1].split('.')[0]) + 1

xs = np.arange(min_epochs, max_epochs, 1000)
xs_left = [x - 20 for x in xs]
xs_right = [x + 20 for x in xs]

mean_ys = [np.mean(scores[f]) for f in logfile_names]
# print(mean_ys)
yerr = [np.std(scores[f]) for f in logfile_names]

# print(yerr)
median_ys = [np.median(scores[f]) for f in logfile_names]
# print(median_ys)
min_ys = [min(scores[f]) for f in logfile_names]
# print(min_ys)
max_ys = [max(scores[f]) for f in logfile_names]
# print(max_ys)

sns.set(style="ticks")

fig,ax = plt.subplots()

ax.errorbar(x=xs, y=median_ys, yerr = [min_ys, max_ys],fmt='_', label = "Median Score w. Min, Max", lw = 1, capsize = 5)
ax.errorbar(x=xs,y=mean_ys,yerr=yerr,fmt='o', label = "Average Score w. SE",  lw = 4, capsize = 5, alpha = 0.3, color = "red")

legend = ax.legend(loc='upper right')
plt.xlabel("# Epochs Trained")
plt.xticks(xs)
plt.ylabel("Score")
plt.title("Score Data: {}".format(model_name))
plt.show()
