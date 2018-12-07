import sys
import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pickle

# python3 -i table3plot.py "../performance_tests/data/first_aws_model/" num_epochs "First AWS Model (1000 Epochs)"
args = (sys.argv[1:])

dir = os.listdir(args[0])
logfile_names = [f for f in dir if f[:7] == "logfile"]
logfile_names = sorted(logfile_names, key = lambda f : float(f.split('--')[-1].split('.')[0]))

results = {}
action_sum = {}
action_dist = {}
for logfile in logfile_names:
    with open(args[0] + logfile, 'rb') as f:
        results[logfile] = pickle.load(f)
        action_sum[logfile]  = np.sum(results[logfile]['actions'], axis=0)
        action_num = np.sum(action_sum[logfile])
        action_dist[logfile] = action_sum[logfile] / action_num

idx_epoch = int((int(args[1]) - 1000) / 1000)
action_counts = action_sum[logfile_names[idx_epoch]]
action_freq = action_dist[logfile_names[idx_epoch]]

fig,ax = plt.subplots()
col_scheme = ['#8dd3c7',
'#ffffb3',
'#bebada',
'#fb8072',
'#80b1d3',
'#fdb462',
'#b3de69',
'#fccde5',
'#d9d9d9']
actions = ['NOOP',
 'UP',
 'RIGHT',
 'LEFT',
 'DOWN',
 'UPRIGHT',
 'UPLEFT',
 'DOWNRIGHT',
 'DOWNLEFT']
colors = [col_scheme[r] for r in range(9) if action_counts[r] != 0]
labels = [(actions[r] +": " +  "%.2E"%action_freq[r]) for r in range(9) if action_counts[r] != 0]
patches, texts = ax.pie([c for c in action_counts if c != 0], colors = colors, startangle=90)
ax.axis('equal')
plt.legend(patches, labels, title = "Action Proportions", loc = "best")
plt.title("{}: Average Distribution of Actions".format(args[2]))
plt.show()
