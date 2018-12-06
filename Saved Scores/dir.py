import os
import sys

args = (sys.argv[1:])
dir = os.listdir(args[0])
logfiles = [f for f in dir if f[:7] == "logfile"]

logfiles = sorted(logfiles, key = lambda f : float(f.split('--')[-1].split('.')[0]))

print(logfiles)
