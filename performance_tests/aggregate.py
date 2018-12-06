import os, pickle
import numpy as np

def main(pickle_file, logfile_pointer):
    with open(pickle_file, "rb") as f:
        p = pickle.load(f)
        score_avg = np.mean(p['scores'])
        score_stddev = np.std(p['scores'])
        score_min = np.min(p['scores'])
        score_max = np.max(p['scores'])
        score_median = np.median(p['scores'])

        steps_avg = np.mean(p['steps'], axis=0)
        steps_std = np.std(p['steps'], axis=0)

        action_sum = np.sum(p['actions'], axis=0)
        action_num = np.sum(action_sum)
        distribution_of_actions = action_sum / action_num

        logfile_pointer.write(pickle_file + "\n")
        logfile_pointer.write("------------------------------------------\n")
        logfile_pointer.write("scores\tavg: {}\tstd: {}\tmed: {}\tmin: {}\tmax: {}\n".format(
            score_avg, score_stddev, score_median, score_min, score_max))

        steps_str = "steps:"
        for i in range(3):
            steps_str += "\t" + str(steps_avg[i]) + "+/-" + str(steps_std[i])
        logfile_pointer.write(steps_str + "\n")

        action_str = "actions:"
        for i in range(9):
            action_str += "\t" + str(distribution_of_actions[i])

        logfile_pointer.write(action_str + "\n")
        logfile_pointer.write("\n")

        logfile_pointer.flush()

if __name__ == "__main__":
    root_base = "/home/knavejack/Documents/School/2018-2019/CS4701/Pacman-Reinforcement-Learning/performance_tests/data/"
    data_dir = "first_aws_model"

    logfile_name = os.path.join(root_base, data_dir, data_dir + ".txt")
    logfile_pointer = open(logfile_name, 'w+')

    listdirs = sorted(os.listdir(os.path.join(root_base, data_dir)))

    for pickle_file in listdirs:
        if pickle_file.endswith(".pickle"):
            main(os.path.join(root_base, data_dir, pickle_file), logfile_pointer)

    logfile_pointer.close()
