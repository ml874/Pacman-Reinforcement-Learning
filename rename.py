import os

def rename():
    data_root = "/home/knavejack/Documents/School/2018-2019/CS4701/Pacman-Reinforcement-Learning/AWS_models"
    data_dir = "first_aws_model_cont"
    file_root = "first_aws_model_cont"

    dir_path = os.path.join(data_root, data_dir)
    for file in os.listdir(dir_path):
        try:
            s1 = file.split("--")
            # s2 = s1[1].split(".")
            s = [file_root] + s1[1]
            # s[0] = file_root

            print("Renaming: {} to {}.".format(file, new_file))

            while len(s[1])<5:
                s[1] = "0" + s[1]

            new_file = s[0] + "--" + s[1]# + "." + s[2]

            os.rename(os.path.join(dir_path, file), os.path.join(dir_path, new_file))
        except:
            pass

if __name__=="__main__":
    rename()
