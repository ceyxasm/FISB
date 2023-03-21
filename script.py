import os
import sys

# Reading all files of fisb dataset
def read_files(path):
    files = os.listdir(path)
    files.sort()
    return files

if __name__ == '__main__':

    path = "fisb_dataset/super"
    super = read_files(path)

    path = "fisb_dataset/sub"
    sub = read_files(path)

    # Creating output directory
    os.system("mkdir output")
    os.system("mkdir output/logs")

    subDir = []
    # For all directories in subDir
    for i in range(len(sub)):
        # For all files in sub, Storing all files in subDir in a list
        sub_folder_files = read_files(os.path.join(path, sub[i]))
        subDir.append(sub_folder_files)
        # For all files in super
        if sub[i] == super[i].split('.')[0]:
            # Run the stitching algorithm
            print("Running stitching algorithm for {}...".format(sub[i]))
            l = len(subDir[i])
            alpha = 10
            os.system("python FISB-Pipeline/autoMain.py {} {} {} {} fisb_dataset/sub/{}/ >> output/logs/log{}.txt".format(1,1,l,alpha,sub[i],i+1))
            # Uncoment to clear the logs
            # os.system("echo '' > output/logs/log{}.txt".format(i+1))
            # os.system("python FISB-Pipeline/opencv.py fisb_dataset/sub/{} >> output/logs/log{}.txt".format(sub[i],i+1))
        else:
            continue