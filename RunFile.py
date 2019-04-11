from Dependencies import DustDetection as dd
from Dependencies import ImageProcessing as ip
import matplotlib.pyplot as plt
import csv
import numpy as np
import os
import matplotlib.image as m
import json
import multiprocessing as mp
from functools import partial
import time
import shutil
import glob


streak = False
threshold_brightness =6
threshold_probability = 0.97
variable_switches = {"sigma_delta_position": True,
                      "mean_delta_position": True,
                      "mean_delta_theta": True,
                      "mean_delta_width": True,
                      "mean_delta_brightness": False,
                      "mean_theta": True,}


folder = "S40"
type = "PSI"

set_1=ip.import_images("InputData/"+type+"/"+folder)[760:785]
a = os.listdir("OutputData/TrackFiles")  # listdir returns a list of the entries in the folder
print("hello")
# implot1 = plt.imshow(set_1[0])
# for tr in a:
#     fname="OutputData/TrackFiles/" + str(tr)
#     X=np.loadtxt(fname=fname,delimiter=',',skiprows=1)
#     plt.plot(X[:,0],X[:,1])
# plt.savefig("OutputData/DIII-D.png")
#
# ip.make_gif(set_1,"OutputData","original",0.1)
dict,bgsub=ip.iterate_frames(set_1,threshold_brightness,False)

def write_training(dict,variable_switches,bgsub,type,folder):
    training_data=dd.train(dict, variable_switches, bgsub,False)
    J = json.dumps(training_data)
    f = open("InputData/TrainingData/"+type+"/"+folder+".json", "w")
    f.write(J)
    f.close()



def get_training(input_data,variable_switches):
    training_data={}

    for variable in variable_switches:
        if variable_switches[variable] == True:
            training_data[str(variable)] = []
    training_data["identifier"] = []


    input_data=os.listdir("InputData/TrainingData/"+type)  # listdir returns a list of the entries in the folder

    for dataset in input_data:
        f1 = open("InputData/TrainingData/"+type+"/"+str(dataset))
        data=json.load(f1)
        for variable in variable_switches:
            if variable_switches[variable]==True:
                training_data[str(variable)].extend((data[str(variable)]))
        training_data["identifier"].extend(data["identifier"])

    features=[]
    labels = []

    for i in range(len(training_data["mean_delta_width"])):
        features.append([])
    for i in range(len(training_data["mean_delta_width"])):
        features[i].append(training_data["sigma_delta_position"][i])
        features[i].append(training_data["mean_delta_position"][i])
        features[i].append(training_data["mean_delta_theta"][i])
        features[i].append(training_data["mean_delta_width"][i])
        features[i].append(training_data["mean_theta"][i])
        # features[i].append(training_data["mean_delta_brightness"][i])

        labels.append(training_data["identifier"][i])
    return(features,labels)

#
# split the number of frames into sections to be handled by seperate cores
#
# frames_per_core = int(len(dict)/mp.cpu_count())
# coredictionaries = np.array_split(np.array(dict), frames_per_core)
#
# corevariables=[]
#
# for i in range(len(coredictionaries)):
#     corevariables.append((coredictionaries[i],features,labels, streak))
#
# if __name__=='__main__':
#     pool = mp.Pool()
#     tracking = partial(dd.track,features=features,labels=labels,streak=streak)
#     results = pool.map(tracking,coredictionaries)
#     tx,ty,tframe = results[0]




def output_tracks(tx,ty,tb,tframe,image_set,total_gif):

    for i in range(len(tx)):
        frame_data=[[],[],[],[]]
        images=[]
        for j in range(len(tx[i])):
            frame_data[0].append(tx[i][j])
            frame_data[1].append(ty[i][j])
            frame_data[2].append(tb[i][j])
            frame_data[3].append(tframe[i][j])
            plt.clf()
            plt.cla()
            plt.close()
            implot1 = plt.imshow(image_set[tframe[i][j]])
            plt.scatter(tx[i][j],ty[i][j],c='r')
            plt.title(str(tframe[i][j]))
            plt.savefig("OutputData/TrackImages/temp.png")
            img = m.imread("OutputData/TrackImages/temp.png")
            images.append(img)
            os.remove("OutputData/TrackImages/temp.png")
        np.savetxt(fname="OutputData/TrackFiles/track"+str(i)+".csv",X=np.transpose(frame_data),header="X,Y,BRIGHTNESS,FRAME",delimiter=",",comments="")
        ip.make_gif(images,"OutputData/TrackImages/","track"+str(i),0.1)

    if total_gif==True:

        images=[]
        for i in range(len(set_1)):
            plt.clf()
            plt.cla()
            plt.close()
            implot1 = plt.imshow(set_1[i])
            for j in range(len(tx)):
                for k in range(len(tx[j])):
                    if tframe[j][k] == i:
                        plt.scatter(tx[j][k], ty[j][k])
            plt.savefig("OutputData/TrackImages/temp"+str(i)+".png")
            img = m.imread("OutputData/TrackImages/temp"+str(i)+".png")
            images.append(img)
            os.remove("OutputData/TrackImages/temp"+str(i)+".png")
        ip.make_gif(images,"OutputData/TrackImages/","surface",0.1)

# write_training(dict,variable_switches,bgsub,type,folder)

input_data = os.listdir("InputData/TrainingData/"+type)
features, labels = get_training(input_data=input_data,variable_switches=variable_switches)

print(len(features))
# coredictionaries = np.array_split(np.array(dict), mp.cpu_count())
# print(len(coredictionaries[0]))
# for i in range(len(coredictionaries)-1):
#     coredictionaries[i] = np.append(coredictionaries[i],coredictionaries[i+1][-1])
#
# if __name__=='__main__':
#     pool = mp.Pool()
#     tracking = partial(dd.track,variable_switches=variable_switches,features=features,labels=labels,streak=streak,threshold_probability=threshold_probability,split_switch=False)
#     results = pool.map(tracking ,coredictionaries)
#     if len(results) > 1:
#         for i in range(1,len(results)):
#             for j in range(0,len(results[i][3])):
#                 print(results[i][3][j])
#                 print(len(coredictionaries[i-1]))
#                 results[i][3][j] = (np.array(results[i][3][j])+len(coredictionaries[i-1])).tolist()
#     print(results)


tx, ty, tb, tframe, = dd.track(dust_dictionary=dict,variable_switches=variable_switches,features=features,labels=labels,streak=streak,threshold_probability=threshold_probability,split_switch=False)

output_tracks(tx,ty,tb,tframe,set_1,True)