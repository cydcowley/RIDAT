from sklearn import tree
import imageprocessing as ip
import dust_detection as dd
import matplotlib.pyplot as plt
import csv
import numpy as np
import json
import os
import matplotlib.image as m
import multiprocessing as mp
from functools import partial
import time

streak = False
Threshold_brightness = 15
Threshold_probability = 0.9

learning_variables = {"sigma_delta_position": True,
                      "mean_delta_position": True,
                      "mean_delta_theta": True,
                      "mean_delta_width": True,
                      "mean_theta": True,



                      }

folder = "167342_11198"
set_1=ip.import_images(folder)
type = "D111-D"
dict,bgsub=ip.iterate_frames(set_1,Threshold_brightness)




training_data=dd.train(dict, bgsub,False)
json = json.dumps(training_data)
f = open("training_data/"+type+"/"+folder+".json", "w")
f.write(json)
f.close()

cv2.setMouseCallback('image',mouse_callback) 
while True:
    cv2.imshow('image',bgsub[5])
    k = cv2.waitKey(4)
    if k == 27:
        break
cv2.destroyAllWindows()
"""

training_data=dd.train(dict, bgsub)



training_data={"sigma_delta_position": [], "mean_delta_position": [], "mean_delta_theta": [],
               "mean_delta_width":[],"mean_theta":[],"identifier": []}

a=os.listdir("training_data/"+type)  # listdir returns a list of the entries in the folder
for image in a:
    f1 = open("training_data/"+type+"/"+str(image))
    data=json.load(f1)
    training_data["sigma_delta_position"].extend(data["sigma_delta_position"])
    training_data["mean_delta_position"].extend(data["mean_delta_position"])
    training_data["mean_delta_theta"].extend(data["mean_delta_theta"])
    training_data["mean_delta_width"].extend(data["mean_delta_width"])
    training_data["mean_theta"].extend(data["mean_theta"])
    # training_data["mean_delta_brightness"].extend(data["mean_delta_brightness"])
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


tx, ty, tb, tframe = dd.track(dict,features,labels,streak,Threshold_probability)



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
        implot1 = plt.imshow(set_1[tframe[i][j]])
        plt.scatter(ty[i][j],tx[i][j],c='r')
        plt.savefig("tracks/temp.png")
        img = m.imread("tracks/temp.png")
        images.append(img)
        os.remove("tracks/temp.png")
    np.savetxt(fname="track"+str(i)+".csv",X=np.transpose(frame_data),header="X,Y,BRIGHTNESS,FRAME",delimiter=",",comments="")
    ip.make_gif(images,"tracks","surface"+str(i),0.1)
# images=[]
# for i in range(len(set_1)):
#     plt.clf()
#     plt.cla()
#     plt.close()
#     implot1 = plt.imshow(set_1[i])
#     for j in range(len(tx)):
#         for k in range(len(tx[j])):
#             if tframe[j][k] == i:
#                 plt.scatter(ty[j][k], tx[j][k])
#     plt.savefig("tracks/temp.png")
#     img = m.imread("tracks/temp.png")
#     images.append(img)
#     os.remove("tracks/temp.png")
# ip.make_gif(images,"tracks","surface",0.1)

    ip.make_gif(images,"tracks","surface"+str(i))
"""
