from sklearn import tree
import imageprocessing as ip
import dust_detection as dd
import matplotlib.pyplot as plt
import csv
import numpy as np
import json
import os


set_1=ip.import_images("KL11-P1DB-92832-1")
dict,bgsub=ip.iterate_frames(set_1,30)
# training_data=dd.train(dict, bgsub)
#
# json = json.dumps(training_data)
# f = open("training_data/KL11-P1DB-92832-1.json", "w")
# f.write(json)
# f.close()



training_data={"delta_width": [], "delta_position": [], "delta_theta": [],
                "identifier": []}

a=os.listdir("training_data")  # listdir returns a list of the entries in the folder
for image in a:
    print(image)
    f1 = open("training_data/"+str(image))
    data=json.load(f1)
    training_data["delta_width"].extend(data["delta_width"])
    training_data["delta_position"].extend(data["delta_position"])
    training_data["delta_theta"].extend(data["delta_theta"])
    training_data["identifier"].extend(data["identifier"])


features=[]
labels = []

for i in range(len(training_data["delta_width"])):
    features.append([])
for i in range(len(training_data["delta_width"])):
    features[i].append(training_data["delta_width"][i])
    features[i].append(training_data["delta_position"][i])
    features[i].append(training_data["delta_theta"][i])
    labels.append(training_data["identifier"][i])

print(features[0])
tx,ty=dd.track(dust_dictionary=dict,images=set_1,features=features,labels=labels)


#

for i in range(len(tx)):
    plt.clf()
    plt.cla()
    plt.close()
    implot1 = plt.imshow(set_1[i])
    plt.scatter(ty[i],tx[i],c='r')
    plt.savefig("tracks/temp"+str(i))

