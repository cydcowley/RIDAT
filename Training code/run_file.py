from sklearn import tree
import imageprocessing as ip
import dust_detection as dd
import matplotlib.pyplot as plt
import csv
import numpy as np


im = ip.import_images("KL11-P1DB-92708-3")

set_1 = ip.Images(im)
set_1.find_bg()
ip.iterate_frames(set_1)
training_data=set_1.train()

features=[]
labels = []

for i in range(len(training_data["delta_width"])):
    features.append([])
for i in range(len(training_data["delta_width"])):
    features[i].append(training_data["delta_width"][i])
    features[i].append(training_data["delta_position"][i])
    labels.append(training_data["identifier"][i])

im2 = ip.import_images("test1")
set_2 = dd.Images(im2)
set_2.find_bg()
dd.iterate_frames(set_2)
tx,ty=set_2.connect_frames(features,labels)

for i in range(len(tx)):
    plt.clf()
    plt.cla()
    plt.close()
    implot1 = plt.imshow(set_2.images[0] - set_2.bg)
    plt.scatter(ty[i],tx[i])
    plt.savefig("tracks/temp"+str(i))

