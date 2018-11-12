import matplotlib
import matplotlib.image as m
import matplotlib.pyplot as plt
import os
import numpy as np
import imageio as io
import os
from sklearn import tree
import cv2

def train(self):
    
    training={"delta_width":[],"delta_position":[],"identifier": []}
    
    """When dust in all frames has been sorted and characterised, this function connects dust particles across frames, forming a trajectory"""
    
    for i in self.dust_every_frame: #check to ensure characterisation has been performed for every frame
        if i==0:
            return 0
    
    for i in range(len(self.images)-1):
        current_frame = i
        next_frame = i + 1
        for j in range(len(self.dust_every_frame[current_frame]["xpositions"])):
            for k in range(len(self.dust_every_frame[next_frame]["xpositions"])):
                
                dw = np.abs(self.dust_every_frame[next_frame]["widths"][k]-
                            self.dust_every_frame[current_frame]["widths"][j])
                dp = np.sqrt((self.dust_every_frame[next_frame]["xpositions"][k]-self.dust_every_frame[current_frame]["xpositions"][j])**2+
                             (self.dust_every_frame[next_frame]["ypositions"][k] -self.dust_every_frame[current_frame]["ypositions"][j]) ** 2)
                
                training["delta_width"].append(dw)
                training["delta_position"].append(dp)
                
                implot0 = plt.imshow(self.images[current_frame]-self.bg)
                plt.scatter([self.dust_every_frame[current_frame]["ypositions"][j]], [self.dust_every_frame[current_frame]["xpositions"][j]])
                plt.savefig("temp0")

                plt.clf()
                plt.cla()
                plt.close()

                implot1 = plt.imshow(self.images[next_frame]-self.bg)
                plt.scatter([self.dust_every_frame[next_frame]["ypositions"][k]],
                            [self.dust_every_frame[next_frame]["xpositions"][k]])
                plt.savefig("temp1")

                plt.clf()
                plt.cla()
                plt.close()

                img0 = cv2.imread('temp0.png')
                img1 = cv2.imread('temp1.png')

                while (1):
                    cv2.imshow('img', img0)
                    cv2.imshow('img1', img1)
                    key = cv2.waitKey(33)
                    if key == 27:  # Esc key to stop
                        break
                    elif key == 121:  # normally -1 returned,so don't print it
                        training["identifier"].append("yes")
                        break
                    elif key == 110:
                        training["identifier"].append("no")
                        break
    return(training)



def connect_frames(self,features,labels):
    """When dust in all frames has been sorted and characterised, this function connects dust particles across frames, forming a trajectory"""
    clf = tree.DecisionTreeClassifier()
    clf = clf.fit(features, labels)


    for i in self.dust_every_frame: #check to ensure characterisation has been performed for every frame
        if i==0:
            return 0


    trackx = []
    tracky = []
    trackw=[]
    trackframe=[]

    for i in range(len(self.dust_every_frame[0]["xpositions"])):
        trackx.append([self.dust_every_frame[0]["xpositions"][i]])
        tracky.append([self.dust_every_frame[0]["ypositions"][i]])
        trackw.append([self.dust_every_frame[0]["widths"][i]])
    for i in range(len(trackx)):
        trackframe.append(0)

    for i in range(1,len(self.images)):
        for j in range(len(self.dust_every_frame[i]["xpositions"])):
            belongto=False
            for k in range(len(trackx)):
                dw = np.abs(self.dust_every_frame[i]["widths"][j] -
                            trackw[k][-1])
                dp = np.sqrt((self.dust_every_frame[i]["xpositions"][j] -
                              trackx[k][-1]) ** 2 +
                             (self.dust_every_frame[i]["ypositions"][j] -
                              tracky[k][-1]) ** 2)
                print(clf.predict([[dw,dp]]))
                if clf.predict([[dw, dp]])=="yes":
                    trackx[k].append(self.dust_every_frame[i]["xpositions"][j])
                    tracky[k].append(self.dust_every_frame[i]["ypositions"][j])
                    trackw[k].append(self.dust_every_frame[i]["widths"][j])
                    belongto=True

            if belongto==False:
                trackx.append([self.dust_every_frame[i]["xpositions"][j]])
                tracky.append([self.dust_every_frame[i]["ypositions"][j]])
                trackw.append([self.dust_every_frame[i]["widths"][j]])



    return(trackx,tracky)

