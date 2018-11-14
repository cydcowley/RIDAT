import matplotlib
import matplotlib.image as m
import matplotlib.pyplot as plt
import os
import numpy as np
import imageio as io
import os
from sklearn import tree
import cv2


def train(dust_dictionary,images):
    """When dust in all frames has been sorted and characterised, this function connects dust particles across frames, forming a trajectory"""

    training = {"delta_width": [], "delta_position": [], "delta_theta": [],
                "identifier": []}  # define an empty dictionary of machine learning parameters

    for i in range(len(dust_dictionary) - 1):

        current_frame = i
        next_frame = i + 1

        for j in range(len(
                dust_dictionary[current_frame]["x0s"])):  # cycle through every dust grain combination across two frames
            for k in range(len(dust_dictionary[next_frame]["x0s"])):

                avx0 = (dust_dictionary[current_frame]["x0s"][j] + dust_dictionary[current_frame]["x1s"][j]) / 2
                avx1 = (dust_dictionary[next_frame]["x0s"][k] + dust_dictionary[next_frame]["x1s"][k]) / 2
                avy0 = (dust_dictionary[current_frame]["y0s"][j] + dust_dictionary[current_frame]["y1s"][j]) / 2
                avy1 = (dust_dictionary[next_frame]["y0s"][k] + dust_dictionary[next_frame]["y1s"][k]) / 2

                theta0 = np.arctan2(
                    (dust_dictionary[current_frame]["y0s"][j] - dust_dictionary[current_frame]["y1s"][j]),
                    (dust_dictionary[current_frame]["x0s"][j] - dust_dictionary[current_frame]["x1s"][j]))
                theta1 = np.arctan2((dust_dictionary[next_frame]["y0s"][k] - dust_dictionary[next_frame]["y1s"][k]),
                                    (dust_dictionary[next_frame]["x0s"][k] - dust_dictionary[next_frame]["x1s"][k]))

                dw = np.abs(dust_dictionary[next_frame]["widths"][k] -
                            dust_dictionary[current_frame]["widths"][j])
                dp = np.sqrt((avx1 - avx0) ** 2 + (avy1 - avy0) ** 2)

                dtheta = np.abs(theta1 - theta0)

                training["delta_width"].append(dw)
                training["delta_position"].append(dp)
                training["delta_theta"].append(dtheta)

                implot0 = plt.imshow(images[current_frame])
                plt.scatter([avy0], [avx0])
                plt.savefig("temp0")
                plt.clf()
                plt.cla()
                plt.close()

                implot1 = plt.imshow(images[next_frame])
                plt.scatter([avy1], [avx1])

                plt.savefig("temp1")
                plt.clf()
                plt.cla()
                plt.close()

                img0 = cv2.imread('temp0.png')
                img1 = cv2.imread('temp1.png')

                while True:  # displays current and next frame until a key is pressed
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
    return training



def track(dust_dictionary,images,features,labels):
    """When dust in all frames has been sorted and characterised, this function connects dust particles across frames, forming a trajectory"""
    clf = tree.DecisionTreeClassifier()
    clf = clf.fit(features, labels)

    trackx = []
    tracky = []
    trackw=[]
    track_theta=[]
    track_lastframe=[] #defines the last frame where a dust grain in a given track was recorded

    for i in range(len(dust_dictionary[0]["x0s"])):
        avx = (dust_dictionary[0]["x0s"][i] + dust_dictionary[0]["x1s"][i]) / 2
        avy = (dust_dictionary[0]["y0s"][i] + dust_dictionary[0]["y1s"][i]) / 2
        theta = np.arctan2((dust_dictionary[0]["y0s"][i] - dust_dictionary[0]["y1s"][i]),
            (dust_dictionary[0]["x0s"][i] - dust_dictionary[0]["x1s"][i]))

        trackx.append([avx])
        tracky.append([avy])
        track_theta.append([theta])
        trackw.append([dust_dictionary[0]["widths"][i]])

    for i in range(len(trackx)):
        track_lastframe.append(0)

    for i in range(1,len(images)):
        for j in range(len(dust_dictionary[i]["x0s"])):
            belong_to = False

            avx = (dust_dictionary[i]["x0s"][j] + dust_dictionary[i]["x1s"][j]) / 2
            avy = (dust_dictionary[i]["y0s"][j] + dust_dictionary[i]["y1s"][j]) / 2
            theta = np.arctan2((dust_dictionary[i]["y0s"][j] - dust_dictionary[i]["y1s"][j]),
                               (dust_dictionary[i]["x0s"][j] - dust_dictionary[i]["x1s"][j]))


            for k in range(len(trackx)):

                dw = np.abs(dust_dictionary[i]["widths"][j] -
                            trackw[k][-1])
                dp = np.sqrt((avx -trackx[k][-1]) ** 2 +(avy -tracky[k][-1]) ** 2)
                dtheta= np.abs(theta-track_theta[k][-1])

                if clf.predict([[dw, dp,dtheta]]) == "yes" and track_lastframe[k] >= i-5:
                    trackx[k].append(avx)
                    tracky[k].append(avy)
                    track_theta[k].append(theta)
                    trackw[k].append(dust_dictionary[i]["widths"][j])
                    belong_to = True
                    track_lastframe[k]=i

            if not belong_to:
                trackx.append([avx])
                tracky.append([avy])
                track_theta.append([theta])
                trackw.append([dust_dictionary[i]["widths"][j]])
                track_lastframe.append(i)

    return (trackx,tracky)

