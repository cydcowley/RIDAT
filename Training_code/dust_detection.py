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

    training = {"sigma_delta_position": [], "mean_delta_position": [], "mean_delta_theta": [], "mean_delta_width": [],
                "mean_delta_brightness": [], "identifier": []}  # define an empty dictionary of machine learning parameters
    
    for frame in range(len(dust_dictionary) - 2):
        
        frame_1 = frame
        frame_2 = frame + 1
        frame_3 = frame + 2
        
        for i in range(len(dust_dictionary[frame_1]["x0s"])):  # cycle through every dust grain combination across three frames
            
            avx0 = (dust_dictionary[frame_1]["x0s"][i] + dust_dictionary[frame_1]["x1s"][i]) / 2
            avy0 = (dust_dictionary[frame_1]["y0s"][i] + dust_dictionary[frame_1]["y1s"][i]) / 2
            theta0 = np.arctan2((dust_dictionary[frame_1]["y0s"][i] - dust_dictionary[frame_1]["y1s"][i]),
                                (dust_dictionary[frame_1]["x0s"][i] - dust_dictionary[frame_1]["x1s"][i]))
            IDed_track = False
            
            for j in range(len(dust_dictionary[frame_2]["x0s"])):
                
                avx1 = (dust_dictionary[frame_2]["x0s"][j] + dust_dictionary[frame_2]["x1s"][j]) / 2
                avy1 = (dust_dictionary[frame_2]["y0s"][j] + dust_dictionary[frame_2]["y1s"][j]) / 2
                dx0 = dust_dictionary[frame_2]["x0s"][j] - dust_dictionary[frame_1]["x0s"][i]
                dy0 = dust_dictionary[frame_2]["y0s"][j] - dust_dictionary[frame_1]["y0s"][i]
                theta1 = np.arctan2((dust_dictionary[frame_2]["y0s"][j] - dust_dictionary[frame_2]["y1s"][j]),
                                    (dust_dictionary[frame_2]["x0s"][j] - dust_dictionary[frame_2]["x1s"][j]))
                dw0 = np.abs(dust_dictionary[frame_2]["widths"][j] - dust_dictionary[frame_1]["widths"][i])
                dp0 = np.sqrt(dx0**2 + dy0**2)
                dtheta0 = np.abs(theta1 - theta0)
                
                for k in range(len(dust_dictionary[frame_3]["x0s"])):
                    
                    avx2 = (dust_dictionary[frame_3]["x0s"][k] + dust_dictionary[frame_3]["x1s"][k]) / 2
                    avy2 = (dust_dictionary[frame_3]["y0s"][k] + dust_dictionary[frame_3]["y1s"][k]) / 2
                    dx1 = dust_dictionary[frame_3]["x0s"][k] - dust_dictionary[frame_2]["x0s"][j]
                    dy1 = dust_dictionary[frame_3]["y0s"][k] - dust_dictionary[frame_2]["y0s"][j]
                    theta2 = np.arctan2((dust_dictionary[frame_3]["y0s"][k] - dust_dictionary[frame_3]["y1s"][k]),
                                        (dust_dictionary[frame_3]["x0s"][k] - dust_dictionary[frame_3]["x1s"][k]))
                    dw1 = np.abs(dust_dictionary[frame_3]["widths"][k] - dust_dictionary[frame_2]["widths"][j])
                    dp1 = np.sqrt(dx1**2 + dy1**2)
                    dtheta1 = np.abs(theta2 - theta1)
                
                    dp_mean = (dp0 + dp1)/2
                    dtheta_mean = (dtheta0 + dtheta1)/2
                    dw_mean = (dw0 + dw1)/2
                    dp_sigma = np.sqrt((dp0 - dp_mean)**2 + (dp1 - dp_mean)**2) # using Bessels correction
                    
                    training["sigma_delta_position"].append(dp_sigma)
                    training["mean_delta_position"].append(dp_mean)
                    training["mean_delta_theta"].append(dtheta_mean)
                    training["mean_delta_width"].append(dw_mean)
                    
                    if IDed_track == False:
                    
                        implot0 = plt.imshow(images[frame_1])
                        plt.scatter([avy0], [avx0])
                        plt.savefig("temp0")
                        plt.clf()
                        plt.cla()
                        plt.close()
                
                        implot1 = plt.imshow(images[frame_2])
                        plt.scatter([avy1], [avx1])
                        plt.savefig("temp1")
                        plt.clf()
                        plt.cla()
                        plt.close()
                        
                        implot2 = plt.imshow(images[frame_3])
                        plt.scatter([avy2], [avx2])
                        plt.savefig("temp2")
                        plt.clf()
                        plt.cla()
                        plt.close()
                
                        img0 = cv2.imread('temp0.png')
                        img1 = cv2.imread('temp1.png')
                        img2 = cv2.imread('temp2.png')
                
                        while True:  # displays current and next frame until a key is pressed
                            cv2.imshow('img0', img0)
                            cv2.imshow('img1', img1)
                            cv2.imshow('img2', img2)
                            key = cv2.waitKey(33)
                            if key == 27:  # Esc key to stop
                                break
                            elif key == 121:  # normally -1 returned,so don't print it
                                training["identifier"].append("yes")
                                IDed_track = True
                                break
                            elif key == 110:
                                training["identifier"].append("no")
                                break
                    else:
                       training["identifier"].append("no")
    return training



def track(dust_dictionary,features,labels):
    """When dust in all frames has been sorted and characterised, this function connects dust particles across frames, forming a trajectory"""
    clf = tree.DecisionTreeClassifier()
    clf = clf.fit(features, labels)

    trackx = []
    tracky = []
    trackw=[]
    track_theta=[]
    track_lastframe=[] #defines the last frame where a dust grain in a given track was recorded
    
    for frame in range(len(dust_dictionary) - 2):
        
        frame_1 = frame
        frame_2 = frame + 1
        frame_3 = frame + 2
        
        for i in range(len(dust_dictionary[frame_1]["x0s"])):  # cycle through every dust grain combination across three frames
            
            x0i = dust_dictionary[frame_1]["x0s"][i]
            y0i = dust_dictionary[frame_1]["y0s"][i]
            theta0 = np.arctan2((dust_dictionary[frame_1]["y0s"][i] - dust_dictionary[frame_1]["y1s"][i]),
                                (dust_dictionary[frame_1]["x0s"][i] - dust_dictionary[frame_1]["x1s"][i]))
            trackx.append([x0i])
            tracky.append([y0i])
            track_lastframe.append([frame])
            prob0 = 0
            
            for j in range(len(dust_dictionary[frame_2]["x0s"])):
                
                dx0 = dust_dictionary[frame_2]["x0s"][j] - dust_dictionary[frame_1]["x0s"][i]
                dy0 = dust_dictionary[frame_2]["y0s"][j] - dust_dictionary[frame_1]["y0s"][i]
                theta1 = np.arctan2((dust_dictionary[frame_2]["y0s"][j] - dust_dictionary[frame_2]["y1s"][j]),
                                    (dust_dictionary[frame_2]["x0s"][j] - dust_dictionary[frame_2]["x1s"][j]))
                dw0 = np.abs(dust_dictionary[frame_2]["widths"][j] - dust_dictionary[frame_1]["widths"][i])
                dp0 = np.sqrt(dx0**2 + dy0**2)
                dtheta0 = np.abs(theta1 - theta0)
                
                for k in range(len(dust_dictionary[frame_3]["x0s"])):
                    
                    dx1 = dust_dictionary[frame_3]["x0s"][k] - dust_dictionary[frame_2]["x0s"][j]
                    dy1 = dust_dictionary[frame_3]["y0s"][k] - dust_dictionary[frame_2]["y0s"][j]
                    theta2 = np.arctan2((dust_dictionary[frame_3]["y0s"][k] - dust_dictionary[frame_3]["y1s"][k]),
                                        (dust_dictionary[frame_3]["x0s"][k] - dust_dictionary[frame_3]["x1s"][k]))
                    dw1 = np.abs(dust_dictionary[frame_3]["widths"][k] - dust_dictionary[frame_2]["widths"][j])
                    dp1 = np.sqrt(dx1**2 + dy1**2)
                    dtheta1 = np.abs(theta2 - theta1)
                
                    dp_mean = (dp0 + dp1)/2
                    dtheta_mean = (dtheta0 + dtheta1)/2
                    dw_mean = (dw0 + dw1)/2
                    dp_sigma = np.sqrt((dp0 - dp_mean)**2 + (dp1 - dp_mean)**2) # using Bessels correction
                    
                    prob1 = clf.predict_proba([[dp_sigma,dp_mean,dtheta_mean,dw_mean]])[0][1]
                    print(clf.predict([[dp_sigma,dp_mean,dtheta_mean,dw_mean]]))
                    print(prob1)
                    if prob1 > prob0:
                        prob0 = prob1
                        a=[j,k]
                        
            #print('frame_1=',len(dust_dictionary[frame_1]["x0s"]))
            #print('frame_2=',len(dust_dictionary[frame_2]["x0s"]))
            #print('j=',a[0])
            #print('frame_3=',len(dust_dictionary[frame_3]["x0s"]))
            #print('k=',a[1])
            if prob0 != float(0):
                trackx[i].append(dust_dictionary[frame_2]["x0s"][a[0]])
                trackx[i].append(dust_dictionary[frame_3]["x0s"][a[1]])
                tracky[i].append(dust_dictionary[frame_2]["y0s"][a[0]])
                tracky[i].append(dust_dictionary[frame_3]["y0s"][a[1]])
                track_lastframe[i].append(frame_2)
                track_lastframe[i].append(frame_3)
                if frame_1 >= 2:
                    for r in range(len(trackx)):
                        if trackx[i][0] == trackx[r][-1] and tracky[i][0] == tracky[r][-1]:
                            trackx[i] += trackx[r]
                            tracky[i] += tracky[r]
                dust_dictionary[frame_2]["x0s"].pop(a[0])
                dust_dictionary[frame_2]["x1s"].pop(a[0])
                dust_dictionary[frame_2]["y0s"].pop(a[0])
                dust_dictionary[frame_2]["y1s"].pop(a[0])
                dust_dictionary[frame_2]["widths"].pop(a[0])
            
    return (trackx,tracky,track_lastframe)

