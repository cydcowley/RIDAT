import matplotlib
import matplotlib.image as m
import matplotlib.pyplot as plt
import os
import numpy as np
import imageio as io
import os
from sklearn.naive_bayes import GaussianNB
import cv2


def find_dp_dtheta_avtheta(position_listx,position_listy):  # uses information on distance travelled within frames as well as between frames
    dp_list=[]
    angle_list=[]
    dytotal=0
    dxtotal=0
    delta_angle_list=[]
    for i in range(0,len(position_listx)-1):
        dy=position_listy[i+1]-position_listy[i]
        dx=position_listx[i+1]-position_listx[i]
        dytotal+=dy
        dxtotal+=dx
        dp_list.append(np.sqrt(dy**2+dx**2))
        angle = np.arctan2(dy,dx)
        angle_list.append(angle)
    
    for i in range(0,len(angle_list)-1):
        d_angle = np.abs(angle_list[i+1]-angle_list[i])
        if d_angle > np.pi:
            d_angle = 2*np.pi-d_angle  
        delta_angle_list.append(d_angle)

    return dp_list,np.mean(delta_angle_list),np.arctan2(dytotal,dxtotal)


def sort_points(unordered_pointsx, unordered_pointsy):
    min_dist = 10000
    for j in range(2):
        for k in range(2, 4):
            dist = np.sqrt((unordered_pointsx[k] - unordered_pointsx[j]) ** 2 + (unordered_pointsy[k] - unordered_pointsy[j])**2)
            if dist < min_dist:
                min_dist = dist
                correct_pair = [j, k]
    x0 = unordered_pointsx.pop(correct_pair[0])
    y0 = unordered_pointsy.pop(correct_pair[0])
    unordered_pointsx.insert(1, x0)
    unordered_pointsy.insert(1, y0)
    x1 = unordered_pointsx.pop(correct_pair[1])
    y1 = unordered_pointsy.pop(correct_pair[1])
    unordered_pointsx.insert(2, x1)
    unordered_pointsy.insert(2, y1)

    for i in range(3,len(unordered_pointsx)-2,2): 
        d1=np.sqrt((unordered_pointsx[i+1]-unordered_pointsx[i])**2+(unordered_pointsy[i+1]-unordered_pointsy[i])**2)
        d2=np.sqrt((unordered_pointsx[i+2]-unordered_pointsx[i])**2+(unordered_pointsy[i+2]-unordered_pointsy[i])**2)
        if d2<d1:  # if d2 is smaller than d1 swap the coordinates
            x0 = unordered_pointsx.pop(i+1)
            y0 = unordered_pointsy.pop(i+1)
            unordered_pointsx.insert(i+2, x0)
            unordered_pointsy.insert(i+2, y0)
    return unordered_pointsx,unordered_pointsy
  

def train(dust_dictionary,variable_switches,images,streak):
    """When dust in all frames has been sorted and characterised, this function connects dust particles across frames, forming a trajectory"""

    def create_tempimage(frame, numberedorder, x, y):
        sizes = np.shape(images[frame])
        height = float(sizes[0])
        width = float(sizes[1])
        fig = plt.figure(frameon=False)
        fig.set_size_inches(width / height, 1, forward=False)
        ax = plt.Axes(fig, [0., 0., 1., 1.])
        ax.set_axis_off()
        fig.add_axes(ax)
        ax.imshow(images[frame])
        plt.scatter(x, y,s=1)
        fig.savefig("temp" + str(numberedorder), dpi=height)
        plt.clf()
        plt.cla()
        plt.close()

    def nearest_point_to_mouse(mousex, mousey, dustx, dusty):
        dist_min = 1000000
        for i in range(len(dustx)):
            dist = np.sqrt((mousex - dustx[i]) ** 2 + (mousey - dusty[i]) ** 2)  
            if dist < dist_min:
                dist_min = dist
                index_min = i
        return index_min

    def onMouse0(event, x, y, flags, param):
        global x0, y0, index0
        if event == cv2.EVENT_LBUTTONDOWN:
            index0 = nearest_point_to_mouse(x,y,dust_dictionary[frame_0]["x0s"],dust_dictionary[frame_0]["y0s"])
            x0=dust_dictionary[frame_0]["x0s"][index0]
            y0=dust_dictionary[frame_0]["y0s"][index0]

    def onMouse1(event, x, y, flags, param):
        global x1, y1,index1
        if event == cv2.EVENT_LBUTTONDOWN:
            index1 = nearest_point_to_mouse(x,y,dust_dictionary[frame_1]["x0s"],dust_dictionary[frame_1]["y0s"])
            x1=dust_dictionary[frame_1]["x0s"][index1]
            y1=dust_dictionary[frame_1]["y0s"][index1]

    def onMouse2(event, x, y, flags, param):
        global x2, y2, index2
        if event == cv2.EVENT_LBUTTONDOWN:
            index2 = nearest_point_to_mouse(x,y,dust_dictionary[frame_2]["x0s"],dust_dictionary[frame_2]["y0s"])
            x2=dust_dictionary[frame_2]["x0s"][index2]
            y2=dust_dictionary[frame_2]["y0s"][index2]

    training = {}  # define an empty dictionary of machine learning parameters
    for variable in variable_switches:
        training[variable]=[]
    training["identifier"] = []

    stopping=False
    for frame in range(len(dust_dictionary) - 2):
        
        frame_0 = frame
        frame_1 = frame + 1
        frame_2 = frame + 2
        nextframe=False

        global x0,y0,x1,y1,x2,y2,index0,index1,index2
        x0,y0,x1,y1,x2,y2,index0,index1,index2=0,0,0,0,0,0,0,0,0

        while nextframe == False:

            create_tempimage(frame_0,0,x0,y0)
            create_tempimage(frame_1, 1, x1, y1)
            create_tempimage(frame_2, 2, x2, y2)  

                
            img0 = cv2.imread('temp0.png')
            img1 = cv2.imread('temp1.png')
            img2 = cv2.imread('temp2.png')
                
            while True:  # displays current and next frame until a key is pressed
                cv2.imshow('img0', img0)
                cv2.imshow('img1', img1)
                cv2.imshow('img2', img2)
                cv2.setMouseCallback("img0", onMouse0)
                cv2.setMouseCallback("img1", onMouse1)
                cv2.setMouseCallback("img2", onMouse2)

                key = cv2.waitKey(33)

                if key ==27:  # esc key
                    stopping = True
                    nextframe=True
                    break
                if key ==13:  # enter
                    break
                elif key == 110:
                    nextframe = True
                    break
                elif key == 115:  # S
                    def append_variables(index0,index1,index2):
                        trackx = []
                        tracky = []
                        trackw = []
                        trackb = []

                        # append unordered positions and widths of particle i frame 0 to track lists
                        trackx.append(dust_dictionary[frame_0]["x0s"][index0])
                        tracky.append(dust_dictionary[frame_0]["y0s"][index0])
                        trackw.append(dust_dictionary[frame_0]["widths"][index0])
                        trackb.append(dust_dictionary[frame_0]["brightness"][index0])
                        if streak == True:
                            trackx.append(dust_dictionary[frame_0]["x1s"][index0])
                            tracky.append(dust_dictionary[frame_0]["y1s"][index0])
                            trackw.append(dust_dictionary[frame_0]["widths"][index0])
                            trackb.append(dust_dictionary[frame_0]["brightness"][index0])

                        # append unordered positions and widths of particle j frame 1 to track lists
    
                        trackx.append(dust_dictionary[frame_1]["x0s"][index1])
                        tracky.append(dust_dictionary[frame_1]["y0s"][index1])
                        trackw.append(dust_dictionary[frame_1]["widths"][index1])
                        trackb.append(dust_dictionary[frame_1]["brightness"][index1])
                        if streak == True:
                            trackx.append(dust_dictionary[frame_1]["x1s"][index1])
                            tracky.append(dust_dictionary[frame_1]["y1s"][index1])
                            trackw.append(dust_dictionary[frame_1]["widths"][index1])
                            trackb.append(dust_dictionary[frame_1]["brightness"][index1])

                        # append unordered positions and widths of particle j frame 1 to track lists
    
                        trackx.append(dust_dictionary[frame_2]["x0s"][index2])
                        tracky.append(dust_dictionary[frame_2]["y0s"][index2])
                        trackw.append(dust_dictionary[frame_2]["widths"][index2])
                        trackb.append(dust_dictionary[frame_2]["brightness"][index2])
                        if streak == True:
                            trackx.append(dust_dictionary[frame_2]["x1s"][index2])
                            tracky.append(dust_dictionary[frame_2]["y1s"][index2])
                            trackw.append(dust_dictionary[frame_2]["widths"][index2])
                            trackb.append(dust_dictionary[frame_2]["brightness"][index2])
                            
                        if streak == True:
                            trackx, tracky = sort_points(trackx, tracky)  
                        track_dist, mean_delta_theta, mean_theta = find_dp_dtheta_avtheta(trackx, tracky)
                        mean_delta_position = np.mean(track_dist)
                        sigma_delta_position = np.std(track_dist)
                        mean_delta_width = np.std(trackw)
                        mean_delta_brightness = np.std(trackb)
                        try:
                            training["sigma_delta_position"].append(sigma_delta_position)
                        except:
                            pass
                        try:
                            training["mean_delta_position"].append(mean_delta_position)
                        except:
                            pass
                        try:
                            training["mean_delta_theta"].append(mean_delta_theta)
                        except:
                            pass
                        try:
                            training["mean_delta_width"].append(mean_delta_width)
                        except:
                            pass
                        try:
                            training["mean_theta"].append(mean_theta)
                        except:
                            pass
                        try:
                            training["mean_delta_brightness"].append(mean_delta_brightness)
                        except:
                            pass
                    append_variables(index0=index0,index1=index1,index2=index2)
                    training["identifier"].append(1)

                    for j in range(len(dust_dictionary[frame_1]["y1s"])):
                        for k in range(len(dust_dictionary[frame_2]["y1s"])):
                            if j == index1 and k==index2: # ensure that the track saved as yes is not also counted as a no
                                continue
                            append_variables(index0=index0, index1=j, index2=k)
                            training["identifier"].append(0)
                    break
                elif key == 110:  # N
                    nextframe=True
                    break
        if stopping==True:
            break
    return training


def track(dust_dictionary,variable_switches,features,labels,streak,threshold_probability,split_switch):
    """When dust in all frames has been sorted and characterised, this function connects dust particles across frames, forming a trajectory"""
    clf = GaussianNB()
    clf = clf.fit(features, labels)

    trackxtotal = [[]]
    trackytotal = [[]]
    trackwtotal=[[]]
    trackbtotal=[[]]
    track_lastframe=[[]] #defines the last frame where a dust grain in a given track was recorded
    
    for frame in range(len(dust_dictionary) - 2):
        print(frame)
        frame_0 = frame
        frame_1 = frame + 1
        frame_2 = frame + 2
        
        for i in range(len(dust_dictionary[frame_0]["x0s"])):  # cycle through every dust grain combination across three frames
            prob0=threshold_probability
            splitting_prob = threshold_probability
            splitting = False
            probs=[]
            trackxfinal=[]
            trackyfinal=[]
            trackwfinal=[]
            trackbfinal=[]
            trackframefinal=[]
            for j in range(len(dust_dictionary[frame_1]["x0s"])):
                for k in range(len(dust_dictionary[frame_2]["x0s"])):
                
                    trackx = []
                    tracky = []
                    trackw = []
                    trackb=[]

                    # append unordered positions and widths of particle i frame 0 to track lists
                    trackx.append(dust_dictionary[frame_0]["x0s"][i])
                    tracky.append(dust_dictionary[frame_0]["y0s"][i])
                    trackw.append(dust_dictionary[frame_0]["widths"][i])
                    trackb.append(dust_dictionary[frame_0]["brightness"][i])
                    if streak == True:
                        trackx.append(dust_dictionary[frame_0]["x1s"][i])
                        tracky.append(dust_dictionary[frame_0]["y1s"][i])
                        trackw.append(dust_dictionary[frame_0]["widths"][i])
                        trackb.append(dust_dictionary[frame_0]["brightness"][i])

                    # append unordered positions and widths of particle j frame 1 to track lists

                    trackx.append(dust_dictionary[frame_1]["x0s"][j])
                    tracky.append(dust_dictionary[frame_1]["y0s"][j])
                    trackw.append(dust_dictionary[frame_1]["widths"][j])
                    trackb.append(dust_dictionary[frame_1]["brightness"][j])
                    if streak == True:
                        trackx.append(dust_dictionary[frame_1]["x1s"][j])
                        tracky.append(dust_dictionary[frame_1]["y1s"][j])
                        trackw.append(dust_dictionary[frame_1]["widths"][j])
                        trackb.append(dust_dictionary[frame_1]["brightness"][j])

                    # append unordered positions and widths of particle k frame 2 to track lists

                    trackx.append(dust_dictionary[frame_2]["x0s"][k])
                    tracky.append(dust_dictionary[frame_2]["y0s"][k])
                    trackw.append(dust_dictionary[frame_2]["widths"][k])
                    trackb.append(dust_dictionary[frame_2]["brightness"][k])
                    if streak == True:
                        trackx.append(dust_dictionary[frame_2]["x1s"][k])
                        tracky.append(dust_dictionary[frame_2]["y1s"][k])
                        trackw.append(dust_dictionary[frame_2]["widths"][k])
                        trackb.append(dust_dictionary[frame_2]["brightness"][k])

                    if streak == True:
                        trackx, tracky = sort_points(trackx, tracky)
                    track_dist, mean_delta_theta, mean_theta = find_dp_dtheta_avtheta(trackx, tracky)  
                    mean_delta_position = np.mean(track_dist)
                    sigma_delta_position = np.std(track_dist)
                    mean_delta_width = np.std(trackw)
                    mean_delta_brightness = np.std(trackb)
                    predicting_features = []
                    if variable_switches["sigma_delta_position"]==True:
                        predicting_features.append(sigma_delta_position)
                    if variable_switches["mean_delta_position"]==True:
                        predicting_features.append(mean_delta_position)
                    if variable_switches["mean_delta_theta"] == True:
                        predicting_features.append(mean_delta_theta)
                    if variable_switches["mean_delta_width"]==True:
                        predicting_features.append(mean_delta_width)
                    if variable_switches["mean_delta_brightness"]==True:
                        predicting_features.append(mean_delta_brightness)
                    if variable_switches["mean_theta"]==True:
                        predicting_features.append(mean_theta)
                    prob1 = clf.predict_proba([predicting_features])[0][1]
                    probs.append(prob1)
                    if split_switch == True:
                        for l in range(len(probs)-1):
                            if abs(probs[-1]-probs[l])<=0.01 and probs[-1]>=splitting_prob and probs[l]>=splitting_prob:
                                splitting = True
                                trackxfinal.append(trackx)
                                trackyfinal.append(tracky)
                                trackwfinal.append(trackw)
                                trackbfinal.append(trackb)
                                splitting_prob = prob1
                                if streak == True:
                                    trackframefinal.append([frame_0,frame_0,frame_1,frame_1,frame_2,frame_2])
                                else:
                                    trackframefinal.append([frame_0, frame_1, frame_2])
                                break
                    else:    
                        if prob1 > prob0 and splitting == False:
                            prob0 = prob1
                            
                            trackxfinal=[trackx]
                            trackyfinal=[tracky]
                            trackwfinal=[trackw]
                            trackbfinal=[trackb]
                            if streak == True:
                                trackframefinal=[[frame_0,frame_0,frame_1,frame_1,frame_2,frame_2]]
                            else:
                                trackframefinal = [[frame_0, frame_1, frame_2]]
        
            if prob0 > threshold_probability:
                contained=False
                for y in range(len(trackxtotal)):
                    for z in range(len(trackxtotal[y])):
                        if (trackxtotal[y][z]==trackxfinal[0][0] and trackytotal[y][z] == trackyfinal[0][0]) and track_lastframe[y][z] == trackframefinal[0][0]:
                            contained = [y,z]                                                           
                if contained == False:  
                    if splitting == True:
                        for t in range(len(trackxfinal)):
                            trackxtotal.append(trackxfinal[t])
                            trackytotal.append(trackyfinal[t])
                            trackwtotal.append(trackwfinal[t])
                            track_lastframe.append(trackframefinal[t])
                            trackbtotal.append(trackbfinal[t])
                        
                    else:
                        trackxtotal.append(trackxfinal[0])
                        trackytotal.append(trackyfinal[0])
                        trackwtotal.append(trackwfinal[0])
                        trackbtotal.append(trackbfinal[0])
                        track_lastframe.append(trackframefinal[0])
                        
                    
                else: #if current total track has overextended slightly, cut it short
                    trackxtotal[contained[0]] = trackxtotal[contained[0]][0:contained[1]]
                    trackytotal[contained[0]] = trackytotal[contained[0]][0:contained[1]]
                    trackwtotal[contained[0]] = trackwtotal[contained[0]][0:contained[1]]
                    trackbtotal[contained[0]] = trackbtotal[contained[0]][0:contained[1]]
                    track_lastframe[contained[0]] = track_lastframe[contained[0]][0:contained[1]]
                    
                    if splitting == True:
                        for t in range(1,len(trackxfinal)):
                            trackxtotal.insert(contained[0]+t,list(trackxtotal[contained[0]]))
                            trackytotal.insert(contained[0]+t,list(trackytotal[contained[0]]))
                            trackwtotal.insert(contained[0]+t,list(trackwtotal[contained[0]]))
                            trackbtotal.insert(contained[0]+t,list(trackbtotal[contained[0]]))
                            track_lastframe.insert(contained[0]+t,list(track_lastframe[contained[0]]))
                        for t in range(len(trackxfinal)):
                            trackxtotal[contained[0]+t].extend(trackxfinal[t])
                            trackytotal[contained[0]+t].extend(trackyfinal[t])
                            trackwtotal[contained[0]+t].extend(trackwfinal[t])
                            trackbtotal[contained[0]+t].extend(trackbfinal[t])
                            track_lastframe[contained[0]+t].extend(trackframefinal[t])
                    
                    else:
                        trackxtotal[contained[0]].extend(trackxfinal[0])
                        trackytotal[contained[0]].extend(trackyfinal[0])
                        trackwtotal[contained[0]].extend(trackwfinal[0])
                        trackbtotal[contained[0]].extend(trackbfinal[0])
                        track_lastframe[contained[0]].extend(trackframefinal[0])
                        
    trackxtotal.pop(0)   # to get rid of the empty list at the start of trackxtotal
    trackytotal.pop(0)
    trackwtotal.pop(0)
    trackbtotal.pop(0)
    track_lastframe.pop(0)
    return (trackxtotal,trackytotal,trackbtotal,track_lastframe)