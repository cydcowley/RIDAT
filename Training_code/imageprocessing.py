import matplotlib
import matplotlib.image as m
import matplotlib.pyplot as plt
import os
import numpy as np
import imageio as io
import os
import cv2



def rgb2gray(rgb):
    r, g, b = rgb[:,:,0], rgb[:,:,1], rgb[:,:,2]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b

    return gray

def import_images(folder):
    images = []
    a=os.listdir(folder)  # listdir returns a list of the entries in the folder
    for image in a[1:]:
        img = m.imread(os.path.join(folder,image))  # imread reads an image from a file into an array
        if 'numpy' in str(type(img[0][0])): #checks whether image is rgba or gray
            img = rgb2gray(img)
        images.append(img)
    return(images)


def find_bg(images):
    """Function that returns the average of all the images"""
    bg = np.zeros_like(images[0])
    try:
        for image in images:
            bg += image
        return(bg/len(images))
    except: #pictures could be an empty list, we don't want an error in that case. we just don't do anything to it then
        return 0



def variable_bg(images,bgres):
    """Function that returns a list of the backgrounds for each image - now each image has a
    unique background given by the average of a range of images either side of the chosen image"""
    backgrounds=[]
    for image in range(len(images)):
        bg = np.zeros_like(images[0])
        if image < bgres:
            for i in images[0:(2*bgres)+1]:
                bg += i
            bg = bg/((2*bgres)+1)
            backgrounds.append(bg)

        elif image > len(images) - (bgres+1):
            for i in images[-1-(2*bgres):]:
                bg += i
            bg = bg/((2*bgres)+1)
            backgrounds.append(bg)

        else:
            for i in images[image-bgres:image+bgres+1]:
                bg += i
            bg = bg/((2*bgres)+1)
            backgrounds.append(bg)

    return backgrounds


def find_dust(images,background,threshold,activeframe):
    """Function that sets dust images to brightness 1, and stores the positions in dust_positions array"""
    dust_positions=[]
    if type(background) == list:
        bgsubtracted_image = images[activeframe]-background[activeframe]
    else:
        bgsubtracted_image = images[activeframe]-background

    for i in range(len(bgsubtracted_image)):
        for j in range(len(bgsubtracted_image[0])):
            if bgsubtracted_image[i][j]>=threshold:
                bgsubtracted_image[i][j]=1.0
                dust_positions.append([i,j])
            else:
                bgsubtracted_image[i][j]=0.0
    return([dust_positions,bgsubtracted_image])

def collect_dust(pixels):
    """Function that lumps dust pixels into dust grains, by checking if the bright pixels neighbor other bright pixels"""

    dust_grains = []
    dust_grains.append([pixels[0]])
    pixels.pop(0)
    while(len(pixels)>=1):
        contained = False
        for i in range(len(dust_grains)):
            for j in range(len(dust_grains[i])):
                if(np.absolute(pixels[0][0]-dust_grains[i][j][0])<=1 and np.absolute(pixels[0][1]-dust_grains[i][j][1])<=1):
                    contained = i
        if type(contained)==int:
            dust_grains[contained].append(pixels[0])
        else:
            dust_grains.append([pixels[0]])
        pixels.pop(0)
    keep_dustgrains=[]

    for grain in (dust_grains):
        if len(grain)>=2:
            keep_dustgrains.append(grain)
    return(keep_dustgrains)



def characterise_dust(pixels):
    """Funciton that takes pixel locations of each dust grain and outpus entire dust grain position and dimensions"""
    dust_this_frame={"x0s":[],"y0s":[],"x1s":[],"y1s":[],"widths":[],
                     "lengths":[],"pixels":[],"name":"undefined"}
    dust_lengths=len(pixels)*[0] # set placeholder positions and dimensions
    dust_widths= len(pixels)*[0]
    dust_x0s = len(pixels)*[0]
    dust_y0s =len(pixels)*[0]
    dust_x1s = len(pixels) * [0]
    dust_y1s = len(pixels) * [0]

    for i in range(len(pixels)):
        for j in range(len(pixels[i])):
            for k in range(j,len(pixels[i])):
                r2= (pixels[i][j][0]-pixels[i][k][0])**2
                + (pixels[i][j][1] - pixels[i][k][1])**2
                if dust_lengths[i] <= np.sqrt(r2):
                    dust_lengths[i] = np.sqrt(r2)
                    dust_x0s[i]= pixels[i][j][0]
                    dust_x1s[i]= pixels[i][k][0]
                    dust_y0s[i]= pixels[i][j][1]
                    dust_y1s[i]= pixels[i][k][1]

    for i in range(len(dust_lengths)):
        dust_widths[i] = len(pixels[i])/(dust_lengths[i]+1)

    # for i in range(len(pixels)):
    #     av_x=0
    #     av_y=0
    #     for j in range(len(pixels[i])):
    #         av_x+=pixels[i][j][0]
    #         av_y+=pixels[i][j][1]
    #     av_x = av_x/len(pixels[i])
    #     av_y = av_y/len(pixels[i])
    #     dust_xpositions[i] = av_x
    #     dust_ypositions[i] = av_y

    dust_this_frame["pixels"]=pixels
    dust_this_frame["x0s"]=dust_x0s
    dust_this_frame["x1s"] = dust_x1s
    dust_this_frame["y0s"]=dust_y0s
    dust_this_frame["y1s"] = dust_y1s
    dust_this_frame["widths"]=dust_widths
    dust_this_frame["lengths"]=dust_lengths
    return(dust_this_frame)

def iterate_frames(images,thresh):
    dust_every_frame=len(images)*[0]
    bgsub=[]
    bg = variable_bg(images,2)
    for i in range(len(images)):
        current_frame={"x0s":[],"y0s":[],"x1s":[],"y1s":[],"widths":[],
                     "lengths":[],"pixels":[]}
        [positions, bgsub_image] = find_dust(images=images,background=bg,threshold=thresh,activeframe=i)
        bgsub.append(bgsub_image)
        current_frame["pixels"]=collect_dust(positions)
        current_frame = characterise_dust(current_frame["pixels"])
        dust_every_frame[i] = current_frame

    return dust_every_frame,bgsub

def make_gif(self,image_list):
    images = []
    for filename in image_list:
        images.append(io.imread("plots/"+filename))
    io.mimsave('surface1.gif', images, duration=0.5)
            
            
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        

   