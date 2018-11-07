import matplotlib.image as m
import matplotlib.pyplot as plt
import os
import numpy as np



def import_images(folder):
    images = []
    a=os.listdir(folder)  # listdir returns a list of the entries in the folder
    for image in a[1:]:    
        img = m.imread(os.path.join(folder,image))  # imread reads an image from a file into an array
        images.append(img)
    return(images)


class Images:
    def __init__(self,images):
        self.images=images # list of all the pixel data of every image
        self.bg = np.zeros_like(self.images[0]) #create a blank slate for background
        self.activeimage=0 # sets the current image worked on for find_dust, collect_dust, and characterise_dust
        self.dust_this_frame={"xpositions":[],"ypositions":[],"widths":[], "lengths":[],"pixels":[],"name":"undefined"}
        self.dust_every_frame=len(self.images)*[0]
        
    def find_bg(self,bgres):
        """Function that returns a list of the backgrounds for each image - now each image has a 
        unique background given by the average of a range of images either side of the chosen image"""
        self.backgrounds=[]
        for image in range(len(self.images)):
            self.bg = np.zeros_like(self.images[0])
            if image < bgres:
                for i in self.images[0:(2*bgres)+1]:
                    self.bg += i
                self.bg = self.bg/((2*bgres)+1)
                self.backgrounds.append(self.bg)
                
            elif image > len(self.images) - (bgres+1):
                for i in self.images[-1-(2*bgres):]:
                        self.bg += i
                self.bg = self.bg/((2*bgres)+1)
                self.backgrounds.append(self.bg)
                
            else:
                for i in self.images[image-bgres:image+bgres+1]:
                    self.bg += i
                self.bg = self.bg/((2*bgres)+1)
                self.backgrounds.append(self.bg)
        
    def find_dust(self,threshold):
        """Function that sets dust images to brightness 1, and stores the positions in dust_positions array"""
        dust_positions=[]
        bgsubtracted_image = self.images[self.activeimage]-self.backgrounds[self.activeimage]
        for i in range(len(bgsubtracted_image)):
            for j in range(len(bgsubtracted_image[0])):
                if bgsubtracted_image[i][j]>=threshold:
                    bgsubtracted_image[i][j]=1.0
                    dust_positions.append([i,j])
                else:
                    bgsubtracted_image[i][j]=0.0
        return([dust_positions,bgsubtracted_image])
        
    def collect_dust(self,positions):
        """Function that lumps dust pixels into dust grains, by checking if the bright pixels neighbour other bright pixels"""
        
        dust_grains = []
        dust_grains.append([positions[0]])
        positions.pop(0)
        
        while(len(positions)>=1):
            contained = False
            for i in range(len(dust_grains)):
                for j in range(len(dust_grains[i])):
                    if(np.absolute(positions[0][0]-dust_grains[i][j][0])<=1 and np.absolute(positions[0][1]-dust_grains[i][j][1])<=1):
                        contained = i
            if type(contained)==int:
                dust_grains[contained].append(positions[0])
            else:
                dust_grains.append([positions[0]])
            positions.pop(0)
        
        keep_dustgrains=[]
        
        for grain in (dust_grains):
            if len(grain)>=2:
                keep_dustgrains.append(grain)
        self.dust_this_frame["pixels"]=keep_dustgrains
        
        

    def characterise_dust(self):
        """Funciton that takes pixel locations of each dust grain and outputs entire dust grain position and dimensions"""
        
        dust_lengths=len(self.dust_this_frame["pixels"])*[0] # set placeholder positions and dimensions
        dust_widths= len(self.dust_this_frame["pixels"])*[0]
        dust_xpositions = len(self.dust_this_frame["pixels"])*[0]
        dust_ypositions =len(self.dust_this_frame["pixels"])*[0]
        
        for i in range(len(self.dust_this_frame["pixels"])):
            for j in range(len( self.dust_this_frame["pixels"][i])):
                for k in range(j,len(self.dust_this_frame["pixels"][i])):
                    r2= (self.dust_this_frame["pixels"][i][j][0]-self.dust_this_frame["pixels"][i][k][0])**2 
                    + (self.dust_this_frame["pixels"][i][j][1]- self.dust_this_frame["pixels"][i][k][1])**2
                    if (dust_lengths[i] <= np.sqrt(r2)):
                        dust_lengths[i]=np.sqrt(r2)
      
        for i in range(len(dust_lengths)):
            dust_widths[i]=len(self.dust_this_frame["pixels"][i])/dust_lengths[i]
            
        for i in range(len(self.dust_this_frame["pixels"])):
            av_x=0
            av_y=0
            for j in range(len(self.dust_this_frame["pixels"][i])):
                av_x+=self.dust_this_frame["pixels"][i][j][0]
                av_y+=self.dust_this_frame["pixels"][i][j][1]
            av_x=av_x/len(self.dust_this_frame["pixels"][i])
            av_y=av_y/len(self.dust_this_frame["pixels"][i])
            dust_xpositions[i]=av_x
            dust_ypositions[i]=av_y
                
        self.dust_this_frame["xpositions"]= dust_xpositions
        self.dust_this_frame["ypositions"]=dust_ypositions
        self.dust_this_frame["widths"]= dust_widths
        self.dust_this_frame["lengths"]=dust_lengths
        
        self.dust_every_frame[self.activeimage]=self.dust_this_frame
        
    def connect_frames(self,distance_cutoff=20,width_cutoff=10):
        """When dust in all frames has been sorted and characterised, this function connects dust particles across frames, forming a trajectory"""
        
        for i in self.dust_every_frame: #check to ensure characterisation has been performed for every frame
            if i==0:
                return 0
        
        for i in range(len(self.images)-1):
            current_frame=i
            next_frame=i+1
            pairing_probabilities=np.zeros(shape=(len(self.dust_every_frame[current_frame]["xpositions"]),len(self.dust_every_frame[next_frame]["xpositions"])))
            print(pairing_probabilities)
            
            

        


im = import_images("Images and Videos")

set_1 = Images(im)
set_1.find_bg()

for i in range(len(im)):
    set_1.activeimage=i
    [pos1,bgsub1]=set_1.find_dust(0.09)
    set_1.collect_dust(pos1)
    set_1.characterise_dust()

set_1.connect_frames()


#for i in range(len(im)):
#    no_background=(im[i]-background(im))    
#    po = detect_dust(no_background,0.09)
#    collect_dust(po)
#    plt.imsave('Bgsubtracted/normal'+str(i)+'.png',im[i],cmap='gray')
#    plt.imsave('Bgsubtracted/bgsubtracted'+str(i)+'.png',no_background,cmap='gray')