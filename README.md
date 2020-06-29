RIDAT - Robust Impurity Detector and Tracker

For use in detecting dust particles and other plasma impurities. 
Outputs a list of CSVs, each one detailing the consecutive positions and other properties of a dust grain.

RIDAT uses the following external libraries, which may need to be installed manually (depending on OS and environment):
- csv
- numpy
- os
- matplotlib
- json


Code will look for a folder containing consecutive images in the directory InputData/folder(variable)/type(variable).
Running the RunFile can either train or track. Instructions for each are as follows


TRAINING

Training will open three consecutive processed frames from the image directory you are working in. Click to select three conecutive positions of the same dust grain, and press enter to see the closest dust grain to where you clicked. If these three positions are truley consecitve positions of the same grain, press 's' to save this data. When all available tracks for a given 3 frames are selected, press 'n' to move onto the next set of 3. When  you are finished collecting data, press 'esc' to save the training data in InputData/TrainingData/Type(variable).


Tracking

Running the tracking function will generate a multiple 2d arrays containing dust track properties. The first axis of a given array corresponds to the track number, the second axis corresponds to the dust grain number inside a track. To export these tracks, use the output_tracks function.
