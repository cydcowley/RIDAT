from sklearn import tree
import imageprocessing as dd
import matplotlib.pyplot as plt

im = dd.import_images("KL11-P1DB-92708-3")

set_1 = dd.Images(im)
set_1.find_bg()
dd.iterate_frames(set_1)
set_1.train()


features = [[2200,4,1000], [1500,1,1000], [1800,1,1000], [900,2,0.4], [1000,2,0.4]]
labels = ['SUV', 'SUV', 'SUV', 'hatchback', 'hatchback']
clf = tree.DecisionTreeClassifier()
clf = clf.fit(features, labels)

print(clf.predict([[1350, 1,100]]))