import numpy as np
import pandas as pd

#%matplotlib inline
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

from matplotlib import interactive
interactive(True)

df = pd.read_csv("train.csv")

labels = df['label']
features = df.drop(['label'],axis=1)

# 7 = 6
# 2 = 22
label_1 = labels[22]
image_1 = features.iloc[22]

image_1 = np.reshape(image_1,(28,28))

rows, columns = np.shape(image_1)
filter = [[2,-1,-1],[-1,2,-1],[-1,-1,2]]

conv_op = np.zeros((rows-2)*(columns-2))

a=0

for i in range(rows - 2):
    
    for j in range(columns - 2):
        
        conv_op[a] = (image_1[i][j]*filter[0][0]) + (image_1[i][j+1]*filter[0][1]) + (image_1[i][j+2]*filter[0][2])+ \
            (image_1[i+1][j]*filter[1][0]) +  (image_1[i+1][j+1]*filter[1][1]) + (image_1[i+1][j+2]*filter[1][2]) + \
            (image_1[i+2][j+1]*filter[2][1]) + (image_1[i+2][j+1]*filter[2][1]) + (image_1[i+2][j+2]*filter[2][2])
        
        if conv_op[a] >= 2:
            conv_op[a] = 1
        else:
            conv_op[a] = 0
            
        a += 1
		
conv_op = np.reshape(conv_op,((rows-2),(columns-2)))

plt.imshow(image_1, cmap='gray')
    
plt.imshow(conv_op, cmap='gray')

plt.imshow(filter, cmap='gray')