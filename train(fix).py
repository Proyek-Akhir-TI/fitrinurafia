import numpy as np
import os
from skimage.feature import greycomatrix,greycoprops
import pandas as pd
import cv2
from skimage.measure import label,regionprops
import skimage

proList = ['contrast', 'dissimilarity', 'homogeneity', 'energy','correlation']#fitur glcm
featlist = ['Contrast', 'Dissimilarity', 'Homogeneity', 'Energy','Correlation','Kelas'] # #header CSV
properties =np.zeros(5)
glcmMatrix = []
final = []
folders = ["gajah oling","kopi pecah","lainnya"]
# folders = ["gajah oling","kopi"]
for folder in folders: 
    print(folder)
    labell=folders.index(folder)
    INPUT_SCAN_FOLDER="C:/Users/fitri nur afia/Pictures/Batik/FIX/data training/"+folder+"/"

    image_folder_list = os.listdir(INPUT_SCAN_FOLDER) 

    for i in range(len(image_folder_list)):

        abc =cv2.imread(INPUT_SCAN_FOLDER+image_folder_list[i])

        gray_image = cv2.cvtColor(abc, cv2.COLOR_BGR2GRAY)

        print(image_folder_list[i])

        glcmMatrix = (greycomatrix(gray_image, [1], [0], levels=2 ** 8))
        for j in range(0, len(proList)):
            properties[j] = (greycoprops(glcmMatrix, prop=proList[j]))

        features = np.array(
            [properties[0], properties[1], properties[2], properties[3], properties[4], labell])
        final.append(features)

df = pd.DataFrame(final, columns=featlist)
filepath =  "Training.csv"
df.to_csv(filepath)
