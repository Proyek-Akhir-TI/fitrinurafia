import numpy as np
import os
from skimage.feature import greycomatrix,greycoprops
import pandas as pd
import cv2
from skimage.measure import label,regionprops
import csv 
import matplotlib.pyplot as plt

INPUT_SCAN_FOLDER="C:/Users/fitri nur afia/Pictures/Batik/FIX/test/"
image_folder_list = os.listdir(INPUT_SCAN_FOLDER)
proList = ['contrast', 'dissimilarity', 'homogeneity', 'energy','correlation']
featlist = ['Contrast', 'Dissimilarity', 'Homogeneity', 'Energy','Correlation','Kelas'] #header CSV
properties =np.zeros(5)
glcmMatrix = []
final = []  

for i in range(len(image_folder_list)):

        abc =cv2.imread(INPUT_SCAN_FOLDER+image_folder_list[i])

        gray_image = cv2.cvtColor(abc, cv2.COLOR_RGB2GRAY)

        # images = images.f.arr_0
        print(image_folder_list[i])

        glcmMatrix = (greycomatrix(gray_image, [1], [0], levels=2 ** 8))
        for j in range(0, len(proList)):
            properties[j] = (greycoprops(glcmMatrix, prop=proList[j]))

        features = np.array(
            [properties[0], properties[1], properties[2], properties[3], properties[4], "belum terklasifikasi"])
        final.append(features)

df = pd.DataFrame(final, columns=featlist)
filepath =  "testing.csv"
df.to_csv(filepath)
batik = pd.read_csv('testing.csv')
batik.plot.bar(y=['Contrast', 'Dissimilarity', 'Homogeneity', 'Energy','Correlation','Kelas']) #subplots=True, layout=(2,4))
plt.putText(image, train_labels[prediction], (20,30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,255,255), 3)
plt.show()


# Menghitung jarak perkiraan dari dua buah titik
def hitung_perkiraan(x, y):
	return abs(x['Contrast'] - y['Contrast'])+ (x['Dissimilarity'] - y['Dissimilarity'])+ (x['Homogeneity'] - y['Homogeneity'])+ (x['Energy'] - y['Energy'])+ (x['Correlation'] - y['Correlation'])


# Memprediksi data dari datasets
def prediksi_data(nilai, data, x):
	daftar_perkiraan = [{'hitung_perkiraan': float('inf')}]
	for dataset in data:
		hasil = hitung_perkiraan(nilai, dataset)
		if hasil < daftar_perkiraan[-1]['hitung_perkiraan']:
			if len(daftar_perkiraan) >= x:
				daftar_perkiraan.pop()
			i = 0
			while i < len(daftar_perkiraan)-1 and hasil >= daftar_perkiraan[i]['hitung_perkiraan']:
				i += 1
			daftar_perkiraan.insert(i, {'hitung_perkiraan': hasil, 'Kelas': dataset['Kelas']})
	daftar_nilai = list(map(lambda x: x['Kelas'], daftar_perkiraan))
	return max(daftar_nilai, key=daftar_nilai.count)



# Klasifikasi datatest berdasarkan data pada file DataTrain
def hasil_klasifikasi(data_test, data_train, k):
	for d_test in data_test:
		d_test['Kelas'] = prediksi_data(d_test, data_train, k)
		hasil = d_test['Kelas']
		if hasil == 0:
			imgplot = plt.imshow(abc)
			plt.title('Gajah Oling')
			plt.show()
		elif hasil== 1:
			imgplot = plt.imshow(abc)
			plt.title('Gajah Oling')
			plt.show()

		else :
			imgplot = plt.imshow(abc)
			plt.title('Gajah Oling')
			plt.show()


		
	#hasil_file_csv('Hasil.csv', map(lambda x: [x['Y']], data_test)) # Generate file csv

# Fungsi untuk membaca data dari file csv 
def baca_input_csv(f, kondisi=False):
	dataset = [] # buat array kosong untuk menampung nilai dari file csv yang dibaca
	with open(f) as csv_input:
		baca_csv = csv.DictReader(csv_input, skipinitialspace=True)
		for baris in baca_csv:
			dataset.append({'i': int(baris['']), 'Contrast': float(baris['Contrast']), 'Dissimilarity': float(baris['Dissimilarity']), 'Homogeneity': float(baris['Homogeneity']), 'Energy': float(baris['Energy']),'Correlation': float(baris['Correlation']),'Kelas': int(baris['Kelas'])if kondisi else baris['Kelas']}) 
	return dataset

# Main program untuk menjalankan fungsi yang sudah dibuat sebelumnya
if __name__ == '__main__':
	hasil_klasifikasi(baca_input_csv('testing.csv'), baca_input_csv('Training.csv', kondisi=True), 23) # Nilai parameter k =11 terbagus

