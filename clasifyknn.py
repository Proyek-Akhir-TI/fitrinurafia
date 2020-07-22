import csv
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
			print('0')
		elif hasil== 1:
			print('1')
		else :
			print('2')

		
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
	hasil_klasifikasi(baca_input_csv('testing.csv'), baca_input_csv('Training.csv', kondisi=True), 1) # Nilai parameter k =11 terbagus
