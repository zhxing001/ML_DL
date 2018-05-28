

import csv

with open(r'C:\\Users\zhxing\Desktop\competition\train.csv',encoding='utf-8') as csvfile:
	readCSV=csv.reader(csvfile)
	for row in readCSV:
		print(row)

