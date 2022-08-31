import os

names = {
	"A",
	"A_34k",
	"B",
	"B_34k",
	"B_270k",
	"C",
	"C_34k",
	"C_270k",
	"D",
	"D_34k",
	"D_270k",
	"CD_2k",
	"CD_34k",
	"CD_270k"}

for name in names:
	command = "py -3.7 C:/Users/C201_ALES_NTB/source/repos/HGR_CNN/HGR_CNN/HGR_CNN.py evaluate_model " + name + ".h5"
	print(command)
	os.system(command)