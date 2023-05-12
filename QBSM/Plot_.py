import csv
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.pyplot import figure
import matplotlib.animation as animation

plt.rcParams.update({'figure.max_open_warning': 0})

# Format plot
fig, (ax0, ax1, ax2) = plt.subplots(3, 1, figsize = (9, 6.75))

def init():

	ax0.clear()
	ax1.clear()
	ax2.clear()

def Read_Data():

	Data = []

	# filename = "D:/上課資料/IME/實驗室研究/Paper/Coverage Control/Quality based switch mode/Data/Data_0.csv"
	filename = "/home/leo/mts/src/QBSM/Data/Cost"
	files = Path(filename).glob("*.csv")

	for file in files:

		Cost, time_ = [], []
		data = []

		with open(file, "r", encoding='UTF8', newline='') as f:

			reader = csv.reader(f)
			
			for mem in reader:
				
				Cost.append(float(mem[0]))
				time_.append(float(mem[1]))

			data = [Cost, time_]

		Data.append(data)

	return Data

def Execute_read():

	Data = Read_Data()

	x_0 = Data[0][1]
	y_0 = Data[0][0]
	x_1 = Data[1][1]
	y_1 = Data[1][0]
	x_2 = Data[2][1]
	y_2 = Data[2][0]

	init()

	ax0.plot(x_0, y_0, color = "#005AB5")
	ax0.set_title('Cost of Agent 0 (Red)', fontdict = {'fontsize': 20})
	ax0.set_xlabel("Time (s)", fontdict = {'fontsize': 20})
	ax0.set_ylabel("Cost", fontdict = {'fontsize': 20})
	ax0.tick_params(axis='both', which='major', labelsize = 20)

	ax1.plot(x_1, y_1, color = "#FF9224")
	ax1.set_title('Cost of Agent 1 (Green)', fontdict = {'fontsize': 20})
	ax1.set_xlabel("Time (s)", fontdict = {'fontsize': 20})
	ax1.set_ylabel("Cost", fontdict = {'fontsize': 20})
	ax1.tick_params(axis='both', which='major', labelsize = 20)

	ax2.plot(x_2, y_2, color = "#007500")
	ax2.set_title('Cost of Agent 2 (Blue)', fontdict = {'fontsize': 20})
	ax2.set_xlabel("Time (s)", fontdict = {'fontsize': 20})
	ax2.set_ylabel("Cost", fontdict = {'fontsize': 20})
	ax2.tick_params(axis='both', which='major', labelsize = 20)

	plt.subplots_adjust(left = 0.1, bottom = 0.1, right = 0.8, top = 0.9, wspace = 0, hspace = 0.45)

	# figure(figsize = (10, 7.5), dpi = 80)
	# parameters = {'legend.fontsize': 15, "xtick.labelsize": 25, "ytick.labelsize": 25,
	# 				"axes.labelsize": 30, "axes.titlesize": 20}
	# plt.rcParams.update(parameters)

	# plt.cla()
	# plt.grid()
	# plt.tight_layout()
	# plt.plot(x, y, linewidth = 1.5, linestyle = '-', label = 'Cost')
	
	# plt.title('Cost Function of Agent 0')
	# plt.xlabel('Time (s)')
	# plt.ylabel('Cost')

	# plt.legend()
	plt.show()

def PlotData():

	Data = []
	count = 0

	filename = "/home/leo/mts/src/QBSM/Data/"
	files = Path(filename).glob("*.csv")

	for file in files:

		data_ = []
		C1, C2, C3, time_ = [], [], [], []

		with open(file, "r", encoding='UTF8', newline='') as f:

			reader = csv.reader(f)
		
			for mem in reader:
				
				C1.append(float(mem[0]))
				C2.append(float(mem[1]))
				C3.append(float(mem[2]))
				time_.append(float(mem[3]))

			if count == 0:

				data_ = [C1, C2, C3, time_]
			elif count == 1:

				data_ = [C1, C2, C3, time_]
			elif count == 2:

				data_ = [C1, C2, C3, time_]

		count += 1

		Data.append(data_)

	return Data

def animate(i):

	Data = readData()

	for (mem, i) in zip(Data, range(len(Data))):

		globals()["x_" + str(i)] = np.ndarray.tolist(np.arange(0, np.shape(mem)[1], 1))
		globals()["y_" + str(i) + str(1)] = mem[0]
		globals()["y_" + str(i) + str(2)] = mem[1]
		globals()["y_" + str(i) + str(3)] = mem[2]

	init()

	if y_01 == []:

		lables_1 = "Cost Function 1, " + "Value: " + str(0)
		lables_2 = "Cost Function 2, " + "Value: " + str(0)
		lables_3 = "Cost Function 3, " + "Value: " + str(0)
		ax0.plot([], [], color = "#005AB5", label = lables_1)
		ax0.plot([], [], color = "#FF9224", label = lables_2)
		ax0.plot([], [], color = "#007500", label = lables_3)
		ax0.legend(loc = "upper right", bbox_to_anchor = (1.25, 0.7), ncol = 1)
	elif (y_01[-1] >= 0) and (y_02[-1] >= 0) and (y_03[-1] >= 0):

		lables_1 = "Cost Function 1, " + "Value: " + str(np.round(y_01[-1], 3))
		lables_2 = "Cost Function 2, " + "Value: " + str(np.round(y_02[-1], 3))
		lables_3 = "Cost Function 3, " + "Value: " + str(np.round(y_03[-1], 3))
		ax0.plot(x_0, y_01, color = "#005AB5", label = lables_1)
		ax0.plot(x_0, y_02, color = "#FF9224", label = lables_2)
		ax0.plot(x_0, y_03, color = "#007500", label = lables_3)
		ax0.legend(loc = "upper right", bbox_to_anchor = (1.25, 0.7), ncol = 1)

	ax0.set_title('Cost Finction of Agent 0 (Red)')
	ax0.set_xlabel("Instant")
	ax0.set_ylabel("Cost")
	ax0.set_ylim((0, 20))

	if y_11 == []:

		lables_1 = "Cost Function 1, " + "Value: " + str(0)
		lables_2 = "Cost Function 2, " + "Value: " + str(0)
		lables_3 = "Cost Function 3, " + "Value: " + str(0)
		ax1.plot([], [], color = "#005AB5", label = lables_1)
		ax1.plot([], [], color = "#FF9224", label = lables_2)
		ax1.plot([], [], color = "#007500", label = lables_3)
		ax1.legend(loc = "upper right", bbox_to_anchor = (1.25, 0.7), ncol = 1)
	elif (y_11[-1] >= 0) and (y_12[-1] >= 0) and (y_13[-1] >= 0):

		lables_1 = "Cost Function 1, " + "Value: " + str(np.round(y_11[-1], 3))
		lables_2 = "Cost Function 2, " + "Value: " + str(np.round(y_12[-1], 3))
		lables_3 = "Cost Function 3, " + "Value: " + str(np.round(y_13[-1], 3))
		ax1.plot(x_1, y_11, color = "#005AB5", label = lables_1)
		ax1.plot(x_1, y_12, color = "#FF9224", label = lables_2)
		ax1.plot(x_1, y_13, color = "#007500", label = lables_3)
		ax1.legend(loc = "upper right", bbox_to_anchor = (1.25, 0.7), ncol = 1)

	ax1.set_title('Cost Finction of Agent 1 (Green)')
	ax1.set_xlabel("Instant")
	ax1.set_ylabel("Cost")
	ax1.set_ylim((0, 20))

	if y_21 == []:

		lables_1 = "Cost Function 1, " + "Value: " + str(0)
		lables_2 = "Cost Function 2, " + "Value: " + str(0)
		lables_3 = "Cost Function 3, " + "Value: " + str(0)
		ax2.plot([], [], color = "#005AB5", label = lables_1)
		ax2.plot([], [], color = "#FF9224", label = lables_2)
		ax2.plot([], [], color = "#007500", label = lables_3)
		ax2.legend(loc = "upper right", bbox_to_anchor = (1.25, 0.7), ncol = 1)
	elif (y_21[-1] >= 0) and (y_22[-1] >= 0) and (y_23[-1] >= 0):

		lables_1 = "Cost Function 1, " + "Value: " + str(np.round(y_21[-1], 3))
		lables_2 = "Cost Function 2, " + "Value: " + str(np.round(y_22[-1], 3))
		lables_3 = "Cost Function 3, " + "Value: " + str(np.round(y_23[-1], 3))
		ax2.plot(x_2, y_21, color = "#005AB5", label = lables_1)
		ax2.plot(x_2, y_22, color = "#FF9224", label = lables_2)
		ax2.plot(x_2, y_23, color = "#007500", label = lables_3)
		ax2.legend(loc = "upper right", bbox_to_anchor = (1.25, 0.7), ncol = 1)

	ax2.set_title('Cost Finction of Agent 2 (Blue)')
	ax2.set_xlabel("Instant")
	ax2.set_ylabel("Cost")
	ax2.set_ylim((0, 20))

	# plt.tight_layout()
	plt.subplots_adjust(left = 0.1, bottom = 0.1, right = 0.8, top = 0.9, wspace = 0, hspace = 0.45)
	parameters = {'legend.fontsize': 12, "xtick.labelsize": 12, "ytick.labelsize": 12,
					"axes.labelsize": 15, "axes.titlesize": 15}
	plt.rcParams.update(parameters)

if __name__ == "__main__":

	# ani = animation.FuncAnimation(plt.gcf(), animate, cache_frame_data = False,\
	# 								init_func = init, interval = 30, blit = False)
	# plt.show()

	Execute_read()