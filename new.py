from sklearn.datasets import load_wine
from sklearn import preprocessing

import numpy as np
import random, math
import copy

#############################################################################################################
#                                        Tempat Meletakan Fungsi                                            #
#############################################################################################################
def checkGrad(sol):
	#Menghitung turunan dW/dMSE, menggunakkan Central Difference
	der = []
	h = 0.00001
	for i in range(solutionSize):
		solOld1 = copy.deepcopy(sol)
		solOld2 = copy.deepcopy(sol)
		solOld1[i] = solOld1[i]+h
		solOld2[i] = solOld2[i]-h
		der.append((testing(solOld1, wine) + testing(solOld2, wine))/2*h)
	a = sum(der)/solutionSize
	print("%.16f" % a)
#	print(float(sum(der)/solutionSize))

def circleMap(lama):
	xbaru = lama + 1.2 - ((0.5/2*math.pi)*math.sin(2*math.pi*lama)%1)
	return xbaru

def gauss_map(x):
    if x == 0:
        return 0
    else:
        xbaru = ((1/x)%1)
        return xbaru

def deriv(x):
	h = 1.00000001
	h_= 0.00000001
	d = (testing(x*h, wine) + testing(x*h, wine))/2*h_
	return d

def Newton(Flower, i, curr_best, V):
	alpha1, alpha2 = np.random.uniform(0,1), np.random.uniform(0, 1)

	left = Flower[i]
	right = Flower[curr_best]

	A1 = left + alpha1 * V
	A2 = right + alpha2 * V
	print(A1)
	print(A2)

#	MSE1 = testing(A1, wine)
#	MSE2 = testing(A2, wine)

#	alpha1_new = alpha1 - (MSE1/deriv(A1))
#	alpha2_new = alpha2 - (MSE1/deriv(A2))
#	MSE1_new = testing((left + alpha1_new * V), wine)
#	MSE2_new = testing((right + alpha2_new * V), wine)

#	grad1 =  (MSE1_new - MSE1)/(alpha1_new - alpha1)
#	grad2 =  (MSE2_new - MSE2)/(alpha2_new - alpha2)

#	C1 = MSE1 - (grad1 * alpha1)
#	C2 = MSE2 - (grad2 * alpha2)

#	alpha_baru = (C2-C1)/(grad1-grad2)
#	MSE_baru = testing((x + alpha_baru * V), wine)

#	alpha_new = alpha_baru - (MSE_baru/MSE_baru_diff)
#	MSE_baru2 = testing((x + alpha_new * V), wine)
#	grad_baru = (MSE_baru2 - MSE_baru)/(alpha_new-alpha_baru)

#	if grad_baru < 0:
#s		alpha1 = alpha_baru
#	elif grad_baru > 0:
#		alpha2 = alpha_baru

#	return alpha_baru

def Levy():
    lambd = 1.5

    sigma = ((math.gamma(1+lambd) * math.sin(math.pi*lambd/2))/(math.gamma((1+lambd)/2*lambd*(2**((lambd-1)/2)))))**1/lambd
    U = np.random.normal(0, sigma)
    V = np.random.normal(0, 1)

    s = U / (abs(V))**1/lambd
    return s

def initClass(jumlah_kelas):
	kelas = np.zeros(jumlah_kelas)
	low = 0
	high = 1/jumlah_kelas
	for place in range(jumlah_kelas):
		kelas[place] = low + high
		low = kelas[place]
		kelas[place] = kelas[place] - high/2
	return kelas

def activate(value):
	return 1 / (1 + math.exp(-1 * value))

def neuralNet(x, data_input):
	# Inisialisasi weight di input ==> hidden
	bot = 0
	hidden = [] 
	for i in range(n_hidden):
		top = bot+kolom
		hidden.append(x[bot:top])
		bot = bot + kolom
	# Inisialisasi weight di hidden ==> output 
	out = x[bot:]

	hidden_value = []
	for j in range(n_hidden):
		hidden_value.append(np.dot(data_input, hidden[j]))
	output_val = activate(np.dot(hidden_value, out))

	return output_val

def hitungMSE(kelas, nilai):
    temp = []
    for n in range(baris):
    	if n<=58:
    		temp.append((kelas[0]-nilai[n])**2)
    	elif n>58 and n<=129:
    		temp.append((kelas[1]-nilai[n])**2)
    	else:
    		temp.append((kelas[2]-nilai[n])**2)
    mse = sum(temp)/baris
    return mse

def FPA(Flower, best, mseBest):
	iterasi = 0
	p = 0.8
	threshold = 0.00000001
	diff = 1
	delta = []
	test = np.zeros(pop_size)
	curr_best = best
	initChaos = 0
	terbaik = Flower[best]
	FlowerNew = np.zeros(shape=(2*pop_size, solutionSize))
	testScore = np.zeros(2*pop_size)

	while diff>threshold:
		iterasi += 1
		print(iterasi)
		for i in range(pop_size):
			rand = 0.8
			FlowerNew[i] = Flower[i]
			if rand < p:
#				V = Flower[curr_best] - Flower[i]
#				FlowerNew = Flower[i] + Newton(Flower, i, curr_best, V) * V
#				c = circleMap(initChaos)
#				initChaos = c
				FlowerNew[i+10] = GlobalPollination(Flower[i], terbaik)
			else:
				tetangga = [np.random.randint(0, pop_size), np.random.randint(0, pop_size)]
				FlowerNew[i+10] = LocalPollination(Flower[i], Flower[tetangga[0]], Flower[tetangga[1]])
			testScore[i] = testing(FlowerNew[i], wine)
			testScore[i+10] = testing(FlowerNew[i+10], wine)

		selected = np.argsort(testScore)
		for j in range(pop_size):
			Flower[j] = FlowerNew[selected[j]]
		terbaik = Flower[0]
		delta.append(testScore[selected[0]])
		if len(delta)<=5:
			pass
		else:
			m = len(delta)
			deltaOld = sum(delta[(m-6):(m-1)])/5
			deltaNow = sum(delta[(m-5):])/5
			diff = deltaOld - deltaNow
			print(diff)

#		diff-=1
	print("%.16f" % testScore[0])
	return Flower[0]

def GlobalPollination(lama, star):
	L = Levy()

	while L>1 or L<-1:
		L = Levy()

	baru = lama + L * (lama - star)
	return baru

def LocalPollination(lama, x1, x2):
	baru = lama + np.random.uniform(0,1) * (x1 - x2)
	return baru

def testing(weight, wine):
	skor = []
	for l in range(baris):
		skor.append(neuralNet(weight, wine[l]))

	error = hitungMSE(kelas, skor)
	
	return error

#############################################################################################################
#                                                                                                           #
#############################################################################################################


#############################################################################################################
#                                        Tempat Define Parameter Awal                                       #
#############################################################################################################

# Meload dataset
data = load_wine()
min_max_scaler = preprocessing.MinMaxScaler()

wine = min_max_scaler.fit_transform(data.data)
baris, kolom = np.shape(wine)

# Neural Network Parameter
n_hidden = 5

# FPA Parameter
pop_size = 10

# Inisialisasi kelas
kelas = initClass(3)

#############################################################################################################
#                                                                                                           #
#############################################################################################################

# Inisialisasi Solusi pada FPA
solutionSize = kolom * n_hidden + n_hidden 
Flower = np.zeros(shape=(pop_size, solutionSize))
for i in range(pop_size):
	for j in range(solutionSize):
		Flower[i][j] = np.random.uniform(-1, 1)

# Mencari g* awal
skorAwal = np.zeros(shape=(pop_size, baris))
mseAwal = []
for pop in range(pop_size):
	for dat in range(baris):
		skorAwal[pop][dat] = neuralNet(Flower[pop], wine[dat])
	mseAwal.append(hitungMSE(kelas, skorAwal[pop]))

best = np.argmin(mseAwal)

Optimized = FPA(Flower, best, mseAwal[best])
checkGrad(Optimized)
