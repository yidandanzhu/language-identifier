import numpy as np
import sys
import math
import random
import matplotlib.pyplot as plt

reload(sys)  
sys.setdefaultencoding('utf-8')

# Part One: Perdict the probable language of each one-hot embeding in the sentence. 
# Use the highest voted language by one-hot embedings to predict the language of the sentence.

train_dataset = sys.argv[1]
dev_dataset = sys.argv[2]
test_dataset = sys.argv[3]

# Load datasets
with open(train_dataset) as f:
	data = f.read().decode('iso-8859-1').encode("latin1").split('\n')
data = [x for x in map(str.strip, data) if len(x)>0 and ' ' in x]

with open(dev_dataset) as f:
	dev_data = f.read().decode('iso-8859-1').encode("latin1").split('\n')
dev_data = [x for x in map(str.strip, dev_data) if len(x)>0 and ' ' in x]

uniq_ls = {}
num = 0
hidden = 100
learn_rate = 0.1

for i in [x.split(' ',1)[1] for x in data]:
	for j in i:
		if j not in uniq_ls.keys():
			uniq_ls[j]=num
			num+=1
len_uniq = len(uniq_ls)

# Create labels
def label(x):
	if x == 'ENGLISH':
		return [1,0,0]
	elif x == 'ITALIAN':
		return [0,1,0]
	elif x == 'FRENCH':
		return [0,0,1]

# One hot embeding
def one_hot_data(data):
	c_sent = []
	sent_lab = {}
	for s in range(len(data)):
		st = data[s].split(' ',1)[1]
		lab = data[s].split(' ',1)[0]
		sent_lab[s] = label(lab)
		for i in range(len(st)-4):
			seg = st[i:i+5]
			c_sent.append([seg,s])
	return c_sent,sent_lab

#sigmoid function
def sigmoid(x):
	return 1.0/(1+np.exp(-x))

#softmax function
def softmax(x):
	return_ls =[]
	sumup = np.sum([np.exp(i) for i in x])
	for i in range(len(x)):
		return_ls.append(np.exp(x[i])/sumup)
	return return_ls

#calculate the accuracy of the classifier
training data and the accuracy of the classifier on the dev data
def accuracy(data,w_1,b_1,w_2,b_2):
	c_sent,sent_lab = one_hot_data(data)
	accur = 0
	total = 0
	for item in sent_lab:
		c_label= sent_lab[item]
		c_all =[k for k,v in c_sent if v == item]
		sent_soft = [0,0,0]
		for c in c_all:
			curr_sent = np.zeros((5,len_uniq))
			for i in range(len(c)):
				try:
					curr_sent[i,:][uniq_ls[c[i]]] =1
				except:
					pass
			softm = NeuroNet_1(curr_sent,w_1,b_1,w_2,b_2)[0]
			softm = softm.tolist()
			max_value = max(softm)
			max_index = softm.index(max_value)
			sent_soft[max_index]+=1

		max_value = max(sent_soft)
		max_index = sent_soft.index(max_value)
		if c_label[max_index]==1:
			accur +=1
			total+=1
		else:
			total+=1
	accuracy = float(accur)/float(total)
	print accuracy
	return accuracy

# forward propagation
def NeuroNet_1(inp,weight_1,bias_1,weight_2,bias_2):
	input_flat = inp.flatten()
	#linear 1
	h_1=weight_1.dot(input_flat)+bias_1
	#sigmoid
	h_2 = sigmoid(h_1)
	y = weight_2.dot(np.transpose(h_2))+bias_2
	#softmax
	soft_y = softmax(np.transpose(y))[0]
	return soft_y,h_2,input_flat

# derivative of softmax
def softmax_back(y,y_h):
	gred_loss = (y-y_h)
	return_ls = []
	for j in range(len(y)):
		sof = 0
		for i in range(len(y)):
			if i!=j:
				sof += gred_loss[i]*y[i]*(0-y[j])
			else:
				sof+= gred_loss[i]*y[i]*(1-y[j])
		return_ls.append(sof)
	return return_ls

# backward propagation
def NeuroNet_2(c,label, weight_1,bias_1,weight_2,bias_2,hidden):
	#label is the correct tag of the sentence
	y,h2,x_1 = NeuroNet_1(c,weight_1,bias_1,weight_2,bias_2)
	y_loss = np.square(y - label)/2
	gred_loss = y-label
	#
	gred_softmax = np.array(softmax_back(y,label))
	#
	gred_w2 = np.array(gred_softmax)*np.transpose(h2)
	#print gred_loss
	gred_b2 = gred_softmax
	gred_h = np.transpose(weight_2).dot(np.array(gred_softmax))
	#
	gred_sigmoid = gred_h*((np.ones(1*hidden)-h2)*h2)

	gred_w1 = np.transpose(gred_sigmoid)*(x_1)

	gred_b1 = gred_sigmoid
	gred_h1 = weight_1*np.transpose(gred_sigmoid)
	return gred_w2,gred_b2,gred_w1,gred_b1


epoch_accur_train = []
epoch_l_train = []
epoch_accur_dev = []
epoch_l_dev = []

# train model
def train_NN(dev_data, data,hidden,learn_rate):
	weight_1 = np.random.rand(hidden,5*len_uniq)
	bias_1 = np.random.rand(1,hidden)
	#print bias_1.shape
	weight_2 = np.random.rand(3,hidden)
	bias_2 = np.random.rand(3,1)
	#accuracy of initial
	print 'random weight:'
	print 'train accuracy:'
	ac_train = accuracy(data,weight_1,bias_1,weight_2,bias_2)
	epoch_accur_train.append(ac_train)
	epoch_l_train.append(0)
	print 'dev accuracy:'
	ac_dev = accuracy(dev_data,weight_1,bias_1,weight_2,bias_2)
	epoch_accur_dev.append(ac_dev)
	epoch_l_dev.append(0)
	for i in range(3):
		print 'epoch: ',i+1
		#data = random_shuffle(data)
		# np.random.shuffle(data)
		one_hot,lab = one_hot_data(data)
		np.random.shuffle(one_hot)
		#each sentence
		for item in one_hot:
			c_label= lab[item[1]]
			c_cont = item[0]
			curr_sent = np.zeros((5,len_uniq))
			for index_i in range(len(c_cont)):
				curr_sent[index_i,:][uniq_ls[c_cont[index_i]]] =1
			sent=j
			g_w2,g_b2,g_w1,g_b1 = NeuroNet_2(curr_sent,c_label,weight_1,bias_1,weight_2,bias_2,hidden)
			#print gred_w1
			weight_2 = weight_2-np.transpose(learn_rate*g_w2)
			weight_1 = weight_1-learn_rate*g_w1
			bias_1 = bias_1-learn_rate*g_b1
			bias_2 = np.transpose(np.transpose(bias_2)-learn_rate*g_b2)
		#after each epoch, run accuracy
		print 'train accuracy:'
		ac_train=accuracy(data,weight_1,bias_1,weight_2,bias_2)
		epoch_accur_train.append(ac_train)
		epoch_l_train.append(i+1)
		print 'dev accuracy:'
		ac_dev=accuracy(dev_data, weight_1,bias_1,weight_2,bias_2)
		epoch_accur_dev.append(ac_dev)
		epoch_l_dev.append(i+1)
	return weight_1,bias_1,weight_2,bias_2,ac_dev


#plot accuracy
w_1,b_1,w_2,b_2,dev_acc = train_NN(dev_data, data,hidden,learn_rate)  
print epoch_accur_train
print epoch_l_train
print epoch_accur_dev
print epoch_l_dev
X,Y = [epoch_l_train,epoch_accur_train]
line_1 = plt.plot(X,Y,label='train',color = 'red')
X,Y = [epoch_l_dev,epoch_accur_dev]
line_2 = plt.plot(X,Y,label='dev',color = 'blue')
plt.legend()
plt.savefig('accuracy.png')

#Accuracy of the language identifier: 0.85


#load test file
with open(test_dataset) as f:
	test_data = f.read().decode('iso-8859-1').encode("latin1").split('\n')
test_data = [x for x in map(str.strip, test_data) if len(x)>0]

#predict label for test dataset
def test_label(data,w_1,b_1,w_2,b_2): 
	output_l = []
	c_data = []
	for s in range(len(data)):
		c_sent = []
		st = data[s]
		for i in range(len(st)-4):
			seg = st[i:i+5]
			c_sent.append(seg)
		c_data.append(c_sent)

	for i_d in range(len(c_data)):
		item_s = [0,0,0]
		sent_soft = [0,0,0]
		for c in c_data[i_d]:
			curr_sent = np.zeros((5,len_uniq))
			for i in range(len(c)):
				try:
					curr_sent[i,:][uniq_ls[c[i]]] =1
				except:
					pass
			softm = NeuroNet_1(curr_sent,w_1,b_1,w_2,b_2)[0]
			softm = softm.tolist()
			max_value = max(softm)
			max_index = softm.index(max_value)
			sent_soft[max_index]+=1
		max_value = max(sent_soft)
		max_index = sent_soft.index(max_value)
		item_s[max_index] = 1
		if item_s == [1,0,0]:
			item_l = 'ENGLISH'
		elif item_s  == [0,1,0]:
			item_l = 'ITALIAN'
		elif item_s == [0,0,1]:
			item_l = 'FRENCH'
		output_l.append([i_d+1, item_l])
	return output_l


test_out =test_label(test_data,w_1,b_1,w_2,b_2)

with open('languageIdentification.data/test_solutions') as f:
	test_solu = (f.read().decode('iso-8859-1').encode("latin1").split('\n'))[:-1]

#calculate accuracy for test dataset
def test_accur(test_solu,test_out):
	corr=0
	total = 0
	for i in range(len(test_solu)):
		line = test_solu[i]
		lab = line.split(' ')[1]

		if test_out[i][1].lower() == lab.lower():
			corr +=1
			total+=1
		else:
			total+=1
	print 'test accuracy: ', float(corr)/float(total)

test_accur(test_solu,test_out)

#predict label for test dataset
with open('prediction.txt','w') as my_f:
	for i in test_out:
		my_f.write('{} {}\n'.format(i[0],i[1]))


# list of hyper parameter: [50,0.1],[50,0.05],[100,0.05],[120,0.05],[120,0.1]
hyper = [[50,0.1],[50,0.05],[100,0.05],[120,0.05],[120,0.1]]
def hyper_opt(hyper):
	best = []
	for n in range(len(hyper)):
		h = hyper[n]
		hidden = h[0]
		learn_rate = h[1]
		print 'hidden layer: ',hidden, ', learn_rate: ',learn_rate
		w_1,b_1,w_2,b_2,dev_acc = train_NN(dev_data, data,hidden,learn_rate)  
		best.append([dev_acc,w_1,b_1,w_2,b_2,n])
	select = sorted(best,key=lambda x:x[0],reverse = True)[0]
	print select
	test_out =test_label(test_data,select[1],select[2],select[3],select[4])
	acc = test_accur(test_solu,test_out)
hyper_opt(hyper)

#The best performing set of hyperparameters (as determined by the dev data) [50,0.1]

# Accuracy of your best language identifier (measured on the test data)
# test accuracy:  1.0 （train accuracy: 0.991121976325， dev accuracy: 0.992992992993）




