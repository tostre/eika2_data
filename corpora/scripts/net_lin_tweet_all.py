import pandas
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim
import pandas as pd
import os
from sklearn.model_selection import train_test_split


class Lin_Net(nn.Module):

	def __init__(self):
		# input = 8, output = 1
		super(Lin_Net, self).__init__()
		self.lin1 = nn.Linear(8, 8)
		self.lin2 = nn.Linear(8, 4)

	def forward(self, x):
		x = F.relu(self.lin1(x))
		x = self.lin2(x)
		return x

	def num_flat_features(self, x):
		self.size = x.size()[1:]
		self.num = 1
		for i in self.size:
			self.num *= i
		return self.num


def train(train_x, train_y):
	for i in range(100):
		# make network-input from csv-data		
		input = torch.Tensor(train_x)
		print(input.shape)
		# create target tensor
		target = Variable(torch.Tensor(train_y))		
		# calc output given the input
		output = lin_net(input)
		
		# define how the error is calculated (mean squared error)
		criterion = nn.MSELoss()
		loss = criterion(output, target)
		print(i, loss)
		# set loss from previous round back to zero
		lin_net.zero_grad()
		# propagate error back through the network
		loss.backward()
		optimizer = optim.SGD(lin_net.parameters(), lr=0.05)
		optimizer.step()

	# save network
	torch.save(lin_net.state_dict(), "net_lin_tweet_all.pt")


def test(test_x, test_y):
	input = torch.Tensor(test_x)
	target = Variable(torch.Tensor(test_y))
	output = lin_net(input)

	# define how the error is calculated (mean squared error)
	criterion = nn.MSELoss()
	loss = criterion(output, target)
	print("test loss", loss)


def make_data(list_of_datasets): 
	print("generating datasets") 
	datasets = []
	target_vec = []
	data = []
	train_x, train_y, test_x, test_y = [], [], [], []

	# load datasets
	for f in list_of_datasets:
		print("... loading", f)
		datasets.append(pd.read_csv("../" + f))
	dataset = pd.concat(datasets, axis=0, ignore_index=True)
	
	# split data into train and test set
	print("... splitting data")
	train_data, test_data = train_test_split(dataset, test_size=0.3)
	
	# build train set
	print("... seperating target and feature vecs") 	
	for index, row in train_data.iterrows():
		if row["affect"] == 0:
			train_y.append([1, 0, 0, 0])
		elif row["affect"] == 1:
			train_y.append([0, 1, 0, 0])
		elif row["affect"] == 2:
			train_y.append([0, 0, 1, 0])
		elif row["affect"] == 3:
			train_y.append([0, 0, 0, 1]) 
		train_x.append([row["word_count"], row["upper_word_count"], row["ent_word_count"], row["h_count"], row["s_count"], row["a_count"], row["f_count"], row["cons_punct_count"]])


	# build test set	
	for index, row in test_data.iterrows():
		if row["affect"] == 0:
			test_y.append([1, 0, 0, 0])
		elif row["affect"] == 1:
			test_y.append([0, 1, 0, 0])
		elif row["affect"] == 2:
			test_y.append([0, 0, 1, 0])
		elif row["affect"] == 3:
			test_y.append([0, 0, 0, 1]) 
		test_x.append([row["word_count"], row["upper_word_count"], row["ent_word_count"], row["h_count"], row["s_count"], row["a_count"], row["f_count"], row["cons_punct_count"]])

	return Variable(torch.Tensor(train_x)), Variable(torch.Tensor(test_x)), Variable(torch.Tensor(train_y)), Variable(torch.Tensor(test_y))



# load datasats
train_x, test_x, train_y, test_y = make_data(["crowdflower_clean.csv", "emoint_clean.csv", "tec_clean.csv"])
lin_net = Lin_Net()
train(train_x, train_y)
test(test_x, test_y)






