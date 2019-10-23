import torch.nn as nn
from torch.nn import functional as F
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np 
import torch.optim as optim
import torch 
from torch.autograd import Variable
from torch.utils.data import TensorDataset, DataLoader
import gc

pos_encoding = {84.0: 1, 85.0: 2, 86.0: 3, 87.0: 4, 89.0: 5, 90.0: 6, 91.0: 7, 92.0: 8, 93.0: 9, 
				94.0: 10, 95.0: 11, 96.0: 12, 97.0: 13, 99.0: 14, 100.0: 15, 101.0: 16, 03.0: 17, np.nan: 18}


class Lstm(nn.Module):
	def __init__(self, vocab_size, output_size, embedding_dim, hidden_dim, n_layers, is_cuda, drop_prob=0.5):
		super(Lstm, self).__init__()
		# Variables
		self.device = torch.device("cuda" if is_cuda else "cpu")		
		self.is_cuda = is_cuda
		self.output_size = output_size
		self.n_layers = n_layers
		self.hidden_dim = hidden_dim
		self.batch_size = 50
		# Layers
		self.embedding = nn.Embedding(vocab_size,embedding_dim)
		self.lstm = nn.LSTM(embedding_dim, hidden_dim, n_layers, batch_first=True)
		self.lin1 = nn.Linear(hidden_dim, hidden_dim) 		
		self.lin2 = nn.Linear(hidden_dim, output_size)

	# return the initial hidden state of the lstm layer (hidden state and cell state) 	
	def init_hidden(self, batch_size):
		if self.is_cuda:
			print("Returning cuda hidden state") 
			return (torch.randn(self.n_layers, batch_size, self.hidden_dim).cuda(), torch.randn(self.n_layers, batch_size, self.hidden_dim).cuda())
		else:
			return (torch.randn(self.n_layers, batch_size, self.hidden_dim), torch.randn(self.n_layers, batch_size, self.hidden_dim))

	# propagate features through netword
	def forward(self, x, hidden):	
		x = x.long()
		
		if self.is_cuda: 
			x = x.to(self.device)
		 
		embeds = self.embedding(x) 		
		out, hidden = self.lstm(embeds, hidden)
		# take only the last hidden_dim output of the lstm 
		out = self.lin1(out[:,-1,:]) 
		out = self.lin2(out)
		return out, hidden


# loads the needed datasets and builds feature and target vectors
def load_datasets(list_of_datasets): 
	print("loading datasets") 
	datasets = []
	data = []
	target = []

	# load all tweet datasets and merge them into one
	for f in list_of_datasets:
		print("...", f)
		datasets.append(pd.read_csv("../" + f + "_clean.csv"))
		data.append(pd.read_csv("../" + f + "_pos.csv"))		
	dataset = pd.concat(datasets, axis=0, ignore_index=True, sort=False)		
	data = pd.concat(data, axis=0, ignore_index=True, sort=False)
	data = data.to_numpy()	

	# build target vector by one-hot-encoding the affects 
	for index, row in dataset.iterrows():
		if row["affect"] == 0:
			target.append([1, 0, 0, 0])
		elif row["affect"] == 1:
			target.append([0, 1, 0, 0])
		elif row["affect"] == 2:
			target.append([0, 0, 1, 0])
		elif row["affect"] == 3:
			target.append([0, 0, 0, 1]) 
	target = pd.DataFrame(target) 	
	target = target.to_numpy()
	target = target.astype(np.float64)
	
	# split data into test and training data and return
	train_x, test_x, train_y, test_y = train_test_split(data, target, test_size=0.3)
	return Variable(torch.from_numpy(train_x)), Variable(torch.from_numpy(test_x)), Variable(torch.from_numpy(train_y)), Variable(torch.from_numpy(test_y))

def train(epochs):
	hidden = lstm.init_hidden(batch_size)
	sum_loss = 0
	for e in range(epochs): 
		for i, (inputs, targets) in enumerate(train_loader): 
			hidden = tuple([item.data for item in hidden])
			lstm.zero_grad()
			output, hidden = lstm(inputs, hidden)
			if(is_cuda):
				ouput = output.to("cuda")
				targets = targets.to("cuda")
			loss = criterion(output.squeeze(), targets.float())
			
			if i % print_every == 0:
				print("epoch", e, "\tbatch", i, "/", len(train_loader)) 
				print("--- loss", loss) 
			# retain_graph ist nötig, weil er sonst nach jedem durchgang im batch die rückpropagation löscht
			loss.backward(retain_graph=True)
			sum_loss += loss.data
			# prevents exploding gradient in lstm 
			nn.utils.clip_grad_norm_(lstm.parameters(), clip)
			optimizer.step()
			gc.collect()
		# save network after every epoch
		torch.save(lstm.state_dict(), "lstm_tweet.pt")
	runs = epochs * len(train_loader) 
	print("Average train loss: ", (sum_loss/runs)) 

def test(): 
	hidden = lstm.init_hidden(batch_size)
	sum_loss = 0
	for i, (inputs, targets) in enumerate(test_loader): 
		hidden = tuple([item.data for item in hidden])
		output, hidden = lstm(inputs, hidden)
		if(is_cuda):
			ouput = output.to("cuda")
			targets = targets.to("cuda")
		loss = criterion(output.squeeze(), targets.float())
		
		if i % print_every == 0:
			print("batch", i, "/", len(test_loader)) 
			print("--- loss", loss) 
		# retain_graph ist nötig, weil er sonst nach jedem durchgang im batch die rückpropagation löscht
		sum_loss += loss.data
		gc.collect()
		
	runs = len(test_loader) 
	print("Average test loss: ", (sum_loss/runs)) 

### script part
train_x, test_x, train_y, test_y = load_datasets(["crowdflower", "emoint", "tec"])

# create Tensor datasets
batch_size = 50
train_data = TensorDataset(train_x, train_y)
test_data = TensorDataset(test_x, test_y)
# drop_last: Löscht den letzten batch wenn der nicht die batch-größe hat. Sonst stürzt das netz ab
train_loader = DataLoader(train_data, shuffle=True, batch_size=batch_size, drop_last=True,)
test_loader = DataLoader(test_data, shuffle=True, batch_size=batch_size,  drop_last=True,)
# init important variables
num_layers = 1
vocab_size = 18 + 1
embedding_dim = 85 
hidden_dim = 256
output_dim = 4
batch_size = 50
lr=0.001
clip = 5
is_cuda = torch.cuda.is_available()
epochs = 10
print_every = 100
# init net
lstm = Lstm(vocab_size, output_dim, embedding_dim, hidden_dim, num_layers, is_cuda, drop_prob=0.5)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(lstm.parameters(), lr=lr)

print("GPU: ", torch.cuda.is_available())
if torch.cuda.is_available(): 
	print("Moving net to gpu")
	lstm = lstm.cuda()
	train_x = train_x.cuda()
	train_y = train_y.cuda()
	test_x = test_x.cuda()
	test_y = test_y.cuda()
	print("... Moved net to gpu")

train(epochs) 
test()

# den ganzen kram noch auf cuda umstellen
# auf dem tower trainieren lassen 








