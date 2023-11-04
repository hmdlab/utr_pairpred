import torch
import torch.nn as nn
from troch.utils.data import Dataset

class PairDataset(Dataset):
	def __init__(self,data):
		super().__init__()
		self.data=data

	def __getitem__(self):
		pass

	def __len__(self):
		return len(self.data)
