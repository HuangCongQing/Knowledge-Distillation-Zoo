from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
import torch
import torch.nn as nn
import torch.nn.functional as F


class SoftTarget(nn.Module):
	'''
	Distilling the Knowledge in a Neural Network
	https://arxiv.org/pdf/1503.02531.pdf
	'''
	def __init__(self, T):
		super(SoftTarget, self).__init__()
		self.T = T # 蒸馏系数

	def forward(self, out_s, out_t):
    	# KL散度loss
		loss = F.kl_div(F.log_softmax(out_s/self.T, dim=1), # student输出
						F.softmax(out_t/self.T, dim=1), # teacher输出
						reduction='batchmean') * self.T * self.T

		return loss