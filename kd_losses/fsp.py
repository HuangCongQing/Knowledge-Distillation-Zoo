from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
import torch
import torch.nn as nn
import torch.nn.functional as F

# 我不拟合大模型的输出，而是去拟合大模型层与层之间的关系，这才是我要转移和蒸馏的知识！这个关系是用层与层之间的内积来定义的
# https://www.yuque.com/huangzhongqing/lightweight/ofs894
class FSP(nn.Module):
	'''
	A Gift from Knowledge Distillation: Fast Optimization, Network Minimization and Transfer Learning
	http://openaccess.thecvf.com/content_cvpr_2017/papers/Yim_A_Gift_From_CVPR_2017_paper.pdf
	'''
	def __init__(self):
		super(FSP, self).__init__()

	# fm_s1, fm_s2之间得到层关系，然后子和父之间再做loss
	def forward(self, fm_s1, fm_s2, fm_t1, fm_t2):
		loss = F.mse_loss(self.fsp_matrix(fm_s1,fm_s2), self.fsp_matrix(fm_t1,fm_t2))

		return loss

	# 层与层之间的内积
	def fsp_matrix(self, fm1, fm2):
		if fm1.size(2) > fm2.size(2): # shape变成小的fm1:(128, 16, 32, 32)-->(128, 16, 16, 16)
			fm1 = F.adaptive_avg_pool2d(fm1, (fm2.size(2), fm2.size(3))) # >>>>输出固定维度 指定输出（H，W）

		# HW两维变成1维
		fm1 = fm1.view(fm1.size(0), fm1.size(1), -1) # (128,16,16*16)
		fm2 = fm2.view(fm2.size(0), fm2.size(1), -1).transpose(1,2) # (128,32,16*16)->(128,16*16,16)

		# 两个输入tensor维度是(b×n×m)和(b×m×p), https://www.yuque.com/huangzhongqing/pytorch/gck7a0#xwo2P
		# (128,16,256) * (128,256,32)-->(128,16,32)    里面所数值再除以256 (128,16,32)/256
		fsp = torch.bmm(fm1, fm2) / fm1.size(2) # bmm??? bmm是两个三维张量相乘, 两个输入tensor维度是(b×n×m)和(b×m×p), 第一维b代表batch size，输出为(b×n×p)

		return fsp
