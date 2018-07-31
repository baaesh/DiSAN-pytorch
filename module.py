import torch
import torch.nn as nn

import torch.nn.init as init


def masked_softmax(vec, mask, dim=1):
	masked_vec = vec * mask.float()
	max_vec = torch.max(masked_vec, dim=dim, keepdim=True)[0]
	exps = torch.exp(masked_vec - max_vec)
	masked_exps = exps * mask.float()
	masked_sums = masked_exps.sum(dim, keepdim=True)
	zeros = (masked_sums == 0)
	masked_sums += zeros.float()
	return masked_exps / (masked_sums + 1e-20)


def get_direct_mask_tile(direction, seq_len, device):
	mask = torch.FloatTensor(seq_len, seq_len).to(device)
	mask.data.fill_(1)
	if direction == 'fw':
		mask = torch.triu(mask, diagonal=1)
	elif direction == 'bw':
		mask = torch.tril(mask, diagonal=-1)
	else:
		raise NotImplementedError('only forward or backward mask is allowed!')
	mask.unsqueeze_(0)
	return mask


def get_rep_mask_tile(rep_mask, device):
	batch_size, seq_len, _ = rep_mask.size()
	mask = rep_mask.view(batch_size, 1, seq_len).expand(batch_size, seq_len, seq_len)

	return mask


class Source2Token(nn.Module):

	def __init__(self, d_h, dropout=0.2):
		super(Source2Token, self).__init__()

		self.d_h = d_h
		self.dropout_rate = dropout

		self.fc1 = nn.Linear(d_h, d_h)
		self.fc2 = nn.Linear(d_h, d_h)

		init.xavier_uniform_(self.fc1.weight.data)
		init.constant_(self.fc1.bias.data, 0)
		init.xavier_uniform_(self.fc2.weight.data)
		init.constant_(self.fc2.bias.data, 0)

		self.elu = nn.ELU()
		self.softmax = nn.Softmax(dim=-2)
		self.dropout = nn.Dropout(dropout)

	def forward(self, x, rep_mask):
		x = self.dropout(x)
		map1 = self.elu(self.fc1(x))
		map2 = self.fc2(self.dropout(map1))

		soft = masked_softmax(map2, rep_mask, dim=1)
		out = torch.sum(x * soft, dim=1)

		return out


class DiSA(nn.Module):

	def __init__(self, args, direction):
		super(DiSA, self).__init__()

		self.d_e = args.d_e
		self.d_h = args.d_h
		self.direction = direction
		self.dropout_rate = args.dropout
		self.device = args.device

		self.fc = nn.Linear(args.d_e, args.d_h)
		init.xavier_uniform_(self.fc.weight.data)
		init.constant_(self.fc.bias.data, 0)

		self.w_1 = nn.Linear(args.d_h, args.d_h)
		self.w_2 = nn.Linear(args.d_h, args.d_h)
		init.xavier_uniform_(self.w_1.weight)
		init.xavier_uniform_(self.w_2.weight)
		init.constant_(self.w_1.bias, 0)
		init.constant_(self.w_2.bias, 0)
		self.w_1.bias.requires_grad = False
		self.w_2.bias.requires_grad = False

		self.b_1 = nn.Parameter(torch.zeros(args.d_h))
		self.c = nn.Parameter(torch.Tensor([5.0]), requires_grad=False)

		self.w_f1 = nn.Linear(args.d_h, args.d_h)
		self.w_f2 = nn.Linear(args.d_h, args.d_h)
		init.xavier_uniform_(self.w_f1.weight)
		init.xavier_uniform_(self.w_f2.weight)
		init.constant_(self.w_f1.bias, 0)
		init.constant_(self.w_f2.bias, 0)
		self.w_f1.bias.requires_grad = False
		self.w_f2.bias.requires_grad = False
		self.b_f = nn.Parameter(torch.zeros(args.d_h))

		self.elu = nn.ELU()
		self.relu = nn.ReLU()
		self.tanh = nn.Tanh()
		self.softmax = nn.Softmax(dim=-2)
		self.sigmoid = nn.Sigmoid()
		self.dropout = nn.Dropout(args.dropout)

	def forward(self, x, rep_mask):
		batch_size, seq_len, d_e = x.size()

		# Make directional mask
		# (batch, seq_len, seq_len)
		rep_mask_tile = get_rep_mask_tile(rep_mask, self.device)
		# (1, seq_len, seq_len)
		direct_mask_tile = get_direct_mask_tile(self.direction, seq_len, self.device)
		# (batch, seq_len, seq_len)
		mask = rep_mask_tile * direct_mask_tile
		# (batch, seq_len, seq_len, 1)
		mask.unsqueeze_(-1)

		# Transform the input sequence to a sequence of hidden state
		x_dp = self.dropout(x)
		rep_map = self.elu(self.fc(x_dp))
		rep_map_tile = rep_map.unsqueeze(1).expand(batch_size, seq_len, seq_len, d_e)
		rep_map_dp = self.dropout(rep_map)

		# Make logits
		# (batch, 1, seq_len, hid_dim)
		dependent_etd = self.w_1(rep_map_dp).unsqueeze(1)
		# (batch, seq_len, 1, hid_dim)
		head_etd = self.w_2(rep_map_dp).unsqueeze(2)

		# (batch, seq_len, seq_len, hid_dim)
		logits = self.c * self.tanh((dependent_etd + head_etd + self.b_1) / self.c)

		# Attention scores
		attn_score = masked_softmax(logits, mask, dim=2)
		attn_score = attn_score * mask

		# Attention results
		attn_result = torch.sum(attn_score * rep_map_tile, dim=2)

		# Fusion gate: combination with input
		fusion_gate = self.sigmoid(self.w_f1(self.dropout(rep_map)) + self.w_f2(self.dropout(attn_result)) + self.b_f)
		out = fusion_gate * rep_map + (1-fusion_gate) * attn_result

		# Mask for high rank
		out = out * rep_mask

		return out


class DiSAN(nn.Module):

	def __init__(self, args):
		super(DiSAN, self).__init__()

		self.d_e = args.d_e
		self.d_h = args.d_h
		self.device = args.device

		self.fw_DiSA = DiSA(args, direction='fw')
		self.bw_DiSA = DiSA(args, direction='bw')

		self.source2token = Source2Token(args.d_h * 2, args.dropout)

	def forward(self, inputs, rep_mask):
		# Forward and backward DiSA
		for_u = self.fw_DiSA(inputs, rep_mask)
		back_u = self.bw_DiSA(inputs, rep_mask)

		# Concat
		u = torch.cat([for_u, back_u], dim=-1)

		# Source2Token
		s = self.source2token(u, rep_mask)

		return s





