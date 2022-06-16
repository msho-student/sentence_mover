import numpy as np
from wmd import WMD
from icecream import ic

from transformers import (
	AutoTokenizer, AutoModel,
	DistilBertTokenizer, DistilBertModel,
	pipeline
)

class SentenceMoverSim():
	def __init__(self):
		self.pipe = pipeline("feature-extraction", model="distilbert-base-uncased")
		self.sent_id = 0
	
	def embed_sentence(self, sent, return_token_length=True):
		pipe_res = self.pipe(sent)
		if return_token_length:
			return pipe_res[0][0], len(pipe_res[0])
		return pipe_res[0][0]

	def compute_helper(self, txt, rep_map, ids, weights):
		for sentence in txt:
			rep_map[self.sent_id], sent_len = self.embed_sentence(sentence)
			ids.append(self.sent_id)
			weights.append(float(sent_len))
			self.sent_id += 1

	def compute(self, ref, hyp):
		# ref and hyp are both lists of sentences
		self.sent_id = 0
		rep_map = {}
		ref_ids, ref_weights, hyp_ids, hyp_weights = [], [], [], []

		self.compute_helper(ref, rep_map, ref_ids, ref_weights)
		self.compute_helper(hyp, rep_map, hyp_ids, hyp_weights)

		doc_dict = {
			"0": ("ref", ref_ids, ref_weights), 
			"1": ("hyp", hyp_ids, hyp_weights)
		}

		calc = WMD(rep_map, doc_dict, vocabulary_min=1)

		try:
			# dist = calc.nearest_neighbors("0", k=1, early_stop=1)[0][1]  # how far is hyp from ref?
			res = calc.nearest_neighbors("0")  # how far is hyp from ref?
			dist = res[0][1]
		except Exception as e:
			ic(e)
			ic(ref, hyp)

		sim = np.exp(-dist) # switch to similarity
		return sim
	
	def score_batch(self, refs, hyps):
		# STUB: finish me
		return [0.0]
		
	def batch_compute(self, refs, hyps, batch_size):
		scores = []

		# determine if batch sizes fits evenly or not
		num_ex = len(hyps)
		extra_batch = bool(num_ex % batch_size)
		end = ((num_ex // batch_size) + int(extra_batch)) * batch_size

		for i in range(0, end, batch_size):
			start, stop = i, min(i + batch_size, num_ex)
			batch_scores = self.score_batch(refs[start:stop], hyps[start:stop])
			scores += batch_scores

		return scores
