# -*- coding: utf-8 -*-

# Small LSTM Network to Generate Text for Alice in Wonderland
import numpy
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM
from keras.callbacks import ModelCheckpoint
from keras.utils import np_utils
import sys 
import codecs
import re

# ===================== Guanglei Deng start ================
reload(sys) 
sys.setdefaultencoding("utf-8")
raw_text = codecs.open('tianlongbabu.txt').read().decode('utf-8')

pattern_prd = re.compile(ur'[？！…]+')
raw_text = re.sub(pattern_prd, '。', raw_text)
pattern_keep = re.compile(u'[^\u4E00-\u9FA5，。：]')
raw_text = re.sub(pattern_keep, '', raw_text)

# build the dictionary of every characters, the value is [0(id), times]
chars = {}
for char in raw_text:
	if char not in chars:
		chars[char] = [0, 1]
	else:
		chars[char][1] += 1

# build the dictionary of characters only appear once
char_cut = {}
for char in chars:
	if chars[char][1] == 1:
		char_cut[char] = 1

# build the dictionary exclude the rare-appeared word. Map each word an id
for char in char_cut:
	chars.pop(char)
i = 0
for char in chars:
	chars[char][0] = i
	i += 1

def get_next_end(text):
	end_pos = text.find('。')
	if end_pos == -1:
		return None, 0
	line = text[0:end_pos+1].strip()
	return line, end_pos

def get_all_lines(text):
	lines = []
	while True:
		line,end_pos = get_next_end(text)
		if line:
			cut = False
			for char in line:
				if char in char_cut:
					text = text[end_pos+1:]
					cut = True
					break
			if cut == False:
				lines.append(line)
				text = text[end_pos+1:]
		else:
			break
	return lines

text_lines = get_all_lines(raw_text)

text_all = ""
for line in text_lines:
	text_all += line

# ===================== Guanglei Deng end ==================

# ===================== Xue Chen start =====================
# def data_pro(lines,minlen,ngram,chardic):
# 	dataX = []
# 	dataY = []
# 	for line in lines:
# 		if len(line)<minlen:
# 			continue
# 		else:
# 			for i in range(0,len(lines)-ngram,1):
# 				numline =[]
# 				for char in line[i:i+ngram]:
# 					numline.append(chardic[char][0])
# 				next_char = line[i+ngram]
# 				dataX.append(numline)
# 				dataY.append(chardic[next_char][0])
# 	return {'dataX':dataX,'dataY':dataY,'vocab_num':len(chardic)}
def data_pro(text, ngram, chardic):
	dataX = []
	dataY = []
	for i in range(0, len(text)-ngram, 1):
		numline =[]
		for char in text[i:i+ngram]:
			numline.append(chardic[char][0])
		next_char = text[i+ngram]
		dataX.append(numline)
		dataY.append(chardic[next_char][0])
	return {'dataX':dataX,'dataY':dataY,'vocab_num':len(chardic)}
# ===================== Xue Chen end =======================

# ===================== Pre-processing =====================
processed_data = data_pro(text_all, 7, chars)
dataX = processed_data['dataX']
dataY = processed_data['dataY']
n_vocab = processed_data['vocab_num']
patterns = len(dataX)
seq_length = len(dataX[0])

# reshape X to be [samples, time steps, features]
X = numpy.reshape(dataX, (patterns, seq_length, 1))
# normalize
X = X / float(n_vocab)
# one hot encode the output variable
y = np_utils.to_categorical(dataY)
# ===================== Pre-processing end =================

# ===================== Weilun Chen start ==================
# define the LSTM model
model = Sequential()
model.add(LSTM(256, input_shape=(X.shape[1], X.shape[2]), return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(256))
model.add(Dropout(0.2))
model.add(Dense(y.shape[1], activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam')
# define the checkpoint
filepath="weights-improvement-{epoch:02d}-{loss:.4f}.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True, mode='min')
callbacks_list = [checkpoint]
# fit the model
model.fit(X, y, epochs=50, batch_size=64, callbacks=callbacks_list)
# ===================== Weilun Chen end ====================

# ===================== Qijia start ========================

# ===================== Qijia end ==========================