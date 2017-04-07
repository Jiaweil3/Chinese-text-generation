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
from sets import Set
import re

# ===================== Guanglei Deng start ================
reload(sys) 
sys.setdefaultencoding("utf-8")
raw_text = codecs.open('tianlongbabu_text.txt').read().decode('utf-8')

pattern_prd = re.compile(ur'[？！……]')
raw_text = re.sub(pattern_prd, '。', raw_text)
pattern_keep = re.compile(u'[^\u4E00-\u9FA5，。：]')
raw_text = re.sub(pattern_keep, '', raw_text)

# build the dictionary of every characters, the value is [id, times]
charcs = {}
i = 0
for charc in raw_text:
	if charc not in charcs:
		charcs[charc] = [i, 1]
		i += 1
	else:
		charcs[charc][1] += 1

# build the dictionary of characters only appear once
char_cut = {}
for charc in charcs:
	if charcs[charc][1] == 1:
		char_cut[charc] = 1

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
# ===================== Guanglei Deng end ==================

# ===================== pre-processing =====================
text_lines = get_all_lines(raw_text)
char_collection = Set()
for line in text_lines:
	for char in line:
		char_collection.add(char)

index = 1
char_dic = {}
for char in char_collection:
	char_dic[char] = [index, 0]
	index += 1
# ===================== pre-processing end =====================

# ===================== Xue Chen start =====================
def data_pro(lines,minlen,ngram,chardic):
	dataX = []
	dataY = []
	for line in lines:
		if len(line)<minlen:
			continue
		else:
			for i in range(0,len(line)-ngram,1):
				numline =[]
				for char in line[i:i+ngram]:
					numline.append(chardic[char][0])
				next_char = line[i+ngram]
				dataX.append(numline)
				dataY.append(chardic[next_char][0])
	return dataX, dataY,len(chardic)
# ===================== Xue Chen end =======================

# ===================== Pre-processing =====================
dataX, dataY, n_vocab = data_pro(text_lines, 10, 10, char_dic)
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
model.add(LSTM(512, input_shape=(X.shape[1], X.shape[2])))
model.add(Dropout(0.2))
model.add(LSTM(512))
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