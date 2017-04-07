#coding:utf-8
from keras.models import load_model
import numpy as np
import sys
from collections import deque

def load_dic(dic, filename):
    """
    dic: dict(). Used to store specific index-character key-value
    load dictionary from file into dic
    """
    with open(filename, 'r') as f:
        # todo

    return

def save_output(results, filename):
    with open(filename, "w") as f:
        for character in results:
            f.write(character)


if __name__ == "__main__":
    models = []
    seq_length = int(sys.argv[1])
    test_max_iter = 1000
    dic = dict()

    # todo
    dic_fn = ""

    load_dic(dic, dic_fn)

    for i in range(50):
        models.append(load_model("weights-improvement-{epoch:" + str(i) + "}.hdf5"))
    while 1:
        seed = raw_input("Please input your " + str(seq_length) +\
             " Chinese characters as a seed:")
        output_fn = raw_input("Please input your output file path:")
        seed = deque(seed.split())
        if seed[-1] == '。':
            print "It's already a sentence.:)"
            continue
        results = []
        results.extend(seed)
        for i, model in enumerate(models):
            print "Testing model#" + str(i+1)
            i = 0
            while 1:
                input_x = np.reshape(seed, (1, seq_length, 1))
                result = (model.predict(input_x, batch_size=64, verbose=1))
                index = np.argmax(result)
                results.append(dic[index])
                if dic[index] == "。" or i >= test_max_iter:
                    break
                else:
                    seed.popleft()
                    seed.append(dic[index])
                i += 1
            print ''.join(results)
            save_output(results, output_fn+"model#"+str(i+1))
