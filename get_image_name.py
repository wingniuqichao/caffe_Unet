import numpy as np
import cv2
import numpy.random as random
import os
import random

TRAIN_RATE = 0.8
def gen():
	imgdir = "../../datasets/baidu/JPEGImages/"
	lines = os.listdir(imgdir)
	imgdir2 = "../../datasets/baidu/humanparsing/JPEGImages/"
	lines2 = [os.listdir(imgdir2)]
	lines.extend(lines2)

	np.random.shuffle(lines)

	test_split = int(TRAIN_RATE*len(lines))
	train = lines[:test_split]
	test = lines[test_split:]

	with open('train_name.txt', 'w') as f:
		for name in train:
			f.write('%s\n'%name)
	with open('test_name.txt', 'w') as f:
		for name in test:
			f.write('%s\n'%name)

if __name__ == '__main__':
	gen()