# -*- coding: utf-8 -*-
import caffe
import numpy as np
import cv2
import numpy.random as random
import os
import random

class DataLayer(caffe.Layer):

    def setup(self, bottom, top):

        # 是否随机
        self.random = False
        self.seed = None

        if len(top) != 2:
            raise Exception("Need to define two tops: data and mask.")

        if len(bottom) != 0:
            raise Exception("Do not define a bottom.")

        self.lines = np.loadtxt("data/train2.txt", dtype=str)
        np.random.shuffle(self.lines)

        self.idx = 0                                       # 初始位置
        self.batch_size = 8                               
        self.width = 256
        self.height = 256
        self.flip = [False for _ in range(self.batch_size)] # 是否镜像
        self.offset = [0 for _ in range(self.batch_size)]
        self.blurList = [3,5,7,9,11,13]                  # 高斯模糊半径
        if self.random:
            random.seed(self.seed)
            self.idx = random.randint(0, len(self.lines) - 1)

    def reshape(self, bottom, top):
        # load image + label image pair
        self.data = self.load_image(self.idx)
        self.mask = self.load_mask(self.idx)
        # reshape tops to fit (leading 1 is for batch dimension)
        top[0].reshape(*self.data.shape)
        top[1].reshape(*self.mask.shape)

    def forward(self, bottom, top):
        # assign output
        top[0].data[...] = self.data
        top[1].data[...] = self.mask

        # pick next input
        if self.random:
            self.idx = random.randint(0, len(self.lines) - 1)
        else:
            self.idx += self.batch_size
            if self.idx >= len(self.lines):
                self.idx = 0

    def backward(self, top, propagate_down, bottom):
        pass

    def load_image(self, idx):
        '''
        读取RGB图像
        '''
        x = []
        for i in range(self.batch_size):
            currIndex = idx + i
            # 如果越界，则从头开始
            if currIndex >= len(self.lines):
                currIndex = currIndex - len(self.lines)

            imname = self.lines[currIndex][0]

            im = cv2.imread(imname)
            
            # 图像增强
            if random.random() > 0.2:
                alpha = np.random.random()*0.6+0.4
                beta = np.random.randint(50)
                blank = np.zeros(im.shape, im.dtype)
                # dst = alpha * img + beta * blank
                im = cv2.addWeighted(im, alpha, blank, 1-alpha, beta)
            if random.random() > 0.5:
                im = cv2.flip(im, 1)
                self.flip[i] = True
            else:
                self.flip[i] = False
            if random.random() > 0.5:
                ksize=self.blurList[np.random.randint(6)]
                im = cv2.GaussianBlur(im, (ksize, ksize), 0)
                
            # 图像四周扩边，使其长宽一致
            if im.shape[0] >= im.shape[1]:
                res = im.shape[0] - im.shape[1]
                if res > 0:
                    left_res = np.random.randint(res//2+1)
                    self.offset[i] = left_res
                    im = cv2.copyMakeBorder(im, 0, 0, left_res, res-left_res, cv2.BORDER_REFLECT)
            else:
                res = im.shape[1] - im.shape[0]
                up_res = np.random.randint(res//2+1)
                self.offset[i] = up_res
                im = cv2.copyMakeBorder(im, up_res, res-up_res, 0, 0, cv2.BORDER_REFLECT)                       
        
            im = cv2.resize(im,(self.width, self.height))        

            x.append(im)
        x = np.array(x, np.float64)
        x /= 255.0
        x -= 0.5

        return x.transpose((0, 3, 1, 2))

    def load_mask(self, idx):
        '''
        读取mask
        '''
        x = []
        for i in range(self.batch_size):
            currIndex = idx + i
            if currIndex >= len(self.lines):
                currIndex = currIndex - len(self.lines)

            imname = self.lines[currIndex][1]
            im = cv2.imread(imname, 0)

            # 如果原图镜像了，那么mask也需要镜像
            if self.flip[i]:
                im = cv2.flip(im, 1)
            # 图像四周扩边，使其长宽一致
            if im.shape[0] >= im.shape[1]:
                res = im.shape[0] - im.shape[1]
                if res > 0:
                    im = cv2.copyMakeBorder(im, 0, 0, self.offset[i], res-self.offset[i], cv2.BORDER_REFLECT)
            else:
                res = im.shape[1] - im.shape[0]
                im = cv2.copyMakeBorder(im, self.offset[i], res-self.offset[i], 0, 0, cv2.BORDER_REFLECT)   

            im = cv2.resize(im,(self.width, self.height))

            outimg = np.empty((1,self.height,self.width))
            outimg[0, ...] = im > 0.5

            x.append(outimg)
        x = np.array(x, np.uint8)
        return x


    
