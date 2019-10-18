#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep  6 17:08:33 2019

@author: AI-Bridge_User
"""

import os
import torch
from torch.autograd import Variable
import torch.utils.data as data
import torchvision.transforms as transforms
from PIL import Image, ImageDraw
from SPPE.src.utils.img import load_image, cropBox, im_to_torch
from opt import opt
from yolo.preprocess import prep_image, prep_frame, inp_to_image
from pPose_nms import pose_nms, write_json
from matching import candidate_reselect as matching
from SPPE.src.utils.eval import getPrediction, getMultiPeakPrediction
from yolo.util import write_results, dynamic_write_results
from yolo.darknet import Darknet
from tqdm import tqdm
import cv2
import json
import numpy as np
import sys
import time
import torch.multiprocessing as mp
from multiprocessing import Process
from multiprocessing import Queue as pQueue
from threading import Thread

torchCuda=0
torch.cuda.set_device(torchCuda)

# import the Queue class from Python 3
if sys.version_info >= (3, 0):
    from queue import Queue, LifoQueue
# otherwise, import the Queue class for Python 2.7
else:
    from Queue import Queue, LifoQueue

if opt.vis_fast:
    from fn import vis_frame_fast as vis_frame
else:
    from fn import vis_frame



class ImageLoader:
    def cnv_img(Frame):
        img = []
        orig_img = []
        im_dim_list = []
#            for k in range(i*self.batchSize, min((i +  1)*self.batchSize, self.datalen)):
        inp_dim = int(opt.inp_dim)

        img_k, orig_img_k, im_dim_list_k = prep_image(Frame, inp_dim)
    
        img.append(img_k)
        orig_img.append(orig_img_k)
        im_dim_list.append(im_dim_list_k)
        with torch.no_grad():
            # Human Detection
            img = torch.cat(img)
            im_dim_list = torch.FloatTensor(im_dim_list).repeat(1,2)
        return img, orig_img, im_dim_list

 


class DetectionLoader2:
    def __init__(self):
        # initialize the file video stream along with the boolean
        # used to indicate if the thread should be stopped or not
        self.det_model = Darknet("./yolo/cfg/yolov3-spp.cfg")
        self.det_model.load_weights('./models/yolo/yolov3-spp.weights')
        self.det_model.net_info['height'] = opt.inp_dim
        self.det_inp_dim = int(self.det_model.net_info['height'])
        assert self.det_inp_dim % 32 == 0
        assert self.det_inp_dim > 32
        self.det_model.cuda(torchCuda)
        self.det_model.eval()

    def load(self,img, orig_img, im_dim_list):
            with torch.no_grad():
                # Human Detection
                img = img.cuda(torchCuda)
                prediction = self.det_model(img, CUDA=True)
                # NMS process
                dets = dynamic_write_results(prediction, opt.confidence,
                                    opt.num_classes, nms=True, nms_conf=opt.nms_thesh)
                dets = dets.cpu()
                im_dim_list = torch.index_select(im_dim_list,0, dets[:, 0].long())
                scaling_factor = torch.min(self.det_inp_dim / im_dim_list, 1)[0].view(-1, 1)

                # coordinate transfer
                dets[:, [1, 3]] -= (self.det_inp_dim - scaling_factor * im_dim_list[:, 0].view(-1, 1)) / 2
                dets[:, [2, 4]] -= (self.det_inp_dim - scaling_factor * im_dim_list[:, 1].view(-1, 1)) / 2

                
                dets[:, 1:5] /= scaling_factor
                for j in range(dets.shape[0]):
                    dets[j, [1, 3]] = torch.clamp(dets[j, [1, 3]], 0.0, im_dim_list[j, 0])
                    dets[j, [2, 4]] = torch.clamp(dets[j, [2, 4]], 0.0, im_dim_list[j, 1])
                boxes = dets[:, 1:5]
                scores = dets[:, 5:6]

#            for k in range(len(orig_img)):
            k=0
            boxes_k = boxes[dets[:,0]==k]
            inps = torch.zeros(boxes_k.size(0), 3, opt.inputResH, opt.inputResW)
            pt1 = torch.zeros(boxes_k.size(0), 2)
            pt2 = torch.zeros(boxes_k.size(0), 2)
                
            return (orig_img[k], boxes_k, scores[dets[:,0]==k], inps, pt1, pt2)

    def read(self):
        # return next frame in the queue
        return self.Q.get()

    def len(self):
        # return queue len
        return self.Q.qsize()


class DetectionProcessor2:
    def process(orig_img,boxes,scores,inps,pt1,pt2):
        with torch.no_grad():

            inp = im_to_torch(cv2.cvtColor(orig_img, cv2.COLOR_BGR2RGB))
            inps, pt1, pt2 = crop_from_dets(inp, boxes, inps, pt1, pt2)
            return (orig_img,boxes,scores,inps,pt1,pt2)





class Mscoco(data.Dataset):
    def __init__(self, train=True, sigma=1,
                 scale_factor=(0.2, 0.3), rot_factor=40, label_type='Gaussian'):
        self.img_folder = '../data/coco/images'    # root image folders
        self.is_train = train           # training set or test set
        self.inputResH = opt.inputResH
        self.inputResW = opt.inputResW
        self.outputResH = opt.outputResH
        self.outputResW = opt.outputResW
        self.sigma = sigma
        self.scale_factor = scale_factor
        self.rot_factor = rot_factor
        self.label_type = label_type

        self.nJoints_coco = 17
        self.nJoints_mpii = 16
        self.nJoints = 33

        self.accIdxs = (1, 2, 3, 4, 5, 6, 7, 8,
                        9, 10, 11, 12, 13, 14, 15, 16, 17)
        self.flipRef = ((2, 3), (4, 5), (6, 7),
                        (8, 9), (10, 11), (12, 13),
                        (14, 15), (16, 17))

    def __getitem__(self, index):
        pass

    def __len__(self):
        pass


def crop_from_dets(img, boxes, inps, pt1, pt2):
    '''
    Crop human from origin image according to Dectecion Results
    '''

    imght = img.size(1)
    imgwidth = img.size(2)
    tmp_img = img
    tmp_img[0].add_(-0.406)
    tmp_img[1].add_(-0.457)
    tmp_img[2].add_(-0.480)
    for i, box in enumerate(boxes):
        upLeft = torch.Tensor(
            (float(box[0]), float(box[1])))
        bottomRight = torch.Tensor(
            (float(box[2]), float(box[3])))

        ht = bottomRight[1] - upLeft[1]
        width = bottomRight[0] - upLeft[0]

        scaleRate = 0.3

        upLeft[0] = max(0, upLeft[0] - width * scaleRate / 2)
        upLeft[1] = max(0, upLeft[1] - ht * scaleRate / 2)
        bottomRight[0] = max(
            min(imgwidth - 1, bottomRight[0] + width * scaleRate / 2), upLeft[0] + 5)
        bottomRight[1] = max(
            min(imght - 1, bottomRight[1] + ht * scaleRate / 2), upLeft[1] + 5)

        try:
            inps[i] = cropBox(tmp_img.clone(), upLeft, bottomRight, opt.inputResH, opt.inputResW)
        except IndexError:
            print(tmp_img.shape)
            print(upLeft)
            print(bottomRight)
            print('===')
        pt1[i] = upLeft
        pt2[i] = bottomRight

    return inps, pt1, pt2


class DataWriter2:
    
    def update(boxes, scores, hm_data, pt1, pt2, orig_img, i):
        
        orig_img = np.array(orig_img, dtype=np.uint8)

        preds_hm, preds_img, preds_scores = getPrediction(
            hm_data, pt1, pt2, opt.inputResH, opt.inputResW, opt.outputResH, opt.outputResW)
        result = pose_nms(
            boxes, scores, preds_img, preds_scores)
        result = {
            'imgname': str(i),
            'result': result
        }
        img = vis_frame(orig_img, result)
        return img, result
