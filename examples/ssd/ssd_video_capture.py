from __future__ import print_function
import caffe
from caffe.model_libs import *
from google.protobuf import text_format

import math
import os
import shutil
import stat
import subprocess
import sys

import numpy as np
import time
import matplotlib.pyplot as plt
import cv2

cap = cv2.VideoCapture(0)
#cap.set(3,640)
#cap.set(4,480)
out = cv2.VideoWriter('output.avi',cv2.cv.CV_FOURCC(*'XVID'), 20, (640,480))

# Use GPU or CPU
caffe.set_mode_gpu()

# load ILSVRC2015 DET labels
voc_labelmap_file = 'data/myDataSet_extended/labelmap.prototxt'
file = open(voc_labelmap_file, 'r')
voc_labelmap = caffe_pb2.LabelMap()
text_format.Merge(str(file.read()), voc_labelmap)

def get_labelname(labelmap, labels):
    num_labels = len(labelmap.item)
    labelnames = []
    if type(labels) is not list:
        labels = [labels]
    for label in labels:
        found = False
        for i in xrange(0, num_labels):
            if label == labelmap.item[i].label:
                found = True
                labelnames.append(labelmap.item[i].display_name)
                break
        assert found == True
    return labelnames

snapshot_dir = "models/VGGNet/myDataSet_extended/SSD_300x300"

# Find most recent snapshot of the model
max_iter = 0
for file in os.listdir(snapshot_dir):
  if file.endswith(".caffemodel"):
    basename = os.path.splitext(file)[0]
    iter = int(basename.split("{}_iter_".format("VGG_myDataSet_extended_SSD_300x300"))[1])
    if iter > max_iter:
      max_iter = iter

if max_iter == 0:
  print("Cannot find snapshot in {}".format(snapshot_dir))
  sys.exit()

# load model
model_def = 'models/VGGNet/myDataSet_extended/SSD_300x300/deploy.prototxt'
model_weights = 'models/VGGNet/myDataSet_extended/SSD_300x300/VGG_myDataSet_extended_SSD_300x300_iter_{}.caffemodel'.format(max_iter)

net = caffe.Net(model_def,      # defines the structure of the model
                model_weights,  # contains the trained weights
                caffe.TEST)     # use test mode (e.g., don't perform dropout)

# input preprocessing: 'data' is the name of the input blob == net.inputs[0]
transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
transformer.set_transpose('data', (2, 0, 1))
transformer.set_mean('data', np.array([104,117,123])) # mean pixel
#transformer.set_raw_scale('data', 255)  # the reference model operates on images in [0,255] range instead of [0,1]
#transformer.set_channel_swap('data', (2,1,0))  # the reference model has channels in BGR order instead of RGB

# reshape data blob
net.blobs['data'].reshape(1,3,300,300)

# load image
while True:
    start_time = time.time()
    ret, image = cap.read()

    # preprocess image
    transformed_image = transformer.preprocess('data', image)
    net.blobs['data'].data[...] = transformed_image
    
    # Forward pass.
    detections = net.forward()['detection_out']

    # Parse the outputs.
    det_label = detections[0,0,:,1]
    det_conf = detections[0,0,:,2]
    det_xmin = detections[0,0,:,3]
    det_ymin = detections[0,0,:,4]
    det_xmax = detections[0,0,:,5]
    det_ymax = detections[0,0,:,6]

    # Get detections with confidence higher than 0.6.
    top_indices = [i for i, conf in enumerate(det_conf) if conf >= 0.3]

    top_conf = det_conf[top_indices]
    top_label_indices = det_label[top_indices].tolist()
    top_labels = get_labelname(voc_labelmap, top_label_indices)
    top_xmin = det_xmin[top_indices]
    top_ymin = det_ymin[top_indices]
    top_xmax = det_xmax[top_indices]
    top_ymax = det_ymax[top_indices]

    colors = [(255,0,0), (0,255,0), (0,0,255), (255,255,0), (255,0,255), (0,255,255), (0,0,0)]

    overlay = image.copy()

    for i in xrange(top_conf.shape[0]):
        xmin = int(round(top_xmin[i] * image.shape[1]))
        ymin = int(round(top_ymin[i] * image.shape[0]))
        xmax = int(round(top_xmax[i] * image.shape[1]))
        ymax = int(round(top_ymax[i] * image.shape[0]))
        score = top_conf[i]
        label = top_labels[i]
        name = '%s: %.2f'%(label, score)
        coords = (xmin, ymin), xmax-xmin+1, ymax-ymin+1
        color = colors[i % len(colors)]
        cv2.rectangle(image,(xmin,ymin),(xmax,ymax),color,3)
        retval, baseline = cv2.getTextSize(name, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)
        cv2.rectangle(overlay, (xmin,ymin), (xmin+retval[0]-25,ymin-retval[1]-baseline), (255,255,255),-1)
    
    cv2.addWeighted(overlay, 0.4, image, 0.6, 0.0, image)
    
    for i in xrange(top_conf.shape[0]):
        xmin = int(round(top_xmin[i] * image.shape[1]))
        ymin = int(round(top_ymin[i] * image.shape[0]))
        xmax = int(round(top_xmax[i] * image.shape[1]))
        ymax = int(round(top_ymax[i] * image.shape[0]))
        score = top_conf[i]
        label = top_labels[i]
        name = '%s: %.2f'%(label, score)
        coords = (xmin, ymin), xmax-xmin+1, ymax-ymin+1
        color = colors[i % len(colors)]
        cv2.putText(image, name,(xmin,ymin-baseline), cv2.FONT_HERSHEY_SIMPLEX, 0.8,(0,0,0),2)
   
    cv2.imshow("detections", image)
    out.write(image)
    print("time for frame: {}".format(time.time() - start_time))
    if cv2.waitKey(1) != -1:
        break
