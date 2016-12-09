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
import argparse

# parse commandline arguments
parser = argparse.ArgumentParser()
parser.add_argument('video_source', type=str, help="path to video used as source")
parser.add_argument('image_size', type=str, help="image size used by SSD-Algorithm")
parser.add_argument('dataset', type=str, help="name of the dataset")
parser.add_argument('overlay_size', type=str, choices=['s', 'm', 'b'], help="size of the overlay in the output video")
parser.add_argument('frame_rate', type=int, help="framerate of output video")
parser.add_argument('-s', action='store_true', help="set this flag if video should be saved")
parser.add_argument('-d', action='store_true', help="set this flag if video should be displayed")
args = parser.parse_args()

image_size = int(args.image_size.split("_", -1)[1].split('x', 1)[0])

# configure webcam input
cap = cv2.VideoCapture(args.video_source)
out = cv2.VideoWriter(args.video_source.split("/", -1)[-1].split(".", 1)[0] + args.image_size + '.avi',cv2.cv.CV_FOURCC(*'XVID'), args.frame_rate, (int(cap.get(cv2.cv.CV_CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT))))

# Use GPU or CPU
caffe.set_mode_gpu()

# load labels
labelmap_file = 'data/{}/labelmap.prototxt'.format(args.dataset)
file = open(labelmap_file, 'r')
labelmap = caffe_pb2.LabelMap()
text_format.Merge(str(file.read()), labelmap)

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

snapshot_dir = "models/VGGNet/{}/{}".format(args.dataset, args.image_size)

# Find most recent snapshot of the model
max_iter = 0
for file in os.listdir(snapshot_dir):
    if file.endswith(".caffemodel"):
        basename = os.path.splitext(file)[0]
        iter = int(basename.split("VGG_{}_{}_iter_".format(args.dataset, args.image_size))[1])
        if iter > max_iter:
          max_iter = iter

if max_iter == 0:
    print("Cannot find snapshot in {}".format(snapshot_dir))
    sys.exit()

# load model
model_def = 'models/VGGNet/{}/{}/deploy.prototxt'.format(args.dataset, args.image_size)
model_weights = 'models/VGGNet/{}/{}/VGG_{}_{}_iter_{}.caffemodel'.format(args.dataset, args.image_size, args.dataset, args.image_size, max_iter)

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
net.blobs['data'].reshape(1, 3, image_size, image_size)

# grab webcam picture, detect objects+visualize, save video
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

    threshold = [0.38, 0.27, 0.32, 0.22]
    # Get detections with confidence higher than 0.6.
    top_indices = [i for i, conf in enumerate(det_conf) if conf >= threshold[int(det_label[i])-1]]

    top_conf = det_conf[top_indices]
    top_label_indices = det_label[top_indices].tolist()
    top_labels = get_labelname(labelmap, top_label_indices)
    top_xmin = det_xmin[top_indices]
    top_ymin = det_ymin[top_indices]
    top_xmax = det_xmax[top_indices]
    top_ymax = det_ymax[top_indices]

    #colors = {"robot": (0,255,0), "base_station": (0,0,255), "mug": (255,0,0), "battery": (0,255,255)}
    #colors = {"robot": (0,255,0), "base_station": (0,0,0), "mug": (0,0,0), "battery": (0,0,0)}
    colors = {"robot": (0,255,0), "base": (0,0,150), "mug": (255,128,0), "battery": (0,150,255)}

    # overlay for outline of objects, box for text and text
    overlay = image.copy()

    for i in xrange(top_conf.shape[0]):
      xmin = int(round(top_xmin[i] * image.shape[1]))
      ymin = int(round(top_ymin[i] * image.shape[0]))
      xmax = int(round(top_xmax[i] * image.shape[1]))
      ymax = int(round(top_ymax[i] * image.shape[0]))
      label = top_labels[i]
      color = colors[label]
      if args.overlay_size == 's': cv2.rectangle(image,(xmin,ymin),(xmax,ymax),color, 2) # 640x480 
      elif args.overlay_size == 'm': cv2.rectangle(image,(xmin,ymin),(xmax,ymax),color, 5) # 1920x1080
      else: cv2.rectangle(image,(xmin,ymin),(xmax,ymax),color, 14) # >1920x1080 
      
    cv2.addWeighted(overlay, 0.2, image, 0.8, 0.0, image)

    overlay = image.copy()

    x_offset = []
    y_offset = []
    for i in xrange(top_conf.shape[0]):
	    xmin = int(round(top_xmin[i] * image.shape[1]))
	    ymin = int(round(top_ymin[i] * image.shape[0]))
	    xmax = int(round(top_xmax[i] * image.shape[1]))
	    ymax = int(round(top_ymax[i] * image.shape[0]))
	    score = top_conf[i]
	    label = top_labels[i]
	    name = '%s: %.2f'%(label, score)
	    if args.overlay_size == 's': retval, baseline = cv2.getTextSize(name, cv2.FONT_HERSHEY_DUPLEX, 0.6, 1)
	    elif args.overlay_size == 'm': retval, baseline = cv2.getTextSize(name, cv2.FONT_HERSHEY_DUPLEX, 1.4, 3)
	    else: retval, baseline = cv2.getTextSize(name, cv2.FONT_HERSHEY_DUPLEX, 4, 6) 
	    # correct boxes with borders beyond image borders
	    if xmin+retval[0] > image.shape[1]: 
		  x_offset.append(image.shape[1] - (xmin+retval[0]))
	    else: x_offset.append(0)
	    if ymax+retval[1]+2*baseline > image.shape[0]:
		  y_offset.append(image.shape[0] - (ymax+retval[1]+2*baseline))
	    else: y_offset.append(0) 
	    cv2.rectangle(image, (xmin+x_offset[i],int(ymax+baseline/3)+y_offset[i]), (xmin+retval[0]+x_offset[i],ymax+retval[1]+2*baseline+y_offset[i]), (255,255,255),-1)

    cv2.addWeighted(overlay, 0.5, image, 0.5, 0.0, image)

    for i in xrange(top_conf.shape[0]):
	    xmin = int(round(top_xmin[i] * image.shape[1]))
	    ymin = int(round(top_ymin[i] * image.shape[0]))
	    xmax = int(round(top_xmax[i] * image.shape[1]))
	    ymax = int(round(top_ymax[i] * image.shape[0]))
	    score = top_conf[i]
	    label = top_labels[i]
	    name = '%s: %.2f'%(label, score)
	    if args.overlay_size == 's': cv2.putText(image, name,(xmin+x_offset[i],ymax+retval[1]+baseline+y_offset[i]), cv2.FONT_HERSHEY_DUPLEX, 0.6,(0,0,0), 1) 
	    elif args.overlay_size == 'm': cv2.putText(image, name,(xmin+x_offset[i],ymax+retval[1]+baseline+y_offset[i]), cv2.FONT_HERSHEY_DUPLEX, 1.4,(0,0,0),2)
	    else: cv2.putText(image, name,(xmin+x_offset[i],ymax+retval[1]+baseline+y_offset[i]), cv2.FONT_HERSHEY_DUPLEX, 4,(0,0,0),5)

    if args.d is True:
      cv2.imshow("detections", image)

    if args.s is True:
      out.write(image)

    print("time for frame: {}".format(time.time() - start_time))
    if cv2.waitKey(1) != -1:
        break
