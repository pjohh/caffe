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
import argparse
import numpy as np
import time
import cv2

# parse commandline arguments
parser = argparse.ArgumentParser()
parser.add_argument('image_path', type=str, help="path to image")
parser.add_argument('image_size', type=int, choices=[100, 200, 300, 301, 302, 303, 304, 500], help="image size used by SSD-Algorithm")
parser.add_argument('overlay_size', type=str, choices=['s', 'm', 'b'], help="choose overlay size between small, medium and big")
parser.add_argument('-s', action='store_true', help="set this flag if images should be saved (into home/$USER/Bilder)")
parser.add_argument('-d', action='store_true', help="set this flag if images should be displayed")
args = parser.parse_args()

if args.image_size == 200:
  ssd_size = "200x200"
if args.image_size == 205:
  ssd_size = "200x200_5%"
if args.image_size == 222:
  ssd_size = "200x200_dropout"

elif args.image_size == 300:
  ssd_size = "300x300"
elif args.image_size == 303:
  ssd_size = "300x300_3%_8%"
elif args.image_size == 304:
  ssd_size = "300x300_4%_8%"
elif args.image_size == 305:
  ssd_size = "300x300_5%_10%"
elif args.image_size == 310:
  ssd_size = "300x300_da"
elif args.image_size == 311:
  ssd_size = "300x300_da2"
elif args.image_size == 333:
  ssd_size = "300x300_dropout"

if args.image_size == 500:
  ssd_size = "500x500"
if args.image_size == 505:
  ssd_size = "500x500_5%"
if args.image_size == 555:
  ssd_size = "500x500_dropout"

if args.image_size >= 300 and args.image_size <=399:
  args.image_size = 300
if args.image_size >= 200 and args.image_size <=299:
  args.image_size = 200
if args.image_size >= 500 and args.image_size <=599:
  args.image_size = 500

ssd_size = "SSD_"+ssd_size

# Use GPU or CPU
caffe.set_mode_gpu()

# load labels
voc_labelmap_file = 'data/robot_dataset/labelmap.prototxt'
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

snapshot_dir = "models/VGGNet/robot_dataset/{}".format(ssd_size)

# Find most recent snapshot of the model
max_iter = 0
for file in os.listdir(snapshot_dir):
  if file.endswith(".caffemodel"):
    basename = os.path.splitext(file)[0]
    iter = int(basename.split("VGG_robot_dataset_{}_iter_".format(ssd_size))[1])
    if iter > max_iter:
      max_iter = iter

if max_iter == 0:
  print("Cannot find snapshot in {}".format(snapshot_dir))
  sys.exit()

# load model
model_def = 'models/VGGNet/robot_dataset/{}/deploy.prototxt'.format(ssd_size)
model_weights = 'models/VGGNet/robot_dataset/{}/VGG_robot_dataset_{}_iter_{}.caffemodel'.format(ssd_size, ssd_size, max_iter)

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
if ssd_size.startswith("SSD_200x200"):
    net.blobs['data'].reshape(1, 3, 200, 200)
elif ssd_size.startswith("SSD_300x300"):
    net.blobs['data'].reshape(1, 3, 300, 300)
else:
    net.blobs['data'].reshape(1, 3, 500, 500)

# load image
#image = caffe.io.load_image(sys.argv[1])
image = cv2.imread(args.image_path)

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
top_indices = [i for i, conf in enumerate(det_conf) if conf >= 0.2]

top_conf = det_conf[top_indices]
top_label_indices = det_label[top_indices].tolist()
top_labels = get_labelname(voc_labelmap, top_label_indices)
top_xmin = det_xmin[top_indices]
top_ymin = det_ymin[top_indices]
top_xmax = det_xmax[top_indices]
top_ymax = det_ymax[top_indices]

colors = {"robot": (0,255,0), "base_station": (0,0,150), "mug": (255,128,0), "battery": (0,150,255)}
abbr = {"robot": "r", "base_station": "s", "mug": "m", "battery": "b"}

overlay = image.copy()

for i in xrange(top_conf.shape[0]):
	xmin = int(round(top_xmin[i] * image.shape[1]))
	ymin = int(round(top_ymin[i] * image.shape[0]))
	xmax = int(round(top_xmax[i] * image.shape[1]))
	ymax = int(round(top_ymax[i] * image.shape[0]))
	label = top_labels[i]
	color = colors[label]
	if args.overlay_size == 's': cv2.rectangle(image,(xmin,ymin),(xmax,ymax),color, 2) 
	if args.overlay_size == 'm': cv2.rectangle(image,(xmin,ymin),(xmax,ymax),color, 10) 
	if args.overlay_size == 'b': cv2.rectangle(image,(xmin,ymin),(xmax,ymax),color, 18) 
    
cv2.addWeighted(overlay, 0.3, image, 0.7, 0.0, image)

overlay = image.copy()

for i in xrange(top_conf.shape[0]):
    xmin = int(round(top_xmin[i] * image.shape[1]))
    ymin = int(round(top_ymin[i] * image.shape[0]))
    xmax = int(round(top_xmax[i] * image.shape[1]))
    ymax = int(round(top_ymax[i] * image.shape[0]))
    score = top_conf[i]
    label = top_labels[i]
    name = '%s: %.2f'%(abbr[label], score)
    if args.overlay_size == 's': retval, baseline = cv2.getTextSize(name, cv2.FONT_HERSHEY_PLAIN, 1.2, 1)
    if args.overlay_size == 'm': retval, baseline = cv2.getTextSize(name, cv2.FONT_HERSHEY_DUPLEX, 3, 6)
    if args.overlay_size == 'b': retval, baseline = cv2.getTextSize(name, cv2.FONT_HERSHEY_DUPLEX, 4, 9)
    cv2.rectangle(overlay, (xmax,ymax), (xmax+retval[0],ymax-retval[1]-baseline), (255,255,255),-1)
        
    
cv2.addWeighted(overlay, 0.6, image, 0.4, 0.0, image)

for i in xrange(top_conf.shape[0]):
    xmin = int(round(top_xmin[i] * image.shape[1]))
    ymin = int(round(top_ymin[i] * image.shape[0]))
    xmax = int(round(top_xmax[i] * image.shape[1]))
    ymax = int(round(top_ymax[i] * image.shape[0]))
    score = top_conf[i]
    label = top_labels[i]
    name = '%s: %.2f'%(abbr[label], score)
    if args.overlay_size == 's': cv2.putText(image, name,(xmax,ymax-baseline+2), cv2.FONT_HERSHEY_PLAIN, 1.2,(0,0,0), 1)
    if args.overlay_size == 'm': cv2.putText(image, name,(xmin,ymin-baseline+5), cv2.FONT_HERSHEY_DUPLEX, 3,(0,0,0),3)
    if args.overlay_size == 'b': cv2.putText(image, name,(xmin,ymin-baseline+5), cv2.FONT_HERSHEY_DUPLEX, 4,(0,0,0),7)

if args.d is True:
    cv2.imshow("detections", image)
    cv2.waitKey()

if args.s is True:
    cv2.imwrite(os.environ['HOME']+'/Bilder/'+args.image_path.split("/")[-1].split(".")[0]
                +'_{}.jpg'.format(ssd_size), image)
