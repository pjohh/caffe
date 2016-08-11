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

plt.ion()

# Use GPU or CPU
caffe.set_mode_gpu()

# load labels
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

for ssd_size in ['SSD_300x300', 'SSD_500x500']:
    snapshot_dir = "models/VGGNet/myDataSet_extended/{}".format(ssd_size)

    # Find most recent snapshot of the model
    max_iter = 0
    for file in os.listdir(snapshot_dir):
      if file.endswith(".caffemodel"):
        basename = os.path.splitext(file)[0]
        iter = int(basename.split("VGG_myDataSet_extended_{}_iter_".format(ssd_size))[1])
        if iter > max_iter:
          max_iter = iter

    if max_iter == 0:
      print("Cannot find snapshot in {}".format(snapshot_dir))
      sys.exit()

    # load model
    model_def = 'models/VGGNet/myDataSet_extended/{}/deploy.prototxt'.format(ssd_size)
    model_weights = 'models/VGGNet/myDataSet_extended/{}/VGG_myDataSet_extended_{}_iter_{}.caffemodel'.format(ssd_size, ssd_size, max_iter)

    net = caffe.Net(model_def,      # defines the structure of the model
                    model_weights,  # contains the trained weights
                    caffe.TEST)     # use test mode (e.g., don't perform dropout)

    # input preprocessing: 'data' is the name of the input blob == net.inputs[0]
    transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
    transformer.set_transpose('data', (2, 0, 1))
    transformer.set_mean('data', np.array([104,117,123])) # mean pixel
    transformer.set_raw_scale('data', 255)  # the reference model operates on images in [0,255] range instead of [0,1]
    transformer.set_channel_swap('data', (2,1,0))  # the reference model has channels in BGR order instead of RGB

    # reshape data blob
    if ssd_size == "SSD_300x300":
        net.blobs['data'].reshape(1, 3, 300, 300)
    else:
        net.blobs['data'].reshape(1, 3, 500, 500)

    # load image
    image = caffe.io.load_image(sys.argv[1])
    #plt.ion()
    #plt.imshow(image)
    #plt.draw()
    #time.sleep(5)

    # preprocess image
    transformed_image = transformer.preprocess('data', image)
    net.blobs['data'].data[...] = transformed_image

    start_time = time.time()
    # Forward pass.
    detections = net.forward()['detection_out']
    print("Time for Forward Step: {}".format(time.time() - start_time))

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

    colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']

    plt.figure()
    #plt.title('VGG_myDataSet_extended_{}'.format(ssd_size))
    currentAxis = plt.gca()
    plt.imshow(image)
    plt.axis('off')

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
        currentAxis.add_patch(plt.Rectangle(*coords, fill=False, edgecolor=color, linewidth=3))
        currentAxis.text(xmin, ymin-30, name, fontsize=21, bbox={'facecolor':'white', 'alpha':0.6})

    plt.draw()

    query = raw_input("Save image (yes/no)? ")
    if query == 'yes':
        plt.savefig('/home/pjoh/Bilder/'+sys.argv[1].split("/")[-1].split(".")[0]
                    +'_{}.jpg'.format(ssd_size), bbox_inches='tight', pad_inches = 0)
