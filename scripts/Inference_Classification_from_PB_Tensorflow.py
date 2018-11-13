#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().system('pip install opencv-python')
get_ipython().system('apt update && apt install -y libsm6 libxext6 libfontconfig1 libxrender1')


# In[2]:


import numpy as np
import os
import sys
import tensorflow as tf
from collections import OrderedDict
import re
from PIL import Image
from matplotlib import pyplot as plt
import matplotlib.patches as patches
import pickle
import cv2

print("Tensorflow version: {}".format(tf.__version__))


# # Load Model and Graph

# In[3]:


"""
PB: the frozen model
pb.txt: the label map file
"""
PB_NAME = "ssd_inception_v2_m_0.2.1"
#PATH_TO_PB = "/Users/jiankaiwang/devops/Fruit_Recognition/models/ssd_inception_v2/res/m200000/{}.pb".format(PB_NAME)
#PATH_TO_LABELS = "/Users/jiankaiwang/devops/Fruit_Recognition/models/ssd_inception_v2/image_label.pbtxt"
PATH_TO_PB = "/notebooks/devops/Auxiliary_Operations/model/inception_single_0.1/inception_single_0.1.pb"
PATH_TO_LABELS = "/notebooks/devops/Auxiliary_Operations/model/inception_single_0.1/inception_single_0.1.txt"

PATH_TO_PICKLE_DIR = "/notebooks/devops/Auxiliary_Operations/model/inception_single_0.1"

if not os.path.exists(PATH_TO_PB): raise FileNotFoundError("PB is not found.")
if not os.path.exists(PATH_TO_LABELS): raise FileNotFoundError("Label is not found")
if not os.path.exists(PATH_TO_PICKLE_DIR): raise FileNotFoundError("PICKLE PATH is not found")


# ## load frozen graph 

# In[4]:


detection_graph = tf.Graph()
with detection_graph.as_default():
    od_graph_def = tf.GraphDef()
    with tf.gfile.GFile(PATH_TO_PB, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')


# ## Show all operation names

# In[5]:


def show_operation_names(count=10):
    with detection_graph.as_default():
        with tf.Session() as sess:
            opts = tf.get_default_graph().get_operations()
            for opt in opts[:count]: 
                for output in opt.outputs: print(output.name)
            print("...")
            for opt in opts[-count:]: 
                for output in opt.outputs: print(output.name)
                    
show_operation_names(10)


# ## Load Label file

# In[6]:


category_index = OrderedDict()
name_indx = OrderedDict()
count = 1
with open(PATH_TO_LABELS, "r") as fin:
    tmpData = ""
    for line in fin:
        tmpData = line.strip()
        category_index[count] = tmpData
        name_indx[tmpData] = count
        count += 1
    print(category_index)
    print(name_indx)


# # Inference 

# In[7]:


def inference_single_image(image, graph):
    with graph.as_default():
        with tf.Session() as sess:
            # handle input and output tensor
            opts = tf.get_default_graph().get_operations()
            all_tensorflow_names = { output.name for opt in opts for output in opt.outputs }
            tensor_dict = {}
            for key in ['final_result']:
                tensor_name = key + ':0'
                if tensor_name in all_tensorflow_names:
                    tensor_dict[key] = tf.get_default_graph().get_tensor_by_name(tensor_name)
                    
            # run for single image            
            # input
            image_tensor = tf.get_default_graph().get_tensor_by_name('Placeholder:0')
            
            # inference
            output_dict = sess.run(tensor_dict, feed_dict={image_tensor: np.expand_dims(image, 0)})
            
            # convert data type float32 to appropriate
            output_dict['final_result'] = output_dict['final_result']
            
        return output_dict


# In[8]:


def single_image(imagePath):
    image_path = imagePath
    if not os.path.exists(image_path): raise FileNotFoundError("{} not found.".format(image_path))
        
    image = cv2.imread(image_path)
    image = cv2.resize(image, (299, 299), interpolation=cv2.INTER_CUBIC)
    image_np = image[:,:,::-1]
    output_dict = inference_single_image(image_np, detection_graph)
    
    return output_dict


# In[9]:


#image_path = '/Users/jiankaiwang/devops/Fruit_Recognition/eval/qnap_fruit_val_00003.JPEG'
image_path = '/notebooks/devops/Auxiliary_Operations/data/IMG_1562_single_crop/screw/IMG_1562_frame_68_597213347968410b3e1b212b7c93a320767da9a1.jpg'
output_dict = single_image(image_path)
print(output_dict)


# In[10]:


cls_idx = int(np.argmax(output_dict['final_result'], axis=1) + 1)
print("Classification: {}".format(category_index[cls_idx]))


# In[ ]:




