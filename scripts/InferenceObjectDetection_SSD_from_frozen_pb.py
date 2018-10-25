
# coding: utf-8

# In[1]:


import numpy as np
import os
import sys
import tensorflow as tf
from collections import OrderedDict
import re
from PIL import Image
from matplotlib import pyplot as plt
import matplotlib.patches as patches


print("Tensorflow version: {}".format(tf.__version__))


# # Load Model and Graph

# In[2]:


"""
PB: the frozen model
pb.txt: the label map file
"""
#PATH_TO_PB = "/Users/jiankaiwang/devops/Fruit_Recognition/models/ssd_mobilenet/res/m98242/ssd_mobilenet_m_0.1.pb"
#PATH_TO_LABELS = "/Users/jiankaiwang/devops/Fruit_Recognition/models/ssd_mobilenet/image_label.pbtxt"
PATH_TO_PB = "/notebooks/devops/Fruit_Recognition/models/ssd_mobilenet/res/m98242/ssd_mobilenet_m_0.1.pb"
PATH_TO_LABELS = "/notebooks/devops/Fruit_Recognition/models/ssd_mobilenet/image_label.pbtxt"

if not os.path.exists(PATH_TO_PB): raise FileNotFoundError("PB is not found.")
if not os.path.exists(PATH_TO_LABELS): raise FileNotFoundError("Label is not found")


# ## load frozen graph 

# In[3]:


detection_graph = tf.Graph()
with detection_graph.as_default():
    od_graph_def = tf.GraphDef()
    with tf.gfile.GFile(PATH_TO_PB, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')


# ## Show all operation names

# In[4]:


def show_operation_names(count=10):
    """
    Show all operation names in the graph, including "input(image_tensor)" 
    and "output(detection_boxes, detection_scores, detection_classes, num_detections)".
    """
    with detection_graph.as_default():
        with tf.Session() as sess:
            opts = tf.get_default_graph().get_operations()
            for opt in opts[:count]: 
                for output in opt.outputs: print(output.name)
            print("...")
            for opt in opts[-count:]: 
                for output in opt.outputs: print(output.name)
                    
show_operation_names(5)


# ## Load Label file

# In[5]:


category_index = OrderedDict()
with open(PATH_TO_LABELS, "r") as fin:
    tmpData = ""
    for line in fin:
        tmpData += line.strip()
    print(tmpData)
    
    pattern = re.compile("item\s+\{id:\s+(\d*)name:\s+'([\d\S]*)'\}", re.I)
    allItems = pattern.findall(tmpData)
    for item_idx in range(len(allItems)):
        category_index[int(allItems[item_idx][0])] = {'id': int(allItems[item_idx][0]), 'name': str(allItems[item_idx][1])}
    print(category_index)


# ## load image data

# In[6]:


def load_image_into_nparray(image):
    (img_width, img_height) = image.size
    return np.array(image.getdata()).reshape(img_height, img_width, 3).astype(np.uint8)


# # Inference 

# In[7]:


def inference_single_image(image, graph):
    """
    image `image_tensor`: data in type uint8 of shape [None, None, None, 3]
    graph: loaded from frozen model file
    `num_detections`: shape [batch] in float32, the number of valid box per images
    `detection_boxes`: shape [batch, box_number, 4] in float32, (ymin, xmin, ymax, xmax)
    `detection_scores`: shape [batch, box_number] in float32, class scores for each box detections
    `detection_classes`: shape [batch, box_number] in float32, classes for each box detections
    """
    with graph.as_default():
        with tf.Session() as sess:
            # handle input and output tensor
            opts = tf.get_default_graph().get_operations()
            all_tensorflow_names = { output.name for opt in opts for output in opt.outputs }
            tensor_dict = {}
            for key in ['num_detections', 'detection_boxes', 'detection_scores', 'detection_classes']:
                tensor_name = key + ':0'
                if tensor_name in all_tensorflow_names:
                    tensor_dict[key] = tf.get_default_graph().get_tensor_by_name(tensor_name)
                    
            # run for single image            
            # input
            image_tensor = tf.get_default_graph().get_tensor_by_name('image_tensor:0')
            
            # inference
            output_dict = sess.run(tensor_dict, feed_dict={image_tensor: np.expand_dims(image, 0)})
            
            # convert data type float32 to appropriate
            output_dict['num_detections'] = int(output_dict['num_detections'][0])
            output_dict['detection_classes'] = output_dict['detection_classes'][0].astype(np.uint8)
            output_dict['detection_boxes'] = output_dict['detection_boxes'][0]
            output_dict['detection_scores'] = output_dict['detection_scores'][0]
                
        return output_dict


# ## Plot

# In[8]:


def calculateCoord(imgPath, points):
    # (ymin, xmin, ymax, xmax)
    (h, w) = plt.imread(imgPath).shape[:2]
    y1, x1 = h * points[0], w * points[1]
    y2, x2 = h * points[2], w * points[3]
    return int(round(x1)), int(round(y1)), int(round(x2)), int(round(y2))


# In[9]:


def showImage(imgPath, points, class_res):
    im = plt.imread(imgPath)
    
    fig,ax = plt.subplots(1, 1, figsize=(15,15))

    # Display the image
    ax.imshow(im)
    
    color = ['r','g','b']

    # Create a Rectangle patch
    for point in range(len(points)):
        rect = patches.Rectangle((points[point][0], points[point][1]),                                 points[point][2]- points[point][0],                                 points[point][3]- points[point][1],                                 linewidth=3,                                 edgecolor=color[(point+1)%len(color)],                                 facecolor='none',                                  label=class_res[point])

        # Add the patch to the Axes
        ax.add_patch(rect)
    
    ax.legend(loc="lower left")    
    plt.show()


# In[10]:


if __name__ == "__main__":
    #image_path = '/Users/jiankaiwang/devops/Fruit_Recognition/eval/qnap_fruit_val_00003.JPEG'
    image_path = '/notebooks/devops/Fruit_Recognition/eval/qnap_fruit_val_00003.JPEG'
    if not os.path.exists(image_path): raise FileNotFoundError("{} not found.".format(image_path))

    image = Image.open(image_path)   # (width, height)
    image_np = load_image_into_nparray(image)   # reshape to (height, width, 3)
    output_dict = inference_single_image(image_np, detection_graph)
 
    threshold = 9e-1
    odList = np.argwhere(output_dict['detection_scores'] > threshold)   # select highly confident

    point_list = []
    class_res = []
    for odidx in odList:
        print("\t", odidx,               output_dict['detection_scores'][odidx],               output_dict['detection_classes'][odidx][0],               category_index[output_dict['detection_classes'][odidx][0]]['name'])
        point_list.append(calculateCoord(image_path, output_dict['detection_boxes'][odidx][0]))
        class_res.append("{}:{}".format(                category_index[output_dict['detection_classes'][odidx][0]]['name'],                 round(output_dict['detection_scores'][odidx][0], 4)))
        print("\t", point_list)

    showImage(image_path, point_list, class_res)

