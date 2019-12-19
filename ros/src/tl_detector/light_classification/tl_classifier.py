#from styx_msgs.msg import TrafficLight
#
#
#class TLClassifier(object):
#    def __init__(self):
#        #TODO load classifier
#        pass
#
#    def get_classification(self, image):
#        """Determines the color of the traffic light in the image
#
#        Args:
#            image (cv::Mat): image containing the traffic light
#
#        Returns:
#            int: ID of traffic light color (specified in styx_msgs/TrafficLight)
#
#        """
#        #TODO implement light color prediction
#        return TrafficLight.UNKNOWN

from styx_msgs.msg import TrafficLight
import tensorflow as tf
import numpy as np
# import matplotlib.pyplot as plt
from PIL import Image
from PIL import ImageDraw
from PIL import ImageColor
# import time
# from scipy.stats import norm
import cv2
from keras.models import load_model

#%matplotlib inline
#plt.style.use('ggplot')

# Import everything needed to edit/save/watch video clips
#from moviepy.editor import VideoFileClip
#from IPython.display import HTML

# Frozen inference graph files. NOTE: change the path to where you saved the models.
SSD_GRAPH_FILE = '/home/student/CarND-Capstone/ros/src/tl_detector/light_classification/ssd_mobilenet_v1_coco_11_06_2017/frozen_inference_graph.pb'
# RFCN_GRAPH_FILE = '/home/student/capstone_ws/CarND-Capstone/ros/src/tl_detector/light_classification/rfcn_resnet101_coco_11_06_2017/frozen_inference_graph.pb'
#FASTER_RCNN_GRAPH_FILE = 'faster_rcnn_inception_resnet_v2_atrous_coco_11_06_2017/frozen_inference_graph.pb'


class TLClassifier(object):
    def __init__(self):
        #TODO load classifier
        self.current_light = TrafficLight.UNKNOWN
        self.light_classes = ['Red', 'Green', 'Yellow']

        self.detection_graph = tf.Graph()
        with self.detection_graph.as_default():
            od_graph_def = tf.GraphDef()
            with tf.gfile.GFile(SSD_GRAPH_FILE, 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')
        # detection_graph = load_graph(SSD_GRAPH_FILE)
        # detection_graph = load_graph(RFCN_GRAPH_FILE)
        # detection_graph = load_graph(FASTER_RCNN_GRAPH_FILE)
        self.category_index = {1: {'id': 1, 'name': 'Green'}, 2: {'id': 2, 'name': 'Red'}, 3: {'id': 3, 'name': 'Yellow'}, 4: {'id': 4, 'name': 'off'}}

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        self.sess = tf.Session(graph=self.detection_graph, config=config)
        # The input placeholder for the image.
        # `get_tensor_by_name` returns the Tensor with the associated name in the Graph.
        self.image_tensor = self.detection_graph.get_tensor_by_name('image_tensor:0')

        # Each box represents a part of the image where a particular object was detected.
        self.detection_boxes = self.detection_graph.get_tensor_by_name('detection_boxes:0')

        # Each score represent how level of confidence for each of the objects.
        # Score is shown on the result image, together with the class label.
        self.detection_scores = self.detection_graph.get_tensor_by_name('detection_scores:0')

        # The classification of the object (integer id).
        self.detection_classes = self.detection_graph.get_tensor_by_name('detection_classes:0')

        self.light_model = load_model('/home/student/CarND-Capstone/ros/src/tl_detector/light_classification/tl_model_5.h5')
        self.graph = tf.get_default_graph()


#		pass
    def filter_boxes(self, min_score, boxes, scores, classes):
        """Return boxes with a confidence >= `min_score`"""
        n = len(classes)
        idxs = []
        for i in range(n):
            if scores[i] >= min_score:
                idxs.append(i)

        filtered_boxes = boxes[idxs, ...]
        filtered_scores = scores[idxs, ...]
        filtered_classes = classes[idxs, ...]
        return filtered_boxes, filtered_scores, filtered_classes

    def draw_boxes(self, image, boxes, classes, thickness=4):
        """Draw bounding boxes on the image"""
        draw = ImageDraw.Draw(image)
        for i in range(len(boxes)):
            bot, left, top, right = boxes[i, ...]
            # class_id = int(classes[i])
            # color = COLOR_LIST[class_id]
            draw.line([(left, top), (left, bot), (right, bot), (right, top), (left, top)], width=thickness)
            
    def get_classification(self, image):
        """Determines the color of the traffic light in the image

        Args:
        	image (cv::Mat): image containing the traffic light

        Returns:
        	int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        """
        #TODO implement light color prediction


        # Load a sample image.
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_np = np.expand_dims(np.asarray(image, dtype=np.uint8), 0)
        width, height, _ = image.shape
        # print('image width=', width)
        # print('image height=', height)
        # with tf.Session(graph=self.detection_graph) as sess:
        with self.detection_graph.as_default():
        # Actual detection.
            (boxes, scores, classes) = self.sess.run([self.detection_boxes, self.detection_scores, self.detection_classes], feed_dict={self.image_tensor: image_np})

            # Remove unnecessary dimensions
            boxes = np.squeeze(boxes)
            scores = np.squeeze(scores)
            classes = np.squeeze(classes).astype(np.int32)

            confidence_cutoff = 0.15
            light_count = 0
            other_count = 0
            # # Filter boxes with a confidence score less than `confidence_cutoff`
            boxes, scores, classes = self.filter_boxes(confidence_cutoff, boxes, scores, classes)
            # self.draw_boxes(image, boxes, classes, thickness=4)
        red_lights = []
        for i in range(boxes.shape[0]):
            if classes[i] == 10:
                y1 = np.int32(boxes[i][0]*height)
                y2 = np.int32(boxes[i][2]*height)
                x1 = np.int32(boxes[i][1]*width)
                x2 = np.int32(boxes[i][3]*width)
                # print('x1 =', x1)
                # print('y1 =', y1)
                h = y2 - y1
                w = x2 - x1
                ratio = h/(w+0.01)
                if (h < 20) or (w < 20) or (ratio < 0.1) or (ratio > 10):
                    img_light = np.zeros((1, 32,32, 3), dtype=np.int32)
                else:
                    img_light = cv2.resize(image[y1:y2, x1:x2], (32,32))
                    print('img_light shape=', img_light.shape)
                    img_light = np.expand_dims(np.asarray(img_light, dtype=np.uint8), 0)
                    with self.graph.as_default():
                        light_predict = self.light_model.predict(img_light)
                        light_color = self.light_classes[np.argmax(light_predict)]
                        print('light color = ', light_color)
                        print('classes =', classes[i])
                    red_lights.append(light_color)
                        # self.current_light = light_color
        print('red_lights =', red_lights)
        # idx = next((i for i, v in enumerate(red_lights) if v == 'Red'), None)
        if 'Red' in red_lights:
            self.current_light = TrafficLight.RED
        elif 'Yellow' in red_lights:
            self.current_light = TrafficLight.YELLOW
        elif 'Green' in red_lights:
            self.current_light = TrafficLight.GREEN
        else:
            self.current_light = TrafficLight.UNKNOWN 


        # for i in range(boxes.shape[0]):
        #     if scores is None or scores[i] > confidence_cutoff:
        #         other_count += 1
        #         # class_name = self.category_index[classes[i]]['name']
        #         # if class_name == 'Red':
        #         if classes[i] == 10:
        #             light_count += 1
        # # if light_count < len(boxes.shape[0])/2:
        # if light_count < other_count/2:
        #     self.current_light = TrafficLight.GREEN
        # else:
        #     self.current_light = TrafficLight.RED

            # # The current box coordinates are normalized to a range between 0 and 1.
            # # This converts the coordinates actual location on the image.
            # width, height = image.size
            # box_coords = to_image_coords(boxes, height, width)

            # # Each class with be represented by a differently colored box
            # draw_boxes(image, box_coords, classes)

            # plt.figure(figsize=(12, 8))
            # plt.imshow(image)

        return self.current_light 


	def to_image_coords(boxes, height, width):
		"""
		The original box coordinate output is normalized, i.e [0, 1].
		
		This converts it back to the original coordinate based on the image
		size.
		"""
		box_coords = np.zeros_like(boxes)
		box_coords[:, 0] = boxes[:, 0] * height
		box_coords[:, 1] = boxes[:, 1] * width
		box_coords[:, 2] = boxes[:, 2] * height
		box_coords[:, 3] = boxes[:, 3] * width
		
		return box_coords


	# def load_graph(graph_file):
	# 	"""Loads a frozen inference graph"""
	# 	graph = tf.Graph()
	# 	with graph.as_default():
	# 		od_graph_def = tf.GraphDef()
	# 		with tf.gfile.GFile(graph_file, 'rb') as fid:
	# 			serialized_graph = fid.read()
	# 			od_graph_def.ParseFromString(serialized_graph)
	# 			tf.import_graph_def(od_graph_def, name='')
	# 	return graph
	
	# TODO: Complete this function.
	# The input is an NumPy array.
	# The output should also be a NumPy array.
	# def pipeline(img):
	# 	#pass
	# 	draw_img = Image.fromarray(img)
	# 	boxes, scores, classes = sess.run([detection_boxes, detection_scores, detection_classes], feed_dict={image_tensor: np.expand_dims(img, 0)})
	# 	# Remove unnecessary dimensions
	# 	boxes = np.squeeze(boxes)
	# 	scores = np.squeeze(scores)
	# 	classes = np.squeeze(classes)

	# 	confidence_cutoff = 0.8
	# 	# Filter boxes with a confidence score less than `confidence_cutoff`
	# 	boxes, scores, classes = filter_boxes(confidence_cutoff, boxes, scores, classes)

	# 	# The current box coordinates are normalized to a range between 0 and 1.
	# 	# This converts the coordinates actual location on the image.
	# 	width, height = draw_img.size
	# 	box_coords = to_image_coords(boxes, height, width)

	# 	# Each class with be represented by a differently colored box
	# 	draw_boxes(draw_img, box_coords, classes)
	# 	return np.array(draw_img)
## Colors (one for each class)
#cmap = ImageColor.colormap
#print("Number of colors =", len(cmap))
#COLOR_LIST = sorted([c for c in cmap.keys()])

#
# Utility funcs
#


    



    
#def time_detection(sess, img_height, img_width, runs=10):
#image_tensor = sess.graph.get_tensor_by_name('image_tensor:0')
#detection_boxes = sess.graph.get_tensor_by_name('detection_boxes:0')
#detection_scores = sess.graph.get_tensor_by_name('detection_scores:0')
#detection_classes = sess.graph.get_tensor_by_name('detection_classes:0')
#
## warmup
#gen_image = np.uint8(np.random.randn(1, img_height, img_width, 3))
#sess.run([detection_boxes, detection_scores, detection_classes], feed_dict={image_tensor: gen_image})
#
#times = np.zeros(runs)
#for i in range(runs):
#	t0 = time.time()
#	sess.run([detection_boxes, detection_scores, detection_classes], feed_dict={image_tensor: image_np})
#	t1 = time.time()
#	times[i] = (t1 - t0) * 1000
#return times

#with tf.Session(graph=detection_graph) as sess:
#times = time_detection(sess, 600, 1000, runs=10)
#
## Create a figure instance
#fig = plt.figure(1, figsize=(9, 6))
#
## Create an axes instance
#ax = fig.add_subplot(111)
#plt.title("Object Detection Timings")
#plt.ylabel("Time (ms)")
#
## Create the boxplot
#plt.style.use('fivethirtyeight')
#bp = ax.boxplot(times)



# HTML("""
# <video width="960" height="600" controls>
#   <source src="{0}" type="video/mp4">
# </video>
# """.format('driving.mp4'))

# clip = VideoFileClip('driving.mp4')


    
# with tf.Session(graph=detection_graph) as sess:
# image_tensor = sess.graph.get_tensor_by_name('image_tensor:0')
# detection_boxes = sess.graph.get_tensor_by_name('detection_boxes:0')
# detection_scores = sess.graph.get_tensor_by_name('detection_scores:0')
# detection_classes = sess.graph.get_tensor_by_name('detection_classes:0')

# new_clip = clip.fl_image(pipeline)

# # write to file
# new_clip.write_videofile('result.mp4')

# HTML("""
# <video width="960" height="600" controls>
#   <source src="{0}" type="video/mp4">
# </video>
# """.format('result.mp4'))
