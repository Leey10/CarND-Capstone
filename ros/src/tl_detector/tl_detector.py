#!/usr/bin/env python
import rospy
from std_msgs.msg import Int32
from geometry_msgs.msg import PoseStamped, Pose
from styx_msgs.msg import TrafficLightArray, TrafficLight
from styx_msgs.msg import Lane
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from light_classification.tl_classifier import TLClassifier
import tf
import cv2
import yaml
from scipy.spatial import KDTree
import time
import os

# para for checking the light status is stable
STATE_COUNT_THRESHOLD = 3

# flag for testing the code with given light status
TEST_ENABLED = False

# flag for saving images from the simulator for training the classifier
SAVE_IMAGES = False

class TLDetector(object):
    def __init__(self):
        rospy.init_node('tl_detector')

        self.pose = None
        self.waypoints = None
        self.camera_image = None
        self.lights = []
        self.waypoints_2d = None
        self.waypoints_tree = None

        rospy.Subscriber('/current_pose', PoseStamped, self.pose_cb)
        rospy.Subscriber('/base_waypoints', Lane, self.waypoints_cb)

        '''
        /vehicle/traffic_lights provides you with the location of the traffic light in 3D map space and
        helps you acquire an accurate ground truth data source for the traffic light
        classifier by sending the current color state of all traffic lights in the
        simulator. When testing on the vehicle, the color state will not be available. You'll need to
        rely on the position of the light and the camera image to predict it.
        '''
        rospy.Subscriber('/vehicle/traffic_lights', TrafficLightArray, self.traffic_cb)
        rospy.Subscriber('/image_color', Image, self.image_cb)

        config_string = rospy.get_param("/traffic_light_config")
        self.config = yaml.load(config_string)

        self.upcoming_red_light_pub = rospy.Publisher('/traffic_waypoint', Int32, queue_size=1)

        self.bridge = CvBridge()
        self.light_classifier = TLClassifier()
        self.listener = tf.TransformListener()

        self.state = TrafficLight.UNKNOWN
        self.last_state = TrafficLight.UNKNOWN
        self.last_wp = -1
        self.state_count = 0
        self.has_image = None 

        rospy.spin()

    def pose_cb(self, msg):
        self.pose = msg

    def waypoints_cb(self, waypoints):
        # TODO: Implement
        # pass
        #self.base_waypoints = waypoints
        self.waypoints = waypoints
        if not self.waypoints_2d:
            self.waypoints_2d = [[waypoint.pose.pose.position.x, waypoint.pose.pose.position.y] for waypoint in waypoints.waypoints]
            self.waypoints_tree = KDTree(self.waypoints_2d) 

    def traffic_cb(self, msg):
        self.lights = msg.lights

        ## Code for first step to test system integration without real light detection function.
        # light_wp, state = self.process_traffic_lights()
        # if state == TrafficLight.RED:
        #     self.upcoming_red_light_pub.publish(Int32(light_wp))

    def image_cb(self, msg):
        """Identifies red lights in the incoming camera image and publishes the index
            of the waypoint closest to the red light's stop line to /traffic_waypoint

        Args:
            msg (Image): image from car-mounted camera

        """
        self.has_image = True
        self.camera_image = msg
        light_wp, state = self.process_traffic_lights()

        '''
        Publish upcoming red lights at camera frequency.
        Each predicted state has to occur `STATE_COUNT_THRESHOLD` number
        of times till we start using it. Otherwise the previous stable state is
        used.
        '''
        if self.state != state:
            self.state_count = 0
            self.state = state
        elif self.state_count >= STATE_COUNT_THRESHOLD:
            self.last_state = self.state
            light_wp = light_wp if state == TrafficLight.RED else -1
            self.last_wp = light_wp
            self.upcoming_red_light_pub.publish(Int32(light_wp))
        else:
            self.upcoming_red_light_pub.publish(Int32(self.last_wp))
        self.state_count += 1

    def get_closest_waypoint(self, x, y):
        # x = self.pose.pose.position.x
        # y = self.pose.pose.position.y
        closest_idx = self.waypoints_tree.query([x,y], 1)[1]

        # closest_coord = self.waypoints_2d[closest_idx]
        # prev_coord = self.waypoints_2d[closest_idx -1]

        # cl_vect = np.array(closest_coord)
        # prev_vect = np.array(prev_coord)
        # pos_vect = np.array([x,y])

        # val = np.dot(cl_vect - prev_vect, pos_vect -cl_vect)

        # if val > 0:
        #     closest_idx = (closest_idx + 1) % len(self.waypoints_2d)
        return closest_idx

    def to_string(self, state):
        out = 'unknown'
        if state == TrafficLight.GREEN:
            out = 'green'
        elif state == TrafficLight.YELLOW:
            out = 'yellow'
        elif state == TrafficLight.RED:
            out = 'red'
        return out

    def get_light_state(self, light):
        """Determines the current color of the traffic light

        Args:
        light (TrafficLight): light to classify

        Returns:
        int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        """

        #for testing, just return the light state
        if TEST_ENABLED:
            ## code to save images from the simulator for training the classifier
            # cv_image = self.bridge.imgmsg_to_cv2(self.camera_image, "bgr8")
            # if SAVE_IMAGES:
            #     save_file = "/home/Newdisk/imgs/{}-{:.0f}.jpeg".format(self.to_string(light.state), (time.time()*100))
            #     cv2.imwrite(save_file, cv_image)
            return light.state
        else:
            if(not self.has_image):
                self.prev_light_loc = None
                return False

            cv_image = self.bridge.imgmsg_to_cv2(self.camera_image, "bgr8")
            light_status = self.light_classifier.get_classification(cv_image)
            ## code to save images from the simulator for training the classifier
            # if SAVE_IMAGES:
            #     save_file = "/home/Newdisk/imgs/{}-{:.0f}.jpeg".format(self.to_string(light_status), (time.time()*100))
            #     cv2.imwrite(save_file, cv_image)
            #Get classification
            return light_status

    def process_traffic_lights(self):
        """Finds closest visible traffic light, if one exists, and determines its
        location and color

        Returns:
        int: index of waypoint closes to the upcoming stop line for a traffic light (-1 if none exists)
        int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        """
        #light = None
        closest_light = None
        line_wp_idx = None

        # List of positions that correspond to the line to stop in front of for a given intersection
        stop_line_positions = self.config['stop_line_positions']
        if(self.pose):
        #car_position = self.get_closest_waypoint(self.pose.pose)
            car_wp_idx = self.get_closest_waypoint(self.pose.pose.position.x, self.pose.pose.position.y)

            #TODO find the closest visible traffic light (if one exists)
            diff = len(self.waypoints.waypoints)
            for i, light in enumerate(self.lights):
                line = stop_line_positions[i]
                temp_wp_idx = self.get_closest_waypoint(line[0], line[1])
                #find closest stop line wapoint index
                d = temp_wp_idx - car_wp_idx
                if d >= 0 and d < diff:
                    diff = d
                    closest_light = light
                    line_wp_idx = temp_wp_idx

        if closest_light:
            state = self.get_light_state(closest_light)
            return line_wp_idx, state


        #        if light:
        #            state = self.get_light_state(light)
        #            return light_wp, state
        #        self.waypoints = None
        return -1, TrafficLight.UNKNOWN

if __name__ == '__main__':
    try:
        TLDetector()
    except rospy.ROSInterruptException:
        rospy.logerr('Could not start traffic node.')


