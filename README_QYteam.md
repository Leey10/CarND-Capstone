# Udacity Self-Driving Car Engineer Nanodegree

* * *


## Final Project - System Integration

## Team QY

For this project, we wrote the ROS nodes to implement core functionality of the autonomous vehicle system, including traffic light detection, control, and waypoint following! We tested our code using the simulator. This software system now can autonomously drive the car in the simulator around the track.

## The Team

### Yundong Qi

m18621058126@163.com
Project Contributions: Team Lead, tl_detector.py and tl_classifier.py, waypoint_udapter.py.

### Li Yang

lyang3740@gmail.com
Project contributions: twist_controller.py, dbw_node.py

## Software Architecture
The System Architecture Diagram is shown as in the diagram final-project-graph-v2.png.
We used the same architecture as in the course instruction, and go through the order as suggested in the class. 
*Note: The obstacle detection node is not implemented yet, and will be implemented in next weeks*!

### Waypoint Updater Node

The purpose of this node is to update the target velocity property of each waypoint based on traffic light and obstacle detection data. This node will subscribe to the /base_waypoints, /current_pose, /obstacle_waypoint, and /traffic_waypoint topics, and publish a list of waypoints ahead of the car with target velocities to the /final_waypoints topic.

Every time the vehicle changes its position, we plan a new path. First, we find the closest waypoint to the vehicle’s current position and build a list containing the next 75 waypoints. Next, we look at the upcoming traffic lights. If there are any upcoming red lights, we adjust the speed of the waypoints immediately before the traffic light to slow down the vehicle to stop at the red light. After the light turn to green, the vehicle start to speed up again till the max speed. 
After the path planning, the final list of waypoints is published on the /final_waypoints topic.

### Drive By Wire (DBW) Node

The purpose of this node is to give the control command to the vehicle, throttle, brake, steering, that will drive the vehicle ultimately.

Once the waypoint updater is publishing /final_waypoints, the waypoint_follower node will start publishing messages to the /twist_cmd topic, and the simulator/car will publish a flag signal /vehicle/dbw_enabled. At this point, the dbw_node has all the messages needed. 

And then the DBW node will use above information, then adjust the vehicle’s controls accordingly. 

### Traffic Light Detection Node

The purpose of this node is to detect the upcoming traffic light's status. It will subscribe to the /base_waypoints, /current_pose, /image_color, and publish the lights position and status to /traffic_waypoint topic.

This work was implemented in 2 steps:
First, use the topic /vehicle/traffic_lights to test code systemly in the simulator, as it contains the exact location and status of all traffic lights in simulator, and use this message information can test that the system integration works fine.
Second, subscribe from the /image_color topic the images from the camera, and then use tl_classifier to classify the light status. 

Once have correctly identified the traffic light and determined its position, convert it to a waypoint index and publish it to the /traffic_waypoint topic. The Waypoint Updater node will uses this information to determine if/when the car should slow down to safely stop at upcoming red lights.  

We choose a single shot localizer-classifier solution for the classifier tl_classifier.py.

#### Single-shot classification solution

We used one end-to-end one-shot approach that provide both bounding box and classification results, and used one frozen_inference_graph.pb model file forked from others.
After the boxes and light classes classification, we put all the light states detected into one list, and to be safe, we 
set the light to be RED once there is any red in the list, and then YELLOW, and then GREEN.
We learned the classification code from CarND-Object-Detection-Lab and SiliconCar/CarND-Capstone project, and give one simple straight forward solution for the classifier to make it work faster, due to lack of GPU resources. 

### Waypoint Loader Node

This node was implemented by Udacity. It loads a CSV file that contains all the waypoints along the track and publishes them to the topic /base_waypoints. The CSV can easily be swapped out based on the test location (simulator vs real world).

### Waypoint Follower Node

This node was given to us by Udacity. It parses the list of waypoints to follow and publishes proposed linear and angular velocities to the /twist_cmd topic

#### Throttle Controller

The throttle controller is a simple PID controller that compares the current velocity with the target velocity and adjusts the throttle accordingly. The throttle gains were tuned using trial and error for allowing reasonable acceleration without oscillation around the set-point.

#### Steering Controller

This controller translates the proposed linear and angular velocities into a steering angle based on the vehicle’s steering ratio and wheelbase length. To ensure our vehicle drives smoothly, we cap the maximum linear and angular acceleration rates. The steering angle computed by the controller is also passed through a low pass filter to reduce possible jitter from noise in velocity data.

#### Braking Controller

This is the simplest controller of the three - we simply proportionally brake based on the difference in the vehicle’s current velocity and the proposed velocity. This proportional gain was tuned using trial and error to ensure reasonable stopping distances while at the same time allowing low-speed driving. Despite the simplicity, we found that it works very well.

## Testing on Simulator

We validated our project in the simulator using the following commands:

git clone https://github.com/Qiyd81/CarND-Capstone.git

Then, install python dependencies

cd CarND-Capstone
pip install -r requirements.txt

We ran our simulator in MacOS with code in VM （the method to setup the communication between VM and MacOS host can be found in "VM_environment_setup_method.txt".

To launch the environment, follow the instructions below:

cd CarND-Capstone

cd ros

catkin_make 

source devel/setup.bash

roslaunch launch/styx.launch

Run the simulator (which can be downloaded here: 
https://github.com/udacity/CarND-Capstone/releases/tag/v1.2

## Testing for Site

We have done the testing mostly in the Udacity simulator, and validated the classified function with modified CarND-Object-Detection-Lab code. 
1. We extracted 2562 images from the rosbag file, steps as follows:
 - *Step 1:* Open a terminal window:
      source devel/setup.sh
      roscore
 - *Step 2:* Open another terminal window:
      source devel/setup.sh
      rosbag play -l traffic_light_bag_files/traffic_light_training.bag
 - *Step 3:* Open a third terminal window:
      source devel/setup.sh
      rostopic list //to identify the current topics
      rosrun image_view image_saver image:=/image_color  //where image_color is the topic we want to listen to.
2. We created a video by using the video.py code carried from Behavior_Cloning class. 

3. We modifed CarND-Object-Detection-Lab code, and used the frozen_inference_graph.pb forked from https://github.com/SiliconCar/CarND-Capstone.git to test on the video and also the single images. But we don't find out how to drive the car in the simulator site track. 

## What to do next

1. Write the obstacle_detection node.
2. Sensor fusion the information from other sensors to detect the distance.

