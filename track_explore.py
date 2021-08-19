"""
Track Explorer that creates a map using FastSLAM

Author: Mahmoud Kamaleldin
"""
# Pip packages
import sys
import os
import time
import collections
import time
import numpy as np
import matplotlib.pyplot as plt
import cv2
import math

# Local Libraries
import fsds
import darknet
from fastslam import *

IMG_DIM = 416
FOV_ANGLE = 120*math.pi/180
MAX_DEPTH = 25

class TrackExplorer():
  """
  A class used to track a path using a Stanley Controller

  ...

  Attributes
  ----------
  slam_particles : list of FastSlAM particles
      a list of initiallzed FstSLAM particles
  fsds_client : fsds Client
      client used to communicate with the FSDS simulator

  Methods
  -------
  image_detection_cones(image_data, depthmap)
      Detect cones in an image
  get_cone_positions(detections, depthmap)
      Get cones 3D position using depthmap
  draw_boxes(detections, image, colors, depthmap)
      Draws boxes around cone detections
  bbox2points(bbox)
      Converts darknet bounding box to four corner points
  log_data()
      Log data in list
  track_explore(time_interval, time_elapsed)
      Perform track exploration
  """


  def __init__(self, slam_particles, fsds_client):
    """
    Parameters
    ----------
    slam_particles : list of FastSlAM particles
        a list of initiallzed FstSLAM particles
    fsds_client : fsds Client
        client used to communicate with the FSDS simulator
    """

    self.slam_particles = slam_particles
    self.fsds_client = fsds_client
    
    self.max_curvature = 4

    # Initiallize erros
    self.center_error = 0
    self.curvature = 0
    self.curvature_deque = collections.deque(maxlen=5)

    # Initiallize data logging lists
    self.curvature_list = []
    self.x_position_list = []
    self.y_position_list = []
    self.theta_position_list = []
    self.center_error_list = []
    self.hxEst = np.zeros((STATE_SIZE, 1))

    # Initiallize car state
    self.car_state = "INIT"

    """
    Darknet Initialization
    """
    base_dir = './yolov4-mit'
    weights = f'{base_dir}/yolov4-tiny-cones-416_best.weights'
    data_file = f'{base_dir}/cones.data'
    config_file = f'{base_dir}/yolov4-tiny-cones-416-infer.cfg'
    batch_size = 1
    self.label_map = {"cone":0, "orangecone": 1, "bluecone":2, "yellowcone":3}

    self.network, self.class_names, self.class_colors = darknet.load_network(
        config_file,
        data_file,
        weights,
        batch_size=batch_size
    )

    self.thresh = 0.7

    # Get one image to load darknet engine to cache before moving
    init_image = fsds_client.simGetImages([fsds.ImageRequest(camera_name = 'cam1', image_type = fsds.ImageType.Scene, pixels_as_float = False, compress = True),fsds.ImageRequest(camera_name = 'cam2', image_type = fsds.ImageType.DepthPerspective, pixels_as_float = True, compress = True)], vehicle_name = 'FSCar')


  def image_detection_cones(self, image_data, depthmap):
    """Detect cones and their 3D positions from an image and depth map

    Parameters
    ----------
    image_data : str
        image string
    depthmap : float array
        2D array of depth

    Returns
    -------
    cones : list
        Cone detections and depths 
    """

    # Darknet doesn't accept numpy images.
    # Create one with image we reuse for each detect
    width = IMG_DIM
    height = IMG_DIM
    darknet_image = darknet.make_image(width, height, 3)

    # Convert image from string to byte aray
    nparr = np.fromstring(image_data, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Resize image if input image is not the same dimensions as darknet network
    #image_resized = cv2.resize(image_rgb, (width, height),
                               #interpolation=cv2.INTER_LINEAR)

    # Perform cone detection
    darknet.copy_image_from_bytes(darknet_image, image.tobytes())
    detections = darknet.detect_image(self.network, self.class_names, darknet_image, thresh=self.thresh)
    darknet.free_image(darknet_image)
    
    cones = self.get_cone_positions(detections, depthmap)
    
    return cones

  def get_cone_positions(self, detections, depthmap):
    """Detect cones and their 3D positions from an image and depth map

    Parameters
    ----------
    detections : list of dictionaries
        Cone detections
    depthmap : float array
        2D array of depth

    Returns
    -------
    cones : list
        Cone detections and depths 
    """


    cones = []
    for label, confidence, bbox in detections:
        # Convert detection bounding box to 4 corner coordinates
        left, top, right, bottom = self.bbox2points(bbox)

        # Obtain depth at bounding box coordinates
        cone = depthmap[top:bottom,left:right]
        depth = round(float(min(cone.flatten())),2)

        # Set maximum depth threshold
        if(depth >= MAX_DEPTH):
            continue

        # Trignometrically obtain angle of cone relative to center of car
        if bbox[0] < IMG_DIM/2:
            angle = -((IMG_DIM/2 - bbox[0])/(IMG_DIM/2))*(math.pi/4)
        else:
            angle = ((bbox[0] - IMG_DIM/2)/(IMG_DIM/2))*(math.pi/4)

        x_cone = depth*math.sin(angle)
        y_cone = depth*math.cos(angle)
        cone_range = depth

        cones.append({'r': cone_range, 'b': angle, 'x': x_cone, 'y': y_cone,  'label':self.label_map[label]})

    return cones


  def draw_boxes(self, detections, image, colors, depthmap):
    """Draw boxes on cones in image

    Parameters
    ----------
    detections : list of dictionaries
        Cone detections
    image : float array
        2D NumPy array
    colors : list
        label colors
    depthmap : float array
        2D array of depth

    Returns
    -------
    image : float array
        2D NumPy array
    """


    for label, confidence, bbox in detections:
        left, top, right, bottom = bbox2points(bbox)
        cone = depthmap[top:bottom,left:right]
        depth = round(float(min(cone.flatten())),2)
        cv2.rectangle(image, (left, top), (right, bottom), colors[label], 1)
        cv2.putText(image, f"{label} [{round(float(confidence), 2)}, {depth}m]", (left, top - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, colors[label], 2)
    return image

  def bbox2points(self, bbox):
    """From bounding box yolo format to corner points cv2 rectangle

    Parameters
    ----------
    bbox : list
        yolo format bounding box

    Returns
    -------
    xmin : float
        left
    xmax : float
        right  
    ymin : float
        bottom
    ymax : float
        top    
    """

    x, y, w, h = bbox
    xmin = max(int(round(x - (w / 2))), 0)
    xmax = max(int(round(x + (w / 2))), 0)
    ymin = max(int(round(y - (h / 2))), 0)
    ymax = max(int(round(y + (h / 2))), 0)

    return xmin, ymin, xmax, ymax

  def log_data(self):
    """Return logged data

    Returns
    -------
    data : list of lists
        list of logged data
    """

    data = [0, self.x_position_list, self.y_position_list, self.theta_position_list,
            self.center_error_list, self.curvature_list, 0, 0,
            self.slam_particles, self.hxEst]
    return data

  def steering_pid(self, error, pid, max_steering, dt):
    """Calculates steering PID control

    Parameters
    ----------
    error : float
      steering error
    pid : PID from simple_pid
      steering PID controller
    dt : float
       Time difference since last call

    Returns
    -------
    steering_control : float
      steering control
    """
    if error != 0:
        control = pid(error, dt=dt)
        steering_control = -control*max_steering
        return steering_control
    else:
        return 0

  def track_explore(self, time_interval, time_elapsed):
    """From bounding box yolo format to corner points cv2 rectangle

    Parameters
    ----------
    time_interval : float
        time at last call
    time_elapsed : float
        time elapsed since start

    Returns
    -------
    curvature_ewma : float
        smoothed curvature
    steering_error : float
        center offset of vehicle  
    time_interval : float
        current time
    max_curvature : float
        max curvature threshold
    dt : float
        Time difference since last call   
    """        

    # Get depth and monoscopic images from simulation
    sim_images = self.fsds_client.simGetImages([fsds.ImageRequest(camera_name = 'cam1', image_type = fsds.ImageType.Scene, pixels_as_float = False, compress = True),fsds.ImageRequest(camera_name = 'cam2', image_type = fsds.ImageType.DepthPerspective, pixels_as_float = True, compress = True)], vehicle_name = 'FSCar')

    # Detect cones and their depths from images
    cones_cam = self.image_detection_cones(sim_images[0].image_data_uint8, fsds.get_pfm_array(sim_images[1]))

    # Get kinematic state of the car
    state = self.fsds_client.getCarState()

    lin_vel = state.kinematics_estimated.linear_velocity
    ang_vel = state.kinematics_estimated.angular_velocity
    position = state.kinematics_estimated.position
    theta = fsds.to_eularian_angles(state.kinematics_estimated.orientation)[2]

    car_position = np.array([[position.x_val], [position.y_val], [theta]])
    car_velocity = np.array([[lin_vel.get_length()], [ang_vel.z_val]])

    cone_array = np.array([[cone['x'], cone['y'], cone['label']] for cone in cones_cam])

    # Convert XY cone observations to range and bearing
    cone_observations, _, cone_labels = observation(car_velocity, cone_array)

    dt = time.time() - time_interval
    time_interval = time.time()

    # Perform SLAM
    self.slam_particles = fast_slam1(self.slam_particles, car_position, cone_observations, cone_labels, dt)
    
    """
    Obtain the center error and the curvature of the track ahead
    to perform the low level controls of steering and speed
    """
    right_cones = {'x':[], 'y':[]}
    left_cones = {'x':[], 'y':[]}

    cones_cam = sorted(cones_cam, key=lambda k: k['y']) 

    ORANGE_DETECT = False
    for cone in cones_cam:
        if cone['label'] == self.label_map['bluecone']:
            #plt.scatter(x=cone['x'], y=cone['y'], c="blue")
            left_cones['x'].append(cone['x'])
            left_cones['y'].append(cone['y'])

        elif cone['label'] == self.label_map['yellowcone']:
            #plt.scatter(x=cone['x'], y=cone['y'], c="yellow")
            right_cones['x'].append(cone['x'])
            right_cones['y'].append(cone['y'])

        elif cone['label'] ==  self.label_map['orangecone']:
            if self.car_state == "INIT":
                print("!!!!!!!! INIT")
                self.car_state = "EXPLORE"
            
            elif self.car_state == "EXPLORE":
                if time_elapsed > 10:
                    self.car_state = "FINISHING" 
            
            elif self.car_state == "FINISHING":
                ORANGE_DETECT = True

    if ORANGE_DETECT == False and self.car_state =="FINISHING":
        self.car_state = "TRACK_EXPLORED"

    if len(right_cones['x']) > 1 and len(left_cones['x']) > 1:
        right_cones['x'] = np.array(right_cones['x'])
        right_cones['y'] = np.array(right_cones['y'])
        left_cones['x'] = np.array(left_cones['x'])
        left_cones['y'] = np.array(left_cones['y'])

        # Find the range of x values in a1
        min_right_cones, max_right_cones = min(right_cones['y']), max(right_cones['y'])
        min_left_cones, max_left_cones = min(left_cones['y']), max(left_cones['y'])

        max_cones = min([max_left_cones, max_right_cones])
        # Create an evenly spaced array that ranges from the minimum to the maximum
        # I used 100 elements, but you can use more or fewer. 
        # This will be used as your new x coordinates
        new_right_cones_y = np.linspace(min_right_cones, max_right_cones, 50)
        # Fit a 3rd degree polynomial to your data
        right_coefs = np.polyfit(right_cones['y'], right_cones['x'], 3)
        # Get your new y coordinates from the coefficients of the above polynomial
        new_right_cones_x = np.polyval(right_coefs, new_right_cones_y)

        # Repeat for array 2:
        # Find the range of x values in a1
        new_left_cones_y = np.linspace(min_left_cones, max_left_cones, 50)
        left_coefs = np.polyfit(left_cones['y'], left_cones['x'], 3)
        new_left_cones_x = np.polyval(left_coefs, new_left_cones_y)

        midx = [np.mean([new_right_cones_x[i], new_left_cones_x[i]]) for i in range(50)]
        midy = [np.mean([new_right_cones_y[i], new_left_cones_y[i]]) for i in range(50)]

        # Obtain curvature as x offset ahead from the center
        self.curvature = abs(np.mean(midx[40:50])-midx[0])

        # Center error is current offset from center of boundaries
        self.center_error = midx[0]

    # if only 1 cone of each color is detected, center error is the center of them
    elif len(right_cones['x']) > 0 and len(left_cones['x']) > 0:
        self.center_error = (right_cones['x'][0] + left_cones['x'][0])/2

    # Max center error if only 1 color cone can be detected
    elif len(right_cones['x']) == 0 and len(left_cones['x']) > 0:
        self.center_error = 3
        self.curvature = self.max_curvature

    elif len(right_cones['x']) > 0 and len(left_cones['x']) == 0:
        self.center_error = -3
        self.curvature = self.max_curvature

    steering_error = self.center_error
    self.curvature_deque.append(self.curvature)

    # EWMA to smooth curvature
    curvature_ewma = fsds.ewma_vectorized(np.array(self.curvature_deque), 0.2)
    print("CURVATURE: ", curvature_ewma)

    # Data logging
    self.theta_position_list.append(theta)
    self.center_error_list.append(steering_error)
    self.x_position_list.append(position.x_val)
    self.y_position_list.append(position.y_val)
    self.curvature_list.append(curvature_ewma[-1])

    xEst = calc_final_state(self.slam_particles)
    x_state = xEst[0: STATE_SIZE]
    self.hxEst = np.hstack((self.hxEst, x_state))

    return curvature_ewma[-1], steering_error, time_interval, self.max_curvature, dt