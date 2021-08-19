"""
Path tracker using Stanley Controller

Author: Mahmoud Kamaleldin
"""
# Pip packages
import sys
import os
import time
import numpy as np
import math

# Local packages
import fsds
import trackmap
import stanley


class PathTracker():
  """
  A class that defines a map of track using cone observations

  ...

  Attributes
  ----------
  slam_particles : list of FastSlAM particles
      a list of initiallzed FstSLAM particles
  hxEst : 2 by ND array
    ND array of XY car throughout track
  fsds_client : fsds Client
      client used to communicate with the FSDS simulator

  Methods
  -------
  track_path()
      Perform path tracking
  log_data()
      Log data in list
  """

  def __init__(self, slam_particles, hxEst, fsds_client):
    """
    Attributes
    ----------
    slam_particles : list of FastSlAM particles
        a list of initiallzed FstSLAM particles
    hxEst : 2 by ND array
      ND array of XY car throughout track
    fsds_client : fsds Client
        client used to communicate with the FSDS simulator
    """

    self.fsds_client = fsds_client

    track_map = trackmap.TrackMap(hxEst, slam_particles[0].mu, slam_particles[0].labels)

    filtered_blue_landmarks, filtered_yellow_landmarks = track_map.filter_landmarks()

    blue_loop = track_map.find_path(filtered_blue_landmarks)
    yellow_loop = track_map.find_path(filtered_yellow_landmarks)

    blue_interp = track_map.gen_boundary(blue_loop)
    yellow_interp = track_map.gen_boundary(yellow_loop)

    self.center_interp = track_map.get_center_path(blue_interp, yellow_interp)

    self.curvature_array = track_map.get_track_curvature(self.center_interp)
    self.max_curvature = max(self.curvature_array)-0.05

    #track_map.plot_track(blue_interp, yellow_interp, center_interp, curvature)

    self.stanley_state = stanley.State(x=0.0, y=0.0, yaw=np.radians(0.0), v=0.0)

    # Initiallize data logging lists
    self.cross_track_error_list = []
    self.heading_error_list = []
    self.curvature_list = []
    self.steering_control_list = []
    self.x_position_list = []
    self.y_position_list = []

  def log_data(self):
    """Return logged data

    Returns
    -------
    data : list of lists
        list of logged data
    """
    return [self.cross_track_error_list, self.heading_error_list, self.curvature_list, self.steering_control_list, self.x_position_list, self.y_position_list, 0, 0, 0]

  def track_path(self):
    """Perform path tracking

    Returns
    -------
    steering_control : float
        control input for steering 
    curvature_ewma : float
        smoothed curvature
    max_curvature : float
        max curvature threshold
    """   

    state = self.fsds_client.getCarState()

    lin_vel = state.kinematics_estimated.linear_velocity
    ang_vel = state.kinematics_estimated.angular_velocity
    position = state.kinematics_estimated.position
    theta = fsds.to_eularian_angles(state.kinematics_estimated.orientation)[2]

    self.stanley_state.x = position.x_val 
    self.stanley_state.y = position.y_val 
    self.stanley_state.yaw = theta
    self.stanley_state.v = np.hypot(lin_vel.x_val, lin_vel.y_val)

    delta, current_target_idx, theta_e, theta_d = stanley.stanley_control(self.stanley_state, self.center_interp[:,0], self.center_interp[:,1])
    delta = np.clip(delta, -stanley.max_steer, stanley.max_steer)

    L = 0.9
    fx = position.x_val + L * np.cos(theta)
    fy = position.y_val + L * np.sin(theta)

    dx = [fx - icx for icx in self.center_interp[:,0]]
    dy = [fy - icy for icy in self.center_interp[:,1]]
    d = np.hypot(dx, dy)
    target_idx = np.argmin(d)

    # Get curvature ahead based on speed of the car
    look_ahead = min(int(self.stanley_state.v*2), 20)
    curve_idx = (target_idx+look_ahead)%10000
    curvature_ewma = self.curvature_array[curve_idx]

    steering_control = -delta/math.pi

    # Data logging
    self.cross_track_error_list.append(theta_d)
    self.heading_error_list.append(theta_e)
    self.curvature_list.append(curvature_ewma)
    self.steering_control_list.append(steering_control)
    self.x_position_list.append(position.x_val)
    self.y_position_list.append(position.y_val)

    return steering_control, curvature_ewma, self.max_curvature