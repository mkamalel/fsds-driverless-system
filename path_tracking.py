"""
Path tracker using Stanley Controller

Author: Mahmoud Kamaleldin
"""
import sys
import os
import time
import numpy as np
import math

#sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import fsds
import trackmap
import stanley


class PathTracker():
  def __init__(self, slam_particles, hxEst, fsds_client):
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

  def track_path(self):
    state = self.fsds_client.getCarState()

    lin_vel = state.kinematics_estimated.linear_velocity
    ang_vel = state.kinematics_estimated.angular_velocity
    position = state.kinematics_estimated.position
    theta = fsds.to_eularian_angles(state.kinematics_estimated.orientation)[2]

    self.stanley_state.x = position.x_val 
    self.stanley_state.y = position.y_val 
    self.stanley_state.yaw = theta
    self.stanley_state.v = np.hypot(lin_vel.x_val, lin_vel.y_val)

    delta, current_target_idx = stanley.stanley_control(self.stanley_state, self.center_interp[:,0], self.center_interp[:,1])
    delta = np.clip(delta, -stanley.max_steer, stanley.max_steer)
    #stanley_state.yaw += stanley_state.v / stanley.L * np.tan(delta) * dt
    #stanley_state.yaw = stanley.normalize_angle(stanley_state.yaw)
    L = 0.9
    fx = position.x_val + L * np.cos(theta)
    fy = position.y_val + L * np.sin(theta)


    #print("stanley: ", delta, "yaw: ", stanley_state.yaw)

    dx = [fx - icx for icx in self.center_interp[:,0]]
    dy = [fy - icy for icy in self.center_interp[:,1]]
    d = np.hypot(dx, dy)
    target_idx = np.argmin(d)

    look_ahead = min(int(self.stanley_state.v*2), 20)
    curve_idx = (target_idx+look_ahead)%10000
    curvature_ewma = self.curvature_array[curve_idx]

    steering_control = -delta/math.pi

    return steering_control, curvature_ewma, self.max_curvature