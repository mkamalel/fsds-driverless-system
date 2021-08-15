import sys
import os
import math

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from fastslam1_cones import *

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pickle
import pandas as pd
from sklearn.cluster import KMeans
from icp import icp
import matplotlib.colors
from scipy.interpolate import splprep, splev

sns.set(rc={'figure.figsize':(14, 8), 'axes.grid' : False})

"""
Load Data
"""
track_name = "Competition Track 1"

with open('data/tracking_competitionTrack1.pickle', 'rb') as handle:
  data = pickle.load(handle)

cross_track_error_list = data[0]
heading_error_list = data[1]
curvature_list = data[2]
steering_control_list = data[3]
x_position_list = data[4]
y_position_list = data[5]
time_delta_list = data[6]
velocity_list = data[7]
velocity_error_list = data[8]

# Ground truth cone positions
processed_cones_df = pd.read_csv("data/processed_cones_competitionTrack1.csv")
x_list = processed_cones_df['x'].tolist()
y_list = processed_cones_df['y'].tolist()
color_list = processed_cones_df['color'].tolist()

gt_cones = np.array([x_list[0], y_list[0]]) 
gt_blue_cones = np.array([x_list[0], y_list[0]]) 
gt_yellow_cones = np.array([x_list[97], y_list[97]]) 
for i in range(1, len(x_list)):
  cone = np.array([x_list[i], y_list[i]])
  gt_cones = np.vstack((gt_cones, cone))

  if color_list[i] == 0:
    gt_blue_cones = np.vstack((gt_blue_cones, cone))
  elif color_list[i] == 1:
    gt_yellow_cones = np.vstack((gt_yellow_cones, cone))

gt_cones_length = len(gt_cones[:,0])
gt_blue_cones_length = len(gt_blue_cones[:,0])
gt_yellow_cones_length = len(gt_yellow_cones[:,0])

plt.figure()
plt.title("Velocity Error vs Time Elapsed")
plt.plot(time_delta_list, velocity_error_list, label="Velocity Error")
#plt.plot(time_delta_list, curvature_list, label="Curvature")
plt.legend()
plt.xlabel("Time(s)")
plt.ylabel("Velocity(m/s) / Curvature(m)")

plt.figure()
plt.title("Cross Track Error vs Time Elapsed")
plt.plot(time_delta_list, cross_track_error_list, label="Cross Track Error")
#plt.plot(time_delta_list, curvature_list, label="Curvature")
plt.legend()
plt.xlabel("Time(s)")
plt.ylabel("Center Error(m) / Curvature(m)")

plt.figure()
plt.title("Heading Error vs Time Elapsed")
plt.plot(time_delta_list, heading_error_list, label="Heading Error")
#plt.plot(time_delta_list, curvature_list, label="Curvature")
plt.legend()
plt.xlabel("Time(s)")
plt.ylabel("Center Error(m) / Curvature(m)")

print(velocity_list)
track_fig = plt.figure()
plt.scatter(gt_cones[:,0], gt_cones[:,1], s=50, c='b', marker='o')
plt.scatter(x_position_list, y_position_list, s=50, c=velocity_list, cmap="Reds")
plt.title("Velocity throughout Known Track")
plt.legend()
cbar = plt.colorbar()
cbar.set_label("Velocity (m/s)")
plt.xlabel("x (m)")
plt.ylabel("y (m)")



plt.show()