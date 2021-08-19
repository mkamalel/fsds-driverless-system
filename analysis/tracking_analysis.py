import sys
import os
import math

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from fastslam import *

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pickle
import pandas as pd
from sklearn.cluster import KMeans
from icp import icp
import matplotlib.colors as colors
from scipy.interpolate import splprep, splev

sns.set(rc={'figure.figsize':(14, 8), 'axes.grid' : False, 'axes.facecolor':'silver'})

"""
Load Data
"""
track_name = "Competition Track 1"

with open('data/tracking_competitionTrack1.pickle', 'rb') as handle:
  data = pickle.load(handle)

cross_track_error_list = data[0]
heading_error_list = [heading*180/np.pi for heading in data[1]]
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
plt.ylabel("Velocity Error(m/s)")

plt.figure()
plt.title("Cross Track Error vs Time Elapsed")
plt.plot(time_delta_list, cross_track_error_list, label="Cross Track Error")
#plt.plot(time_delta_list, curvature_list, label="Curvature")
plt.legend()
plt.xlabel("Time(s)")
plt.ylabel("Cross Track Error(m)")

plt.figure()
plt.title("Heading Error vs Time Elapsed")
plt.plot(time_delta_list, heading_error_list, label="Heading Error")
#plt.plot(time_delta_list, curvature_list, label="Curvature")
plt.legend()
plt.xlabel("Time(s)")
plt.ylabel("Heading Error(deg)")


track_fig = plt.figure()
plt.scatter(gt_cones[:,0], gt_cones[:,1], s=50, c='black', marker='o')
plt.scatter(x_position_list, y_position_list, s=20, c=velocity_list, cmap="Reds")
plt.title("Velocity throughout Known Track")
plt.legend()
cbar = plt.colorbar()
cbar.set_label("Velocity(m/s)")
plt.xlabel("x (m)")
plt.ylabel("y (m)")


track_fig = plt.figure()
plt.scatter(gt_cones[:,0], gt_cones[:,1], s=50, c='black', marker='o')
plt.scatter(x_position_list[200:], y_position_list[200:], s=20, c=velocity_error_list[200:], cmap="RdYlBu", norm=colors.TwoSlopeNorm(0.0))
plt.title("Velocity Error throughout Known Track")
plt.legend()
cbar = plt.colorbar()
cbar.set_label("Velocity Error(m/s)")
plt.xlabel("x (m)")
plt.ylabel("y (m)")


track_fig = plt.figure()
plt.scatter(gt_cones[:,0], gt_cones[:,1], s=50, c='black', marker='o')
plt.scatter(x_position_list, y_position_list, s=20, c=cross_track_error_list, cmap="RdYlBu", norm=colors.TwoSlopeNorm(0.0))
plt.title("Cross Track Error throughout Known Track")
plt.legend()
cbar = plt.colorbar()
cbar.set_label("Cross Track Error(m)")
plt.xlabel("x (m)")
plt.ylabel("y (m)")

track_fig = plt.figure()
plt.scatter(gt_cones[:,0], gt_cones[:,1], s=50, c='black', marker='o')
plt.scatter(x_position_list, y_position_list, s=20, c=heading_error_list, cmap="RdYlBu", norm=colors.TwoSlopeNorm(0.0))
plt.title("Heading Error throughout Known Track")
plt.legend()
cbar = plt.colorbar()
cbar.set_label("Heading Error(deg)")
plt.xlabel("x (m)")
plt.ylabel("y (m)")

plt.show()