import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from fastslam1_cones import *
import math

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pickle
import pandas as pd
from sklearn.cluster import KMeans
from icp import icp
import matplotlib.colors
from scipy.interpolate import splprep, splev

label_map = {"cone":0, "orangecone": 1, "bluecone":2, "yellowcone":3}

def distance(x1, y1, x2, y2):
    return math.sqrt(math.pow(abs(x1-x2), 2) + math.pow(abs(y1-y2), 2))

def get_side(a, b, p):
  return (p[0]-a[0])*(b[1]-a[1]) - (p[1]-a[1])*(b[0]-a[0])

def filter_landmarks(particles):
  filtered_x = []
  filtered_y = []
  filtered_blue_x = []
  filtered_blue_y = []

  filtered_yellow_x = []
  filtered_yellow_y = []
  car_path = [(hxEst[0,i], hxEst[1,i]) for i in range(0, len(hxEst[1,:]))]

  for i in range(0, len(particles[0].mu)):
    landmark = [particles[0].mu[i, 0], particles[0].mu[i, 1], particles[0].labels[i, 0]]

    car_pos = min(car_path, key=lambda x: distance(landmark[0],landmark[1],x[0],x[1]))
    nearest_index = car_path.index(car_pos)

    dist = distance(landmark[0],landmark[1],car_pos[0],car_pos[1])

    if not dist > 2 and not dist < 0.4:
      filtered_x.append(landmark[0])
      filtered_y.append(landmark[1])
      prev_car_pos = [car_path[nearest_index-1][0], car_path[nearest_index-1][1]]
      
      side = get_side(prev_car_pos, car_pos, landmark)

      if landmark[2] == label_map['bluecone'] and side < 0:
        filtered_blue_x.append(landmark[0])
        filtered_blue_y.append(landmark[1])

      elif landmark[2] == label_map['yellowcone'] and side > 0:
        filtered_yellow_x.append(landmark[0])
        filtered_yellow_y.append(landmark[1])

  filtered_landmarks = np.array([filtered_x[0], filtered_y[0]]) 
  filtered_blue_landmarks = np.array([filtered_blue_x[0], filtered_blue_y[0]])
  filtered_yellow_landmarks = np.array([filtered_yellow_x[0], filtered_yellow_y[0]])
  for i in range(1, len(filtered_x)):
    landmark = np.array([filtered_x[i], filtered_y[i]])
    filtered_landmarks = np.vstack((filtered_landmarks, landmark))

  for i in range(1, len(filtered_blue_x)):
    landmark = np.array([filtered_blue_x[i], filtered_blue_y[i]])
    filtered_blue_landmarks = np.vstack((filtered_blue_landmarks, landmark))

  for i in range(1, len(filtered_yellow_x)):
    landmark = np.array([filtered_yellow_x[i], filtered_yellow_y[i]])
    filtered_yellow_landmarks = np.vstack((filtered_yellow_landmarks, landmark))
  plt.figure()
  plt.scatter(filtered_x, filtered_y)
  return filtered_landmarks, filtered_blue_landmarks, filtered_yellow_landmarks


sns.set(rc={'figure.figsize':(14, 8), 'axes.grid' : False})

"""
Load Data
"""
track_name = "Competition Track 2"

with open('data/particles_competitionTrack2.pickle', 'rb') as handle:
  data = pickle.load(handle)

x_position_list = data[1]
y_position_list = data[2]
theta_position_list = data[3]
center_error_list = data[4]
curvature_list = data[5]
particles = data[8]
hxEst = data[9]

time_delta_list = data[0][0:len(curvature_list)]
velocity_error_list = data[6][0:len(curvature_list)]
velocity_list = data[7][0:len(curvature_list)]
#xEst = data[10]

print(len(hxEst[:,0]), len(hxEst[0,:]))

# Ground truth cone positions
processed_cones_df = pd.read_csv("data/processed_cones_competitionTrack2.csv")
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


#velocity_error_series = pd.Series(velocity_error_list)
#filtered_vel_error = velocity_error_series.ewm(halflife=8).mean()

# Scatter plots
plt.figure()
plt.title("Velocity Error and Curvature vs Time Elapsed")
plt.plot(time_delta_list, velocity_list, label="Velocity Error")
plt.plot(time_delta_list, curvature_list, label="Curvature")
plt.legend()
plt.xlabel("Time(s)")
plt.ylabel("Velocity(m/s) / Curvature(m)")

plt.figure()
plt.title("Center Error and Curvature vs Time Elapsed")
plt.plot(time_delta_list, center_error_list, label="Center Error")
plt.plot(time_delta_list, curvature_list, label="Curvature")
plt.legend()
plt.xlabel("Time(s)")
plt.ylabel("Center Error(m) / Curvature(m)")


# Track Analysis
filtered_landmarks, filtered_blue_landmarks, filtered_yellow_landmarks = filter_landmarks(particles)
clustered_landmarks = KMeans(n_clusters=gt_cones_length).fit(filtered_landmarks)

plt.figure()
plt.scatter(filtered_blue_landmarks[:,0], filtered_blue_landmarks[:,1], s=50, c='b', marker='o')
plt.scatter(filtered_yellow_landmarks[:,0], filtered_yellow_landmarks[:,1], s=50, c='y', marker='o')
#plt.scatter(x_position_list, y_position_list, s=50, c=velocity_list, cmap="Reds")

colors = ["blue", "yellow"]
plt.scatter(x_list, y_list, s=150, c=color_list, cmap=matplotlib.colors.ListedColormap(colors))


transform, dist_errors, icp_iter, icp_indices = icp.icp(gt_cones, clustered_landmarks.cluster_centers_)

track_fig = plt.figure()
plt.scatter(hxEst[0,0:-1], hxEst[1,0:-1], s=50, c=velocity_list, cmap="Reds")
plt.title("Performance on Track")
plt.legend()
cbar = plt.colorbar()
cbar.set_label("Velocity (m/s)")
plt.xlabel("x (m)")
plt.ylabel("y (m)")

plt.scatter(clustered_landmarks.cluster_centers_[:, 0], clustered_landmarks.cluster_centers_[:, 1], s = 100, c = dist_errors*100, cmap="viridis")
cbar2 = plt.colorbar()
cbar2.set_label("Distance error(cm)")

track_fig.tight_layout()
#track_fig.savefig('figures/track_performance.png')

# Cone Filtering Analysis
plt.figure()
plt.title(f"FastSLAM Generated Landmarks in {track_name}")
plt.plot(particles[0].mu[:, 0], particles[0].mu[:, 1], "xb")
plt.xlabel("x (m)")
plt.ylabel("y (m)")

plt.figure()
plt.title(f"Filtered Landmarks in {track_name}")
plt.plot(filtered_landmarks[:, 0], filtered_landmarks[:, 1], "xb")
plt.xlabel("x (m)")
plt.ylabel("y (m)")

plt.figure()
plt.title(f"Clustered Landmarks in {track_name}")
plt.plot(clustered_landmarks.cluster_centers_[:, 0], clustered_landmarks.cluster_centers_[:, 1], "o")
plt.xlabel("x (m)")
plt.ylabel("y (m)")

# Colored cones
colors = ["orange", "blue", "yellow"]

plt.figure()
plt.scatter(filtered_blue_landmarks[:,0], filtered_blue_landmarks[:,1], s=50, c='b', marker='o')
plt.scatter(filtered_yellow_landmarks[:,0], filtered_yellow_landmarks[:,1], s=50, c='y', marker='o')

plt.figure()
plt.scatter(particles[0].mu[:,0], particles[0].mu[:,1], s=50, c=particles[0].labels, cmap=matplotlib.colors.ListedColormap(colors), marker="o")

clustered_blue_landmarks = KMeans(n_clusters=gt_blue_cones_length).fit(filtered_blue_landmarks)
clustered_yellow_landmarks = KMeans(n_clusters=gt_yellow_cones_length).fit(filtered_yellow_landmarks)

plt.figure()
plt.title(f"Clustered colored Landmarks in {track_name}")
plt.plot(clustered_blue_landmarks.cluster_centers_[:, 0], clustered_blue_landmarks.cluster_centers_[:, 1], "o")
plt.plot(clustered_yellow_landmarks.cluster_centers_[:, 0], clustered_yellow_landmarks.cluster_centers_[:, 1], "o")

print(len(gt_blue_cones), len(gt_yellow_cones))

transform, blue_dist_errors, icp_iter, _ = icp.icp(gt_blue_cones, clustered_blue_landmarks.cluster_centers_)
transform, yellow_dist_errors, icp_iter, _ = icp.icp(gt_yellow_cones, clustered_yellow_landmarks.cluster_centers_)

labeled_track_fig = plt.figure()
plt.scatter(x_position_list, y_position_list, s=50, c=velocity_list, cmap="Reds")
plt.title(f"Performance with Segmented Cones in {track_name}")
plt.legend()
cbar = plt.colorbar()
cbar.set_label("Velocity (m/s)")
plt.xlabel("x (m)")
plt.ylabel("y (m)")

plt.scatter(clustered_blue_landmarks.cluster_centers_[:, 0], clustered_blue_landmarks.cluster_centers_[:, 1], s = 100, c = blue_dist_errors*100, cmap="viridis")
plt.scatter(clustered_yellow_landmarks.cluster_centers_[:, 0], clustered_yellow_landmarks.cluster_centers_[:, 1], s = 100, c = yellow_dist_errors*100, cmap="viridis")
cbar2 = plt.colorbar()
cbar2.set_label("Distance error(cm)")

labeled_track_fig.tight_layout()

# Interpolate Track boundaries
plt.figure()
tck, u = splprep(filtered_blue_landmarks.T, u=None, s=0.0, per=1) 
u_new = np.linspace(u.min(), u.max(), 1000)
x_new, y_new = splev(u_new, tck, der=0)

plt.plot(clustered_blue_landmarks.cluster_centers_[:,0], clustered_blue_landmarks.cluster_centers_[:,1], 'ro')
plt.plot(x_new, y_new, 'b--')
plt.title(f"Raw generated boundary in {track_name}")
plt.xlabel("x (m)")
plt.ylabel("y (m)")
plt.show()


