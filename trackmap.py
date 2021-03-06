"""
Track Map to process XY positions of cones to a map

Author: Mahmoud Kamaleldin
"""
# Pip packages
import math
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pickle
from scipy.interpolate import splprep, splev

# Local packages
from fastslam import *

def distance(x1, y1, x2, y2):
  """Get euclidean distance between two points"""
  return math.sqrt(math.pow(abs(x1-x2), 2) + math.pow(abs(y1-y2), 2))

def get_side(a, b, p):
  """Get which side cone is on relative to vehicle absolute coordinates"""
  return (p[0]-a[0])*(b[1]-a[1]) - (p[1]-a[1])*(b[0]-a[0])

class TrackMap():
  """
  A class that defines a map of track using cone observations

  ...

  Attributes
  ----------
  lap_path : 2 by ND array
    ND array of XY car throughout track
  landmarks : 2 by ND array
    ND array of XY cone coordinates
  labels : 1 by ND array
    ND array of cone labels

  Methods
  -------
  filter_landmarks()
      Perform landmark filtering algorithm
  find_path(coords, start=None)
      Order list of coordinates using nearest neighbor algorithm
  gen_boundary(loop)
      Interpolate boundary
  get_center_path(blue_interp, yellow_interp)
      Interpolate center path between two boundaries
  get_track_curvature(center_interp)
      Get curvature of path
  plot_track(self, blue_interp, yellow_interp, center_interp, curvature, track_name)
      Plot interpolated track and center path
  """

  label_map = {"cone":0, "orangecone": 1, "bluecone":2, "yellowcone":3}

  def __init__(self, lap_path, landmarks, labels):
    """
    Attributes
    ----------
    lap_path : 2 by ND array
      ND array of XY car throughout track
    landmarks : 2 by ND array
      ND array of XY cone coordinates
    labels : 1 by ND array
      ND array of cone labels
    """

    self.lap_path = lap_path
    self.landmarks = landmarks
    self.landmark_labels = labels

  def filter_landmarks(self):
    """Perform landmark filtering algorithm

    Returns
    -------
    filtered_blue_landmarks : 2 by ND array
      ND array of blue cone coordinates
    filtered_yellow_landmarks : 2 by ND array
      ND array of yellow cone coordinates
    """

    filtered_blue_landmarks = np.empty((0,2), np.float32)
    filtered_yellow_landmarks = np.empty((0,2), np.float32)

    car_path = [(self.lap_path[0,i], self.lap_path[1,i]) for i in range(0, len(self.lap_path[1,:]))]

    # Iterate over cones
    for i in range(0, len(self.landmarks)):
      landmark = [self.landmarks[i, 0], self.landmarks[i, 1]]
      label = self.landmark_labels[i, 0]

      # Get closest point on car path to cone
      car_pos = min(car_path, key=lambda x: distance(landmark[0],landmark[1],x[0],x[1]))
      nearest_index = car_path.index(car_pos)

      dist = distance(landmark[0],landmark[1],car_pos[0],car_pos[1])

      if not dist > 2 and not dist < 0.4:
        prev_car_pos = [car_path[nearest_index-1][0], car_path[nearest_index-1][1]]
        
        # Filter blue cones on the right and yellow cones on the left
        side = get_side(prev_car_pos, car_pos, landmark)

        if label == self.label_map['bluecone'] and side < 0:
          # side less than 0, cone on the left
          filtered_blue_landmarks = np.vstack((filtered_blue_landmarks, np.array(landmark)))

        elif label == self.label_map['yellowcone'] and side > 0:
          # side greater than 0, cone on the right
          filtered_yellow_landmarks = np.vstack((filtered_yellow_landmarks, np.array(landmark)))

    return filtered_blue_landmarks, filtered_yellow_landmarks

  def find_path(self, coords, start=None):
    """Detect cones and their 3D positions from an image and depth map

    Parameters
    ----------
    coords : 2 by ND array
      ND array of XY coordinates
    start : 2 by 1 array
      XY coordinate to start from

    Returns
    -------
    path : 2 by ND array
      Ordered paht array
    """

    coords = coords.tolist()

    if start is None:
      start = min(coords, key=lambda x: distance(0, 0, x[0], x[1]))

    # Order points by nearest neighbor
    pass_by = coords
    path = [start]
    pass_by.remove(start)
    while pass_by:
          nearest = min(pass_by, key=lambda x: distance(path[-1][0], path[-1][1], x[0], x[1]))
          path.append(nearest)
          pass_by.remove(nearest)

    return np.array(path)

  def gen_boundary(self, loop):
    """Interpolate closed boundary from loop coordinates

    Parameters
    ----------
    loop : 2 by ND array
      ND array of XY coordinates

    Returns
    -------
    interp : 2 by ND array
      interpolated boundary
    """

    tck, u = splprep(loop.T, u=None, s=10, per=1) 
    u_new = np.linspace(u.min(), u.max(), 1000)

    x_new, y_new = splev(u_new, tck, der=0)
    interp = np.array(list(zip(x_new, y_new)))    

    return interp

  def get_center_path(self, blue_interp, yellow_interp):
    """Interpolate center path from blue and yellow boundaries

    Parameters
    ----------
    blue_interp : 2 by ND array
      ND array of blue boundary
    yellow_interp : 2 by ND array
      ND array of yellow boundary

    Returns
    -------
    center_interp : 2 by ND array
      interpolated center path
    """

    center_path = np.empty((0,2), np.float32)

    # Find nearest blue and yellow points and take their mean as the center path
    for i in range(len(blue_interp[:,0])):
      nearest_yellow = min(yellow_interp, key=lambda x: distance(blue_interp[i,0], blue_interp[i,1], x[0], x[1]))

      mid = np.array([np.mean([nearest_yellow[0], blue_interp[i,0]]), np.mean([nearest_yellow[1], blue_interp[i,1]])])
      center_path = np.vstack((center_path, mid))

    tck, u = splprep(center_path.T, u=None, s=50, per=1) 
    u_new = np.linspace(u.min(), u.max(), 10000)

    center_x_new, center_y_new = splev(u_new, tck, der=0)
    center_interp = np.array(list(zip(center_x_new, center_y_new)))

    return center_interp

  def get_track_curvature(self, center_interp):
    """Calculate track curvature

    Parameters
    ----------
    center_interp : 2 by ND array
      ND array of interpolated center path

    Returns
    -------
    curvature : ND array
      calculated curvature along center path
    """

    #first derivatives 
    dx = np.gradient(np.array(center_interp[:,0]))
    dy = np.gradient(np.array(center_interp[:,1]))

    #second derivatives 
    d2x = np.gradient(dx)
    d2y = np.gradient(dy)

    #calculation of curvature from the typical formula
    curvature = np.asarray(np.abs(dx * d2y - d2x * dy) / (dx * dx + dy * dy)**1.5)   
    
    return curvature 


  def plot_track(self, blue_interp, yellow_interp, center_interp, curvature, track_name):
    """Plot generated track map

    Parameters
    ----------
    blue_interp : 2 by ND array
      ND array of blue boundary
    yellow_interp : 2 by ND array
      ND array of yellow boundary
    center_interp : 2 by ND array
      ND array of interpolated center path
    curvature : ND array
      calculated curvature along center path
    track_name : str
      name of the track
    """

    plt.figure()
    plt.plot(filtered_blue_landmarks[:,0], filtered_blue_landmarks[:,1], 'bo')
    plt.plot(blue_interp[:,0], blue_interp[:,1], 'b--')
    plt.plot(filtered_yellow_landmarks[:,0], filtered_yellow_landmarks[:,1], 'yo')
    plt.plot(yellow_interp[:,0], yellow_interp[:,1], 'y--')
  
    plt.scatter(center_interp[:,0], center_interp[:,1], s=5, c=curvature, cmap="magma_r", marker="o")
    plt.title(f"Interpolated center path with curvature in {track_name}")
    plt.xlabel("x (m)")
    plt.ylabel("y (m)")


    cbar = plt.colorbar()
    cbar.set_label("Curvature")


if __name__=='__main__':
  sns.set(rc={'figure.figsize':(14, 8), 'axes.grid' : False})

  """
  Load Data
  """
  with open('analysis/data/particles_competitionTrack2.pickle', 'rb') as handle:
    data = pickle.load(handle)

  time_delta_list = data[0]
  x_position_list = data[1]
  y_position_list = data[2]
  theta_position_list = data[3]
  center_error_list = data[4]
  curvature_list = data[5]
  velocity_error_list = data[6]
  velocity_list = data[7]
  particles = data[8]
  hxEst = data[9]

  track_map = TrackMap(hxEst, particles[0].mu, particles[0].labels)

  filtered_blue_landmarks, filtered_yellow_landmarks = track_map.filter_landmarks()

  blue_loop = track_map.find_path(filtered_blue_landmarks)
  yellow_loop = track_map.find_path(filtered_yellow_landmarks)

  blue_interp = track_map.gen_boundary(blue_loop)
  yellow_interp = track_map.gen_boundary(yellow_loop)

  center_interp = track_map.get_center_path(blue_interp, yellow_interp)


  curvature = track_map.get_track_curvature(center_interp)
  print(center_interp[9990:10])
  track_map.plot_track(blue_interp, yellow_interp, center_interp, curvature, "Competition Track 2")
  #plt.plot(x_position_list, y_position_list)
  plt.show()