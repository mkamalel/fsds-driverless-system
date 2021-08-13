from fastslam1_cones import *
import math
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pickle
from scipy.interpolate import splprep, splev


def distance(x1, y1, x2, y2):
  return math.sqrt(math.pow(abs(x1-x2), 2) + math.pow(abs(y1-y2), 2))

def get_side(a, b, p):
  return (p[0]-a[0])*(b[1]-a[1]) - (p[1]-a[1])*(b[0]-a[0])

class TrackMap():
  label_map = {"cone":0, "orangecone": 1, "bluecone":2, "yellowcone":3}

  def __init__(self, lap_path, landmarks, labels):
    self.lap_path = lap_path
    self.landmarks = landmarks
    self.landmark_labels = labels

  def filter_landmarks(self):
    filtered_blue_landmarks = np.empty((0,2), np.float32)
    filtered_yellow_landmarks = np.empty((0,2), np.float32)

    car_path = [(self.lap_path[0,i], self.lap_path[1,i]) for i in range(0, len(self.lap_path[1,:]))]

    for i in range(0, len(self.landmarks)):
      landmark = [self.landmarks[i, 0], self.landmarks[i, 1]]
      label = self.landmark_labels[i, 0]

      car_pos = min(car_path, key=lambda x: distance(landmark[0],landmark[1],x[0],x[1]))
      nearest_index = car_path.index(car_pos)

      dist = distance(landmark[0],landmark[1],car_pos[0],car_pos[1])

      if not dist > 2 and not dist < 0.4:
        prev_car_pos = [car_path[nearest_index-1][0], car_path[nearest_index-1][1]]
        
        side = get_side(prev_car_pos, car_pos, landmark)

        if label == self.label_map['bluecone'] and side < 0:
          # side less than 0, cone on the left
          filtered_blue_landmarks = np.vstack((filtered_blue_landmarks, np.array(landmark)))

        elif label == self.label_map['yellowcone'] and side > 0:
          # side greater than 0, cone on the right
          filtered_yellow_landmarks = np.vstack((filtered_yellow_landmarks, np.array(landmark)))

    return filtered_blue_landmarks, filtered_yellow_landmarks

  def find_path(self, coords, start=None):
    #coords = np.unique(coords, axis=1)
    coords = coords.tolist()

    if start is None:
      start = min(coords, key=lambda x: distance(0, 0, x[0], x[1]))

    pass_by = coords
    path = [start]
    pass_by.remove(start)
    while pass_by:
          nearest = min(pass_by, key=lambda x: distance(path[-1][0], path[-1][1], x[0], x[1]))
          path.append(nearest)
          pass_by.remove(nearest)

    return np.array(path)

    path = [start]
    pass_by.remove(start)

  def gen_boundary(self, loop):
    tck, u = splprep(loop.T, u=None, s=10, per=1) 
    u_new = np.linspace(u.min(), u.max(), 1000)

    x_new, y_new = splev(u_new, tck, der=0)
    interp = np.array(list(zip(x_new, y_new)))    

    return interp

  def get_center_path(self, blue_interp, yellow_interp):
    center_path = np.empty((0,2), np.float32)

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
    #first derivatives 
    dx = np.gradient(np.array(center_interp[:,0]))
    dy = np.gradient(np.array(center_interp[:,1]))

    #second derivatives 
    d2x = np.gradient(dx)
    d2y = np.gradient(dy)

    #calculation of curvature from the typical formula
    curvature = np.asarray(np.abs(dx * d2y - d2x * dy) / (dx * dx + dy * dy)**1.5)   
    
    return curvature 


  def plot_track(self, blue_interp, yellow_interp, center_interp, curvature):
    #plt.figure()
    #plt.plot(filtered_blue_landmarks[:,0], filtered_blue_landmarks[:,1], 'bo')
    plt.plot(blue_interp[:,0], blue_interp[:,1], 'b--')

    #plt.plot(filtered_yellow_landmarks[:,0], filtered_yellow_landmarks[:,1], 'yo')
    plt.plot(yellow_interp[:,0], yellow_interp[:,1], 'y--')

    plt.plot(center_interp[:,0], center_interp[:,1], '--', c='black')    

    plt.scatter(center_interp[:,0], center_interp[:,1], s=10, c=curvature, cmap="magma_r", marker="o")
    cbar2 = plt.colorbar()


if __name__=='__main__':
  sns.set(rc={'figure.figsize':(14, 8), 'axes.grid' : False})

  """
  Load Data
  """
  with open('images/particles.pickle', 'rb') as handle:
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
  track_map.plot_track(blue_interp, yellow_interp, center_interp, curvature)
  #plt.plot(x_position_list, y_position_list)
  plt.show()