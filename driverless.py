"""
Main driverless application for use with Formula Student Driverless Simulation

Author: Mahmoud Kamaleldin
"""

import sys
import os
import time

import collections
import numpy
import math
import matplotlib.pyplot as plt
import pickle 
import cv2
## adds the fsds package located the parent directory to the pyhthon path
#sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import fsds

#sys.path.insert(0, "/home/mkamalel/fsds_src/yolov4/darknet")
import darknet
from fastslam1_cones import *
from simple_pid import PID
import trackmap
import stanley

import track_explore
import path_tracking

plt.rc('figure', figsize=(3, 2))

def distance(x1, y1, x2, y2):
    return math.sqrt(math.pow(abs(x1-x2), 2) + math.pow(abs(y1-y2), 2))


def calculate_throttle():
    gps = client.getGpsData()

    # Calculate the velocity in the vehicle's frame
    velocity = math.sqrt(math.pow(gps.gnss.velocity.x_val, 2) + math.pow(gps.gnss.velocity.y_val, 2))

    # the lower the velocity, the more throttle, up to max_throttle
    return max_throttle * max(1 - velocity / target_speed, 0)

def calculate_throttle_cam(pid, curvature, max_curvature, dt):
    # TODO: take into account error
    gps = client.getGpsData()

    # Calculate the velocity in the vehicle's frame
    velocity = math.sqrt(math.pow(gps.gnss.velocity.x_val, 2) + math.pow(gps.gnss.velocity.y_val, 2))

    #velocity_target = min(2/curvature, 4)
    #if curvature >= max_curvature:
    #    velocity_target = 2
    #else:
    print(f"curvature: {1-curvature/max_curvature}")
    velocity_target = max_velocity * max(1-curvature/max_curvature, 0.3)
    print(f"Velocity: {velocity}, Velocity Target: {velocity_target}")

    error = velocity_target - velocity
    
    #control = pid(error, dt=dt)
    #print(f"Error: {error}")

    if error > 0:
        throttle = max_throttle * max(1 - velocity / velocity_target, 0)
        brake = 0
    elif error < 0:
        brake = max_braking * max(1 - velocity_target / velocity, 0)
        throttle = 0
    else:
        throttle = 0
        brake = 0

    print(f"Throttle: {throttle}, Brake: {brake}")

    return throttle, brake, error, velocity


def calculate_steering_cam(error, pid, dt):
    # If there are more cones on the left, go to the left, else go to the right.
    if error != 0:
        control = pid(error, dt=dt)

        return -control*max_steering
    else:
        return 0

# connect to the simulator 
client = fsds.FSDSClient()

# Check network connection, exit if not connected
client.confirmConnection()

# After enabling setting trajectory setpoints via the api. 
client.enableApiControl(True)

client.reset()

client.enableApiControl(True)

# Autonomous system constatns
max_throttle = 0.3 # m/s^2
max_braking = 0.3
#max_curvature = 5
max_velocity = 16
target_speed = 2 # m/s
max_steering = 0.5
cones_range_cutoff = 10 # meters

tick = 10
i = 0
car_controls = fsds.CarControls()

N_PARTICLE = 10  # number of particle

particles = [Particle() for _ in range(N_PARTICLE)]

steering_pid = PID(0.25, 0.001, 0.001, setpoint=0, output_limits=(-1.0,1.0))
throttle_pid = PID(1, 0.001, 0, setpoint=2)

center_error_list = []

"""
Trackmap
"""
"""
with open('images/particles_trackday2.pickle', 'rb') as handle:
    data = pickle.load(handle)

particles = data[8]
hxEst = data[9]
"""
############################################################################


plt.pause(0.02)
plt.clf()

start_time = time.time()
start = time.time()
time_elapsed = 0
steering_error = 0
#curvature = 0.001
L = 0.9
# Saved Data
curvature_list = []
velocity_error_list = []
velocity_list = []
x_position_list = []
y_position_list = []
theta_position_list = []
time_delta_list = []

track_explorer = track_explore.TrackExplorer(particles, client)
#path_tracker = path_tracking.PathTracker(particles, hxEst, client)
PATH_TRACK_INITIALIZED = False
#track_explorer.car_state = "TRACK_EXPLORED"
while True:

    if track_explorer.car_state == "TRACK_EXPLORED":
        if PATH_TRACK_INITIALIZED == False:

            car_controls.throttle = 0
            car_controls.brake = 0.6
            client.setCarControls(car_controls)
            car_controls.brake = 0

            path_tracker = path_tracking.PathTracker(track_explorer.slam_particles, track_explorer.hxEst, client)
            PATH_TRACK_INITIALIZED = True

        max_throttle = 0.5 # m/s^2
        max_braking = 0.8
        #max_curvature = 5
        max_velocity = 16
        steering_control, curvature_ewma, max_curvature = path_tracker.track_path()
        max_curvature = 0.1
        car_controls.steering = steering_control

    else:
        max_throttle = 0.3 # m/s^2
        max_braking = 0.8
        max_curvature = 5
        max_steering = 0.85
        max_velocity = 12
        curvature_ewma, steering_error, start_time, max_curvature, dt = track_explorer.track_explore(start_time, time_elapsed)
        car_controls.steering = calculate_steering_cam(steering_error, steering_pid, dt)
        print(car_controls.steering)
        data = track_explorer.log_data()




    dt = time.time() - start
    time_elapsed+=dt
    start_time = time.time()
    start = time.time()
    print(f'Time: {dt}')
    print(f'!!!! CAR STATE: {track_explorer.car_state}')
    #start = time.time()
    #cones = find_cones()

    #if len(cones) == 0:
        #continue

    #car_controls.steering = calculate_steering_cam(steering_error, steering_pid, dt)
    print(car_controls.steering)
    #print("curvature: ", curvature_ewma)
    throttle, brake, vel_error, velocity = calculate_throttle_cam(throttle_pid, curvature_ewma, max_curvature, dt)
    car_controls.throttle = throttle
    car_controls.brake = brake
    client.setCarControls(car_controls)

    velocity_error_list.append(vel_error)
    velocity_list.append(velocity)
    time_delta_list.append(time_elapsed)

    data[0] = time_delta_list
    data[6] = velocity_list
    data[7] = velocity_error_list

    #if i%5 == 0:
        #with open(f'images/particles_competitionTrack2.pickle', 'wb') as handle:
            #pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)


    # draw cones
    #for cone in cones_cam:
        #plt.scatter(x=cone['x'], y=cone['y'])

    #plt.plot(x_states, y_states)
    i+=1


#plt.show()