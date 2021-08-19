"""
Main driverless application for use with Formula Student Driverless Simulation

Author: Mahmoud Kamaleldin
"""
# Pip packages
import sys
import os
import time
import collections
import numpy
import math
import matplotlib.pyplot as plt
import pickle 
import cv2
from simple_pid import PID

# Local packages
import fsds
import darknet
from fastslam1_cones import *
import trackmap
import stanley
import track_explore
import path_tracking


# Configure plot size for dynamic figures
plt.rc('figure', figsize=(3, 2))

def velocity_controller(curvature, max_curvature, dt):
    """Calculates velocity target based on curvature of path and inputs throttle or brakes
       to reach target

    Parameters
    ----------
    pid : PID from simple_pid
        Velocity PID controller
    curvature : float
        Curvature of path ahead
    max_curvature : float
        Max curvature of track as a reference point
    dt : float
        Time difference since last call

    Returns
    -------
    throttle : float
        Throttle used
    brake : float
        Brakes used   
    error : float
        Difference between actual and target velocity
    Velocity : float
        Current velocity    
    """

    # Get GPS data
    gps = client.getGpsData()

    # Calculate the velocity in the vehicle's frame
    velocity = math.sqrt(math.pow(gps.gnss.velocity.x_val, 2) + math.pow(gps.gnss.velocity.y_val, 2))

    # Calculate velocity target based on curvature of path ahead
    velocity_target = max_velocity * max(1-curvature/max_curvature, 0.3)
    print(f"Velocity: {velocity}, Velocity Target: {velocity_target}")

    # Calculate velocity error
    error = velocity_target - velocity
    
    # Increase throttle if error is positive or increase brake if error is negative
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


# connect to the simulator 
client = fsds.FSDSClient()

# Check network connection, exit if not connected
client.confirmConnection()

# After enabling setting trajectory setpoints via the api. 
client.enableApiControl(True)

client.reset()

client.enableApiControl(True)

# Initialize car controls where throttle, brakes, and steering are controlled
car_controls = fsds.CarControls()

# Initiallize PID controller
steering_pid = PID(0.25, 0.001, 0.001, setpoint=0, output_limits=(-1.0,1.0))

start_time = time.time()
time_elapsed = 0

# Saved Data
velocity_error_list = []
velocity_list = []
time_delta_list = []

tracking_velocity_list = []
tracking_velocity_error_list = []
tracking_time_delta_list = []

N_PARTICLE = 10  # number of particles
particles = [Particle() for _ in range(N_PARTICLE)]

track_explorer = track_explore.TrackExplorer(particles, client)

PATH_TRACK_INITIALIZED = False

i = 0

# Initiallize plot for any updating plots
plt.pause(0.02)
plt.clf()

# Uncomment to use previous track data to control vehicle
"""
track_explorer.car_state = "TRACK_EXPLORED"
with open('analysis/data/particles_competitionTrack1.pickle', 'rb') as handle:
    data = pickle.load(handle)

track_explorer.slam_particles = data[8]
track_explorer.hxEst = data[9]
"""
while True:
    # Switch between exploring and tracking
    if track_explorer.car_state == "TRACK_EXPLORED":
        if PATH_TRACK_INITIALIZED == False:

            car_controls.throttle = 0
            car_controls.brake = 0.6
            client.setCarControls(car_controls)
            car_controls.brake = 0

            path_tracker = path_tracking.PathTracker(track_explorer.slam_particles, track_explorer.hxEst, client)
            PATH_TRACK_INITIALIZED = True

        # Autonomous vehicle constraints
        max_throttle = 0.5
        max_braking = 0.8
        max_velocity = 16

        # Perform path tracking
        steering_control, curvature_ewma, max_curvature = path_tracker.track_path()
        max_curvature = 0.1 # Hard-coded to limit speed at curves
        car_controls.steering = steering_control

        tracking_data = path_tracker.log_data()

    else:
        # Autonomous vehicle constraints
        max_throttle = 0.3
        max_braking = 0.8
        max_steering = 0.85
        max_velocity = 12

        # Perform track exploration
        curvature_ewma, steering_error, start_time, max_curvature, dt = track_explorer.track_explore(start_time, time_elapsed)
        car_controls.steering = track_explorer.steering_pid(steering_error, steering_pid, max_steering, dt)

        data = track_explorer.log_data()



    dt = time.time() - start_time
    time_elapsed+=dt
    start_time = time.time()
    #start = time.time()

    print(f'Time: {dt}')
    print(f'!!!! CAR STATE: {track_explorer.car_state}')

    # Perform velocity control and set CarControls object with inputs
    throttle, brake, vel_error, velocity = velocity_controller(curvature_ewma, max_curvature, dt)
    car_controls.throttle = throttle
    car_controls.brake = brake
    client.setCarControls(car_controls)


    # Logging Section
    velocity_error_list.append(vel_error)
    velocity_list.append(velocity)
    time_delta_list.append(time_elapsed)


    if PATH_TRACK_INITIALIZED == False:
        data[0] = time_delta_list
        data[6] = velocity_list
        data[7] = velocity_error_list

        #if i%5 == 0:
            #with open(f'images/particles_competitionTrack2.pickle', 'wb') as handle:
                #pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)

    if PATH_TRACK_INITIALIZED == True:
        tracking_velocity_error_list.append(vel_error)
        tracking_velocity_list.append(velocity)
        tracking_time_delta_list.append(time_elapsed)

        tracking_data[6] = tracking_time_delta_list
        tracking_data[7] = tracking_velocity_list
        tracking_data[8] = tracking_velocity_error_list

        #if i%5 == 0:
            #with open(f'analysis/data/tracking_competitionTrack1.pickle', 'wb') as handle:
                #pickle.dump(tracking_data, handle, protocol=pickle.HIGHEST_PROTOCOL)

    i+=1

#plt.show()