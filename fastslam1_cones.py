"""
FastSLAM 1.0 for use with ROS 2
Copyright 2020 Isaac Vander Sluis
Author: Isaac Vander Sluis
Starter code: Atsushi Sakai
"""

#!/usr/bin/env python3

import os
import sys

import numpy as np
import matplotlib.pyplot as plt
import time
from copy import deepcopy

# Covariance matrix of measurement noise
Q = np.diag([9.0, np.deg2rad(9.0)]) ** 2
# Covariance matrix of control noise
R = np.diag([1.0, np.deg2rad(20.0)]) ** 2

#  Simulation parameter
R_sim = np.diag([0.5, np.deg2rad(10.0)]) ** 2
OFFSET = 0.01

DT = 0.0  # Time tick (s)
STATE_SIZE = 3  # State size [x, y, yaw]
LM_SIZE = 2  # LM state size [x, y, color]
N_PARTICLE = 10  # Number of particles
THRESHOLD = 0.065 # Likelihood threshold for data association
NTH = N_PARTICLE / 1.5  # Number of particle for re-sampling
CAM_ANGLE = np.pi/2 # Camera angle (radians)
CAM_DIST = 10 # Distance for camera perception (metres)
PLOTTING = True
DEBUGGING = False

# Definition of variables

    # x: by itself, refers to the state vector [x, y, theta]
    # xd: state with noise

    # u: the control vector [linear velocity, angular velocity]
    # ud: controls with noise

    # z: a set of observations, each of form [distance, angle]
    # z_hat: the predicted observation for a landmark
    # dz: diff between expected and actual observation (z - z_hat)

    # xEst: the estimated true state vector of the vehicle
    # H: Jacobian matrix

    # d: distance (metres)
    # d_sq: distance squared

    # dx: delta x (change in x pos)
    # dy: delta y (change in y pos)
    # dpos: a vector [dx, dy]

    # Particles: a single hypothesis of the pose and landmarks
        # Particle.w: the weight of the particle
        # Particle.x: the robot's pose [x, y, theta]
            # Particle.x[0, 0]: x value of pose
            # Particle.x[1, 0]: y value of pose
            # Particle.x[2, 0]: theta of pose
        # Particle.mu: an array of EKF mean values as x-y coords
        # Particle.sigma: array of 2x2 cov matrices for EKFs
        # Particle.i: a measure of confidence in the landmark

# --- CODE ADAPTED FROM PYTHON ROBOTICS / ATSUSHI SAKAI ---

# Python Robotics
#   https://pythonrobotics.readthedocs.io/en/latest/
# Atsushi Sakai on GitHub
#   https://github.com/AtsushiSakai

class Particle:

    def __init__(self):
        """
        Construct a new particle
        :return: Returns nothing
        """
        
        self.w = 1.0 / N_PARTICLE # Initialise weight evenly
        self.x = np.zeros((3, 1)) # State vector [x, y, theta]
        self.mu = np.zeros((0, LM_SIZE)) # Landmark positions
        self.sigma = np.zeros((0, LM_SIZE, LM_SIZE)) # Covariance
        self.labels = np.zeros((0, 1))
        self.i = np.zeros((0, 1)) # Tracks landmark confidence

def fast_slam1(particles, u, z, labels, dt):
    """
    Updates beliefs about position and landmarks using FastSLAM 1
    :param particles: An array of particles
    :param u: The controls (velocity and orientation)
    :param z: The observation
    :return: Returns new particles sampled from updated
             particles according to weight
    """
    print('RUNNING SLAM')

    # Step 1: predict
    particles = predict_particles(particles, u, dt)

    # Step 2: update
    particles = update_with_observation(particles, z, labels)

    # Step 3: resample
    particles = resampling(particles)

    return particles

def calc_final_state(particles):
    """
    Calculates the final state vector
    :param particles: An array of particles
    :return: xEst, the state vector
    """
    print('CALCULATING FINAL STATE')
    xEst = np.zeros((STATE_SIZE, 1)) # Empty state vector

    particles = normalize_weight(particles)

    for i in range(N_PARTICLE):
        xEst += particles[i].w * particles[i].x

    xEst[2, 0] = pi_2_pi(xEst[2, 0])

    return xEst

# STEP 1: PREDICT

def motion_model(x, u, dt):
    """
    Compute predictions for a particle
    :param x: The state vector [x, y, theta]
    :param u: The input vector [linear vel Vt, angular vel Wt]
    :return: Returns predicted state vector x
    """
    # A 3x2 matrix to calculate change in x, y, yaw given controls
    B = np.array([[dt * np.cos(x[2, 0]), 0],
                  [dt * np.sin(x[2, 0]), 0],
                  [0.0, dt]])

    x = x + B @ u # New pose = old pose + change in pose

    x[2, 0] = pi_2_pi(x[2, 0]) # Ensure Theta is under pi radians

    return x

def predict_particles(particles, u, dt):
    """
    Predict x, y, yaw values for new particles using motion model
    :param particles: An array of particles
    :param u: An input vector [linear vel, angular vel]
    :return: Returns predictions as particles
    """
    print('PREDICTING PARTICLES')

    for i in range(N_PARTICLE):
        #ud = u + (np.random.randn(1, 2) @ R).T  # Add noise
        #particles[i].x = motion_model(particles[i].x, ud, dt)
        particles[i].x = u

    return particles

def pi_2_pi(angle):
    """
    Ensure the angle is under +/- PI radians
    :param angle: Angle in radians
    :return: Returns the angle ensuring it is under +/- PI radians
    """
    return (angle + np.pi) % (2 * np.pi) - np.pi

# STEP 2: UPDATE

def observation(u, data):
    """
    Record an observation in terms of distance and angle
    :param u: Control vector (linear and angular velocity)
    :param data: The landmarks seen by the camera
    :return:
        z - The observation
        ud - Input with noise
    """
    print('MAKING OBSERVATION')

    # Initialize np array for observed cones
    try:
        z = np.zeros_like(data[:,0:2])
        # For each landmark compute distance and angle
        z[:, 0] = np.hypot(data[:, 0], data[:, 1])
        z[:, 1] = pi_2_pi(np.arctan2(data[:, 1], data[:, 0]) - np.pi/2)
        labels = data[:, 2]
    except:
        z = np.empty(shape=(0, 3))
        labels = z = np.empty(shape=(0, 1))

    # Add noise to input
    ud1 = u[0, 0] + np.random.randn() * R_sim[0, 0]**0.5
    ud2 = u[1, 0] + np.random.randn() * R_sim[1, 1]**0.5 + OFFSET
    ud = np.array([ud1, ud2]).reshape(2, 1)


    return z, ud, labels

def update_with_observation(particles, z, labels):
    """
    Update particles using an observation by either matching
    the landmark to an existing landmark or by adding a new landmark
    :param particles: An array of particles
    :param z: An observation (array of landmarks, each [dist, theta])
    :return: Returns updated particles
    """
    for particle in particles:
        # If no landmarks exist yet, add all currently observed landmarks
        if (particle.mu.size == 0):
            particle = add_landmarks(particle, z[:, 0], z[:, 1], labels)            
        else:
            z_hat = np.zeros_like(particle.mu)
            # Calculate dx and dy for each landmark
            dpos = particle.mu - particle.x[0:2, 0]
            d_sq = dpos[:, 0]**2 + dpos[:, 1]**2
            z_hat[:, 0] = np.sqrt(d_sq)
            z_hat[:, 1] = pi_2_pi(np.arctan2(dpos[:, 1], dpos[:, 0])
                                  - particle.x[2, 0])

            # Calculate Jacobians
            H = calc_H(particle, dpos, d_sq, z_hat[:, 0])

            # Calculate covariances
            Qj = H @ particle.sigma @ H.transpose((0, 2, 1)) + Q

            try:
                invQ = np.linalg.inv(Qj)
            except np.linalg.linalg.LinAlgError:
                print("singular")
                return 1.0

            # For each cone observed, determine data association
            for iz in range(len(z)):
                dz = calc_dz(z_hat, z[iz])

                wj = compute_likelihoods(dz, invQ, Qj)

                wj_max = np.max(wj) # Get max likelihood

                # If cone hasn't been seen before, add landmark
                if (wj_max < THRESHOLD):
                    # ! Attempting to place this section into a
                    #   function results in strange outliers
                    # Calculate sine and cosine for the landmark
                    s = np.sin(pi_2_pi(particle.x[2, 0] + z[iz, 1]))
                    c = np.cos(pi_2_pi(particle.x[2, 0] + z[iz, 1]))

                    # Add landmark location to mu
                    particle.mu = np.vstack((particle.mu,
                                            [particle.x[0, 0] + z[iz, 0] * c,
                                             particle.x[1, 0] + z[iz, 0] * s]))

                    particle.labels = np.vstack((particle.labels,
                                                [labels[iz]]))

                    dx = z[iz, 0] * c
                    dy = z[iz, 0] * s
                    d_sq = dx**2 + dy**2
                    d = np.sqrt(d_sq) # Get distance
                    Hj = np.array([[dx / d, dy / d],
                                   [-dy / d_sq, dx / d_sq]])
                    Hj = np.linalg.inv(Hj) @ Q @ np.linalg.inv(Hj.T)
                    particle.sigma = np.vstack((particle.sigma,
                                                Hj.reshape((1, 2, 2))))
                    particle.i = np.append(particle.i, 1)

                # If the cone matches a landmark, update the EKF
                else:
                    cj = np.argmax(wj) # Get ID for highest likelihood
                    particle.w *= wj_max # Adjust particle weight
                    mu_temp, sigma_temp = update_ekf(
                                        particle.mu[cj].reshape((2, 1)),
                                        particle.sigma[cj], dz[cj], Q, H[cj])
                    particle.mu[cj] = mu_temp.T # Update EKF mean
                    particle.sigma[cj] = sigma_temp # Replace cov. matrix
                    particle.i[cj] += 2 # Increase confidence in landmark
            
            # Determine which landmarks should have been observed
            expected = np.argwhere((z_hat[:, 0] < CAM_DIST) &
                                   (np.abs(z_hat[:, 1]) < CAM_ANGLE/2))
            # Decrease confidence by 1
            particle.i[expected] -= 1
            # Remove all landmarks with confidence below zero
            remove = np.argwhere(particle.i < 0)
            particle.mu = np.delete(particle.mu, remove, axis=0)
            particle.sigma = np.delete(particle.sigma, remove, axis=0)
            particle.labels = np.delete(particle.labels, remove, axis=0)
            particle.i = np.delete(particle.i, remove)


    return particles

def update_ekf(mu, sigma, dz, Q_cov, H):
    """
    Updates extended Kalman filter for landmarks
    :param mu: The mean of a landmark EKF
    :param sigma: The 2x2 covariance of a landmark EKF
    :param dz: The difference between the actual and expected observation
    :param Q_cov: A covariance matrix of process noise
    :param H: Jacobian matrix
    :return:
        mu - New EKF mean as x-y coordinates
        sigma - New EKF covariance matrix
    """
    PHt = sigma @ H.T
    S = H @ PHt + Q_cov

    S = (S + S.T) * 0.5
    s_chol = np.linalg.cholesky(S).T
    s_chol_inv = np.linalg.inv(s_chol)
    W1 = PHt @ s_chol_inv
    W = W1 @ s_chol_inv.T

    mu = mu + W @ dz
    sigma = sigma - W1 @ W1.T

    return mu, sigma

def add_landmarks(particle, d, angle, labels):
    """
    Adds a set of landmarks to the particle. Only used on first SLAM cycle
    when no landmarks have been added.
    :param particle: The particle to be updated
    :param d: An array of distances to the landmarks
    :param angle: An array of observation angles for the landmarks
    :return: Returns the updated particle with landmarks added
    """
    # Evaluate sine and cosine values for each observation in z
    s = np.sin(pi_2_pi(particle.x[2, 0] + angle))
    c = np.cos(pi_2_pi(particle.x[2, 0] + angle))

    # Add new landmark locations to mu
    particle.mu = np.vstack((particle.mu, np.array(
                                [particle.x[0, 0] + d * c,
                                 particle.x[1, 0] + d * s]).T))

    particle.labels = np.vstack((particle.labels, np.atleast_2d(labels).T))

    # Distance values
    dpos = np.zeros((len(d), 2))
    dpos[:, 0] = d * c # dx
    dpos[:, 1] = d * s # dy
    d_sq = dpos[:, 0]**2 + dpos[:, 1]**2
    d = np.sqrt(d_sq)

    H = calc_H(particle, dpos, d_sq, d)

    # Add covariance matrices for landmarks
    particle.sigma = np.vstack((particle.sigma, 
                                np.linalg.inv(H) @ Q
                                @ np.linalg.inv(H.transpose((0, 2, 1)))))

    particle.i = np.append(particle.i, np.full(len(d), 1))

    return particle

def calc_H(particle, dpos, d_sq, d):
    """
    Calculate series of H 2x2 Jacobian matrices after the formula
            H = np.array([[dx / d, dy / d],
                          [-dy / d_sq, dx / d_sq]])
    :param particle: The particle being evaluated
    :param dpos: An array of [dx, dy] between the vehicle and landmarks
    :param d_sq: An array of squared distances to each landmark
    :param d: An array of expected observation distances rel. to the vehicle
    :return: The Jacobian matrix H
    """
    dpos_mod = np.flip(dpos, axis=1) # Reverse dpos column order
    dpos_mod[:, 0] = -dpos_mod[:, 0] # Negate dy column
    Ha = dpos/np.vstack(d) # Calculate [dx / d, dy / d]
    Hb = dpos_mod/np.vstack(d_sq) # Calculate [-dy / d_sq, dx / d_sq]
    H = np.vstack((zip(Ha, Hb))).reshape(d_sq.size, 2, 2) # Weave together (3D)
    
    return H

def calc_dz(z_hat, z):
    """
    Compute dz, the difference between the observation expectation
    and the observation
    :param z_hat: An array of expected relative observations for each
                  landmark as distance/angle pairs
    :param z: The distances and angles for the observed landmarks
    """
    dz = z_hat - z
    dz[:, 1] = pi_2_pi(dz[:, 1])
    dz = dz.reshape((len(dz), 2, 1)) # Reshape as array of 2x1 vectors

    return dz

def compute_likelihoods(dz, invQ, Qj):
    """
    Calculates the likelihoods of a given observation matching an existing
    landmark
    :param dz: The difference between z_hat and z
    :param invQ: The inverse of Qj
    :param Qj: An array of covariance matrixes
    :return: Returns wj, a list of likelihoods for each existing landmark
    """

    # Prob Robotics p. 461
    num = np.exp(-0.5 * dz.transpose((0, 2, 1)) @ invQ @ dz)
    den = 2.0 * np.pi * np.sqrt(np.linalg.det(Qj)).reshape((num.size, 1, 1))

    wj = num / den # Calculate likelihoods

    return wj

# STEP 3: RESAMPLE

def normalize_weight(particles):
    """
    Adjusts particle weights such that all weights sum to 1
    :param particles: An array of particles
    :return: An array of particles with reassigned weights
    """
    sum_w = sum([p.w for p in particles]) # Get sum of particle weights

    try:
        for i in range(N_PARTICLE):
            # Turn weight into percentage of total particle weights
            particles[i].w /= sum_w
    except ZeroDivisionError:
        for i in range(N_PARTICLE):
            particles[i].w = 1.0 / N_PARTICLE

        return particles

    return particles


def resampling(particles):
    """
    Low-variance resampling
    :param particles: An array of particles
    :return: An array of particles
    """

    # Normalize weights
    particles = normalize_weight(particles)

    # Get particle weights
    pw = []
    for i in range(N_PARTICLE):
        pw.append(particles[i].w)

    # Create a 1D array of the current particle weights
    pw = np.array(pw)

    n_eff = 1.0 / (pw @ pw.T)  # Effective particle number

    if n_eff.all() < NTH:  # Resampling
        w_cum = np.cumsum(pw) # Sum of all particle weights

        base = np.cumsum(pw * 0.0 + 1 / N_PARTICLE) - 1 / N_PARTICLE
        
        resample_id = base + np.random.rand(base.shape[0]) / N_PARTICLE

        inds = []
        ind = 0

        for ip in range(N_PARTICLE):
            while (ind < w_cum.shape[0] - 1) \
                    and (resample_id[ip] > w_cum[ind]):
                ind += 1
            inds.append(ind)

        tmp_particles = particles[:]
        for i in range(len(inds)):
            particles[i].x = tmp_particles[inds[i]].x
            particles[i].mu = tmp_particles[inds[i]].mu
            particles[i].sigma = tmp_particles[inds[i]].sigma
            particles[i].i = tmp_particles[inds[i]].i
            particles[i].w = 1.0 / N_PARTICLE

    return particles