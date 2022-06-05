#!/usr/bin/env python
import os
import struct
import numpy as np
import math

main_path = os.getcwd()


def get_imu_data(t_start,t_stop):
	imu1_log = main_path + "/IMU1.txt"
	imu_data = np.loadtxt(imu1_log,delimiter=",")
	num_datapoints = np.shape(imu_data)[0]

	#Find index of nearest measurement to "start time", not rigorous for before/after; https://www.geeksforgeeks.org/find-the-nearest-value-and-the-index-of-numpy-array/
	start_index = np.absolute(imu_data[:,0]-t_start).argmin()
	print("Start index = %i" % (start_index))

	#Find index of nearest measurement to "start time", not rigorous for before/after
	stop_index = np.absolute(imu_data[:,0]-t_stop).argmin()
	print("Stop index = %i" % (stop_index))

	#Slice out data that we don't need; before start, after stop
	imu_data_red = imu_data[start_index:stop_index+1,:]
	
	num_datapoints_red = np.shape(imu_data_red)[0]

	#TODO: What about starting angular velocity?
	ang_vel = np.zeros((num_datapoints_red,3))
	'''
	for i in range(1,num_datapoints_red):
		#m.TimeUS,m.GyrX,m.GyrY,m.GyrZ,m.AccX,m.AccY,m.AccZ
		dt = float((imu_data_red[i,0]-imu_data_red[i-1,0])/1e6) #get dt in seconds, is recorded in us
		ang_vel[i,0] = (imu_data_red[i,1]-imu_data_red[i-1,1])/dt # get angular velocity measurement at time t in body frame (X)
		ang_vel[i,1] = (imu_data_red[i,2]-imu_data_red[i-1,2])/dt # get angular velocity measurement at time t in body frame (Y)
		ang_vel[i,2] = (imu_data_red[i,3]-imu_data_red[i-1,3])/dt # get angular velocity measurement at time t in body frame (Z)
	'''
	return imu_data_red, ang_vel



def get_initial_conditions(t_start,t_stop):

	#Get inital conditions from logged ekf data
	ekf1_log = main_path + "/ekf1.txt"
	ekf2_log = main_path + "/ekf2.txt"
	ekf3_log = main_path + "/ekf3.txt"

	ekf1_data = np.loadtxt(ekf1_log,delimiter=",")
	ekf2_data = np.loadtxt(ekf2_log,delimiter=",")
	ekf3_data = np.loadtxt(ekf3_log,delimiter=",")
	
	num_datapoints = np.shape(ekf1_data)[0]

	#Find index of nearest measurement to "start time", not rigorous for before/after; https://www.geeksforgeeks.org/find-the-nearest-value-and-the-index-of-numpy-array/
	start_index = np.absolute(ekf1_data[:,0]-t_start).argmin()
	print("Start index = %i" % (start_index))

	# All ekf variables are logged at the same timestamp, so this index can be applied to all ekf logs

	# Combine initial state variables into single numpy array, only get euler angles, position, velocity
	i_s = ekf1_data[start_index,1:10]
	s_p = np.hstack((ekf1_data[start_index,10:],ekf2_data[start_index,1:])) # This is all the state variable that have a staic process model
	#Order should be Gyro biases (XYZ), Acclerometer biases (XYZ), Wind velocity estimates (NE), Earth Magnetic field vector (XYZ), magnetometer bias errors (XYZ)
	print("Initial State:")
	print(i_s)
	print("Static params:")
	print(s_p)

	true_traj = ekf1_data[start_index:,1:10]
	ekf_time = ekf1_data[start_index:,0]

	var = ekf3_data[start_index,1:] #Get logged variances (velocity, position, height, magnetic field, airspeed)

	return i_s, s_p, var, true_traj, ekf_time


def get_mag_data(t_start,t_stop):
	mag_log = main_path + "/MAG.txt"
	mag_data = np.loadtxt(mag_log,delimiter=",")
	num_datapoints = np.shape(mag_data)[0]

	#Find index of nearest measurement to "start time", not rigorous for before/after; https://www.geeksforgeeks.org/find-the-nearest-value-and-the-index-of-numpy-array/
	start_index = np.absolute(mag_data[:,0]-t_start).argmin()
	print("Start index = %i" % (start_index))

	#Find index of nearest measurement to "start time", not rigorous for before/after
	stop_index = np.absolute(mag_data[:,0]-t_stop).argmin()
	print("Stop index = %i" % (stop_index))

	#Slice out data that we don't need; before start, after stop
	mag_data_red = mag_data[start_index:stop_index+1,:]

	return mag_data_red


def get_baro_data(t_start,t_stop):
	baro_log = main_path + "/BARO.txt"
	baro_data = np.loadtxt(baro_log,delimiter=",")
	num_datapoints = np.shape(baro_data)[0]

	#Find index of nearest measurement to "start time", not rigorous for before/after; https://www.geeksforgeeks.org/find-the-nearest-value-and-the-index-of-numpy-array/
	start_index = np.absolute(baro_data[:,0]-t_start).argmin()
	print("Start index = %i" % (start_index))

	#Find index of nearest measurement to "start time", not rigorous for before/after
	stop_index = np.absolute(baro_data[:,0]-t_stop).argmin()
	print("Stop index = %i" % (stop_index))

	#Slice out data that we don't need; before start, after stop
	baro_data_red = baro_data[start_index:stop_index+1,:]

	baro_update_rate = float(baro_data_red[-1,0] - baro_data_red[0,0])/np.size(baro_data_red[:,0])
	print("baro update rate:")
	print(baro_update_rate)

	return baro_data_red


def get_arspd_data(t_start,t_stop):
	arspd_log = main_path + "/ARSP.txt"
	arspd_data = np.loadtxt(arspd_log,delimiter=",")
	num_datapoints = np.shape(arspd_data)[0]

	#Find index of nearest measurement to "start time", not rigorous for before/after; https://www.geeksforgeeks.org/find-the-nearest-value-and-the-index-of-numpy-array/
	start_index = np.absolute(arspd_data[:,0]-t_start).argmin()
	print("Start index = %i" % (start_index))

	#Find index of nearest measurement to "start time", not rigorous for before/after
	stop_index = np.absolute(arspd_data[:,0]-t_stop).argmin()
	print("Stop index = %i" % (stop_index))

	#Slice out data that we don't need; before start, after stop
	arspd_data_red = arspd_data[start_index:stop_index+1,:]

	arspd_update_rate = float(arspd_data_red[-1,0] - arspd_data_red[0,0])/np.size(arspd_data_red[:,0])
	print("baro update rate:")
	print(arspd_update_rate)

	return arspd_data_red