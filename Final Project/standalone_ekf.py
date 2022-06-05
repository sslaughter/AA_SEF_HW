#!/usr/bin/env python
import os
import struct
import numpy as np
import math
import Process_Log
import process_measurements as pm
from scipy.linalg import sqrtm, det, inv, norm, block_diag
import matplotlib.pyplot as plt




#wrapToPi function taken from code provided through AA274A coursework
def wrapToPi(a):
    if isinstance(a, list):    # backwards compatibility for lists (distinct from np.array)
        return [(x + np.pi) % (2*np.pi) - np.pi for x in a]
    return (a + np.pi) % (2*np.pi) - np.pi


def run_ekf(mu_i,sig_i, static_p, imu_t_initial):

	#for now, Q is simply set, though it should probably be informed by sensor specs, which could be derived from simulation parameters
	Q = .1*np.identity(9)

	#Breakout static parameters
	gyro_bias = static_p[:3]
	accel_bias = static_p[3:6]
	wind_est = static_p[6:8]
	e_mfield = static_p[8:11]
	mag_bias = static_p[11:]

	num_measurements = 3 #skip over magnetometer integration for right now

	yt = np.zeros((1,num_measurements)) #create history of measurements - this will be somewhat processed sensor data
	xt = np.zeros((1,9)) # Create history of predicted state vector

	xt[0,:] = mu_i.T
	time = np.zeros(1)

	mu_tt = mu_i
	sig_tt = sig_i

	#sigs = np.zeros((s_s,s_s,np.size(t_)))
	#print(np.shape(sigs))
	#sigs[:,:,0] = sig_tt

	#Create copies of sensor data which will be pared down during ekf run
	imu_remaining = np.copy(imu_meas)
	mag_remaining = np.copy(mag_meas)
	baro_remaining = np.copy(baro_meas)
	arspd_remaining = np.copy(arsp_meas)
	print(np.shape(ek_t.reshape((-1,1))))
	print(np.shape(x_s[:,2].reshape(((-1,1)))))
	mag_false = np.hstack((ek_t.reshape((-1,1)),x_s[:,2].reshape((-1,1))))
	print("mag")
	print(mag_false)


	imu_t = imu_t_initial[1:] #intial value for "past" imu data, used to propogate dynamics at time t (for prediction of t+1)
	print("imut before loop")
	print(imu_t)

	#IMU is sampled at 1Khz, take it in chunks of 100 measurements to produce ekf solution at 10Hz
	samp_scale = 100
	itera = 1
	while itera < 1200:
			
		#Check that we still have a full step worth of data left
		rem_points = np.size(imu_remaining[:,0])
		print("Remaining points = %i" % rem_points)
		if rem_points < 10:
			break
		
		#Get data for step
		imu_step = imu_remaining[:samp_scale,:]
		imu_remaining = imu_remaining[samp_scale:,:]

		#Get last timestamp of imu measurements
		t_last_samp = imu_step[-1,0]

		# get timestep
		dt = float(imu_step[-1,0] - imu_step[0,0])
		#print("dt1 = %f" % (dt))
		dt = dt/1e6
		#print("dt = %f" % (dt))

		#integrate gyro over timestep to get total angle changes
		gyr = np.array([np.trapz(imu_step[:,1],x=imu_step[:,0]/1.0e6),np.trapz(imu_step[:,2],x=imu_step[:,0]/1.0e6),np.trapz(imu_step[:,3],x=imu_step[:,0]/1.0e6)])

		#print("full imu step")
		#print(imu_step)
		#print("imu step for vertical accel")
		#print(imu_step[:,6])
		#integrate accelration over timestep to get total velocity change (pre-rotation)
		acc = np.array([np.trapz(imu_step[:,4],x=imu_step[:,0]/1.0e6),np.trapz(imu_step[:,5],x=imu_step[:,0]/1.0e6),np.trapz(imu_step[:,6]+9.81,x=imu_step[:,0]/1.0e6)])

		imu_t = np.hstack((gyr,acc)) # save this for future use as "past" imu data


		#Predict
		mu_t1_t, ac_r = state_dynamics(mu_tt,dt,imu_t,gyro_bias,accel_bias) #return predicted mu and transformed basic integration of acceleration (for velocity calc), used in A calc

		A = get_A(mu_tt,dt,ac_r)
		#print("A: ")
		#print(A)

		sig_t1_t = A@sig_tt@A.T + Q

		#Measure
		'''
		print("time_samp")
		print(t_last_samp)
		print(baro_remaining)
		'''
		#Get all measurements since the last interation up to the final timestep for this iteration
		baro_step = baro_remaining[baro_remaining[:,0] <= t_last_samp][-1,:]
		#print("baro slice")
		#print(baro_step)
		mag_step = mag_remaining[mag_remaining[:,0] <= t_last_samp][-1,:]
		arspd_step = arspd_remaining[arspd_remaining[:,0] <= t_last_samp][-1,:]
		mf_false = mag_false[mag_false[:,0] <= t_last_samp][-1,:]
		#mag = get_heading_from_mag(mu_tt,mag_step)

		#Average available measurements within this step
		#baro_step_avg = np.average(baro_step[:,1])
		#mag_step_avg = np.average(mag_step[:,1]) #need to fix this - what mag values do we want? Are we going to process the magnetometer values first so measurement is just a heading? - Yes! - How?
		#arspd_step_avg = np.average(arspd_step[:,1])

		#y = np.array([baro_step_avg,mag_step_avg,arspd_step_avg])
		#y = np.array([baro_step_avg,arspd_step_avg])
		y = np.array([-baro_step[1],arspd_step[1],1.5*np.random.randn()+mf_false[1]])
		#y = np.array([-baro_step[1],arspd_step[1]])
		yt = np.vstack((yt,y)) #add measurements to history

		# Need to update C to only account for measured indices
		C = get_C(mu_t1_t,dt,num_measurements,wind_est,0,0)
		'''


		'''

		#Update
		#For now, measurement noise is set, should probably be informed by sensor specs as well
		R_n = .1*np.identity(num_measurements)

		#print("C:")
		#print(C)

		#print("sig:")
		#print(sig_t1_t)

		#print("inverse:")
		#print(C@sig_t1_t@C.T+R_n)

		K = sig_t1_t@C.T@inv(C@sig_t1_t@C.T+R_n) #Kalman Gain

		y_inov = y.reshape((-1,1))-g_x(mu_t1_t,wind_est)
		#print("iteration #: %i" % (itera))
		#print("mut1t")
		#print(mu_t1_t)

		#print("K")
		#print(K)

		#print("y = ")
		#print(y)

		#print("predicted meas")
		#print(g_x(mu_t1_t,wind_est))

		#print("yinov")
		#print(y_inov)
		mu_tt = mu_t1_t + K@(y_inov)
		#print("Mu = ")
		#print(mu_tt)

		time = np.append(time,t_last_samp)
		xt = np.vstack((xt,np.transpose(mu_tt)))
		sig_tt = sig_t1_t-K@C@sig_t1_t
		itera+=1
		#if(itera == 4):
		#	break
	return xt, yt, time


#TODO define C function
def get_C(x,dt,num_meas,wind,e_m,m_b):

	C = np.zeros((num_meas,9))


	#measurement states - [mag, baro, airspeed]
	# mag = psi (heading)
	# baro = altitude
	# airspeed = velocity w/ components of NED since it's measured in xyz
	norm_arspd = np.sqrt((x[3]-wind[0])**2 + (x[4]-wind[1])**2 + x[5]**2)

	'''
	C[0,2] = 1 #magnetomer - direct measurement of psi
	C[1,8] = 1 #baro - direct measurement of altitude (Pd)
	C[2,4] = (x[3]-wind[0])/norm_arspd #dg(arspd)/dVn
	C[2,5] = (x[4]-wind[1])/norm_arspd #dg(arspd)/dVe
	C[2,6] = x[5]/norm_arspd #dg(arspd)/dVd
	'''
	#skipping mag for now
	C[0,8] = 1 #baro - direct measurement of altitude (Pd)
	C[1,4] = (x[3]-wind[0])/norm_arspd #dg(arspd)/dVn
	C[1,5] = (x[4]-wind[1])/norm_arspd #dg(arspd)/dVe
	C[1,6] = x[5]/norm_arspd #dg(arspd)/dVd
	C[2,2] = 1

	return C

def g_x(x,wind):
	baro_alt = x[8] #baro should equal the altitude
	mag_f = x[2]
	arspd = np.sqrt((x[3]-wind[0])**2 + (x[4]-wind[1])**2 + x[5]**2)
	g = np.array([baro_alt,arspd, mag_f]).reshape((-1,1))
	#g = np.array([baro_alt,arspd]).reshape((-1,1))
	return g


def state_dynamics(x,dt,imu_t, gyro_b, accel_b):

	# state = [attitude, velocity, position].T
	#imu_t/imu_bias = [gx, gy, gz, ax, ay, az]

	#print("imu_t")
	#print(imu_t)
	gyro = imu_t[:3].reshape((-1,1))
	accel = imu_t[3:].reshape((-1,1))
	#print("accel before rot")
	#print(accel)

	#print("x")
	#print(x)
	#Get rotation matrix from state euler angles
	R = get_Rotation_Matrix(x[0,0],x[1,0],x[2,0])
	#print("Rotation matrix:")
	#print(np.shape(R))

	gyro_c = gyro-dt*gyro_b.reshape((-1,1)) #correct gyro measurements for bias
	#print("gyroc:")
	#print(np.shape(gyro_c))
	accel_c = accel-dt*accel_b.reshape((-1,1)) #correct accel measurements for bias
	#print("accelc:")
	#print(np.shape(accel_c))
	accel_r = R.T@accel_c #rotate accel into NED frame (transpose of defined rotation matrix)

	att = np.array((x[:3,0])).reshape((-1,1))
	vel = np.array((x[3:6,0])).reshape((-1,1))
	pos = np.array((x[6:,0])).reshape((-1,1))

	#print("delta velocities")
	#print(accel_r)
	#propgate state dynamics - simple integration over timestep dt
	att_p = att + gyro_c
	if (att_p[2,0] > 360):
		att_p[2,0] = att_p[2,0] - 360
	elif (att_p[2,0] < 0):
		att_p[2,0] = att_p[2,0] + 360
	else:
		pass
	vel_p = vel + accel_r
	pos_p = pos + dt*x[3:6].reshape((-1,1))

	'''
	print("State dynamics: ********************")
	print(att_p)
	print(np.shape(vel_p))
	print(pos_p)
	'''

	x_t1 = np.vstack((att_p,vel_p,pos_p))

	return x_t1, accel_r


def get_A(x,dt,ac_r):
	#Need to pass delta velocities, time step, biases

	# state = [attitude, velocity, position].T
	phi_r = np.radians(x[0])
	theta_r = np.radians(x[1])
	psi_r = np.radians(x[2])

	c_psi = np.cos(psi_r)
	c_theta = np.cos(theta_r)
	s_psi = np.sin(psi_r)
	s_theta = np.sin(theta_r)
	s_phi = np.sin(phi_r)
	c_phi = np.cos(phi_r)

	'''
	if (x[2] < 185.0 and x[2] > 175.0):
		s_psi = psi_r
	else:
		pass
	'''

	A = np.identity(9)

	#Jacobian of process model, primary complication is in derivative of rotation matrix for Velocity calculation:
	
	VXn = dt*ac_r[0]
	VXe = dt*ac_r[1]
	VXd = dt*ac_r[2]

	#Need to confirm this rotation is correct
	A[3,0] = -s_psi*c_theta*dt*VXn + c_psi*c_theta*VXe #dVN/dpsi
	A[3,1] = -c_psi*s_theta*VXn - s_psi*s_theta*VXe - c_theta*VXd #dVN/dtheta

	A[4,0] = (-s_phi*s_theta*s_phi-c_psi*c_phi)*VXn + (c_psi*s_theta*s_phi-s_psi*c_phi)*VXe #dVE/dpsi
	A[4,1] = c_psi*c_theta*s_phi*VXn + s_psi*c_theta*s_phi*VXe - s_theta*s_phi*VXd #dVE/dtheta
	A[4,2] = (c_psi*s_theta*c_phi+s_psi*s_phi)*VXn + (s_psi*s_theta*c_phi-c_psi*s_phi)*VXe + c_theta*c_phi*VXd  #dVE/dpsi

	A[5,0] = (-s_phi*s_theta*c_phi)*VXn + (c_psi*s_theta*c_phi + s_psi*s_phi)*VXe #dVD/dpsi
	A[5,1] = (c_psi*c_theta*c_phi)*VXn + (s_psi*c_theta*c_phi)*VXe - s_theta*c_phi*VXd #dVD/dtheta
	A[5,2] = (-c_psi*s_theta*s_phi)*VXn + (-s_psi*s_theta*s_phi-c_psi*c_phi)*VXe - c_theta*s_phi*VXd #dVD/dphi 

	A[6,3] = dt #dPN/dVN
	A[7,4] = dt #dPE/dVN
	A[8,5] = dt #dPD/dVD

	return A

def get_Rotation_Matrix(phi,theta,psi):

	#Rotation from NED to body (XYZ)
	#Derived from "Quaternions and Rotation Sequences"; J.B.Kuipers, pg. 86
	phi_r = np.radians(phi)
	theta_r = np.radians(theta)
	psi_r = np.radians(psi)

	c_psi = np.cos(psi_r)
	c_theta = np.cos(theta_r)
	s_psi = np.sin(psi_r)
	s_theta = np.sin(theta_r)
	s_phi = np.sin(phi_r)
	c_phi = np.cos(phi_r)

	R1 = np.array([c_psi*c_theta,s_psi*c_theta,-s_theta])
	R2 = np.array([c_psi*s_theta*s_phi-s_psi*c_phi,s_psi*s_theta*s_phi+c_psi*c_phi,c_theta*s_phi])
	R3 = np.array([c_psi*s_theta*c_phi+s_psi*s_phi,s_psi*s_theta*c_phi-c_psi*s_phi,c_theta*c_phi])

	R = np.vstack((R1,R2,R3))
	#R = np.array([[c_psi*c_theta,s_psi*c_theta,-s_theta],[c_psi*s_theta*s_phi-s_psi*c_phi,s_psi*s_theta*s_phi+c_psi*c_phi,c_theta*s_phi],[c_psi*s_theta*c_phi+s_psi*s_phi,s_psi*s_theta*c_phi-c_psi*s_phi,c_theta*c_phi]])
	return R


#def get_heading_from_mag(x,m):




#Time in seconds to let the vehicle have gps after entering a constant heading course.
time_on_constant_heading = 120*1e6 #us

main_path = os.getcwd()

have_datfile = True

if not have_datfile:
	t_start, t_stop, orgn = Process_Log.process(main_path + "\\b_mis.bin")
else:
	stats_path = main_path + "\\stats.txt"
	stats_read = open(stats_path,'r')
	stats = stats_read.readlines()
	for i in range(len(stats)):
		vals = stats[i].split(',')
		print(len(vals))
		if i == 0:
			t_start = int(vals[0])
			t_stop = int(vals[1])
		elif i == 1:
			orgn = np.array([float(vals[0]),float(vals[1]), float(vals[2])])

print("orgin:")
print(orgn)

print("Time start: %i, time stop: %i" % (t_start,t_stop))

ekf_s_start_time = t_start + time_on_constant_heading
print("Start time: %i, stop time: %i" % (ekf_s_start_time, t_stop))

#ekf_s_start_time = 1644082937
#t_stop = 3374126476
#Collect measurements from simulated data, only for relevant time steps (after GNSS is lost)
imu_meas, ang_vel_hist = pm.get_imu_data(ekf_s_start_time,t_stop)
print("imu_measurements")
print(imu_meas[0,:])
mag_meas = pm.get_mag_data(ekf_s_start_time,t_stop)
baro_meas = pm.get_baro_data(ekf_s_start_time,t_stop)
arsp_meas = pm.get_arspd_data(ekf_s_start_time,t_stop)

#Get initial state from simulated EKF data, including static parameters (biases, wind)
initial_states, static_params, variances, x_s, ek_t = pm.get_initial_conditions(ekf_s_start_time,t_stop)


num_states = 9
# state = [attitude, velocity, position].T
mu_o = initial_states.reshape((-1,1))
print("mu_o")
print(mu_o)
sig_o = block_diag(.1*np.identity(3), variances[0]*np.identity(3), variances[1]*np.identity(3))
imu_t_o = imu_meas[0,:]


xt,yt, t = run_ekf(mu_o,sig_o,static_params, imu_t_o)

t[0] = ekf_s_start_time

plt.figure(1)
plt.plot(x_s[:,7],x_s[:,6],color = 'black') #p1/p2(t) predicted
plt.plot(xt[:,7],xt[:,6],color = 'red')
plt.legend(("True", "Estimate"))
plt.ylabel("PN (m, relative to home)")
plt.xlabel("PE (m, relative to home)")
plt.show()


plt.figure(2)
plt.plot((ek_t-ekf_s_start_time)/1e6,x_s[:,2],color = 'black') #p1/p2(t) predicted
plt.plot((t-ekf_s_start_time)/1e6,xt[:,2],color = 'red')
plt.legend(("True", "Estimate"))
plt.ylabel("Heading Angle (degrees)")
plt.xlabel("Time (s)")
plt.show()


plt.figure(3)
plt.plot((ek_t-ekf_s_start_time)/1e6,x_s[:,3],color = 'black') #p1/p2(t) predicted
plt.plot((t-ekf_s_start_time)/1e6,xt[:,3],color = 'red')
plt.legend(("True", "Estimate"))
plt.ylabel("Velocity North (m/s)")
plt.xlabel("Time (s)")
plt.show()


plt.figure(4)
plt.plot((ek_t-ekf_s_start_time)/1e6,-x_s[:,4],color = 'black') #p1/p2(t) predicted
plt.plot((t-ekf_s_start_time)/1e6,-xt[:,4],color = 'red')
plt.legend(("True", "Estimate"))
plt.ylabel("Velocity East (m/s)")
plt.xlabel("Time (s) ")
plt.show()


plt.figure(5)
plt.plot((ek_t-ekf_s_start_time)/1e6,x_s[:,8],color = 'black') #p1/p2(t) predicted
plt.plot((t-ekf_s_start_time)/1e6,xt[:,8],color = 'red')
plt.legend(("True", "Estimate"))
plt.ylabel("PD (altitude, m, relative to home)")
plt.xlabel("Time (s) ")
plt.show()


plt.figure(5)
plt.plot((ek_t-ekf_s_start_time)/1e6,np.sqrt(x_s[:,6]**2 + x_s[:,7]**2),color = 'black') #p1/p2(t) predicted
plt.plot((t-ekf_s_start_time)/1e6,np.sqrt(xt[:,6]**2 + xt[:,7]**2),color = 'red')
plt.legend(("True", "Estimate"))
plt.ylabel("PD (altitude, m, relative to home)")
plt.xlabel("Time (s) ")
plt.show()

#separation distance at end?