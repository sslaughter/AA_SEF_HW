import os
import numpy as np
from scipy.linalg import sqrtm, det, inv, norm
import matplotlib.pyplot as plt

import time


#wrapToPi function taken from code provided through AA274A coursework
def wrapToPi(a):
    if isinstance(a, list):    # backwards compatibility for lists (distinct from np.array)
        return [(x + np.pi) % (2*np.pi) - np.pi for x in a]
    return (a + np.pi) % (2*np.pi) - np.pi


def get_random_vector(mu,cov):
	# Sample normal distribution and then transform per desired parameters to output samples on desired distribution
	num_points = np.size(mu)
	a = np.zeros((num_points,1))
	#print(mu)
	#print(cov)
	#expects mu is a column vector

	rand_points = np.random.randn(num_points).reshape((-1,1))

	if num_points == 1:
		a = cov*rand_points + mu
	else:
		a[:,0] = np.transpose(sqrtm(cov)@rand_points+mu)

	#a comes out as a column vector
	return a

def state_dynamics(x,t,dt,Q, include_noise=True):

	#State vector is [px,py,theta].T

	phi = np.sin(t) #Input fuction
	vt = 1.0
	if(include_noise):
		wt = get_random_vector(np.zeros(3).reshape((-1,1)),Q)
	else:
		wt = np.zeros(3)
	p_x1 = x[0]+dt*vt*np.cos(x[2]) + wt[0]
	p_y1 = x[1]+dt*vt*np.sin(x[2]) + wt[1]
	th_1 = x[2] + dt*phi + wt[2]

	x_t1 = np.array([p_x1,p_y1,th_1]).reshape((-1,1))
	return x_t1

def state_dynamics_f(x,t,dt,Q, include_noise=True):

	#State vector is [m1x,m1y,...].T


	if(include_noise):
		wt = get_random_vector(np.zeros(8).reshape((-1,1)),Q)
	else:
		wt = np.zeros((8,1))

	x_t1 = x+wt # simply add noise to the state estimate for the prediction step
	return x_t1


def get_A_f(x,dt,t):

	A = np.identity(8) # A is constant identity matrix

	return A



def get_C_f(x,dt,t):

	C = np.zeros((4,8))

	for i in range(4):
		#Negative of what I initially thought
		norm = (x[2*i,0] - x_s[t,0])**2 + (x[2*i+1,0] - x_s[t,1])**2 # not a norm, (mx-px) + (my-py)
		C[i,2*i] = (-x[2*i+1,0]+x_s[t,1] )/norm #miy - py
		C[i,2*i+1] = (-x_s[t,0]+x[2*i,0])/norm #px - mix
	return C


def get_measurement_f(x,t,dt,R):


	vt = get_random_vector(np.zeros(4).reshape((-1,1)),R)
	y = np.zeros((4,1))
	# We know the postion, need to measure the bearing to estimated/true features
	#print(x)
	print("t = ")
	print(t)
	print("state for measurement")
	print(x)

	print("simulated position")
	print(x_s[t,:])
	for i in range(4):
		y[i,0] = -x_s[t,2] + np.arctan2((x[2*i+1,0]-x_s[t,1]),(x[2*i,0]-x_s[t,0])) + vt[i] #same measurement function, just with a known position
		#y[i,0] = wrapToPi(y[i,0])
	return y


def run_ekf(t_,x_true,Q,R,num_states, sig_i, iterative=False):

	yt = np.zeros((np.size(t_),4))
	xt = np.zeros((np.size(t_),num_states)) # Create history of predicted state vector
	#xt[0,:] = x_true[0,:] #set initial state - already sampled to generate trajectory
	print(x_o.T)
	xt[0,:] = x_o.T #x_o is a global variable here because I'm lazy

	mu_tt = xt[0,:].reshape((-1,1))
	sig_tt = sig_i

	sigs = np.zeros((s_s,s_s,np.size(t_)))
	#print(np.shape(sigs))
	sigs[:,:,0] = sig_tt



	start_time = time.time()
	for t in range(1, np.size(t_)):
		
		#Predict
		#mu_t1_t = state_dynamics(mu_tt,t_[t],dt,Q, False)
		#print(mu_tt)
		mu_t1_t = state_dynamics_f(mu_tt,t_[t],dt,Q, False)
		#print("predicted mean")
		#print(mu_t1_t)
		#mu_t1_t = dummy_state_dynamics(mu_tt,t,dt,Q, False)

		A = get_A_f(mu_tt,dt,t)
		
		sig_t1_t = A@sig_tt@A.T + Q
		print("Predicted covariance")
		print(sig_t1_t)
		#sig_t1_t = A@sig_tt@A.T
		#Measure
		yt[t,:] = np.transpose(get_measurement_f(m,t,dt,R)) #measure given simulated states and true feature location

		
		C = get_C_f(mu_t1_t,dt,t)
		#print("C:")
		#print(C)
		#Update
		if not iterative:
			K = sig_t1_t@C.T@inv(C@sig_t1_t@C.T+R) #Kalman Gain
			print("K = ")
			print(K)

			#print("Measurement model")
			#print(yt[t,:].reshape((-1,1))-get_measurement(mu_t1_t,t,dt,R))
			print("True measeure")
			print(yt[t,:])
			y_inov = yt[t,:].reshape((-1,1))-get_measurement_f(mu_t1_t,t,dt,np.zeros((4,4)))
			print("Yinov")
			print(y_inov)
			
			#print(y_inov)
			#keep phi within [-pi,pi] thanks to suggestions in edstem post #160/158 and code from AA274A
			for j in range(np.size(y_inov)):
				y_inov[j,0] = wrapToPi(y_inov[j,0])
			#mu_tt = mu_t1_t + K@(yt[t,:].reshape((-1,1))-get_measurement_1a(mu_t1_t,t,dt,np.zeros((4,4))))
			print("Mu correction")
			print(K@y_inov)

			mu_tt = mu_t1_t + K@(y_inov)
			#print("K:")
			#print(K)
			#print(mu_tt)
			xt[t,:] = np.transpose(mu_tt)
			sig_tt = sig_t1_t-K@C@sig_t1_t
			print(sig_tt)

		sigs[:,:,t] = sig_tt


	end_time = time.time()
	avg_time = (end_time-start_time)/np.size(t_)
	return xt, yt, sigs, avg_time



R_x= .01*np.identity(4) #4 dimensional covariance for measurement noise of feature ranges
#R_x= np.zeros((4,4))
dt = .1
s_s = 8 #dimension of the state vector

Q = .01*np.identity(8) # Noise covariance for state positions
Q_p = .01*np.identity(3) # Noise covariance for true state simulation

#m = np.zeros((4,2)) #feature locations, each row is a feature
m = np.zeros((8,1)) #  This is m_true
m[2,0] = 10.0# m2 = [10,0]; m1 = [0,0]
m[4,0] = 10.0# m3 = [10,10]
m[5,0] = 10.0# m3 = [10,10]
m[7,0] = 10.0 # m4 = [0,10]


steps = 300
t_ = np.arange(0,steps*dt,step = dt)

mu_o= np.array([0,0,10,0,10,10,0,10]).reshape((-1,1)) # True states as mean for initial state
sig_o = 1*np.identity(8) # Initial covariance for the feature locations

# Sample initial pose for feature positions
x_o = get_random_vector(mu_o,sig_o)
print("Initial state")
print(x_o)

# get_initial state for robot trajectory
mu_o_p= np.array([0,0,0]).reshape((-1,1))
sig_o_p = .1*np.identity(3)

# Sample initial pose
x_o_p = get_random_vector(mu_o_p,sig_o_p).T
print("Initial position")
print(x_o_p)



x_s = np.zeros((np.size(t_),3)) # ground-truth trajectory for robot position
x_s[0,:] = x_o_p

#print(x_s)

# Get true state propogation
for ts in range(1,np.size(t_)):
	x_s[ts,:] = np.transpose(state_dynamics(x_s[ts-1,:].reshape((-1,1)),t_[ts],dt,Q_p))
	#x_s[t,:] = np.transpose(dummy_state_dynamics(x_s[t-1,:].reshape((-1,1)),t,dt,Q))
print(x_s)


# No dynamics to the feature locations - true state is fixed


a_xt,a_yt,a_sigs,a_time = run_ekf(t_,x_s,Q,R_x,s_s,sig_o)



filter_states = [a_xt]
filter_sigs = [a_sigs]
filter_meas = [a_yt]
filter_times = [a_time]

num_filters = 1
state_list = ["m1x","m1y","m2x","m2y","m3x","m3y","m4x","m4y"]
filter_list = ["EKF w/ Bearing Measurements", "iekf", "ukf", "pf"]


print("Average calculation times: ")
for n in range(num_filters):
	print("%s: %f" % (filter_list[n],filter_times[n]))

fig, axs = plt.subplots(num_filters,s_s)

for n in range(s_s):
	axs[n].plot((t_[0],t_[-1]),(m[n,0],m[n,0]), color='black')
	axs[n].scatter(t_,filter_states[0][:,n], color='red',s=5)
	#axs[n].fill_between(t_, filter_states[0][:,n] - 1.96*np.sqrt(filter_sigs[0][n,n,:]), #95% confidence interval shading, adapted from test.py file that was provided
    #              filter_states[0][:,n] + 1.96 * np.sqrt(filter_sigs[0][n,n,:]),
    #                color='red', alpha=0.3)
	axs[n].set_xlabel("Time, t")
	axs[n].set_ylabel("%s,%s" % (filter_list[0],state_list[n]))
	#axs[f][n].title.set_text("Filter %s" % filter_list[f])
	axs[n].legend(("True", "Estimate"))


plt.show()

plt.figure(2)
plt.plot(t_,x_s[:,2],color = 'black') #p1/p2(t) predicted
plt.legend(("True", "EKF"))
plt.ylabel("Px")
plt.xlabel("Time")
plt.show()
