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
	#print(cov)
	#print(mu)

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
		wt = get_random_vector(np.zeros(11).reshape((-1,1)),Q)
	else:
		wt = np.zeros((11,1))
	p_x1 = x[0]+dt*vt*np.cos(x[2])
	p_y1 = x[1]+dt*vt*np.sin(x[2])
	th_1 = x[2] + dt*phi

	xp_t1 = np.array([p_x1,p_y1,th_1]).reshape((-1,1))
	x_t1 = np.vstack((xp_t1,x[3:])) + wt
	return x_t1


def get_A(x,dt,t):

	A = np.identity(11) # A is constant identity matrix
	vt = 1.0
	A[0,2] = -dt*vt*np.sin(x[2])
	A[1,2] = dt*vt*np.cos(x[2])

	return A



def get_C(x,dt,t):

	C = np.zeros((8,11))

	for i in range(4):
		#Range points, C1a
		norm_r = np.sqrt((x[2*i+3,0] - x[0,0])**2 + (x[2*i+4,0] - x[1,0])**2) # norm of mi - pt
		C[i,0] = (x[0,0] - x[2*i+3,0])/norm_r
		C[i,1] = (x[1,0] - x[2*i+4,0])/norm_r
		#Range measurements - new C, negative elements of C1a
		C[i,2*i+3] = (-x[0,0] + x[2*i+3,0])/norm_r
		C[i,2*i+4] = (-x[1,0] + x[2*i+4,0])/norm_r

		#Bearing points
		norm_b = (x[2*i+3,0] - x[0,0])**2 + (x[2*i+4,0] - x[1,0])**2 # not a norm, (mx-px) + (my-py)


		#C1b
		C[i+4,2*i+3] = (x[2*i+4,0]-x[1,0] )/norm_b #py - miy
		C[i+4,2*i+4] = (x[0,0]-x[2*i+3,0])/norm_b #mix - px
		C[i+4,2] = -1

		#C2
		C[i+4,2*i+3] = (-x[2*i+4,0]+x[1,0])/norm_b #miy - py
		C[i+4,2*i+4] = (-x[0,0]+x[2*i+3,0])/norm_b #px - mix
	return C


def get_measurement(x,t,dt,R):


	vt = get_random_vector(np.zeros(8).reshape((-1,1)),R)
	y = np.zeros((8,1))
	#print(x)
	for i in range(4):
		
		#Rnage measurement
		y[i,0] = np.sqrt((x[2*i,0] - x[0,0])**2 + (x[2*i+1,0] - x[1,0])**2) + vt[i] # Measure distance between state positions and state feature locations

		y[i+4,0] = x[2,0] - np.arctan2((x[2*i+4,0]-x[1,0]),(x[2*i+3,0]-x[0,0])) + vt[i+4] #
		#y[i,0] = wrapToPi(y[i,0])
	return y


def run_ekf(t_,x_true,Q,R,num_states, sig_i, iterative=False, smooth_eks=False):

	yt = np.zeros((np.size(t_),8))
	xt = np.zeros((np.size(t_),num_states)) # Create history of predicted state vector
	#xt[0,:] = x_true[0,:] #set initial state - already sampled to generate trajectory
	print(x_o.T)
	xt[0,:] = x_o.T #x_o is a global variable here because I'm lazy

	mu_tt = xt[0,:].reshape((-1,1))
	sig_tt = sig_i

	sigs = np.zeros((s_s,s_s,np.size(t_)))
	#print(np.shape(sigs))
	sigs[:,:,0] = sig_tt

	A_hist = sigs = np.zeros((s_s,s_s,np.size(t_)))

	start_time = time.time()
	for t in range(1, np.size(t_)):
		
		#Predict
		print("Mutt")
		print(np.shape(mu_tt))

		mu_t1_t = state_dynamics(mu_tt,t_[t],dt,Q, False)
		print("Initial Mu")
		print(np.shape(mu_t1_t))


		A = get_A(mu_tt,dt,t)
		A_hist[:,:,t] = A

		sig_t1_t = A@sig_tt@A.T + Q

		#Measure
		yt[t,:] = np.transpose(get_measurement(x_true[t,:].reshape((-1,1)),t_[t],dt,R))

		
		C = get_C(mu_t1_t,dt,t)

		#Update
		if not iterative:
			print("C = ")
			print(C)
			K = sig_t1_t@C.T@inv(C@sig_t1_t@C.T+R) #Kalman Gain
			print("K = ")
			print(np.shape(K))



			y_inov = yt[t,:].reshape((-1,1))-get_measurement(mu_t1_t,t,dt,np.zeros((8,8)))
			print("y_inov: ")
			print(np.shape(y_inov))

			#keep phi within [-pi,pi] thanks to suggestions in edstem post #160/158 and code from AA274A
			for j in range(4,np.size(y_inov)):
				y_inov[j,0] = wrapToPi(y_inov[j,0])

			mu_tt = mu_t1_t + K@(y_inov)
			print("Mu = ")
			print(mu_tt)

			xt[t,:] = np.transpose(mu_tt)
			sig_tt = sig_t1_t-K@C@sig_t1_t
			print(sig_tt)


		if iterative:
			conv = False
			max_steps = 5
			step = 1
			mu_temp_i = mu_t1_t #intialize the iterative mean to the predicted mean
			ek_mu = mu_t1_t
			while not conv:
				if step > max_steps:
					break

				if step!=1:
					temp_C = get_C(mu_temp_i,dt,t)
				else:
					temp_C = C
				K = sig_t1_t@temp_C.T@inv(temp_C@sig_t1_t@temp_C.T+R)

				if step == 1:
					ek_mu = mu_temp_i + K@(yt[t,:].reshape((-1,1))-get_measurement(mu_temp_i,t,dt,np.zeros((8,8))))

				
				mu_temp_i1 = mu_temp_i + K@(yt[t,:].reshape((-1,1))-get_measurement(mu_temp_i,t,dt,np.zeros((8,8)))) + K@temp_C@(mu_temp_i-mu_t1_t)
				mu_diff = mu_temp_i1-mu_temp_i
				if norm(mu_diff) < .01:
					conv = True

				mu_temp_i = mu_temp_i1
				C = temp_C
				step = step+1

		
			mu_tt = mu_temp_i

			xt[t,:] = np.transpose(mu_tt)
			sig_tt = sig_t1_t-K@C@sig_t1_t

		sigs[:,:,t] = sig_tt


	end_time = time.time()
	avg_time = (end_time-start_time)/np.size(t_)

	xt_s = np.zeros((np.size(t_),num_states))
	xt_s[-1,:] = xt[-1,:]

	sigs_s = np.zeros((s_s,s_s,np.size(t_)))
	sigs_s[:,:,-1] = sigs[:,:,-1]

	if smooth_eks:
		
		#backward pass
		for t in reversed(range(0, np.size(t_)-1)):
			K_t_s = sigs[:,:,t]@A_hist[:,:,t]@inv(sigs[:,:,t+1])

			f_t_mu_t = state_dynamics(xt[t,:].reshape((-1,1)),t_[t],dt,Q, False) #This is previously calculated mean at time t passed through dynamics
			xt_s[t,:] = np.transpose(xt[t,:].reshape((-1,1))+K_t_s@(xt_s[t+1,:].reshape((-1,1))-f_t_mu_t))

			sig_s_t = A_hist[:,:,t]@sigs[:,:,t]@A_hist[:,:,t].T #Progressions of previously calculated sigma passed through dynamics
			sigs_s[:,:,t] = sigs[:,:,t] + K_t_s@(sigs_s[:,:,t+1] - sig_s_t)


	if smooth_eks:
		return xt, yt, sigs, avg_time, A_hist, xt_s, sigs_s
	else:
		return xt, yt, sigs, avg_time



R= .1*np.identity(8) #4 dimensional covariance for measurement noise of feature ranges
#R = np.zeros((8,8))
dt = .1
s_s = 11 #dimension of the state vector

# Noise covariance for state positions
q_noise = .1
Q = np.zeros((11,11))
Q[0,0] = q_noise
Q[1,1] = q_noise
Q[2,2] = q_noise

#m = np.zeros((4,2)) #feature locations, each row is a feature
m = np.zeros((8,1)) #  This is m_true
m[2,0] = 10.0# m2 = [10,0]; m1 = [0,0]
m[4,0] = 10.0# m3 = [10,10]
m[5,0] = 10.0# m3 = [10,10]
m[7,0] = 10.0 # m4 = [0,10]


steps = 300
t_ = np.arange(0,steps*dt,step = dt)

mu_o= np.array([0,0,0,0,0,10,0,10,10,0,10]).reshape((-1,1)) # True states as mean for initial state
sig_o = .1*np.identity(11) # Initial covariance for the state

# Sample initial pose for feature positions
x_o = get_random_vector(mu_o,sig_o)
print("Initial state")
print(x_o)

# get_initial state for robot trajectory
#mu_o_p= np.array([0,0,0]).reshape((-1,1))
#sig_o_p = .01*np.identity(3)

# Sample initial pose
#x_o_p = get_random_vector(mu_o_p,sig_o_p).T
#print("Initial position")
#print(x_o_p)



x_s = np.zeros((np.size(t_),11)) # ground-truth trajectory for robot position
x_s[0,:] = x_o.T

#print(x_s)

# Get true state propogation
for t in range(1,np.size(t_)):
	x_s[t,:] = np.transpose(state_dynamics(x_s[t-1,:].reshape((-1,1)),t_[t],dt,Q))
	#x_s[t,3:] = m
	#x_s[t,:] = np.transpose(dummy_state_dynamics(x_s[t-1,:].reshape((-1,1)),t,dt,Q))


# No dynamics to the feature locations - true state is fixed


a_xt,a_yt,a_sigs,a_time = run_ekf(t_,x_s,Q,R,s_s,sig_o) #EKF SLAM
b_xt,b_yt,b_sigs,b_time = run_ekf(t_,x_s,Q,R,s_s,sig_o,False) #iEKF SLAM - this is more broken than the others right now
c_xt,c_yt,c_sigs,c_time, c_a_hist, c_xs, c_sigs_s = run_ekf(t_,x_s,Q,R,s_s,sig_o,False,True) #EKS


filter_states = [a_xt,b_xt, c_xs]
filter_sigs = [a_sigs,b_sigs, c_sigs_s]
filter_meas = [a_yt,b_yt, c_yt]
filter_times = [a_time,b_time, c_time]

num_filters = 3
state_list = ["px","py","theta","m1x","m1y","m2x","m2y","m3x","m3y","m4x","m4y"]
filter_list = ["EKF SLAM", "iEKF SLAM", "EKS", "pf"]


print("Average calculation times: ")
for n in range(num_filters):
	print("%s: %f" % (filter_list[n],filter_times[n]))

#fig, axs = plt.subplots(4,1)

for f in range(num_filters):
	fig, axs = plt.subplots(4,1)

	for n in range(3):
		axs[n].plot(t_,x_s[:,n], color='black')
		axs[n].plot(t_,filter_states[f][:,n], color='red')
		axs[n].fill_between(t_, filter_states[f][:,n] - 1.96*np.sqrt(filter_sigs[f][n,n,:]), #95% confidence interval shading, adapted from test.py file that was provided
	               filter_states[f][:,n] + 1.96 * np.sqrt(filter_sigs[f][n,n,:]),
	                color='red', alpha=0.3)

		axs[n].set_xlabel("Time, s")
		axs[n].set_ylabel("%s,%s" % (filter_list[f],state_list[n]))
		#axs[f][n].title.set_text("Filter %s" % filter_list[f])
		axs[n].legend(("True", "Estimate"))

	#Plot feature locations

	for i in range(4):
		axs[3].scatter(m[2*i,0],m[2*i+1,0], color='black',s=12)
		axs[3].scatter(filter_states[f][:,2*i+3],filter_states[f][:,2*i+4], color='red',s=5)

	axs[3].plot(filter_states[f][:,0],filter_states[f][:,1],color = 'orange')
	axs[3].plot(x_s[:,0],x_s[:,1],color = 'blue')

	plt.show()

'''
plt.figure(2)
plt.plot(t_,x_s[:,0],color = 'black') #p1/p2(t) predicted
plt.legend(("True", "EKF"))
plt.ylabel("Px")
plt.xlabel("Time")
plt.show()
'''