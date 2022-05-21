import os
import numpy as np
from scipy.linalg import sqrtm, det, inv, norm
import matplotlib.pyplot as plt

import time

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

	for i in range(num_points):
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


def dummy_state_dynamics(x,t,dt,Q, include_noise=True):

	#State vector is [px,py,theta].T

	phi = np.sin(t) #Input fuction
	vt = 1.0

	wt = get_random_vector(np.zeros(3).reshape((-1,1)),Q)
	p_x1 = x[0]+dt*vt + wt[0]
	p_y1 = x[1]+dt*vt + wt[1]
	th_1 = x[2] + dt*phi + wt[2]

	x_t1 = np.array([p_x1,p_y1,th_1]).reshape((-1,1))
	return x_t1

def get_A_1a(x,dt,t):
	vt = 1.0
	A = np.identity(3)
	A[0,2] = -dt*vt*np.sin(x[2])
	A[1,2] = dt*vt*np.cos(x[2])

	#Check observability?
	return A

def get_C_1a(x,dt,t):

	C = np.zeros((4,3))

	for i in range(4):
		norm = np.sqrt((m[i,0] - x[0])**2 + (m[i,1] - x[1])**2) # norm of mi - pt
		C[i,0] = (x[0] - m[i,0])/norm
		C[i,1] = (x[1] - m[i,1])/norm

	return C


def get_measurement_1a(x,t,dt,R):


	vt = get_random_vector(np.zeros(4).reshape((-1,1)),R)
	y = np.zeros((4,1))
	for i in range(4):
		y[i,0] = np.sqrt((m[i,0] - x[0])**2 + (m[i,1] - x[1])**2) + vt[i]
	return y



def run_ekf(t_,x_true,Q,R,num_states, sig_i, iterative=False):

	yt = np.zeros((np.size(t_),4))
	xt = np.zeros((np.size(t_),num_states)) # Create history of predicted state vector
	xt[0,:] = x_true[0,:] #set initial state - already sampled to generate trajectory

	mu_tt = xt[0,:].reshape((-1,1))
	sig_tt = sig_i

	sigs = np.zeros((s_s,s_s,np.size(t_)))
	#print(np.shape(sigs))
	sigs[:,:,0] = sig_tt

	start_time = time.time()
	for t in range(1, np.size(t_)):
		
		#Predict
		mu_t1_t = state_dynamics(mu_tt,t_[t],dt,Q, False)
		#mu_t1_t = dummy_state_dynamics(mu_tt,t,dt,Q, False)

		A = get_A_1a(mu_tt,dt,t)
		sig_t1_t = A@sig_tt@A.T + Q

		#Measure
		yt[t,:] = np.transpose(get_measurement_1a(x_true[t,:].reshape((-1,1)),t_[t],dt,R))

		
		C = get_C_1a(mu_t1_t,dt,t)
		#Update
		if not iterative:
			K = sig_t1_t@C.T@inv(C@sig_t1_t@C.T+R) #Kalman Gain

			#print("Measurement model")
			#print(yt[t,:].reshape((-1,1))-get_measurement(mu_t1_t,t,dt,R))

			mu_tt = mu_t1_t + K@(yt[t,:].reshape((-1,1))-get_measurement_1a(mu_t1_t,t,dt,np.zeros((4,4))))
			xt[t,:] = np.transpose(mu_tt)
			sig_tt = sig_t1_t-K@C@sig_t1_t


		if iterative:
			conv = False
			max_steps = 5
			step = 1
			mu_temp_i = mu_t1_t #intialize the iterative mean to the predicted mean
			ek_mu = mu_t1_t
			while not conv:
				if step > max_steps:
					#print("Reached max steps")
					#print("Number of steps: %i, %f" % (step-1,norm(mu_diff)))
					break

				if step!=1:
					temp_C = get_C_1a(mu_temp_i,dt,t)
				else:
					temp_C = C
				K = sig_t1_t@temp_C.T@inv(temp_C@sig_t1_t@temp_C.T+R)

				if step == 1:
					ek_mu = mu_temp_i + K@(yt[t,:].reshape((-1,1))-get_measurement_1a(mu_temp_i,t,dt,0))

				
				mu_temp_i1 = mu_temp_i + K@(yt[t,:].reshape((-1,1))-get_measurement_1a(mu_temp_i,t,dt,0)) + K@temp_C@(mu_temp_i-mu_t1_t)
				mu_diff = mu_temp_i1-mu_temp_i
				if norm(mu_diff) < .001:
					#print("Number of steps: %i, %f" % (step,norm(mu_diff)))
					conv = True

				mu_temp_i = mu_temp_i1
				C = temp_C
				step = step+1

		
			mu_tt = mu_temp_i
			#print(ek_mu - mu_tt)
			xt[t,:] = np.transpose(mu_tt)
			sig_tt = sig_t1_t-K@C@sig_t1_t

		sigs[:,:,t] = sig_tt

		'''
		rank = np.linalg.matrix_rank((np.vstack((C,C@A,C@A@A))))
		print(rank)

		if rank == num_states:
			print("It's observable!")
		else:
			print("Not observable!")
		'''

	end_time = time.time()
	avg_time = (end_time-start_time)/np.size(t_)
	return xt, yt, sigs, avg_time



R_1a= .1*np.identity(4) #4 dimensional covariance for measurement noise of feature ranges
dt = .1
s_s = 3 #dimension of the state vector

Q = .1*dt*np.identity(3)

m = np.zeros((4,2)) #feature locations, each row is a feature
m[1,0] = 10.0# m2 = [10,0]; m1 = [0,0]
m[2,0] = 10.0# m3 = [10,10]
m[2,1] = 10.0# m3 = [10,10]
m[3,1] = 10.0 # m4 = [0,10]


steps = 1000
t_ = np.arange(0,steps*dt,step = dt)

mu_o= np.array([0,0,0]).reshape((-1,1))
sig_o = .01*np.identity(3)

# Sample initial pose
x_o = get_random_vector(mu_o,sig_o).T
print("Initial state")
print(x_o)

x_s = np.zeros((np.size(t_),s_s)) # ground-truth trajectory
x_s[0,:] = x_o

#print(x_s)

for t in range(1,np.size(t_)):
	x_s[t,:] = np.transpose(state_dynamics(x_s[t-1,:].reshape((-1,1)),t_[t],dt,Q))
	#x_s[t,:] = np.transpose(dummy_state_dynamics(x_s[t-1,:].reshape((-1,1)),t,dt,Q))



a_xt,a_yt,a_sigs,a_time = run_ekf(t_,x_s,Q,R_1a,s_s,sig_o)



filter_states = [a_xt]
filter_sigs = [a_sigs]
filter_meas = [a_yt]
filter_times = [a_time]

num_filters = 1
state_list = ["px","py","theta"]
filter_list = ["EKF w/ Range Measurements", "iekf", "ukf", "pf"]


print("Average calculation times: ")
for n in range(num_filters):
	print("%s: %f" % (filter_list[n],filter_times[n]))

fig, axs = plt.subplots(num_filters,3)

for n in range(s_s):
	axs[n].plot(t_,x_s[:,n], color='black')
	axs[n].plot(t_,filter_states[0][:,n], color='red')
	axs[n].fill_between(t_, filter_states[0][:,n] - 1.96*np.sqrt(filter_sigs[0][n,n,:]), #95% confidence interval shading, adapted from test.py file that was provided
                   filter_states[0][:,n] + 1.96 * np.sqrt(filter_sigs[0][n,n,:]),
                    color='red', alpha=0.3)
	axs[n].set_xlabel("Time, t")
	axs[n].set_ylabel("%s,%s" % (filter_list[0],state_list[n]))
	#axs[f][n].title.set_text("Filter %s" % filter_list[f])
	axs[n].legend(("True", "Estimate"))


plt.show()

plt.figure(2)
plt.plot(t_,x_s[:,0],color = 'black') #p1/p2(t) predicted
plt.plot(t_,a_xt[:,0],color = 'blue')  #p1/p2(t) simulate
plt.legend(("True", "EKF"))
plt.ylabel("Px")
plt.xlabel("Time")
plt.show()
