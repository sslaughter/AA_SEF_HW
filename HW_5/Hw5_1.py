import os
import numpy as np
from scipy.linalg import sqrtm, det, inv
import matplotlib.pyplot as plt

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

def state_dynamics(x,t,dt,Q, include_noise = True):
	u = np.sin(t) #Input fuction
	if(include_noise):
		wt = get_random_vector(np.zeros(3).reshape((-1,1)),Q)
	else:
		wt = np.zeros(3)

	p_t1 = x[0]*(1-x[1]*dt)+x[2]*u*dt + wt[0]
	a_t1 = x[1]+wt[1]
	b_t1 = x[2]+wt[2]

	x_t1 = np.array([p_t1,a_t1,b_t1]).reshape((-1,1))
	#print(x_t1)
	#print(x_t1)
	return x_t1


def get_A(x,dt,t):
	u = np.sin(t)
	A = np.identity(3)
	A[0,0] = (1-x[1]*dt)
	A[0,1] = -x[0]*dt
	A[0,2] = u*dt

	return A

def get_C(x,dt,t):

	C = np.zeros((1,3))
	C[0,0] = 1

	return C


def get_measurement(x,t,dt,R):

	wt = get_random_vector(0,R)
	y = x[0]+wt

	return y

#True parameters
a_t = .5
b_t = 2

#input functionn u = sin(t)

Q = np.zeros((3,3))
Q[0,0] = .1

#R = np.zeros((3,3))
#R[0,0] = .1
R = .1
dt = .1

mu_tt = np.array([0,0,10]).reshape((-1,1))
sig_tt = np.identity(3)

steps = 500
t_ = np.arange(0,steps*dt,step = dt)

# Create history of measurement vector
yt = np.zeros((np.size(t_),1))

s_s = 3 #dimension of the state vector

xt = np.zeros((np.size(t_),s_s)) # Create history of predicted state vector
xt[0,:] = get_random_vector(mu_tt,sig_tt).T
print("it's getting here")

x_s = np.zeros((np.size(t_),s_s)) # Create history of simulated(true) state vector
x_s[0,0] = xt[0,0]
x_s[0,1] = a_t
x_s[0,2] = b_t


sigs = []
sigs.append(sig_tt)

Q_e = .01*np.identity(3)
Q_e[0,0] = .1
print("Q_e: ")
print(Q_e)

mu_tt = xt[0,:].reshape((-1,1))

#Simulate trajectory
for t in range(1, np.size(t_)):
	
	#Predict
	mu_t1_t = state_dynamics(mu_tt,t,dt,Q, False) # No noise in the prediction step of the state

	A = get_A(mu_tt,dt,t)
	sig_t1_t = A@sig_tt@A.T + Q_e
	
	#Get "true" state
	x_s[t,:] = np.transpose(state_dynamics(x_s[t-1,:].reshape((-1,1)),t,dt,Q)) # Noise included in "true" state propogation

	#Measure
	yt[t,:] = np.transpose(get_measurement(x_s[t,:].reshape((-1,1)),t,dt,R)) # Noise included in measurement of "true" state

	C = get_C(mu_t1_t,dt,t)

	#Update
	K = sig_t1_t@C.T@inv(C@sig_t1_t@C.T+R) #Kalman Gain

	#print("Measurement model")
	#print(yt[t,:].reshape((-1,1))-get_measurement(mu_t1_t,t,dt,R))

	mu_tt = mu_t1_t + K@(yt[t,:].reshape((-1,1))-get_measurement(mu_t1_t,t,dt,0)) # No noise included in measurement model given the predicted state
	xt[t,:] = np.transpose(mu_tt)
	sig_tt = sig_t1_t-K@C@sig_t1_t
	#sig_tt = sig_t1_t
	#sig_tt[0:2,0:2] = sig_tt_
	sigs.append(sig_tt)








fig, axs = plt.subplots(1,3)
state_list = ["p","a","b"]


for n in range(s_s):
	axs[n].plot(t_,x_s[:,n], color='black')
	axs[n].plot(t_,xt[:,n], color='red')
	axs[n].set_xlabel("Time, t")
	axs[n].set_ylabel("State Estimate")
	axs[n].title.set_text("State: %s" % state_list[n])
	axs[n].legend(("True", "Estimate"))


#axs[0].plot(t_,yt,color="blue")

plt.show()

