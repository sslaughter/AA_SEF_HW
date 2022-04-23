import os
import numpy as np
from scipy.linalg import sqrtm, det, inv
import matplotlib.pyplot as plt

def get_random_vector(mu,cov):
	# Sample normal distribution and then transform per desired parameters to output samples on desired distribution
	num_points = np.size(mu)
	a = np.zeros((num_points,1))

	for i in range(num_points):
		rand_points = np.random.randn(num_points).reshape((-1,1))
		#print(rand_points)
		#print(mu)
		#print(sqrtm(cov)@rand_points)
		#print(sqrtm(cov))
		a[:,0] = np.transpose(sqrtm(cov)@rand_points+mu)

	return a



dt = 1
Q = np.identity(2) # m^2/s^2
R = 9.0*np.identity(2)  # m^2
print(R)
steps = 10
t_ = np.arange(0,steps*dt,step = dt)

A = np.identity(4)
A[0,2] = dt
A[1,3] = dt
print(A)

B = np.zeros((4,4))
B[2,2] = dt
B[3,3] = dt
print(B)

C = np.zeros((2,4))
C[0,0] = 1
C[1,1] = 1

x_o = np.array([[1000],[0],[0],[50]]) # initial state, p0/v0 in state vector
xt = np.zeros((np.size(t_),4)) # Create history of state vector
xt[0,:] = x_o.T
yt = np.zeros((np.size(t_)-1,2)) # Create history of measurement vector


#Simulate trajectory
for t in range(1, np.size(t_)):

	ut = -2.5*np.array([[0],[0],[np.cos(.05*t_[t])],[np.sin(0.5*t_[t])]])
	Wt = get_random_vector(np.zeros((2,1)),Q)
	#print((Wt))
	Wt = np.vstack((np.zeros((2,1)),Wt)) # Make Wt have 4 elements, noise only in velocity
	xt[t,:] = np.transpose(A@xt[t-1,:].reshape((-1,1))+B@ut+Wt)

	Vt = get_random_vector(np.zeros((2,1)),R)
	#print(Vt)
	#Vt = np.vstack((Vt,np.zeros((2,1))))
	print(C)
	print(xt[t,:])
	print(C@xt[t,:].reshape((-1,1)))
	print(np.transpose(C@xt[t,:].reshape((-1,1))+Vt))
	yt[t-1,:] = np.transpose(C@xt[t,:].reshape((-1,1))+Vt)
	
print(xt)
print(yt)

fig, axs = plt.subplots(1,3)
axs[0].plot(t_, xt[:,0], color = 'red') #p1(t)
axs[0].plot(t_[1:], yt[:,0], color = 'black') #y1(t)
axs[1].plot(t_,xt[:,1], color = 'red') #p2(t)
axs[1].plot(t_[1:], yt[:,1], color = 'black') #y2(t)

axs[0].set_xlabel("Time, t")
axs[0].set_ylabel("p1")
axs[0].legend(("Trajectory", "Measurement"))
axs[1].set_xlabel("Time, t")
axs[1].set_ylabel("p2")
axs[1].legend(("Trajectory", "Measurement"))

axs[2].plot(xt[:,0],xt[:,1], color = 'black') #x1,x2; (t)
axs[2].set_xlabel("p1(t)")
axs[2].set_ylabel("p2(t)")
axs[2].legend(("Trajectory"))

plt.show()

#fig.show()



"""
mu_tt = np.array([[1000],[0],[0],[50]])
sig_tt = 
for t in range(steps-1):
	
	t = t+1 # please ignore
	
	#Predict
	ut = -2.5*np.array([[np.cos(.05*t)],[np.sin(0.5*t)]])
	mu_t1_t = A@mu_tt+B@ut
	sig_t1_t = A@sig_tt@A.T + Q

	#Update
	mu_tt = mu_t1_t + 

"""










