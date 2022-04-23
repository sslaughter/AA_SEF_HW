import os
import numpy as np
from scipy.linalg import sqrtm, det, inv
import matplotlib.pyplot as plt

def get_random_vector(mu,cov):
	# Sample normal distribution and then transform per desired parameters to output samples on desired distribution
	num_points = np.size(mu)
	a = np.zeros((num_points,1))

	for i in range(num_points):
		rand_points = np.random.rand(num_points).reshape((-1,1))
		a[:,0] = np.transpose(sqrtm(cov)@rand_points+mu)

	return a


P = .95
dt = .5
Q = np.identity(2) # m^2/s^2
R = 9*np.identity(2)  # m^2

steps = 20
t_ = np.arange(0,steps*dt,step = dt)

A = np.identity(4)
A[0,2] = dt
A[1,3] = dt
print("A: \n")
print(A)

B = np.zeros((4,4))
B[2,2] = dt
B[3,3] = dt
print("B: \n")
print(B)

C = np.zeros((2,4))
C[:,2:4] = np.identity(2)
#C[3,3] = 0
print("C: \n")
print(C)
print("C.T: \n")
print(np.transpose(C))

#yt = np.zeros((np.size(t_),4)) # Create history of measurement vector
yt = np.zeros((np.size(t_),2))

mu_tt = np.array([[1000],[0],[0],[50]])

xt = np.zeros((np.size(t_),4)) # Create history of predicted state vector
xt[0,:] = mu_tt.T

x_s = np.zeros((np.size(t_),4)) # Create history of simulated(true) state vector
x_s[0,:] = mu_tt.T

sig_tt = np.identity(4)
sig_tt[0,0] = 1
sig_tt[1,1] = 1

sigs = []
sigs.append(sig_tt)

R_b = np.zeros((4,4))
R_b[2:4,2:4] = R

Q_b = np.zeros((4,4))
Q_b[2:4,2:4] = Q

#Simulate trajectory
for t in range(1, np.size(t_)):
	
	#Predict
	ut = -2.5*np.array([[0],[0],[np.cos(.05*t_[t])],[np.sin(0.5*t_[t])]])
	mu_t1_t = A@mu_tt+B@ut
	print(B@ut)
	#x_s[t,:] = np.transpose(mu_t1_t)
	sig_t1_t = A@sig_tt@A.T + Q_b
	
	#Get "true" state
	Wt = get_random_vector(np.zeros((2,1)),Q)
	#print((Wt))
	Wt = np.vstack((np.zeros((2,1)),Wt)) # Make Wt have 4 elements, noise only in velocity
	x_s[t,:] = np.transpose(A@x_s[t-1,:].reshape((-1,1))+B@ut+Wt)

	#Measure
	Vt = get_random_vector(np.zeros((2,1)),R)
	#Vt = np.vstack((Vt,np.zeros((2,1))))
	yt[t,:] = np.transpose(C@x_s[t,:].reshape((-1,1))+Vt)


	#Update
	K = sig_t1_t@C.T@inv(C@sig_t1_t@C.T+R) #Kalman Gain

	print("Kalman Gain: \n")
	print(K)

	mu_tt = mu_t1_t + K@(yt[t].reshape((-1,1))-C@mu_t1_t.reshape((-1,1)))
	xt[t,:] = np.transpose(mu_tt)
	sig_tt = (np.identity(4)-K@C)@sig_t1_t
	#sig_tt = sig_t1_t
	#sig_tt[0:2,0:2] = sig_tt_
	sigs.append(sig_tt)



"""
	ut = -2.5*np.array([[0],[0],[np.cos(.05*t_[t])],[np.sin(0.5*t_[t])]])
	Wt = get_random_vector(np.zeros((2,1)),Q)
	#print((Wt))
	Wt = np.vstack((np.zeros((2,1)),Wt)) # Make Wt have 4 elements, noise only in velocity
	xt[t,:] = np.transpose(A@xt[t-1,:].reshape((-1,1))+B@ut+Wt)

	Vt = get_random_vector(np.zeros((2,1)),R)
	#print(Vt)
	Vt = np.vstack((Vt,np.zeros((2,1))))
	yt[t-1,:] = np.transpose(C@xt[t,:].reshape((-1,1))+Vt)
"""

"""


"""
plt.figure(1)
plt.plot(xt[:,0],xt[:,1],color = 'blue') #p1/p2(t) predicted
plt.plot(x_s[:,0],x_s[:,1],color = 'green') #p1/p2(t) simulate

#plt.xlim([1100,1600])
#plt.ylim([100,600])



#plot error ellipses for trajectory
for j in range(1,np.size(t_)):
	# Compute values for border of "error circle.", implementation taken from ps2 solution
	r = np.sqrt(-2*np.log(1-P))
	theta = np.linspace(0, 2*np.pi)
	w = np.stack((r*np.cos(theta), r*np.sin(theta)))
	x = sqrtm(sigs[j][0:2,0:2])@w + xt[j,0:2].reshape((-1,1))

	#vel_endpoints[j] = x_s[j,0:2] + x_s[j,2:4]

	#v = sqrtm(sigs[j][2:4,2:4])@w + vel_endpoints[j].reshape((-1,1))

	plt.plot(x[0,:], x[1,:], color="red", linestyle='--')
	#plt.plot(v[0,:], v[1,:], color="black", linestyle = "--")
	#plt.plot(ell_p[:,0],ell_p[:,1], color = 'red')
	#plt.plot(v_ell[:,0],v_ell[:,1], color = 'black',linestyle="--")
	#print(vel_endpoints)
	#plt.plot((x_s[j,0],vel_endpoints[j,0]),(x_s[j,1],vel_endpoints[j,1]), color = "orange", linestyle="--")


plt.legend(("Predicted State", "Simulated State", "Position Covariance"))
plt.xlabel("p1(t)")
plt.ylabel("p2(t)")
plt.show()


plt.figure(2)
plt.plot(xt[:,0],xt[:,1],color = 'blue') #p1/p2(t) predicted
plt.plot(x_s[:,0],x_s[:,1],color = 'green') #p1/p2(t) simulate


vel_endpoints = np.zeros((len(t_),2))

for k in range(1,np.size(t_)):
	# Compute values for border of "error circle.", implementation taken from ps2 solution
	r = np.sqrt(-2*np.log(1-P))
	theta = np.linspace(0, 2*np.pi)
	w = np.stack((r*np.cos(theta), r*np.sin(theta)))

	vel_endpoints[k] = x_s[k,0:2] + x_s[k,2:4]

	v = sqrtm(sigs[k][2:4,2:4])@w + vel_endpoints[k].reshape((-1,1))

	plt.plot(v[0,:], v[1,:], color="black", linestyle = "--")
	plt.plot((x_s[k,0],vel_endpoints[k,0]),(x_s[k,1],vel_endpoints[k,1]), color = "black", linestyle="--")


plt.legend(("Predicted State", "Simulated State", "Velocity Covariance", "Velocity Vector"))
plt.xlabel("p1(t)")
plt.ylabel("p2(t)")
plt.show()













