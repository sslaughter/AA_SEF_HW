import os
import numpy as np
from scipy.linalg import sqrtm, det, inv, norm, block_diag
import matplotlib.pyplot as plt

import time



def get_points(r_i,r_o,num_f):

	a = r_i
	f = r_i

	#theta = np.linspace(0,2*np.pi,int(num_f/2))
	#c1 = 5*np.sin(f*theta) + r_i
	#c1 = np.hstack((c1.reshape((-1,1)),theta.reshape((-1,1))))

	#c2 = 8*np.sin(f*theta) + r_o
	#c2 = np.hstack((c2.reshape((-1,1)),theta.reshape((-1,1))))
	#p_points = np.vstack((c1,c2))

	#c_points = np.zeros((2*np.size(theta),2))

	#Convert to cartesian coordinates
	'''
	for p in range(2*np.size(theta)):
		c_points[p,0] = p_points[p,0]*np.cos(p_points[p,1])
		c_points[p,1] = p_points[p,0]*np.sin(p_points[p,1])
	'''
	theta = np.linspace(0,2*np.pi,int(num_f),endpoint=False)
	c_points = np.zeros((np.size(theta),2))
	for p in range(np.size(theta)):
		c_points[p,0] = 20*np.cos(theta[p])
		c_points[p,1] = 20*np.sin(theta[p])


	return c_points

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

	#print(a)
	#a comes out as a column vector
	return a

def state_dynamics(x,t,dt,Q, include_noise=True):

	#State vector is [px,py,theta].T

	phi = np.sin(t) #Input fuction
	vt = 1.0
	if(include_noise):
		wt = get_random_vector(np.zeros(np.size(x)).reshape((-1,1)),Q)
	else:
		wt = np.zeros((np.size(x),1))
	p_x1 = x[0]+dt*vt*np.cos(x[2])
	p_y1 = x[1]+dt*vt*np.sin(x[2])
	th_1 = x[2] + dt*phi

	xp_t1 = np.array([p_x1,p_y1,th_1]).reshape((-1,1))
	x_t1 = np.vstack((xp_t1,x[3:])) + wt
	#print(wt)
	return x_t1


def get_A(x,dt,t):

	A = np.identity(np.size(x)) # A is constant identity matrix
	vt = 1.0
	A[0,2] = -dt*vt*np.sin(x[2])
	A[1,2] = dt*vt*np.cos(x[2])

	return A



def get_C(x,dt,t,feat_ind):

	C = np.zeros((20,np.size(x)))

	n_f = int((np.size(x)-3)/2) #get number of features
	j_f = x[3:].reshape((n_f,2))#get the feature points into mx,my vector

	for i in range(10):
		#Range points, C1a
		norm_r = np.sqrt((j_f[feat_ind[i],0] - x[0,0])**2 + (j_f[feat_ind[i],1] - x[1,0])**2) # norm of mi - pt
		C[i,0] = (x[0,0] - j_f[feat_ind[i],0])/norm_r
		C[i,1] = (x[1,0] - j_f[feat_ind[i],1])/norm_r
		#Range measurements - new C, negative elements of C1a
		C[i,2*feat_ind[i]+3] = (-x[0,0] + j_f[feat_ind[i],0])/norm_r
		C[i,2*feat_ind[i]+4] = (-x[1,0] + j_f[feat_ind[i],1])/norm_r

		#Bearing points
		norm_b = (j_f[feat_ind[i],0] - x[0,0])**2 + (j_f[feat_ind[i],1] - x[1,0])**2 # not a norm, (mx-px) + (my-py)


		#C1b
		C[i+10,0] = (j_f[feat_ind[i],1]-x[1,0] )/norm_b #py - miy
		C[i+10,1] = (x[0,0]-j_f[feat_ind[i],0])/norm_b #mix - px
		C[i+10,2] = -1

		#C2
		C[i+10,2*feat_ind[i]+3] = (-j_f[feat_ind[i],1]+x[1,0])/norm_b #miy - py
		C[i+10,2*feat_ind[i]+4] = (-x[0,0]+j_f[feat_ind[i],0])/norm_b #px - mix
		#print(C)
	return C

def get_C_p(x,dt,t,feat_ind):

	C = np.zeros((2,2))

	mx = x[2*feat_ind+3] #feature x (we only passed a single index)
	my = x[2*feat_ind+4] # feature y
	norm_r = np.sqrt((mx - x[0,0])**2 + (my - x[1,0])**2)
	norm_b = (mx - x[0,0])**2 + (my - x[1,0])**2 # not a norm, (mx-px) + (my-py)
	C[0,0] = (mx-x[0,0])/norm_r#range measurment, dg/dmx
	C[0,1] = (my-x[1,0])/norm_r#range measurment, dg/dmy
	C[1,0] = (x[1,0]-my)/norm_b#bearing measurement, dg/dmx
	C[1,1] = (mx-x[0,0])/norm_b#bearing mesurement, dg/dmy

	return C


def get_measurement(x,t,dt,R,true_state = False,feat_ind = None):

	vt = get_random_vector(np.zeros(20).reshape((-1,1)),R)
	y = np.zeros((20,1))
	#print(x)
	n_f = int((np.size(x)-3)/2) #get number of features
	j_f = x[3:].reshape((n_f,2))#get the feature points into mx,my vector

	#print(j_f)
	ra = np.zeros((n_f,3))
	
	#print("shape x")
	#print(np.shape(x))
	ra[:,:2] = j_f
	for j in range(n_f):
		ra[j,2] = np.sqrt((ra[j,0]-x[0,0])**2 +(ra[j,1]-x[1,0])**2) # get distance between state position and feature j
	#Get 10 nearest neighbors
	if true_state:
		#print("Actual feature locations")
		#print(j_f)
		#print(x[:3])
		ra_s = np.argsort(ra[:,2]) # get sorted list of indeces based on range to feature from robot
		nearest_f = ra_s[:10]
	else:
		#use nearest neighbor indices from true state measurement
		##print("Estimated feature locations")
		#print(j_f)
		#print(x[:3])
		nearest_f = feat_ind

	#Get measurements for nearest features
	for i in range(10):
		#Rnage measurement
		#y[i,0] = np.sqrt((x[2*nearest_f[i]+3,0] - x[0,0])**2 + (x[2*nearest_f[i]+4,0] - x[1,0])**2) + vt[i] # Measure distance between state positions and state feature locations
		#ra still holds all of the feature positions with the range to the current state that we're measuring from!

		y[i,0] = ra[nearest_f[i],2] + vt[i]
		#Bearing measurement
		y[i+10,0] = -x[2,0] + np.arctan2((ra[nearest_f[i],1]-x[1,0]),(ra[nearest_f[i],0]-x[0,0])) + vt[i+10]
		#y[i+10,0] = wrapToPi(y[2*i+1,0])
	#print("y =")
	#print(np.shape(y))
	return y, nearest_f # return measurements and indices for nearest neighbor


def get_measurement_f(x,t,dt,R,true_state = False,feat_ind = None):

	vt = get_random_vector(np.zeros(20).reshape((-1,1)),R)
	y = np.zeros((20,1))
	#print(x)
	n_f = int((np.size(x)-3)/2) #get number of features
	j_f = x[3:].reshape((n_f,2))#get the feature points into mx,my vector

	#print(j_f)
	ra = np.zeros((n_f,3))
	
	#print("shape x")
	#print(np.shape(x))
	ra[:,:2] = j_f
	for j in range(n_f):
		ra[j,2] = np.sqrt((ra[j,0]-x[0,0])**2 +(ra[j,1]-x[1,0])**2) # get distance between state position and feature j
	#Get 10 nearest neighbors
	if true_state:
		#print("Actual feature locations")
		#print(j_f)
		#print(x[:3])
		ra_s = np.argsort(ra[:,2]) # get sorted list of indeces based on range to feature from robot
		nearest_f = ra_s[:10]
	else:
		#use nearest neighbor indices from true state measurement
		##print("Estimated feature locations")
		#print(j_f)
		#print(x[:3])
		nearest_f = feat_ind

	#Get measurements for nearest features
	for i in range(10):
		#Rnage measurement
		#y[i,0] = np.sqrt((x[2*nearest_f[i]+3,0] - x[0,0])**2 + (x[2*nearest_f[i]+4,0] - x[1,0])**2) + vt[i] # Measure distance between state positions and state feature locations
		#ra still holds all of the feature positions with the range to the current state that we're measuring from!
		y[2*i,0] = ra[nearest_f[i],2] + vt[2*i]
		#Bearing measurement
		y[2*i+1,0] = -x[2,0] + np.arctan2((ra[nearest_f[i],1]-x[1,0]),(ra[nearest_f[i],0]-x[0,0])) + vt[2*i+1]
		y[2*i+1,0] = wrapToPi(y[2*i+1,0])
	#print("y =")
	#print(np.shape(y))
	return y, nearest_f # return measurements and indices for nearest neighbor


def run_ekf(t_,x_true,Q,R,num_states, sig_i, iterative=False, smooth_eks=False):

	yt = np.zeros((np.size(t_),20))
	xt = np.zeros((np.size(t_),num_states)) # Create history of predicted state vector
	#xt[0,:] = x_true[0,:] #set initial state - already sampled to generate trajectory
	print(x_o.T)
	xt[0,:] = x_o.T #x_o is a global variable here because I'm lazy

	mu_tt = xt[0,:].reshape((-1,1))
	sig_tt = sig_i

	sigs = np.zeros((s_s,s_s,np.size(t_)))
	#print(np.shape(sigs))
	sigs[:,:,0] = sig_tt

	A_hist = np.zeros((s_s,s_s,np.size(t_)))

	start_time = time.time()
	for t in range(1, np.size(t_)):
		
		#Predict
		#print("Mutt")
		#print(np.shape(mu_tt))

		mu_t1_t = state_dynamics(mu_tt,t_[t],dt,Q, False)
		#print("Initial Mu")
		#print(np.shape(mu_t1_t))


		A = get_A(mu_tt,dt,t)
		A_hist[:,:,t] = A

		sig_t1_t = A@sig_tt@A.T + Q

		#Measure
		y_temp,f_i = get_measurement(x_true[t,:].reshape((-1,1)),t_[t],dt,R,True)
		yt[t,:] = y_temp.T

		# Need to update C to only account for measured indices
		C = get_C(mu_t1_t,dt,t,f_i)

		#Update
		if not iterative:

			K = sig_t1_t@C.T@inv(C@sig_t1_t@C.T+R) #Kalman Gain
			#print("K = ")
			#print(K)
 
			y_inov = yt[t,:].reshape((-1,1))-get_measurement(mu_t1_t,t,dt,np.zeros((20,20)), False,f_i)[0]
			#print("y_inov: ")
			#print(y_inov)

			#keep phi within [-pi,pi] thanks to suggestions in edstem post #160/158 and code from AA274A
			for j in range(10,np.size(y_inov)):
				y_inov[j,0] = wrapToPi(y_inov[j,0])

			mu_tt = mu_t1_t + K@(y_inov)
			#print("Mu = ")
			#print(mu_tt)

			xt[t,:] = np.transpose(mu_tt)
			sig_tt = sig_t1_t-K@C@sig_t1_t
			#print(sig_tt)

		sigs[:,:,t] = sig_tt


	end_time = time.time()
	avg_time = (end_time-start_time)/np.size(t_)

	xt_s = np.zeros((np.size(t_),num_states))
	xt_s[-1,:] = xt[-1,:]

	sigs_s = np.zeros((s_s,s_s,np.size(t_)))
	sigs_s[:,:,-1] = sigs[:,:,-1]

	print(sigs[:,:,0])


	return xt, yt, sigs, avg_time


def do_imp_resample_fast(x,f_x,w, num_particles):


	x_new = np.copy(x)
	f_xnew = np.copy(f_x)
	p_w_bins = np.cumsum(w)
	for i in range(np.size(w)):
		samp = np.random.uniform()
		#print(samp)
		for j in range(np.size(w)):
			if samp >= p_w_bins[j]:
				pass
			else:
				x_new[i,:] = x[j,:]
				f_xnew[i,:] = f_x[j,:]
				break

	w_new = (1.0/num_particles)*np.ones(num_particles)

	return x_new, f_xnew, w_new





def run_pf(t_,x_true,Q,R,num_states, sig_i):
	num_particles = 4

	yt = np.zeros((np.size(t_),20))
	xt = np.zeros((np.size(t_),num_states)) # Create history of predicted state vector
	xt[0,:] = x_true[0,:] #set initial state - already sampled to generate trajectory

	mu_tt = xt[0,:].reshape((-1,1))
	sig_tt = sig_i

	sigs = np.zeros((3,3,np.size(t_))) # time history of pose covariances
	sigs[:,:,0] = sig_tt[:3,:3]

	num_features = int((num_states-3)/2)

	f_sigs = np.zeros((2,2,num_features,np.size(t_))) #time history of feature location covariances

	p_w = (1.0/num_particles)*np.ones(num_particles) #initialize particle weights as uniform 1/N
	#print(p_w)
	
	p_x = np.zeros((num_particles,3)) #list of particle robot poses
	p_m_m = np.zeros((num_particles,2*num_features)) #list of particle feature location means 
	p_m_cov = np.zeros((2,2,num_features,num_particles)) #list of particle feature location covariances

	#initialize particles
	for i in range(num_particles):
		p_x[i,:] = get_random_vector(mu_tt[:3],sig_tt[:3,:3]).T #intialize robot pose
		#print("mutt")
		#print(np.shape(mu_tt[:3]))
		p_m_m[i,:] = get_random_vector(mu_tt[3:],sig_tt[3:,3:]).T #initialize feature locations
		for j in range(num_features):
			p_m_cov[:,:,j,i] = np.identity(2) #initialize covariances for feature locations
			f_sigs[:,:,j,0] = p_m_cov[:,:,j,i]


	start_time = time.time()
	for t in range(1, np.size(t_)):
		p_x_p = np.zeros((num_particles,3))
		p_m_p = np.zeros((num_particles,2*num_features))
		#Predict
		for j in range(num_particles):
			p_x_p[j,:] = state_dynamics(p_x[j,:].reshape((-1,1)),t_[t],dt,Q[:3,:3]).T #send robot pose through dynamics
			p_m_p[j,:] = p_m_m[j,:] #feature locations are static

		#Update
		#print("predicted particle states:")
		#print(p_x_p)
		#Get nearest features and true measurement
		y_temp,f_i = get_measurement_f(x_true[t,:].reshape((-1,1)),t_[t],dt,R,True) #get sample measurement based on true state and nearest feature indices
		yt[t,:] = y_temp.T

		#Update through each particle
		for k in range(num_particles):
			mu_p = np.vstack((p_x_p[k,:].reshape((-1,1)),p_m_p[k,:].reshape((-1,1)))) #full predicted state
			
			g_p = get_measurement_f(mu_p,t_[t],dt,np.zeros((20,20)),False,f_i)[0] # Get measurement at 10 nearest features based on predicted mu
			y_inov = yt[t,:].reshape((-1,1))-g_p
			w_f = np.zeros(np.size(f_i))
			#Go through the prediction step on the measured features
			for f in range(np.size(f_i)):
				#mu_p_f = p_m_p[2*f_i[f]:2*f_i[f]+2].reshape((-1,1)) #get just the feature position of one of the nearest features
				R_p = .1*np.identity(2)
				C_p = get_C_p(mu_p,dt,t,f_i[f]) #get C based on that single feature
				#print("feature covariance:")
				#print(p_m_cov[:,:,f_i[f],k])
				#print("feature C")
				#print(C_p)
				#print("before inverse:")
				#print(C_p@p_m_cov[:,:,f_i[f],k]@C_p.T)
				sig_prev = p_m_cov[:,:,f_i[f],k]
				sig_y = C_p@sig_prev@C_p.T+R_p
				#sig_y = inv(C_p@sig_prev@C_p.T+R_p) # get sig_y for weighting function
				#print("sigma y")
				#print(sig_y)
				#print(det(sig_y))
				#print(inv(sig_y))

				y_inov_f = y_inov[2*f:2*f+2,0].reshape((-1,1))#full yt-g is calculated, so only take 2x1 portion pertaining to this feature
				y_inov_f[1,0] = wrapToPi(y_inov_f[1,0]) #wrap bearing measurement to [-pi,pi]

				#print("feature index")
				##print(f_i[f])
				#print("y innovation")
				#print(y_inov_f)
				eta = 1/(np.sqrt((4*np.pi**2)*det(sig_y)))
				#print("eta = %f" % eta)
				w_f[f] = eta*np.exp(-.5*(y_inov_f.T@inv(sig_y)@y_inov_f)) #calculate weight component for this feature/measurement

				K = sig_prev@C_p.T@inv(C_p@sig_prev@C_p.T+R_p) #Kalman Gain

				cor = K@y_inov_f 

				p_m_p[k,2*f_i[f]] = p_m_p[k,2*f_i[f]] + cor[0,0] #mx prediction mux + k*inov_x
				p_m_p[k,2*f_i[f]+1] = p_m_p[k,2*f_i[f]+1] +cor[1,0] #my prediction muy + k*inov_y
				##print("Old covariance")
				#print(sig_prev)
				#print("Kalman gain")
				#print(K)
				#print("C@sig_p")
				#print(C_p@sig_prev)
				#print("correction:")
				#print(K@C_p@sig_prev)
				p_m_cov[:,:,f_i[f],k] = sig_prev-K@C_p@sig_prev #predicted cov for feature
				#print("new covariance:")
				#print(p_m_cov[:,:,f_i[f],k])

			#print(w_f)
			p_w[k] = np.prod(w_f)*p_w[k]
			#print("Weight before normalization:")
			#print(p_w[k])
			p_w[k] = p_w[k]/np.sum(w_f) # normalize particle weight
			#print("Weight after normalization")
			#print(p_w[k])
			##print("feature weights:")
			#print(w_f)
		#print("sum of weights:")
		#print(p_w)
		# Find new mean for position
		#print("pw before normalization")
		#print(p_w)
		p_w_norm = np.sum(p_w)
		p_w = np.multiply(p_w,(1/p_w_norm))

		#print("p_w after normalization")
		#print(p_w)

		p_m_t = np.zeros(3) #pose mean
		for z in range(num_particles):
			p_m_t = p_m_t + np.multiply(p_x_p[z,:3],p_w[z]).T

		#find new cov for position
		sig_tt_p = np.zeros((3,3))
		for z in range(num_particles):
			#x-mub
			d_mu = p_x_p[z,:3] - p_m_t
			#wrapToPi(d_mu[2]) #wrap bearing difference
			sig_tt_p = sig_tt_p + p_w[z]*np.multiply((d_mu).reshape((-1,1)),((d_mu).reshape((-1,1)).T))

		xt[t,:3] = p_m_t
		sigs[:,:,t] = sig_tt_p

		#Find new mean for feature locations
		f_m_t = np.zeros(2*num_features)
		for z in range(num_particles):
			f_m_t = f_m_t + p_m_p[z,:]*p_w[z]

		xt[t,3:] = f_m_t

		#find res of covariance for feature locations
		p_m_cov_t = np.zeros((2,2,num_features)) #one list of feature covariances
		for z in range(num_particles):
			for i in range(num_features):
				p_m_cov_t[:,:,i] = p_m_cov_t[:,:,i]+p_m_cov[:,:,i,z]*p_w[z]

		f_sigs[:,:,:,t] = p_m_cov_t

		p_x,p_m_m,p_w = do_imp_resample_fast(p_x_p,p_m_p,p_w, num_particles)


	end_time = time.time()
	avg_time = (end_time-start_time)/np.size(t_)
	#print("State mean history")
	#print(xt[:,:3])
	#print("Feature covariance history")
	#print(f_sigs)
	return xt,yt,sigs,avg_time







num_features = 100

R = np.diag(np.hstack((.1*np.ones(10),.1*np.ones(10))))
print(R)
#R= .1*np.identity(20) #20 dimensional covariance for measurement noise of feature measurements
#R = np.zeros((8,8))
dt = .1
s_s = 2*num_features + 3 #dimension of the state vector, features + px, py, theta

# Noise covariance for state positions
q_noise = .1
Q = np.zeros((s_s,s_s))
Q[0,0] = q_noise
Q[1,1] = q_noise
Q[2,2] = q_noise*.1


m = get_points(6,15,num_features) # Returns nx2 vector of feature locations
#print(m)


steps = 300
t_ = np.arange(0,steps*dt,step = dt)

mu_o_pos = np.zeros((3,1))

mu_o= np.vstack((mu_o_pos,m.flatten().reshape((-1,1)))) # True states as mean for initial state; m is flattened to alternate mx_i,my_i vertically
sig_o = .1*np.identity(s_s) # Initial covariance for the state

#print(np.shape(mu_o))
# Sample initial pose for robot and feature positions
x_o = get_random_vector(mu_o,sig_o)
#print("Initial state")
#print(x_o)


#initial true state
x_s_o = np.vstack((x_o[:3,0].reshape((-1,1)),m.flatten().reshape((-1,1)))) #initial true state is sampled initial pose plus true feature locations (I got this wrong on hw6)
x_s = np.zeros((np.size(t_),s_s)) # ground-truth trajectory for robot position
x_s[0,:] = x_s_o.T

#print("Initial_true_state")
#print(x_s[0,:])



#print(x_s)

# Get true state propogation
for t in range(1,np.size(t_)):
	x_s[t,:] = np.transpose(state_dynamics(x_s[t-1,:].reshape((-1,1)),t_[t],dt,Q))
	#x_s[t,3:] = m
	#x_s[t,:] = np.transpose(dummy_state_dynamics(x_s[t-1,:].reshape((-1,1)),t,dt,Q))

# No dynamics to the feature locations - true state is fixed



#b = np.argsort(x_s[:,2])
#print(b)

#for n in range(np.size(b)):
#	print(x_s[b[n],2])


ekfs_xt,ekfs_yt,ekfs_sigs,ekfs_time = run_ekf(t_,x_s,Q,R,s_s,sig_o) #EKF SLAM
fs_xt,fs_yt,fs_sigs,fs_time = run_pf(t_,x_s,Q,R,s_s,sig_o) #EKF SLAM
#b_xt,b_yt,b_sigs,b_time = run_ekf(t_,x_s,Q,R,s_s,sig_o,True) #iEKF SLAM - this is more broken than the others right now
#c_xt,c_yt,c_sigs,c_time, c_a_hist, c_xs, c_sigs_s = run_ekf(t_,x_s,Q,R,s_s,sig_o,False,True) #EKS


filter_states = [ekfs_xt,fs_xt]
filter_sigs = [ekfs_sigs,fs_sigs]
filter_meas = [ekfs_yt,fs_yt]
filter_times = [ekfs_time,fs_time]

num_filters = 2
state_list = ["px","py","theta"]
filter_list = ["EKF SLAM", "FastSLAM"]


print("Average calculation times: ")
for n in range(num_filters):
	print("%s: %f" % (filter_list[n],filter_times[n]))

#fig, axs = plt.subplots(4,1)

for f in range(num_filters):
	fig, axs = plt.subplots(3,1)
	plt.figure(2)

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

	for i in range(int(num_features)):
		plt.scatter(m[i,0],m[i,1], color='black',s=12)
		plt.scatter(filter_states[f][:,2*i+3],filter_states[f][:,2*i+4], color='red',s=5)

	plt.plot(filter_states[f][:,0],filter_states[f][:,1],color = 'orange')
	plt.plot(x_s[:,0],x_s[:,1],color = 'blue')
	plt.xlabel("x")
	plt.ylabel("y")
	plt.legend(("Estimate", "True"))

	plt.show()
'''
plt.figure(3)
plt.plot(t_,x_s[:,2]-fs_xt[:,2], color='black')
plt.show()
'''