import os
import numpy as np
from scipy.linalg import sqrtm, det, inv, norm
import matplotlib.pyplot as plt

import time

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

def get_A(x,dt,t):
	vt = 1.0
	A = np.identity(3)
	A[0,2] = -dt*vt*np.sin(x[2])
	A[0,1] = dt*vt*np.cos(x[2])

	#Check observability?
	return A

def get_C(x,dt,t):

	C = np.zeros((1,3))
	norm = np.sqrt(x[0]**2 + x[1]**2)
	C[0,0] = x[0]/norm
	C[0,1] = x[1]/norm

	return C


def get_measurement(x,t,dt,R):

	vt = get_random_vector(0,R)
	y = np.sqrt(x[0]**2 + x[1]**2) + vt

	return y



def run_ekf(t_,x_true,Q,R,num_states, sig_i, iterative=False):

	yt = np.zeros((np.size(t_),1))
	xt = np.zeros((np.size(t_),num_states)) # Create history of predicted state vector
	xt[0,:] = x_true[0,:] #set initial state - already sampled to generate trajectory

	mu_tt = xt[0,:].reshape((-1,1))
	sig_tt = sig_i

	sigs = np.zeros((s_s,s_s,np.size(t_)))
	print(np.shape(sigs))
	sigs[:,:,0] = sig_tt

	start_time = time.time()
	for t in range(1, np.size(t_)):
		
		#Predict
		mu_t1_t = state_dynamics(mu_tt,t_[t],dt,Q, False)
		#mu_t1_t = dummy_state_dynamics(mu_tt,t,dt,Q, False)

		A = get_A(mu_tt,dt,t)
		sig_t1_t = A@sig_tt@A.T + Q

		#Measure
		yt[t,:] = np.transpose(get_measurement(x_true[t,:].reshape((-1,1)),t,dt,R))

		
		C = get_C(mu_t1_t,dt,t)
		#Update
		if not iterative:
			K = sig_t1_t@C.T@inv(C@sig_t1_t@C.T+R) #Kalman Gain

			#print("Measurement model")
			#print(yt[t,:].reshape((-1,1))-get_measurement(mu_t1_t,t,dt,R))

			mu_tt = mu_t1_t + K@(yt[t,:].reshape((-1,1))-get_measurement(mu_t1_t,t,dt,0))
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
					temp_C = get_C(mu_temp_i,dt,t)
				else:
					temp_C = C
				K = sig_t1_t@temp_C.T@inv(temp_C@sig_t1_t@temp_C.T+R)

				if step == 1:
					ek_mu = mu_temp_i + K@(yt[t,:].reshape((-1,1))-get_measurement(mu_temp_i,t,dt,0))

				
				mu_temp_i1 = mu_temp_i + K@(yt[t,:].reshape((-1,1))-get_measurement(mu_temp_i,t,dt,0)) + K@temp_C@(mu_temp_i-mu_t1_t)
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

		rank = np.linalg.matrix_rank((np.vstack((C,C@A,C@A@A))))
		print(rank)
		if rank == num_states:
			print("It's observable!")
		else:
			print("Not observable!")

	end_time = time.time()
	avg_time = (end_time-start_time)/np.size(t_)
	return xt, yt, sigs, avg_time


def get_sigma_points(mu,cov,num_states):

	lam = 2.0
	num_points = 2*num_states+1
	w = np.zeros(num_points)
	x_sig_temp = np.zeros((num_points,num_states))

	w[0] = lam/(lam+num_states)
	x_sig_temp[0,:] = mu.T

	b = 1.0

	for i in range(1,np.size(w)):
		c = i-1
		if i > num_states:
			b = -1.0
			c = i-num_states-1
		x_sig_temp[i,:] = mu.T+np.multiply(b,sqrtm(np.multiply((lam+num_states),cov))[:,c]) #TODO switch this to np.multiply
		#print(x_sig_temp)
		w[i] = 1/(2*(lam+num_states))
	return x_sig_temp,w





def run_ukf(t_,x_true,Q,R,num_states, sig_i):
	yt = np.zeros((np.size(t_),1))
	xt = np.zeros((np.size(t_),num_states)) # Create history of predicted state vector
	xt[0,:] = x_true[0,:] #set initial state - already sampled to generate trajectory

	mu_tt = xt[0,:].reshape((-1,1))
	sig_tt = sig_i

	sigs = np.zeros((s_s,s_s,np.size(t_)))
	sigs[:,:,0] = sig_tt

	n_p = 2*num_states+1 #number of sigma points


	start_time = time.time()
	for t in range(1, np.size(t_)):

		mu_t1_t = np.zeros((num_states,1))
		sig_t1_t = np.zeros((num_states,num_states))

		#Predict:
		x_sigs, w_sigs = get_sigma_points(mu_tt,sig_tt,num_states) #get sigma points and corresponding weights using prior distribution parameters

		x_s_p = np.zeros((n_p,num_states))
		for x in range(n_p):
			x_s_p[x,:] = state_dynamics(x_sigs[x,:].reshape((-1,1)),t_[t],dt,Q,False).T #Send each sigma point through the state dynamics, no noise
			mu_t1_t = mu_t1_t + np.multiply(w_sigs[x],x_s_p[x,:].reshape((-1,1))) # Build up emprical mean from weighted sigma points (inverse transform)

		for x in range(n_p):
			sig_t1_t = sig_t1_t + w_sigs[x]*(x_s_p[x,:].reshape((-1,1))-mu_t1_t)@(x_s_p[x,:].reshape((-1,1))-mu_t1_t).T + Q # Build up emprical covariance from weighted sigma points (inverse transform)

		x_sigs_u, w_sigs_u = get_sigma_points(mu_t1_t,sig_t1_t,num_states) #Get sigma points from predicted mean/cov
		y_s_u = np.zeros((n_p,1))

		for y in range(n_p):
			y_s_u[y] = get_measurement(x_sigs_u[y,:],t_[t],dt,0) #predict measurement for each sigma point
			yt[t] = yt[t] + w_sigs_u[y]*y_s_u[y] #build up expected measurement based on weighted sum of the predicted measurements

		sig_y = np.zeros((1,1))
		sig_xy = np.zeros((3,1))

		for y in range(n_p):
			sig_y = sig_y + w_sigs_u[y]*(y_s_u[y]-yt[t])*(y_s_u[y]-yt[t]).T + R
			#sig_xy = w_sigs_u[y]*np.multiply(x_sigs_u[y,:].reshape((-1,1))-mu_t1_t.reshape((-1,1)),(y_s_u[y]-yt[t]).T) # Scalar multiplication, will not work if y>1D
			sig_xy = sig_xy + w_sigs_u[y]*np.multiply(x_sigs_u[y,:].reshape((-1,1))-mu_t1_t,(y_s_u[y]-yt[t]).T)


		y_t = get_measurement(x_true[t],t_[t],dt,R) #measure true state

		mu_tt = mu_t1_t + np.multiply(sig_xy,(1/sig_y)*(y_t-yt[t]))

		sig_tt = sig_t1_t - (1/sig_y)*sig_xy@sig_xy.T # Again, scalar multiplications

		xt[t,:] = mu_tt.T
		sigs[:,:,t] = sig_tt

	end_time = time.time()
	avg_time = (end_time-start_time)/np.size(t_)
	return xt,yt,sigs, avg_time


def do_imp_resample(x,w, num_particles):


	x_new = np.copy(x)
	p_w_bins = np.cumsum(w)
	for i in range(np.size(w)):
		samp = np.random.uniform()
		#print(samp)
		for j in range(np.size(w)):
			if samp >= p_w_bins[j]:
				pass
			else:
				x_new[i,:] = x[j,:]
				break

	w_new = (1.0/num_particles)*np.ones(num_particles)

	return x_new, w_new





def run_pf(t_,x_true,Q,R,num_states, sig_i):
	num_particles = 1000

	yt = np.zeros((np.size(t_),1))
	xt = np.zeros((np.size(t_),num_states)) # Create history of predicted state vector
	xt[0,:] = x_true[0,:] #set initial state - already sampled to generate trajectory

	mu_tt = xt[0,:].reshape((-1,1))
	sig_tt = sig_i

	sigs = np.zeros((s_s,s_s,np.size(t_)))
	sigs[:,:,0] = sig_tt

	p_w = (1.0/num_particles)*np.ones(num_particles) #initialize particle weights as uniform 1/N
	#print(p_w)
	p_x = np.zeros((num_particles,num_states))

	for i in range(num_particles):
		p_x[i,:] = get_random_vector(mu_tt,sig_tt).T

	start_time = time.time()
	for t in range(1, np.size(t_)):
	#Sample particles from initial mean/cov
		#print(p_x)
		p_x_p = np.zeros((num_particles,num_states))
		#Predict
		for j in range(num_particles):
			p_x_p[j,:] = state_dynamics(p_x[j,:].reshape((-1,1)),t_[t],dt,Q).T #t is not used in the function at all

		#Update
		yt[t] = get_measurement(x_true[t,:],t_[t],dt,R) #get sample measurement based on true state t is not used in the function at all
		for k in range(num_particles):	
			y_x = np.sqrt(p_x_p[k,0]**2 + p_x_p[k,1]**2) # find y given the predicted particle
			p_w[k] = (1/np.sqrt(2*np.pi*R**2))*np.exp((-.5)*(((yt[t]-y_x)**2)/(R**2))) #find the probability of the measurement given x, based on p(vt)

		p_w_norm = np.sum(p_w)
		p_w = np.multiply(p_w,(1/p_w_norm))

		# Find new mean
		mean_temp = np.zeros(3)
		for z in range(num_particles):
			mean_temp = mean_temp + np.multiply(p_x_p[z,:],p_w[z]).T

		sig_tt = np.zeros((3,3))
		for z in range(num_particles):
			sig_tt = sig_tt + p_w[z]*np.multiply((p_x_p[z,:]-mean_temp).reshape((-1,1)),((p_x_p[z,:]-mean_temp).reshape((-1,1)).T))

		xt[t,:] = mean_temp
		sigs[:,:,t] = sig_tt

		p_x,p_w = do_imp_resample(p_x_p,p_w, num_particles)


	end_time = time.time()
	avg_time = (end_time-start_time)/np.size(t_)
	return xt,yt,sigs,avg_time



R = .1
dt = .1
s_s = 3 #dimension of the state vector

Q = .1*dt*np.identity(3)


steps = 500
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



ekf_xt,ekf_yt,ekf_sigs,ekf_time = run_ekf(t_,x_s,Q,R,s_s,sig_o)
iekf_xt,iekf_yt,iekf_sigs,iekf_time = run_ekf(t_,x_s,Q,R,s_s,sig_o, True)
ukf_xt,ukf_yt,ukf_sigs,ukf_time = run_ukf(t_,x_s,Q,R,s_s,sig_o)
pf_xt,pf_yt,pf_sigs,pf_time = run_pf(t_,x_s,Q,R,s_s,sig_o)



filter_states = [ekf_xt,iekf_xt,ukf_xt,pf_xt]
filter_sigs = [ekf_sigs,iekf_sigs,ukf_sigs, pf_sigs]
filter_meas = [ekf_yt,iekf_yt, ukf_yt,pf_yt]
filter_times = [ekf_time, iekf_time, ukf_time, pf_time]

filter_states = [ekf_xt]
filter_sigs = [ekf_sigs]
filter_meas = [ekf_yt]
filter_times = [ekf_time]

num_filters = 1
state_list = ["px","py","theta"]
filter_list = ["ekf", "iekf", "ukf", "pf"]


print("Average calculation times: ")
for n in range(num_filters):
	print("%s: %f" % (filter_list[n],filter_times[n]))

fig, axs = plt.subplots(num_filters,3)

for f in range(num_filters):
	for n in range(s_s):
		axs[f][n].plot(t_,x_s[:,n], color='black')
		axs[f][n].plot(t_,filter_states[f][:,n], color='red')
		axs[f][n].fill_between(t_, filter_states[f][:,n] - 1.96*np.sqrt(filter_sigs[f][n,n,:]), #95% confidence interval shading, adapted from test.py file that was provided
                       filter_states[f][:,n] + 1.96 * np.sqrt(filter_sigs[f][n,n,:]),
                        color='red', alpha=0.3)
		axs[f][n].set_xlabel("Time, t")
		axs[f][n].set_ylabel("%s,%s" % (filter_list[f],state_list[n]))
		#axs[f][n].title.set_text("Filter %s" % filter_list[f])
		axs[f][n].legend(("True", "Estimate"))


plt.show()

plt.figure(2)
plt.plot(t_,x_s[:,0],color = 'black') #p1/p2(t) predicted
plt.plot(t_,ekf_xt[:,0],color = 'blue')  #p1/p2(t) simulate
plt.plot(t_,iekf_xt[:,0],color = 'green')
plt.legend(("True", "EKF", "iEKF"))
plt.ylabel("Px")
plt.xlabel("Time")
plt.show()
