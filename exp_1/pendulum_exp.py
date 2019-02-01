from casadi import *
import numpy as np 
import math
import gym
import time as t

def dynamics_propagation(x, u, sigma_factor, dt):	
	
	w = sigma_factor*np.random.normal(0.0, 1.0, 2)
	g = 10
	l = 1
	x = blockcat([[x[0] + dt*x[1] + np.sqrt(dt)*w[0]], [x[1] + dt*((-g/l)*sin(x[2]) + u[0]) + np.sqrt(dt)*w[1]]])

	return 	x


def OC(X_0, n_x,n_u, W_x, W_x_f, W_u, K, dt):

	Obj = 0
	opti = Opti()
	g = 9.8
	l = 1

	# Slicing the U and X vectors
	Sl1 = []
	for l in range(0, K + 1):
		Sl1.append(l)

	#  decision variables ----------------

	U = opti.variable(n_u, K) # control trajectory
	U = horzsplit(U, Sl1)

	X = opti.variable(n_x, K) # state trajectory 
	X = horzsplit(X, Sl1)

	#Define objective function
	for t in range(0, K-1):
		Obj = Obj + mtimes(mtimes((X[t] - x_g).T, W_x), (X[t] - x_g)) + mtimes(mtimes(U[t].T, W_u), U[t])

	Obj = Obj + mtimes(mtimes((X[K-1] - x_g).T, W_x_f), (X[K-1] - x_g)) + mtimes(mtimes(U[K-1].T, W_u), U[K-1])


	#constraints
	
	for t in range(0, K):
	
		if t==0:
			opti.subject_to(X[0] == blockcat([[X_0[0] + dt*X_0[1]], [X_0[1] + dt*((-g/l)*sin(X_0[0]) + U[0])]]))
		
		else:
			opti.subject_to(X[t] == blockcat([[X[t-1][0] + dt*X[t-1][1]], [X[t-1][1] + dt*((-g/l)*sin(X[t-1][0]) + U[t])]]))
		
	# Minimize the objective function
	opti.minimize(Obj)
	opts = {}
	opts['ipopt.print_level'] = 0
	opti.solver("ipopt", opts) # set numerical backend
	sol = opti.solve()   # actual solve		# ---- solve NLP              ------
	print("Cost incurred:",sol.value(Obj))
	return [sol.value(X[l]) for l in range(0,K)], [sol.value(U[l]) for l in range(0,K)]

#Initial posiiton
X_0 = DM([pi, 0]) # Initial state
x_g = DM([0, 0]) # goal state
dt = 0.1
K = 30
n_x = 2
n_u = 1


ENV_NAME = 'Pendulum-v0'

# Get the environment and extract the number of actions.
env = gym.make(ENV_NAME)
np.random.seed(123)
env.seed(123)
assert len(env.action_space.shape) == 1
nb_actions = env.action_space.shape[0]


W_x = DM([[10,0],[0,10]])
W_x_f = DM([[10,0],[0,10]])

W_u = DM([[1]])
X_o, U_o = OC(X_0, n_x, n_u, W_x, W_x_f, W_u, K, dt)

a=env.reset()
print(a)
for i in range(0,K):
	env.render()
	observation, r, d, info = env.step([0])
	t.sleep(0)

env.render()
print(X_o, U_o)