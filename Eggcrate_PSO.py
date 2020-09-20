import random
import time
import matplotlib.pyplot as plt
import timeit
import numpy as np


# ------------------------------------------------------------------------------
# this is the real "objective function" of the problem without penalty terms
def costFct(x1, x2):
    return x1**2 + x2**2 + 25 * (np.sin(x1)**2 + np.sin(x2)**2)

def computeCost(posList, costFct):
    cost = []
    for each_position in posList:
        c = costFct(*each_position)
        cost.append(c)
    return cost

# Here is the "fitness function", composed of cost function of the problem and a related penalty function
def objective_function(O):
    # the following are variables, number of variables = 2
    x1 = O[0]
    x2 = O[1]

    # the following equations are Inequality Constraints, number of constraints = 0
    constraint_1 = 0
    constraint_2 = 0

    # penalty functions of each of the constraint
    p = 1
    if constraint_1 > 0:
        penalty1 = p
    else:
        penalty1 = 0

    if constraint_2 > 0:
        penalty2 = p
    else:
        penalty2 = 0

    # here goes the cost function
    global_penalty = penalty1 + penalty2
    z = x1**2 + x2**2 + 25 * (np.sin(x1)**2 + np.sin(x2)**2) + global_penalty
    return z

# The variables bounds for the problem are as follows:
bounds = [(-2 * np.pi, 2 * np.pi),   # upper and lower bounds of x1
          (-2 * np.pi, 2 * np.pi)]   # upper and lower bounds of x2

nv = 2  # number of variables
mm = -1  # if minimization problem, mm = -1; if maximization problem, mm = 1

# PARAMETERS OF PSO
particle_size = 30  # number of particles, normally 10 to 60
iterations = 60  # max number of iterations
w = 0.8  # inertia constant (0.4 to 1.4)
c1 = 1.5  # cognative constant (1.5 to 2)
c2 = 2.2 # social constant (2 to 2.5)

# Visualization
fig = plt.figure()
ax = fig.add_subplot()
fig.show()
plt.title('Evolutionary process of Eggcrate Function')
plt.xlabel("Iteration")
plt.ylabel("Cost function")


# ------------------------------------------------------------------------------
class Particle:
    def __init__(self, bounds):
        self.particle_position = []  # particle position
        self.particle_velocity = []  # particle velocity
        self.local_best_particle_position = []  # best position of the particle
        self.fitness_local_best_particle_position = initial_fitness  # initial objective function value of the best particle position
        self.fitness_particle_position = initial_fitness  # objective function value of the particle position

        for i in range(nv):
            self.particle_position.append(
                random.uniform(bounds[i][0], bounds[i][1]))  # generate random initial position
            self.particle_velocity.append(random.uniform(-1, 1))  # generate random initial velocity

    def evaluate(self, objective_function):
        self.fitness_particle_position = objective_function(self.particle_position)
        if mm == -1:
            if self.fitness_particle_position < self.fitness_local_best_particle_position:
                self.local_best_particle_position = self.particle_position  # update the local best
                self.fitness_local_best_particle_position = self.fitness_particle_position  # update the fitness of the local best
        if mm == 1:
            if self.fitness_particle_position > self.fitness_local_best_particle_position:
                self.local_best_particle_position = self.particle_position  # update the local best
                self.fitness_local_best_particle_position = self.fitness_particle_position  # update the fitness of the local best

    def update_velocity(self, global_best_particle_position):
        for i in range(nv):
            r1 = random.random()
            r2 = random.random()

            cognitive_velocity = c1 * r1 * (self.local_best_particle_position[i] - self.particle_position[i])
            social_velocity = c2 * r2 * (global_best_particle_position[i] - self.particle_position[i])
            self.particle_velocity[i] = w * self.particle_velocity[i] + cognitive_velocity + social_velocity

    def update_position(self, bounds):
        for i in range(nv):
            self.particle_position[i] = self.particle_position[i] + self.particle_velocity[i]

            # check and repair to satisfy the upper bounds
            if self.particle_position[i] > bounds[i][1]:
                self.particle_position[i] = bounds[i][1]
            # check and repair to satisfy the lower bounds
            if self.particle_position[i] < bounds[i][0]:
                self.particle_position[i] = bounds[i][0]


class PSO:
    print('Running PSO Algorithm ...')
    def __init__(self, objective_function, bounds, particle_size, iterations):
        fitness_global_best_particle_position = initial_fitness
        global_best_particle_position = []
        swarm_particle = []
        for i in range(particle_size):
            swarm_particle.append(Particle(bounds))
        A = []
        particle_hist = []  # to store all the navigated particle positions
        cost_hist = []  # to store the cost values corresponding to navigated particle positions

        for i in range(iterations):
            for j in range(particle_size):
                swarm_particle[j].evaluate(objective_function)

                if mm == -1:
                    if swarm_particle[j].fitness_particle_position < fitness_global_best_particle_position:
                        global_best_particle_position = list(swarm_particle[j].particle_position)
                        fitness_global_best_particle_position = float(swarm_particle[j].fitness_particle_position)
                if mm == 1:
                    if swarm_particle[j].fitness_particle_position > fitness_global_best_particle_position:
                        global_best_particle_position = list(swarm_particle[j].particle_position)
                        fitness_global_best_particle_position = float(swarm_particle[j].fitness_particle_position)
            for j in range(particle_size):
                swarm_particle[j].update_velocity(global_best_particle_position)
                swarm_particle[j].update_position(bounds)
                particle_hist.append(swarm_particle[j].particle_position)

            A.append(fitness_global_best_particle_position)  # record the best fitness
            cost_hist = computeCost(particle_hist, costFct)  # returns a list of costs of navigated positions
            optimal_cost = costFct(*global_best_particle_position)

            # Visualization
            ax.plot(A, color='b')
            fig.canvas.draw()
            ax.set_xlim(left=max(0, i - iterations), right=i + 3)
            time.sleep(0.001)

        print('RESULT:')
        print('Optimal solution:', global_best_particle_position)
        print('Cost function value:', optimal_cost)
        print('Fitness function value:', fitness_global_best_particle_position)



# ------------------------------------------------------------------------------
if mm == -1:
    initial_fitness = float("inf")  # for minimization problem
if mm == 1:
    initial_fitness = -float("inf")  # for maximization problem
# ------------------------------------------------------------------------------
# Main PSO
# Call the PSO model with given parameters and show me the plot of progress
start = timeit.default_timer()
PSO(objective_function, bounds, particle_size, iterations)
stop = timeit.default_timer()
execution_time = stop - start
print("Program Executed in seconds: ", str(execution_time))
plt.show()