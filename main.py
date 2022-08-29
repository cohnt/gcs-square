import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, Circle
import cvxpy as cp

start = np.array([0.5, 0.1])
goal = np.array([3.9, 3.9])
obstacle = np.array([
	[2, 1],
	[3, 2],
	[2, 3],
	[1, 2]
])
world_bounds = np.array([
	[0, 4],
	[0, 4]
])
iris_seed_points = np.array([
	[1, 1],
	[3, 1],
	[1, 3],
	[3, 3]
])

shortest_path = []

def draw_output():
	fig, ax = plt.subplots()
	ax.set_xlim(world_bounds[0])
	ax.set_ylim(world_bounds[1])

	ax.add_patch(Polygon(obstacle, color="red"))
	ax.add_patch(Circle(start, radius=0.1, color="green"))
	ax.add_patch(Circle(goal, radius=0.1, color="blue"))

	ax.scatter(iris_seed_points[:,0], iris_seed_points[:,1], color="black")

	ax.set_aspect("equal")
	plt.show()

draw_output()