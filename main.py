import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, Circle
import cvxpy as cp
from scipy.spatial import HalfspaceIntersection

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
tris = [
	[[0, 0], [4, 0], [2, -1]], # South
	[[0, 0], [0, 4], [-1, 2]], # West
	[[0, 4], [4, 4], [2, 5]], # North
	[[4, 4], [4, 0], [5, 2]], # East
	[[1, 2], [3, 2], [2, 1]], # Obstacle Bottom
	[[1, 2], [3, 2], [2, 3]] # Obstacle Top
]
iris_seed_points = np.array([
	[1, 1],
	[3, 1],
	[1, 3],
	[3, 3]
])
tolerance = 0.00001
max_iters = 10

def draw_output(shortest_path, halfspace_reps):
	fig, ax = plt.subplots()
	ax.set_xlim(world_bounds[0])
	ax.set_ylim(world_bounds[1])

	ax.add_patch(Polygon(obstacle, color="red"))
	ax.add_patch(Circle(start, radius=0.1, color="green"))
	ax.add_patch(Circle(goal, radius=0.1, color="blue"))

	ax.scatter(iris_seed_points[:,0], iris_seed_points[:,1], color="black")

	if len(shortest_path) > 0:
		shortest_path = np.asarray(shortest_path)
		ax.plot(np.asarray(shortest_path)[:,0], np.asarray(shortest_path)[:,1])

	for idx, halfspace_rep in enumerate(halfspace_reps):
		color = plt.get_cmap("Set3")(float(idx) / 12.)
		draw_halfspace_rep(ax, halfspace_rep, color=color)
		
	ax.set_aspect("equal")
	plt.show()

def compute_halfspace(A, b, d):
	ineq = np.hstack((A.T, -b))
	hs = HalfspaceIntersection(ineq, d, incremental=False)
	return hs

def draw_halfspace_rep(ax, halfspace_rep, color):
	points = halfspace_rep.intersections
	center = np.mean(points, axis=0)
	centered_points = points - center
	thetas = np.arctan2(centered_points[:,1], centered_points[:,0])
	idxs = np.argsort(thetas)
	current_region = points[idxs]
	ax.add_patch(Polygon(current_region, color=color, alpha=0.25))
	plt.plot(current_region[:,0], current_region[:,1], color=color, alpha=0.75)
	plt.plot(current_region[[0,-1],0], current_region[[0,-1],1], color=color, alpha=0.75)

def SeparatingHyperplanes(C, d, O):
	C_inv = np.linalg.inv(C)
	C_inv2 = C_inv @ C_inv.T
	O_excluded = []
	O_remaining = O
	ais = []
	bis = []
	while len(O_remaining) > 0:
		xs = []
		dists = []
		for o in O_remaining:
			x, dist = ClosestPointOnObstacle(C, C_inv, d, o)
			xs.append(x)
			dists.append(dist)
		best_idx = np.argmin(dists)
		x_star = xs[best_idx]
		ai, bi = TangentPlane(C, C_inv2, d, x_star)
		ais.append(ai)
		bis.append(bi)
		idx_list = []
		for i, li in enumerate(O_remaining):
			redundant = [np.dot(ai.flatten(), xj) >= bi for xj in li]
			if i == best_idx or np.all(redundant):
				idx_list.append(i)
		for i in reversed(idx_list):
			O_excluded.append(O_remaining[i])
			O_remaining.pop(i)
	A = np.array(ais).T[0]
	b = np.array(bis).reshape(-1,1)
	return (A, b)

def ClosestPointOnObstacle(C, C_inv, d, o):
	v_tildes = C_inv @ (o - d).T
	n = 2
	m = len(o)
	x_tilde = cp.Variable(n)
	w = cp.Variable(m)
	prob = cp.Problem(cp.Minimize(cp.sum_squares(x_tilde)), [
		v_tildes @ w == x_tilde,
		w @ np.ones(m) == 1,
		w >= 0
	])
	prob.solve()
	x_tilde_star = x_tilde.value
	dist = np.sqrt(prob.value) - 1
	x_star = C @ x_tilde_star + d
	return x_star, dist

def TangentPlane(C, C_inv2, d, x_star):
	a = 2 * C_inv2 @ (x_star - d).reshape(-1, 1)
	b = np.dot(a.flatten(), x_star)
	return a, b

def InscribedEllipsoid(A, b):
	n = 2
	C = cp.Variable((n,n), symmetric=True)
	d = cp.Variable(n)
	constraints = [C >> 0]
	constraints += [
		cp.atoms.norm2(ai.T @ C) + (ai.T @ d) <= bi for ai, bi in zip(A.T, b)
	]
	prob = cp.Problem(cp.Maximize(cp.atoms.log_det(C)), constraints)
	prob.solve()
	return C.value, d.value

def solve_iris_region(seed_point):
	print("Growing convex region for seed point %s" % seed_point)
	As = []
	bs = []
	Cs = []
	ds = []
	
	C0 = np.eye(2) * 0.01
	Cs.append(C0)
	ds.append(seed_point.copy())
	O = tris

	iters = 0
	while True:
		# print("Iteration %d" % iters)

		A, b = SeparatingHyperplanes(Cs[-1], ds[-1], O.copy())
		if np.any(A.T @ seed_point >= b.flatten()):
			# print("Terminating early to keep seed point in region.")
			break

		As.append(A)
		bs.append(b)

		C, d = InscribedEllipsoid(As[-1], bs[-1])
		Cs.append(C)
		ds.append(d)

		iters += 1

		if (np.linalg.det(Cs[-1]) - np.linalg.det(Cs[-2])) / np.linalg.det(Cs[-2]) < tolerance:
			break

		if iters > max_iters:
			break

	print("Done")
	return As[-1], bs[-1], Cs[-1], ds[-1]

region_tuples = [solve_iris_region(seed_point) for seed_point in iris_seed_points]
halfspace_reps = [compute_halfspace(A, b, d) for A, b, _, d, in region_tuples]

draw_output([], halfspace_reps)