import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, Circle
import cvxpy as cp
from scipy.spatial import HalfspaceIntersection
from shapely.geometry import LineString
from scipy.optimize import linprog

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

def draw_output(shortest_path, halfspace_reps, adj_mat):
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

	start_idx = len(halfspace_reps)
	goal_idx = len(halfspace_reps) + 1
	for i in range(len(halfspace_reps)):
		for j in range(i, len(halfspace_reps)):
			if adj_mat[i,j]:
				plt.plot(iris_seed_points[[i,j],0], iris_seed_points[[i,j],1], color="black", linestyle="dashed")
		if adj_mat[i,start_idx]:
			plt.plot([iris_seed_points[i,0],start[0]], [iris_seed_points[i,1],start[1]], color="black", linestyle="dashed")
		if adj_mat[i,goal_idx]:
			plt.plot([iris_seed_points[i,0],goal[0]], [iris_seed_points[i,1],goal[1]], color="black", linestyle="dashed")
		
	ax.set_aspect("equal")
	plt.show()

def compute_halfspace(A, b, d):
	ineq = np.hstack((A.T, -b))
	hs = HalfspaceIntersection(ineq, d, incremental=False)
	return hs

def order_vertices(points):
	center = np.mean(points, axis=0)
	centered_points = points - center
	thetas = np.arctan2(centered_points[:,1], centered_points[:,0])
	idxs = np.argsort(thetas)
	return points[idxs]

def draw_halfspace_rep(ax, halfspace_rep, color):
	points = halfspace_rep.intersections
	current_region = order_vertices(points)
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

def point_inside_region(point, region):
	#
	return not np.any(region.halfspaces @ np.hstack((point, 1)) > 0)

def do_regions_intersect(region1, region2):
	points1 = order_vertices(region1.intersections)
	points2 = order_vertices(region2.intersections)
	for i in range(len(points1)):
		for j in range(len(points2)):
			if edge_edge_intersection(points1[[i, (i+1) % len(points1)]], points2[[j, (j+1) % len(points2)]]):
				return True
	return False

def edge_edge_intersection(edge1, edge2):
	l1 = LineString(edge1)
	l2 = LineString(edge2)
	return not l1.intersection(l2).is_empty

def construct_gcs_adj_mat(regions):
	foo = 2 + len(region_tuples)
	adj_mat = np.zeros((foo, foo))
	start_idx = len(regions)
	goal_idx = len(regions) + 1
	for i in range(len(regions)):
		adj_mat[i,start_idx] = adj_mat[start_idx,i] = point_inside_region(start, regions[i])
		adj_mat[i,goal_idx] = adj_mat[goal_idx,i] = point_inside_region(goal, regions[i])
		for j in range(i+1, len(regions)):
			adj_mat[i,j] = adj_mat[j,i] = do_regions_intersect(regions[i], regions[j])
	return adj_mat

def find_point_in_intersection(region1, region2):
	# See: https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.HalfspaceIntersection.html
	halfspaces = np.vstack((region1.halfspaces, region2.halfspaces))
	norm_vector = np.reshape(np.linalg.norm(halfspaces[:, :-1], axis=1), (halfspaces.shape[0], 1))
	c = np.zeros((halfspaces.shape[1],))
	c[-1] = -1
	A = np.hstack((halfspaces[:, :-1], norm_vector))
	b = - halfspaces[:, -1:]
	res = linprog(c, A_ub=A, b_ub=b, bounds=(None, None))
	x = res.x[:-1]
	y = res.x[-1]
	return x

region_tuples = [solve_iris_region(seed_point) for seed_point in iris_seed_points]
halfspace_reps = [compute_halfspace(A, b, d) for A, b, _, d, in region_tuples]

adj_mat = construct_gcs_adj_mat(halfspace_reps)
find_point_in_intersection(halfspace_reps[0], halfspace_reps[1])

draw_output([], halfspace_reps, adj_mat)