import numpy as np
from math import sqrt, acos, cos
from matplotlib.path import Path
from shapely.geometry import Point
from scipy.stats import multivariate_normal
from shapely.geometry.polygon import Polygon
from scipy.optimize import linear_sum_assignment

if __name__ == '__main__':

	# cost_1 = np.array([[4, 5, 100],
	# 					[7, 13, 100], 
	# 					[3, 6, 100]])
	# row_ind, col_ind = linear_sum_assignment(cost_1)
	# print(row_ind)
	# print(col_ind, "\n")

	# cost_2 = np.array([[2, 0, 5], [4, 1, 3], [3, 2, 2]])
	# row_ind, col_ind = linear_sum_assignment(cost_2)
	# print(row_ind)
	# print(col_ind, "\n")

	# cost_3 = np.array([[3, 2, 2], [2, 0, 5], [4, 1, 3]])
	# row_ind, col_ind = linear_sum_assignment(cost_3)
	# print(row_ind)
	# print(col_ind)

	# Check for AtoT-1
	'''
	cost_matrix = [self.dist_to_cluster]
	Teammate_matrix = [self.Cluster_Teammate]
	count = np.ones(len(Cluster))
	count[self.Cluster_Teammate[1]] = 0

	for neighbor in self.neighbors:

		temp1 = neighbor.dist_to_cluster
		cost_matrix = np.concatenate((cost_matrix, [temp1]), axis = 0)

		if (neighbor.Cluster_Teammate != None).all() and\
			(neighbor.Cluster_Teammate[0] == 1):

			temp2 = neighbor.Cluster_Teammate
			Teammate_matrix = np.concatenate((Teammate_matrix, [temp2]), axis = 0)
			count[neighbor.Cluster_Teammate[1]] = 0
		elif (neighbor.Cluster_Teammate != None).all() and\
				(neighbor.Cluster_Teammate[0] == 0 and\
					neighbor.Cluster_Teammate[1] == escape_index):
			temp2 = np.array([1, 1])
			Teammate_matrix = np.concatenate((Teammate_matrix, [temp2]), axis = 0)
			count[temp2[1]] = 0
		else:
			temp2 = np.array([None, None])
			Teammate_matrix = np.concatenate((Teammate_matrix, [temp2]), axis = 0)

	if not (count == 0).all() and (Teammate_matrix != None).all():

		dist_untracked = cost_matrix[:,np.nonzero(count)[0]]
		dispatch_index = np.argmin(dist_untracked)

		if dispatch_index == self.id:

			self.target = [Cluster[np.nonzero(count)[0]]]
			self.Cluster_Teammate = np.array([1, np.nonzero(count)[0]])

	if ((count == 0).all() and (Teammate_matrix != None).all()) and\
		(self.Cluster_Teammate[0] == 1 and self.Cluster_Teammate[1] == 1):

		cost_matrix = [self.dist_to_cluster]

		for neighbor in self.neighbors:

			if (neighbor.Cluster_Teammate[0] == 1 and\
					neighbor.Cluster_Teammate[1] == 1):

				temp1 = neighbor.dist_to_cluster
				cost_matrix = np.concatenate((cost_matrix, [temp1]), axis = 0)

				dist_untracked = cost_matrix[:,0]
				dispatch_index = np.argmin(dist_untracked)

				if dispatch_index == self.id:

					self.target = [Cluster[0]]
					self.Cluster_Teammate = np.array([1, 0])
	'''
	
	L = 8
	Wi = 8
	x_range = np.arange(-L/2, L/2, 0.01)
	y_range = np.arange(0, L, 0.01)
	X, Y = np.meshgrid(x_range, y_range)

	W = np.vstack([X.ravel(), Y.ravel()])
	W = W.transpose()

	A = np.array([0, 0])
	B = np.array([2, 7])
	C = np.array([-2, 7])

	pos = np.array([0, 0])
	perspective = np.array([0, 1])
	range_ = 5; lambda_ = 2; AOV = np.pi/6; alpha = AOV*0.5;
	range_max = ( (lambda_+1)/lambda_ )*range_*cos(alpha);

	F1 = multivariate_normal([0.0, 5.5], [[0.5, 0.0], [0.0, 0.5]]) # row with shape 1*25000
	F2 = multivariate_normal([0.3, 5.0], [[0.5, 0.0], [0.0, 0.5]])
	F3 = multivariate_normal([-0.3, 5.0], [[0.5, 0.0], [0.0, 0.5]])

	P = lambda d, a, IoO: np.multiply(np.multiply(\
						np.divide(np.power(np.subtract(d, range_max), 2).transpose()[0], range_max**2),\
						np.divide(np.power(np.subtract(abs(a), alpha), 2), alpha**2)), IoO)

	P_ = lambda d, a, IoO, P0: P0*np.multiply(np.multiply(\
						np.exp(-np.divide(np.power(np.subtract(d, range_max), 2).transpose()[0], 2*0.5**2)),\
						np.exp(-np.divide(np.power(np.subtract(abs(a), alpha), 2), 2*0.5**2))), IoO)

	pt = [A+np.array([0, 0.1]), B+np.array([0.1, -0.1]), C+np.array([-0.1, -0.1]), A+np.array([0, 0.1])]
	polygon = Path(pt)
	In_polygon = polygon.contains_points(W)

	d = np.linalg.norm(np.subtract(W, pos), axis = 1)
	d = np.array([d]).transpose()
	a = np.arccos( np.dot(np.divide(np.subtract(W, pos),np.concatenate((d,d), axis = 1)), perspective) )

	# dist_prob_I = np.divide(np.power(np.subtract(d, range_max), 2).transpose()[0], range_max**2)
	# persp_prob_I = np.divide(np.power(np.subtract(abs(a), alpha), 2), alpha**2)
	# JP_Interior = np.multiply(np.multiply(dist_prob_I, persp_prob_I), In_polygon)
	JP_Interior = P(d, a, In_polygon)
	HW_Interior = np.sum(np.multiply(F1.pdf(W), JP_Interior))\
				+ np.sum(np.multiply(F2.pdf(W), JP_Interior))\
				+ np.sum(np.multiply(F3.pdf(W), JP_Interior))

	P0 = 0.5
	# dist_prob_B = np.exp(-np.divide(np.power(np.subtract(d, range_max), 2).transpose()[0], 2*0.5**2))
	# persp_prob_B = np.exp(-np.divide(np.power(np.subtract(abs(a), alpha), 2), 2*0.5**2))
	# JP_Boundary = P0*np.multiply(np.multiply(dist_prob_B, persp_prob_B), In_polygon)
	# print( (1/(np.sqrt(2*np.pi)*0.5))*np.exp(-(abs(a) - alpha)**2/(alpha**2)/(2*0.5**2)) )
	d = d.transpose()[0]
	# print( (1/(np.sqrt(2*np.pi)*0.5))*np.exp(-((d-0.5*range_max)**2-(0.5*range_max**2))/(0.5*range_max**2)/(2*0.5**2)) )
	# JP_Boundary = P_(d, a, In_polygon, P0)
	JP_Boundary = (1/(np.sqrt(2*np.pi)*1.0))*np.exp(-(abs(a) - alpha)**2/(alpha**2)/(2*1.0**2))*\
	(1/(np.sqrt(2*np.pi)*1.0))*np.exp(-((d-0.5*range_max)**2-(0.5*range_max**2))/(0.5*range_max**2)/(2*1.0**2))*\
	In_polygon
	HW_Boundary = np.sum(np.multiply(F1.pdf(W), JP_Boundary))\
				+ np.sum(np.multiply(F2.pdf(W), JP_Boundary))\
				+ np.sum(np.multiply(F3.pdf(W), JP_Boundary));

	# print(HW_Interior)
	# print(HW_Boundary)

	Q = lambda W, d, IoO, P0: P0*np.multiply(np.multiply((np.divide(\
						np.dot(np.subtract(W, pos), perspective), d) - np.cos(alpha))/(1 - np.cos(alpha)),\
						np.multiply((range_*np.cos(alpha) - lambda_*(d - range_*np.cos(alpha))),\
						(np.power(d, lambda_)/(range_**(lambda_+1))))), IoO)

	d = d.transpose()[0]
	# q_persp = np.divide(np.subtract(np.divide(np.dot(np.subtract(W, pos), perspective), d), alpha), 1-cos(alpha))
	# q_persp = (np.divide(np.dot(np.subtract(W, pos), perspective), d) - np.cos(alpha))/(1 - np.cos(alpha))
	# q_res = np.divide(np.multiply((range_*cos(alpha) - lambda_*np.subtract(d, range_*cos(alpha))), np.power(d, lambda_)), range_**(lambda_+1))
	# q_res = np.multiply((range_*np.cos(alpha) - lambda_*(d - range_*np.cos(alpha))),(np.power(d, lambda_)/(range_**(lambda_+1))))
	# SQ = Q(W, d, In_polygon, 1.0)

	d = np.linalg.norm(np.subtract(W, pos), axis = 1)
	d = np.array([d]).transpose()
	d = d.transpose()[0]

	q_persp = (np.divide(np.dot(np.subtract(W, pos), perspective), d) - np.cos(alpha))\
			/(1 - np.cos(alpha))

	q_res = np.multiply((range_*np.cos(alpha) - lambda_*(d - range_*np.cos(alpha))),\
							(np.power(d, lambda_)/(range_**(lambda_+1))))

	Q = np.multiply(q_persp, q_res)

	print(W)