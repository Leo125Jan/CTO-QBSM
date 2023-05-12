import numpy as np
import numexpr as ne
import timeit
from math import sqrt, acos, cos
from matplotlib.path import Path
from shapely.geometry import Point
from scipy.stats import multivariate_normal
from shapely.geometry.polygon import Polygon
from scipy.optimize import linear_sum_assignment

a = np.arange(0, 30, 0.1)
b = np.arange(0, 30, 0.1)
X, Y = np.meshgrid(a, b)

W = np.vstack([X.ravel(), Y.ravel()])
W = W.transpose()

def npn():

	return np.linalg.norm(W)

def nen():

	x = W[:,0]; y = W[:,1]
	return ne.evaluate('sqrt(x**2 + y**2)')

if __name__ == '__main__':

	number = 1

	# x = timeit.timeit(stmt = "npn()", number = number, globals = globals())
	# print(x)
	# x = timeit.timeit(stmt = "nen()", number = number, globals = globals())
	# print(x)

	a = np.array([[np.cos(0.5*np.pi), -np.sin(0.5*np.pi)],[np.sin(0.5*np.pi), np.cos(0.5*np.pi)]])
	b = np.array([1, 0])
	# c = ne.evaluate("a/b")

	c = np.array([1, 0, 3])
	print(np.nonzero(c)[0])