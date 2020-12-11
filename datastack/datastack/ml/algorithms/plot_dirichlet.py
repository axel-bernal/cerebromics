import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as tri

corners = np.array([[0, 0], [1, 0], [0.5, 0.75**0.5]])
triangle = tri.Triangulation(corners[:, 0], corners[:, 1])


refiner = tri.UniformTriRefiner(triangle)
trimesh = refiner.refine_triangulation(subdiv=4)

# Mid-points of triangle sides opposite of each corner
midpoints = [(corners[(i + 1) % 3] + corners[(i + 2) % 3]) / 2.0 \
			 for i in range(3)]
def xy2bc(xy, tol=1.e-3):
	'''Converts 2D Cartesian coordinates to barycentric.'''
	s = [(corners[i] - midpoints[i]).dot(xy - midpoints[i]) / 0.75 \
		 for i in range(3)]
	return np.clip(s, tol, 1.0 - tol)

def bc2xy(vals, x=np.array([0.0,1.0,0.5]), y=np.array([0.0,0.0,np.sqrt(1.0-0.025)])):
	if len(vals.shape) == 1:
		resx = (vals * x).sum()
		resy = (vals * y).sum()
	else:
		resx = (vals * x[np.newaxis,:]).sum(1)
		resy = (vals * y[np.newaxis,:]).sum(1)
	return resx, resy

class Dirichlet(object):
	def __init__(self, alpha):
		from math import gamma
		from operator import mul
		self._alpha = np.array(alpha)
		self._coef = gamma(np.sum(self._alpha)) / \
					 reduce(mul, [gamma(a) for a in self._alpha])
	def pdf(self, x):
		'''Returns pdf value for `x`.'''
		from operator import mul
		return self._coef * reduce(mul, [xx ** (aa - 1)
										 for (xx, aa)in zip(x, self._alpha)])

def draw_pdf_contours(dist, nlevels=200, subdiv=8, colormap='hot', **kwargs):
	import math

	refiner = tri.UniformTriRefiner(triangle)
	trimesh = refiner.refine_triangulation(subdiv=subdiv)
	pvals = [dist.pdf(xy2bc(xy)) for xy in zip(trimesh.x, trimesh.y)]
	sdfd
	plt.tricontourf(trimesh, pvals, nlevels, cmap=plt.get_cmap(colormap), **kwargs)
	plt.axis('equal')
	plt.xlim(0, 1)
	plt.ylim(0, 0.75**0.5)
	plt.axis('off')



if __name__ == '__main__':
	
	plt.figure(figsize=(8, 4))
	for (i, mesh) in enumerate((triangle, trimesh)):
		plt.subplot(1, 2, i+ 1)
		plt.triplot(mesh)
		plt.axis('off')
		plt.axis('equal')
	plt.figure(figsize=(4, 4)); draw_pdf_contours(Dirichlet([1, 1, 1]))
	plt.figure(figsize=(4, 4)); draw_pdf_contours(Dirichlet([0.999, 0.999, 0.999]))
	plt.figure(figsize=(4, 4)); draw_pdf_contours(Dirichlet([5, 5, 5]))
	plt.figure(figsize=(4, 4)); draw_pdf_contours(Dirichlet([30, 30, 50]))
	plt.figure(figsize=(4, 4)); draw_pdf_contours(Dirichlet([30, 30, 50]))
	plt.figure(figsize=(4, 4)); draw_pdf_contours(Dirichlet([2, 5, 15]))





	draw_pdf_contours(Dirichlet([1, 1, 1]))




