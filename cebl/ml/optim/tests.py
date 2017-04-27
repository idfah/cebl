import matplotlib.pyplot as plt
import matplotlib.cm as pltcm
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

from sciopt import sciopt


class Rosen():
    def __init__(self, a=1.0, b=100.0, initialSolution=None,
                 optimFunc=sciopt, *args, **kwargs):
        self.a = a
        self.b = b

        if initialSolution is None:
            self.solution = np.array((
                np.random.uniform(-3.0, 3.0),
                np.random.uniform(-4.0, 8.0)))
        else:
            self.solution = np.asarray(initialSolution)

        self.train(optimFunc, *args, **kwargs)

    def train(self, optimFunc, *args, **kwargs):
        self.trainResult = optimFunc(self, *args, pTrace=True, **kwargs)

    def parameters(self):
        return self.solution

    def eval(self, points):
        xs = points[:,0]
        ys = points[:,1]

        return (self.a - xs)**2 + self.b * (ys - (xs**2))**2

    def error(self):
        x = self.solution[0]
        y = self.solution[1]
        return self.eval(self.solution[None,:])

    def gradient(self, returnError=False):
        x = self.solution[0]
        y = self.solution[1]
        dx = -2.0 * (self.a - x) + -2.0*x * 2.0 * self.b * (y - (x**2))
        dy = 2.0 * self.b * (y - (x**2))

        grad = np.array((dx,dy))

        if returnError:
            return self.error(), grad
        else:
            return grad

    def plot(self, n=200, rng=(-3.0,3.0, -4.0,8.0)):
        x = np.linspace(rng[0], rng[1], n)
        y = np.linspace(rng[2], rng[3], n)

        xx, yy = np.meshgrid(x, y)

        points = np.vstack((xx.ravel(), yy.ravel())).T
        values = self.eval(points)
        zz = values.reshape((xx.shape[0], yy.shape[1]))

        fig = plt.figure(figsize=(12,6))
        axSurf = fig.add_subplot(1,2,1, projection='3d')

        surf = axSurf.plot_surface(xx, yy, zz, linewidth=1.0, cmap=pltcm.jet)
        surf.set_edgecolor('black')

        axCont = fig.add_subplot(1,2,2)
        axCont.contour(x, y, zz, 40, color='black')
        axCont.scatter(self.a, self.a**2, color='black', marker='o', s=400, linewidth=3)
        axCont.scatter(*self.solution, color='red', marker='x', s=400, linewidth=3)

        paramTrace = np.array(self.trainResult['pTrace'])
        axCont.plot(paramTrace[:,0], paramTrace[:,1], color='red', linewidth=2)

        fig.tight_layout()

class Ackley():
    def __init__(self, a=1.0, b=100.0, optimFunc=sciopt, *args, **kwargs):
        self.a = a
        self.b = b

        self.solution = np.random.uniform(-5.0, 5.0, size=2)

        self.train(optimFunc, *args, **kwargs)

    def train(self, optimFunc, *args, **kwargs):
        self.trainResult = optimFunc(self, *args, pTrace=True, **kwargs)

    def parameters(self):
        return self.solution

    def eval(self, points):
        xs = points[:,0]
        ys = points[:,1]

        return -20.0 * np.exp(-0.2 * np.sqrt(0.5 * (xs**2 + ys**2))) - \
            np.exp(0.5 * (np.cos(2.0*np.pi*xs) + np.cos(2.0*np.pi*ys))) + np.e + 20.0

    def error(self):
        x = self.solution[0]
        y = self.solution[1]
        return self.eval(self.solution[None,:])

    #def gradient(self, returnError=False):
    #    x = self.solution[0]
    #    y = self.solution[1]
    #    dx = -2.0 * (self.a - x) + -2.0*x * 2.0 * self.b * (y - (x**2))
    #    dy = 2.0 * self.b * (y - (x**2))
    #    return np.array((dx,dy))

    def plot(self, n=500, rng=(-5.0,5.0, -5.0,5.0)):
        x = np.linspace(rng[0], rng[1], n)
        y = np.linspace(rng[2], rng[3], n)

        xx, yy = np.meshgrid(x, y)

        points = np.vstack((xx.ravel(), yy.ravel())).T
        values = self.eval(points)
        zz = values.reshape((xx.shape[0], yy.shape[1]))

        fig = plt.figure()
        axSurf = fig.add_subplot(1,2,1, projection='3d')

        surf = axSurf.plot_surface(xx, yy, zz, linewidth=1.0, cmap=pltcm.jet)
        surf.set_edgecolor('black')

        axCont = fig.add_subplot(1,2,2)
        axCont.contour(x, y, zz, 40, color='black')
        axCont.scatter(0.0, 0.0, color='black', marker='o', s=400, linewidth=3)
        axCont.scatter(*self.solution, color='red', marker='x', s=400, linewidth=3)

        paramTrace = np.array(self.trainResult['pTrace'])
        axCont.plot(paramTrace[:,0], paramTrace[:,1], color='red', linewidth=2)

if __name__ == '__main__':
    test = Ackley(method='Powell', verbose=True)
    test.plot()
    plt.show()
