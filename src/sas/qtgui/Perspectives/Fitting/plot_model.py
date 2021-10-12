from math import radians
import numpy as np
import matplotlib
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

class plotModel:

    def __init__(self, params):#, param):
#        params = {
#            'model_name':['sphere'],
#            'radius': ['False', '60', 'None', '0.0', 'inf', '()']}
        self.param = {}

        for key in params:
            if len(params[key]) > 2:
                self.param[key] = float(params[key][1])
            elif key == 'model_name':
                self.param[key] = params[key][0] 

        if self.param['model_name'] != 'raspberry':
            # Discretization sphere
            discr_phi = np.linspace(0., 2 * np.pi, 30)
            discr_teta = np.linspace(0.,np.pi,30)
            self.discr = {'phi':discr_phi, 'teta':discr_teta}
            self.phi, self.teta = np.meshgrid(discr_phi, discr_teta)

        else:
            # Discretization small and large spheres
            n_lg = 30  # Number of discretization elements large sphere
            n_sm = 6  # Number of discretization elements small sphere
            
            discr_phi_lg = np.linspace(0., 2 * np.pi, n_lg)
            discr_teta_lg = np.linspace(0., np.pi, n_lg)

            discr_phi_sm = np.linspace(0., 2 * np.pi, n_sm)
            discr_teta_sm = np.linspace(0., np.pi, n_sm)
            self.discr = {'phi_sm':discr_phi_sm, 'teta_sm':discr_teta_sm, 'phi_lg':discr_phi_lg, 'teta_lg':discr_teta_lg}
            self.phi_sm, self.teta_sm = np.meshgrid(discr_phi_sm, discr_teta_sm)
            self.phi_lg, self.teta_lg = np.meshgrid(discr_phi_lg, discr_teta_lg)
        if self.param['model_name'] == 'sphere':
            self.sphere()
#    x,y,z=self.sphere(param, phi, teta)

    def sphere(self):
        print (self.param['radius'])
        self.x = self.param['radius'] * np.cos(self.phi) * np.sin(self.teta)
        self.y = self.param['radius'] * np.sin(self.phi) * np.sin(self.teta)
        self.z = self.param['radius'] * np.cos(self.teta)

        self.fig = plt.figure('3d-model')
        self.ax = plt.axes(projection = '3d')
        self.surf = self.ax.plot_surface(self.x, self.y, self.z, color='b', antialiased=False, linewidth=0)

        self.fig.canvas.draw()
        self.fig.show()

#def plotSurface(param, model3d):

#    fig = plt.figure('3d-model')
#    ax = plt.axes(projection = '3d')
#    surf = ax.plot_surface(model3d.x, model3d.y, model3d.z, color='b', antialiased=False, linewidth=0)
#    fig.show()