from math import radians
import numpy as np
import matplotlib
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

class plotModelParallelepiped:

    def __init__(self, params):#, param):
#        params = {
#            'model_name':['sphere'],
#            'radius': ['False', '60', 'None', '0.0', 'inf', '()']}
        self.param = {}

        for key in params:
            if len(params[key]) >= 2:
                self.param[key] = float(params[key][1])
            elif key == 'model_name':
                self.param[key] = params[key][0]



        if (self.param['model_name'] == 'parallelepiped' or self.param['model_name'] == 'rectangular_prism' or
            self.param['model_name'] == 'hollow_rectangular_prism_thin_walls'):
            self.parallelepiped()

        if self.param['model_name'] == 'core_shell_parallelepiped':
            self.core_shell_parallelepiped()

        if self.param['model_name'] == 'hollow_rectangular_prism':
            self.hollow_rectangular_prism()



    def parallelepiped(self):       # AND rectangular_prism AND hollow_rectangular_prism_thin_walls

        self.fig = plt.figure('3d-model')
        self.ax = plt.axes(projection='3d')
        self.ax.set_xlabel('A')
        self.ax.set_ylabel('A')
        self.ax.set_zlabel('A')

        self.ax.view_init(25, 45)



        if self.param['model_name'] != 'parallelepiped':

            len_a = self.param['length_a']  # Side a of cuboid
            b2a = self.param['b2a_ratio']  # Ratio sides b/a
            c2a = self.param['c2a_ratio']  # Ratio sides c/a

            len_b = b2a * len_a
            len_c = c2a * len_a

            if self.param['model_name'] == 'rectangular_prism':
                self.ax.set_title('rectangular_prism plot')
            else:
                self.ax.set_title('hollow_rectangular_prism_thin_walls plot')

        else:

            len_a = self.param['length_a']   # side a of cuboid
            len_b = self.param['length_b']   # side b of cuboid
            len_c = self.param['length_c']   # side c of cuboid

            self.ax.set_title('parallelepiped plot')


        # Discretization parallelepiped

        discr_len_a = np.linspace(0., len_a, 2)
        discr_len_b = np.linspace(0., len_b, 2)
        discr_len_c = np.linspace(0., len_c, 2)

        # Plot sides

        ab_a, ab_b = np.meshgrid(discr_len_a, discr_len_b)

        ac_a, ac_c = np.meshgrid(discr_len_a, discr_len_c)

        bc_b, bc_c = np.meshgrid(discr_len_b, discr_len_c)

        ab_c1 = 0. * np.ones((2, 2))
        ab_c2 = len_c * np.ones((2, 2))

        ac_b1 = 0 * np.ones((2, 2))
        ac_b2 = len_b * np.ones((2, 2))

        bc_a1 = 0 * np.ones((2, 2))
        bc_a2 = len_a * np.ones((2, 2))

        if self.param['model_name'] != 'hollow_rectangular_prism_thin_walls':
            self.surf = self.ax.plot_surface(ab_a, ab_b, ab_c1, alpha=1, color="b")
            self.surf = self.ax.plot_surface(ab_a, ab_b, ab_c2, alpha=1, color="b")

            self.surf = self.ax.plot_surface(ac_a, ac_b1, ac_c, alpha=1, color="b")
            self.surf = self.ax.plot_surface(ac_a, ac_b2, ac_c, alpha=1, color="b")

            self.surf = self.ax.plot_surface(bc_a1, bc_b, bc_c, alpha=1, color="b")
            self.surf = self.ax.plot_surface(bc_a2, bc_b, bc_c, alpha=1, color="b")
        else:
            self.surf = self.ax.plot_surface(ab_a, ab_b, ab_c1, alpha=0.5, color="r")
            self.surf = self.ax.plot_surface(ab_a, ab_b, ab_c2, alpha=0.5, color="r")

            self.surf = self.ax.plot_surface(ac_a, ac_b1, ac_c, alpha=0.5, color="r")
            self.surf = self.ax.plot_surface(ac_a, ac_b2, ac_c, alpha=0.5, color="r")

            self.surf = self.ax.plot_surface(bc_a1, bc_b, bc_c, alpha=0.5, color="r")
            self.surf = self.ax.plot_surface(bc_a2, bc_b, bc_c, alpha=0.5, color="r")

        self.fig.canvas.draw()
        self.fig.show()

    def core_shell_parallelepiped(self):

        len_a = self.param['length_a']  # side a of cuboid
        len_b = self.param['length_b']  # side b of cuboid
        len_c = self.param['length_c']  # side c of cuboid

        w_rim_a = self.param['thick_rim_a']  # Thickness of side a
        w_rim_b = self.param['thick_rim_b'] # Thickness of side b
        w_rim_c = self.param['thick_rim_c']  # Thickness if side c


        self.fig = plt.figure('3d-model')
        self.ax = plt.axes(projection='3d')
        self.ax.set_xlabel('A')
        self.ax.set_ylabel('A')
        self.ax.set_zlabel('A')
        self.ax.set_title('core_shell_paralellepiped (half a) plot')
        self.ax.view_init(25, 45)

        # Discretization parallelepiped

        len_a = len_a / 2

        discr_len_a = np.linspace(0., len_a, 2)
        discr_len_b = np.linspace(0., len_b, 2)
        discr_len_c = np.linspace(0., len_c, 2)

        discr_len_a_tot = np.linspace(0, len_a, 2)
        discr_len_b_tot = np.linspace(0, len_b, 2)
        discr_len_c_tot = np.linspace(-w_rim_c, len_c + w_rim_c, 2)

        discr_len_a_tot1 = np.linspace(-w_rim_a, len_a, 2)
        discr_len_b_tot1 = np.linspace(-w_rim_b, len_b + w_rim_b, 2)
        discr_len_c_tot1 = np.linspace(0, len_c, 2)

        discr_w_rim_b1 = np.linspace(-w_rim_b, 0, 2)
        discr_w_rim_b2 = np.linspace(len_b, len_b + w_rim_b, 2)

        discr_w_rim_c1 = np.linspace(-w_rim_c, 0, 2)
        discr_w_rim_c2 = np.linspace(len_c, len_c + w_rim_c, 2)

        # Plot sides

        # Core

        ab_a, ab_b = np.meshgrid(discr_len_a, discr_len_b)

        ac_a, ac_c = np.meshgrid(discr_len_a, discr_len_c)

        bc_b, bc_c = np.meshgrid(discr_len_b, discr_len_c)

        ab_c1 = 0. * np.ones((2, 2))
        ab_c2 = len_c * np.ones((2, 2))

        ac_b1 = 0 * np.ones((2, 2))
        ac_b2 = len_b * np.ones((2, 2))

        bc_a1 = 0 * np.ones((2, 2))
        bc_a2 = len_a * np.ones((2, 2))

        # Total

        ab_a_tot, ab_b_tot = np.meshgrid(discr_len_a_tot, discr_len_b_tot)

        ac_a_tot, ac_c_tot = np.meshgrid(discr_len_a_tot, discr_len_c_tot)

        bc_b_tot, bc_c_tot = np.meshgrid(discr_len_b_tot, discr_len_c_tot)

        ab_c1_tot = -w_rim_c * np.ones((2, 2))
        ab_c2_tot = (len_c + w_rim_c) * np.ones((2, 2))

        ac_b1_tot = 0 * np.ones((2, 2))
        ac_b2_tot = len_b * np.ones((2, 2))

        bc_a1_tot = 0 * np.ones((2, 2))
        bc_a2_tot = len_a * np.ones((2, 2))

        # Total 1

        ab_a_tot1, ab_b_tot1 = np.meshgrid(discr_len_a_tot1, discr_len_b_tot1)

        ac_a_tot1, ac_c_tot1 = np.meshgrid(discr_len_a_tot1, discr_len_c_tot1)

        bc_b_tot1, bc_c_tot1 = np.meshgrid(discr_len_b_tot1, discr_len_c_tot1)

        ab_c1_tot1 = 0 * np.ones((2, 2))
        ab_c2_tot1 = len_c * np.ones((2, 2))

        ac_b1_tot1 = -w_rim_b * np.ones((2, 2))
        ac_b2_tot1 = (len_b + w_rim_b) * np.ones((2, 2))

        bc_a1_tot1 = -w_rim_a * np.ones((2, 2))
        bc_a2_tot1 = len_a * np.ones((2, 2))

        # Crossection

        bc_rim_b1, bc_rim_c1 = np.meshgrid(discr_w_rim_b1, discr_len_c)
        bc_rim_b2, bc_rim_c2 = np.meshgrid(discr_w_rim_b2, discr_len_c)

        bc_rim_c3, bc_rim_b3 = np.meshgrid(discr_w_rim_c1, discr_len_b)
        bc_rim_c4, bc_rim_b4 = np.meshgrid(discr_w_rim_c2, discr_len_b)

        bc_rim_a = len_a * np.ones((2, 2))

        alfa = 0.15

        self.surf = self.ax.plot_surface(ab_a, ab_b, ab_c1, alpha=alfa, color="b")
        self.surf = self.ax.plot_surface(ab_a, ab_b, ab_c2, alpha=alfa, color="b")

        self.surf = self.ax.plot_surface(ac_a, ac_b1, ac_c, alpha=alfa, color="b")
        self.surf = self.ax.plot_surface(ac_a, ac_b2, ac_c, alpha=alfa, color="b")

        self.surf = self.ax.plot_surface(bc_a1, bc_b, bc_c, alpha=alfa, color="b")
        self.surf = self.ax.plot_surface(bc_a2, bc_b, bc_c, alpha=1, color="b")

        self.surf = self.ax.plot_surface(ab_a_tot, ab_b_tot, ab_c1_tot, alpha=alfa, color="r")
        self.surf = self.ax.plot_surface(ab_a_tot, ab_b_tot, ab_c2_tot, alpha=alfa, color="r")

        self.surf = self.ax.plot_surface(ac_a_tot, ac_b1_tot, ac_c_tot, alpha=alfa, color="r")
        self.surf = self.ax.plot_surface(ac_a_tot, ac_b2_tot, ac_c_tot, alpha=alfa, color="r")

        self.surf = self.ax.plot_surface(bc_a1_tot, bc_b_tot, bc_c_tot, alpha=alfa, color="r")
        self.surf = self.ax.plot_surface(bc_a2_tot, bc_b_tot, bc_c_tot, alpha=alfa, color="r")

        self.surf = self.ax.plot_surface(ab_a_tot1, ab_b_tot1, ab_c1_tot1, alpha=alfa, color="r")
        self.surf = self.ax.plot_surface(ab_a_tot1, ab_b_tot1, ab_c2_tot1, alpha=alfa, color="r")

        self.surf = self.ax.plot_surface(ac_a_tot1, ac_b1_tot1, ac_c_tot1, alpha=alfa, color="r")
        self.surf = self.ax.plot_surface(ac_a_tot1, ac_b2_tot1, ac_c_tot1, alpha=alfa, color="r")

        self.surf = self.ax.plot_surface(bc_a1_tot1, bc_b_tot1, bc_c_tot1, alpha=alfa, color="r")
        self.surf = self.ax.plot_surface(bc_a2_tot1, bc_b_tot1, bc_c_tot1, alpha=alfa, color="r")

        self.surf = self.ax.plot_surface(bc_rim_a, bc_rim_b1, bc_rim_c1, alpha=1, color="r")
        self.surf = self.ax.plot_surface(bc_rim_a, bc_rim_b2, bc_rim_c2, alpha=1, color="r")

        self.surf = self.ax.plot_surface(bc_rim_a, bc_rim_b3, bc_rim_c3, alpha=1, color="r")
        self.surf = self.ax.plot_surface(bc_rim_a, bc_rim_b4, bc_rim_c4, alpha=1, color="r")


        self.fig.canvas.draw()
        self.fig.show()

    def hollow_rectangular_prism(self):

        len_a = self.param['length_a']  # core side a of cuboid
        b2a = self.param['b2a_ratio']  # Ratio sides b/a
        c2a = self.param['c2a_ratio']  # Ratio sides c/a
        w_thick = self.param['thickness']  # Wall thickness

        self.fig = plt.figure('3d-model')
        self.ax = plt.axes(projection='3d')
        self.ax.set_xlabel('A')
        self.ax.set_ylabel('A')
        self.ax.set_zlabel('A')
        self.ax.set_title('hollow_rectangular prism')
        self.ax.view_init(25, 45)


        ##############
        # Re-use previous code for core shell

        len_b = len_a * b2a  # core side b of cuboid
        len_c = len_a * c2a  # core side c of cuboid

        w_rim_a = w_thick  # Thickness of side a
        w_rim_b = w_thick  # Thickness of side b
        w_rim_c = w_thick  # Thickness if side c

        # Discretization parallelepiped shell

        len_a = len_a / 2

        discr_len_a_tot = np.linspace(0, len_a, 2)
        discr_len_b_tot = np.linspace(0, len_b, 2)
        discr_len_c_tot = np.linspace(0, len_c, 2)

        discr_len_a_tot1 = np.linspace(-w_rim_a, len_a, 2)
        discr_len_b_tot1 = np.linspace(-w_rim_b, len_b + w_rim_b, 2)
        discr_len_c_tot1 = np.linspace(-w_rim_c, len_c + w_rim_c, 2)

        discr_w_rim_b1 = np.linspace(-w_rim_b, 0, 2)
        discr_w_rim_b2 = np.linspace(len_b, len_b + w_rim_b, 2)

        discr_w_rim_c1 = np.linspace(-w_rim_c, 0, 2)
        discr_w_rim_c2 = np.linspace(len_c, len_c + w_rim_c, 2)

        # Plot sides

        # Total

        ab_a_tot, ab_b_tot = np.meshgrid(discr_len_a_tot, discr_len_b_tot)

        ac_a_tot, ac_c_tot = np.meshgrid(discr_len_a_tot, discr_len_c_tot)

        bc_b_tot, bc_c_tot = np.meshgrid(discr_len_b_tot, discr_len_c_tot)

        ab_c1_tot = 0 * np.ones((2, 2))
        ab_c2_tot = len_c * np.ones((2, 2))

        ac_b1_tot = 0 * np.ones((2, 2))
        ac_b2_tot = len_b * np.ones((2, 2))

        bc_a1_tot = 0 * np.ones((2, 2))
        bc_a2_tot = len_a * np.ones((2, 2))

        # Total 1

        ab_a_tot1, ab_b_tot1 = np.meshgrid(discr_len_a_tot1, discr_len_b_tot1)

        ac_a_tot1, ac_c_tot1 = np.meshgrid(discr_len_a_tot1, discr_len_c_tot1)

        bc_b_tot1, bc_c_tot1 = np.meshgrid(discr_len_b_tot1, discr_len_c_tot1)

        ab_c1_tot1 = -w_rim_c * np.ones((2, 2))
        ab_c2_tot1 = (len_c + w_rim_c) * np.ones((2, 2))

        ac_b1_tot1 = -w_rim_b * np.ones((2, 2))
        ac_b2_tot1 = (len_b + w_rim_b) * np.ones((2, 2))

        bc_a1_tot1 = -w_rim_a * np.ones((2, 2))
        bc_a2_tot1 = len_a * np.ones((2, 2))

        # Crossection

        bc_rim_b1, bc_rim_c1 = np.meshgrid(discr_w_rim_b1, discr_len_c_tot)
        bc_rim_b2, bc_rim_c2 = np.meshgrid(discr_w_rim_b2, discr_len_c_tot)

        bc_rim_c3, bc_rim_b3 = np.meshgrid(discr_w_rim_c1, discr_len_b_tot1)
        bc_rim_c4, bc_rim_b4 = np.meshgrid(discr_w_rim_c2, discr_len_b_tot1)

        bc_rim_a = len_a * np.ones((2, 2))

        alfa = 0.15

        self.surf = self.ax.plot_surface(ab_a_tot, ab_b_tot, ab_c1_tot, alpha=alfa, color="r")
        self.surf = self.ax.plot_surface(ab_a_tot, ab_b_tot, ab_c2_tot, alpha=alfa, color="r")

        self.surf = self.ax.plot_surface(ac_a_tot, ac_b1_tot, ac_c_tot, alpha=alfa, color="r")
        self.surf = self.ax.plot_surface(ac_a_tot, ac_b2_tot, ac_c_tot, alpha=alfa, color="r")

        self.surf = self.ax.plot_surface(bc_a1_tot, bc_b_tot, bc_c_tot, alpha=alfa, color="r")
        self.surf = self.ax.plot_surface(bc_a2_tot, bc_b_tot, bc_c_tot, alpha=alfa, color="r")

        self.surf = self.ax.plot_surface(ab_a_tot1, ab_b_tot1, ab_c1_tot1, alpha=alfa, color="r")
        self.surf = self.ax.plot_surface(ab_a_tot1, ab_b_tot1, ab_c2_tot1, alpha=alfa, color="r")

        self.surf = self.ax.plot_surface(ac_a_tot1, ac_b1_tot1, ac_c_tot1, alpha=alfa, color="r")
        self.surf = self.ax.plot_surface(ac_a_tot1, ac_b2_tot1, ac_c_tot1, alpha=alfa, color="r")

        self.surf = self.ax.plot_surface(bc_a1_tot1, bc_b_tot1, bc_c_tot1, alpha=alfa, color="r")
        self.surf = self.ax.plot_surface(bc_a2_tot1, bc_b_tot1, bc_c_tot1, alpha=alfa, color="r")

        self.surf = self.ax.plot_surface(bc_rim_a, bc_rim_b1, bc_rim_c1, alpha=1, color="r")
        self.surf = self.ax.plot_surface(bc_rim_a, bc_rim_b2, bc_rim_c2, alpha=1, color="r")

        self.surf = self.ax.plot_surface(bc_rim_a, bc_rim_b3, bc_rim_c3, alpha=1, color="r")
        self.surf = self.ax.plot_surface(bc_rim_a, bc_rim_b4, bc_rim_c4, alpha=1, color="r")

        self.fig.canvas.draw()
        self.fig.show()

    def hollow_rectangular_prism_thin_walls(self):

        len_a = self.param['length_a']  # core side a of cuboid
        b2a = self.param['b2a_ratio']  # Ratio sides b/a
        c2a = self.param['c2a_ratio']  # Ratio sides c/a

        self.fig = plt.figure('3d-model')
        self.ax = plt.axes(projection='3d')
        self.ax.set_xlabel('A')
        self.ax.set_ylabel('A')
        self.ax.set_zlabel('A')
        self.ax.set_title('hollow_rectangular_prism_thin_walls')
        self.ax.view_init(25, 45)

        # Discretization parallelepiped

        len_b = len_a * b2a  # side b of cuboid
        len_c = len_a * c2a  # side c of cuboid

        discr_len_a = np.linspace(0., len_a, 2)
        discr_len_b = np.linspace(0., len_b, 2)
        discr_len_c = np.linspace(0., len_c, 2)


        self.fig.canvas.draw()
        self.fig.show()









