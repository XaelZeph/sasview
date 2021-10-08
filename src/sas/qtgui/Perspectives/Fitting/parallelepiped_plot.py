import matplotlib
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np


"""Plotfunctions for the paralellepiped category"""


class parallelepiped:

    def __init__(self):
        """Construct parameters for parallelepiped"""

        self.len_a = 1       # side a of cuboid
        self.len_b = 1       # side b of cuboid
        self.len_c = 1       # side c of cuboid


        # Discretization parallelepiped

        self.discr_len_a = np.linspace(0., self.len_a, 2)
        self.discr_len_b = np.linspace(0., self.len_b, 2)
        self.discr_len_c = np.linspace(0., self.len_c, 2)

    def plot(self):
        """Plot parallelepiped"""

        fig = plt.figure()
        ax = plt.axes(projection='3d')
        ax.set_aspect('auto')
        ax.set_xlabel('A')
        ax.set_ylabel('A')
        ax.set_zlabel('A');
        # Set viev by elevation and azimuth angles
        # Eelevation above x-y plane and counter clockwiserotation on z axis
        ax.view_init(20, 45)

        # Plot sides

        ab_a, ab_b = np.meshgrid(self.discr_len_a, self.discr_len_b)

        ac_a, ac_c = np.meshgrid(self.discr_len_a, self.discr_len_c)

        bc_b, bc_c = np.meshgrid(self.discr_len_b, self.discr_len_c)

        ab_c1 = 0. * np.ones((2,2))
        ab_c2 = self.len_c * np.ones((2,2))

        ac_b1 = 0 * np.ones((2,2))
        ac_b2 = self.len_b * np.ones((2,2))

        bc_a1 = 0 * np.ones((2,2))
        bc_a2 = self.len_a * np.ones((2,2))



        ax.plot_surface(ab_a, ab_b, ab_c1, alpha=1, color="b")
        ax.plot_surface(ab_a, ab_b, ab_c2, alpha=1, color="b")

        ax.plot_surface(ac_a, ac_b1, ac_c, alpha=1, color="b")
        ax.plot_surface(ac_a, ac_b2, ac_c, alpha=1, color="b")

        ax.plot_surface(bc_a1, bc_b, bc_c, alpha=1, color="b")
        ax.plot_surface(bc_a2, bc_b, bc_c, alpha=1, color="b")


        ax.set_title('paralellepiped plot')
        plt.show()


class core_shell_parallelepiped:

    def __init__(self):
        """Construct parameters for core_shell_parallelepiped"""

        self.len_a = 1       # core side a of cuboid
        self.len_b = 1       # core side b of cuboid
        self.len_c = 1       # core side c of cuboid

        self.w_rim_a = 0.1     # Thickness of side a
        self.w_rim_b = 0.1     # Thickness of side b
        self.w_rim_c = 0.1     # Thickness if side c


        # Discretization parallelepiped

        self.len_a = self.len_a / 2

        self.discr_len_a = np.linspace(0., self.len_a, 2)
        self.discr_len_b = np.linspace(0., self.len_b, 2)
        self.discr_len_c = np.linspace(0., self.len_c, 2)

        self.discr_len_a_tot = np.linspace(0, self.len_a, 2)
        self.discr_len_b_tot = np.linspace(0, self.len_b, 2)
        self.discr_len_c_tot = np.linspace(-self.w_rim_c, self.len_c + self.w_rim_c, 2)

        self.discr_len_a_tot1 = np.linspace(-self.w_rim_a, self.len_a, 2)
        self.discr_len_b_tot1 = np.linspace(-self.w_rim_b, self.len_b + self.w_rim_b, 2)
        self.discr_len_c_tot1 = np.linspace(0, self.len_c, 2)



        self.discr_w_rim_b1 = np.linspace(-self.w_rim_b, 0, 2)
        self.discr_w_rim_b2 = np.linspace(self.len_b, self.len_b + self.w_rim_b, 2)

        self.discr_w_rim_c1 = np.linspace(-self.w_rim_c, 0, 2)
        self.discr_w_rim_c2 = np.linspace(self.len_c, self.len_c + self.w_rim_c, 2)


    def plot(self):
        """Plot core_shell_parallelepiped"""

        fig = plt.figure()
        ax = plt.axes(projection='3d')
        ax.set_aspect('auto')
        ax.set_xlabel('A')
        ax.set_ylabel('A')
        ax.set_zlabel('A');
        # Set viev by elevation and azimuth angles
        # Eelevation above x-y plane and counter clockwiserotation on z axis
        ax.view_init(20, 30)

        # Plot sides

        # Core

        ab_a, ab_b = np.meshgrid(self.discr_len_a, self.discr_len_b)

        ac_a, ac_c = np.meshgrid(self.discr_len_a, self.discr_len_c)

        bc_b, bc_c = np.meshgrid(self.discr_len_b, self.discr_len_c)

        ab_c1 = 0. * np.ones((2,2))
        ab_c2 = self.len_c * np.ones((2,2))

        ac_b1 = 0 * np.ones((2,2))
        ac_b2 = self.len_b * np.ones((2,2))

        bc_a1 = 0 * np.ones((2,2))
        bc_a2 = self.len_a * np.ones((2,2))

        # Total

        ab_a_tot, ab_b_tot = np.meshgrid(self.discr_len_a_tot, self.discr_len_b_tot)

        ac_a_tot, ac_c_tot = np.meshgrid(self.discr_len_a_tot, self.discr_len_c_tot)

        bc_b_tot, bc_c_tot = np.meshgrid(self.discr_len_b_tot, self.discr_len_c_tot)

        ab_c1_tot = -self.w_rim_c * np.ones((2, 2))
        ab_c2_tot = (self.len_c + self.w_rim_c) * np.ones((2, 2))

        ac_b1_tot = 0 * np.ones((2, 2))
        ac_b2_tot = (self.len_b) * np.ones((2, 2))

        bc_a1_tot = 0 * np.ones((2, 2))
        bc_a2_tot = self.len_a * np.ones((2, 2))

        # Total 1

        ab_a_tot1, ab_b_tot1 = np.meshgrid(self.discr_len_a_tot1, self.discr_len_b_tot1)

        ac_a_tot1, ac_c_tot1 = np.meshgrid(self.discr_len_a_tot1, self.discr_len_c_tot1)

        bc_b_tot1, bc_c_tot1 = np.meshgrid(self.discr_len_b_tot1, self.discr_len_c_tot1)

        ab_c1_tot1 = 0 * np.ones((2, 2))
        ab_c2_tot1 = (self.len_c) * np.ones((2, 2))

        ac_b1_tot1 = -self.w_rim_b * np.ones((2, 2))
        ac_b2_tot1 = (self.len_b + self.w_rim_b) * np.ones((2, 2))

        bc_a1_tot1 = -self.w_rim_a * np.ones((2, 2))
        bc_a2_tot1 = self.len_a * np.ones((2, 2))

        # Crossection

        bc_rim_b1, bc_rim_c1 = np.meshgrid(self.discr_w_rim_b1, self.discr_len_c)
        bc_rim_b2, bc_rim_c2 = np.meshgrid(self.discr_w_rim_b2, self.discr_len_c)

        bc_rim_c3, bc_rim_b3 = np.meshgrid(self.discr_w_rim_c1, self.discr_len_b)
        bc_rim_c4, bc_rim_b4 = np.meshgrid(self.discr_w_rim_c2, self.discr_len_b)

        bc_rim_a = self.len_a * np.ones((2,2))



        alfa = 0.15

        ax.plot_surface(ab_a, ab_b, ab_c1, alpha=alfa, color="b")
        ax.plot_surface(ab_a, ab_b, ab_c2, alpha=alfa, color="b")

        ax.plot_surface(ac_a, ac_b1, ac_c, alpha=alfa, color="b")
        ax.plot_surface(ac_a, ac_b2, ac_c, alpha=alfa, color="b")

        ax.plot_surface(bc_a1, bc_b, bc_c, alpha=alfa, color="b")
        ax.plot_surface(bc_a2, bc_b, bc_c, alpha=1, color="b")

        ax.plot_surface(ab_a_tot, ab_b_tot, ab_c1_tot, alpha=alfa, color="r")
        ax.plot_surface(ab_a_tot, ab_b_tot, ab_c2_tot, alpha=alfa, color="r")

        ax.plot_surface(ac_a_tot, ac_b1_tot, ac_c_tot, alpha=alfa, color="r")
        ax.plot_surface(ac_a_tot, ac_b2_tot, ac_c_tot, alpha=alfa, color="r")

        ax.plot_surface(bc_a1_tot, bc_b_tot, bc_c_tot, alpha=alfa, color="r")
        ax.plot_surface(bc_a2_tot, bc_b_tot, bc_c_tot, alpha=alfa, color="r")

        ax.plot_surface(ab_a_tot1, ab_b_tot1, ab_c1_tot1, alpha=alfa, color="r")
        ax.plot_surface(ab_a_tot1, ab_b_tot1, ab_c2_tot1, alpha=alfa, color="r")

        ax.plot_surface(ac_a_tot1, ac_b1_tot1, ac_c_tot1, alpha=alfa, color="r")
        ax.plot_surface(ac_a_tot1, ac_b2_tot1, ac_c_tot1, alpha=alfa, color="r")

        ax.plot_surface(bc_a1_tot1, bc_b_tot1, bc_c_tot1, alpha=alfa, color="r")
        ax.plot_surface(bc_a2_tot1, bc_b_tot1, bc_c_tot1, alpha=alfa, color="r")

        ax.plot_surface(bc_rim_a, bc_rim_b1, bc_rim_c1, alpha=1, color="r")
        ax.plot_surface(bc_rim_a, bc_rim_b2, bc_rim_c2, alpha=1, color="r")

        ax.plot_surface(bc_rim_a, bc_rim_b3, bc_rim_c3, alpha=1, color="r")
        ax.plot_surface(bc_rim_a, bc_rim_b4, bc_rim_c4, alpha=1, color="r")



        ax.set_title('core_shell_paralellepiped (half a-side crossection) plot')
        plt.show()


class hollow_rectangular_prism:

    def __init__(self):
        """Construct parameters for hollow_rectangular_prism"""

        self.len_a = 1            # core side a of cuboid
        self.b2a = 1              # Ratio sides b/a
        self.c2a = 1              # Ratio sides c/a
        self.w_thick = 0.1        # Wall thickness




        ##############
        # Re-use/modify previous code for core shell

        self.len_b = self.len_a * self.b2a       # core side b of cuboid
        self.len_c = self.len_a * self.c2a       # core side c of cuboid

        self.w_rim_a = self.w_thick              # Thickness of side a
        self.w_rim_b = self.w_thick              # Thickness of side b
        self.w_rim_c = self.w_thick              # Thickness if side c


        # Discretization parallelepiped shell

        self.len_a = self.len_a / 2


        self.discr_len_a_tot = np.linspace(0, self.len_a, 2)
        self.discr_len_b_tot = np.linspace(0, self.len_b, 2)
        self.discr_len_c_tot = np.linspace(0, self.len_c, 2)

        self.discr_len_a_tot1 = np.linspace(-self.w_rim_a, self.len_a, 2)
        self.discr_len_b_tot1 = np.linspace(-self.w_rim_b, self.len_b + self.w_rim_b, 2)
        self.discr_len_c_tot1 = np.linspace(-self.w_rim_c, self.len_c + self.w_rim_c, 2)



        self.discr_w_rim_b1 = np.linspace(-self.w_rim_b, 0, 2)
        self.discr_w_rim_b2 = np.linspace(self.len_b, self.len_b + self.w_rim_b, 2)

        self.discr_w_rim_c1 = np.linspace(-self.w_rim_c, 0, 2)
        self.discr_w_rim_c2 = np.linspace(self.len_c, self.len_c + self.w_rim_c, 2)


    def plot(self):
        """Plot hollow_rectangular_prism"""

        fig = plt.figure()
        ax = plt.axes(projection='3d')
        ax.set_aspect('auto')
        ax.set_xlabel('A')
        ax.set_ylabel('A')
        ax.set_zlabel('A');
        # Set viev by elevation and azimuth angles
        # Eelevation above x-y plane and counter clockwiserotation on z axis
        ax.view_init(20, 30)

        # Plot sides

        # Total

        ab_a_tot, ab_b_tot = np.meshgrid(self.discr_len_a_tot, self.discr_len_b_tot)

        ac_a_tot, ac_c_tot = np.meshgrid(self.discr_len_a_tot, self.discr_len_c_tot)

        bc_b_tot, bc_c_tot = np.meshgrid(self.discr_len_b_tot, self.discr_len_c_tot)

        ab_c1_tot = 0 * np.ones((2, 2))
        ab_c2_tot = (self.len_c) * np.ones((2, 2))

        ac_b1_tot = 0 * np.ones((2, 2))
        ac_b2_tot = (self.len_b) * np.ones((2, 2))

        bc_a1_tot = 0 * np.ones((2, 2))
        bc_a2_tot = (self.len_a) * np.ones((2, 2))

        # Total 1

        ab_a_tot1, ab_b_tot1 = np.meshgrid(self.discr_len_a_tot1, self.discr_len_b_tot1)

        ac_a_tot1, ac_c_tot1 = np.meshgrid(self.discr_len_a_tot1, self.discr_len_c_tot1)

        bc_b_tot1, bc_c_tot1 = np.meshgrid(self.discr_len_b_tot1, self.discr_len_c_tot1)

        ab_c1_tot1 = -self.w_rim_c * np.ones((2, 2))
        ab_c2_tot1 = (self.len_c + self.w_rim_c) * np.ones((2, 2))

        ac_b1_tot1 = -self.w_rim_b * np.ones((2, 2))
        ac_b2_tot1 = (self.len_b + self.w_rim_b) * np.ones((2, 2))

        bc_a1_tot1 = -self.w_rim_a * np.ones((2, 2))
        bc_a2_tot1 = self.len_a * np.ones((2, 2))

        # Crossection

        bc_rim_b1, bc_rim_c1 = np.meshgrid(self.discr_w_rim_b1, self.discr_len_c_tot)
        bc_rim_b2, bc_rim_c2 = np.meshgrid(self.discr_w_rim_b2, self.discr_len_c_tot)

        bc_rim_c3, bc_rim_b3 = np.meshgrid(self.discr_w_rim_c1, self.discr_len_b_tot1)
        bc_rim_c4, bc_rim_b4 = np.meshgrid(self.discr_w_rim_c2, self.discr_len_b_tot1)

        bc_rim_a = self.len_a * np.ones((2,2))



        alfa = 0.15

        ax.plot_surface(ab_a_tot, ab_b_tot, ab_c1_tot, alpha=alfa, color="r")
        ax.plot_surface(ab_a_tot, ab_b_tot, ab_c2_tot, alpha=alfa, color="r")

        ax.plot_surface(ac_a_tot, ac_b1_tot, ac_c_tot, alpha=alfa, color="r")
        ax.plot_surface(ac_a_tot, ac_b2_tot, ac_c_tot, alpha=alfa, color="r")

        ax.plot_surface(bc_a1_tot, bc_b_tot, bc_c_tot, alpha=alfa, color="r")
        ax.plot_surface(bc_a2_tot, bc_b_tot, bc_c_tot, alpha=alfa, color="r")

        ax.plot_surface(ab_a_tot1, ab_b_tot1, ab_c1_tot1, alpha=alfa, color="r")
        ax.plot_surface(ab_a_tot1, ab_b_tot1, ab_c2_tot1, alpha=alfa, color="r")

        ax.plot_surface(ac_a_tot1, ac_b1_tot1, ac_c_tot1, alpha=alfa, color="r")
        ax.plot_surface(ac_a_tot1, ac_b2_tot1, ac_c_tot1, alpha=alfa, color="r")

        ax.plot_surface(bc_a1_tot1, bc_b_tot1, bc_c_tot1, alpha=alfa, color="r")
        ax.plot_surface(bc_a2_tot1, bc_b_tot1, bc_c_tot1, alpha=alfa, color="r")

        ax.plot_surface(bc_rim_a, bc_rim_b1, bc_rim_c1, alpha=1, color="r")
        ax.plot_surface(bc_rim_a, bc_rim_b2, bc_rim_c2, alpha=1, color="r")

        ax.plot_surface(bc_rim_a, bc_rim_b3, bc_rim_c3, alpha=1, color="r")
        ax.plot_surface(bc_rim_a, bc_rim_b4, bc_rim_c4, alpha=1, color="r")



        ax.set_title('hollow_rectangular_prism (half a-side crossection) plot')
        plt.show()



class rectangular_prism:

    def __init__(self):
        """Construct parameters for rectangular_prism"""

        self.len_a = 1       # Side a of cuboid
        self.b2a = 2         # Ratio sides b/a
        self.c2a = 2         # Ratio sides c/a



        # Discretization parallelepiped

        self.len_b = self.len_a * self.b2a  # side b of cuboid
        self.len_c = self.len_a * self.c2a  # side c of cuboid

        self.discr_len_a = np.linspace(0., self.len_a, 2)
        self.discr_len_b = np.linspace(0., self.len_b, 2)
        self.discr_len_c = np.linspace(0., self.len_c, 2)

    def plot(self):
        """Plot rectangular_prism"""

        fig = plt.figure()
        ax = plt.axes(projection='3d')
        ax.set_aspect('auto')
        ax.set_xlabel('A')
        ax.set_ylabel('A')
        ax.set_zlabel('A');
        # Set viev by elevation and azimuth angles
        # Eelevation above x-y plane and counter clockwiserotation on z axis
        ax.view_init(20, 45)

        # Plot sides

        ab_a, ab_b = np.meshgrid(self.discr_len_a, self.discr_len_b)

        ac_a, ac_c = np.meshgrid(self.discr_len_a, self.discr_len_c)

        bc_b, bc_c = np.meshgrid(self.discr_len_b, self.discr_len_c)

        ab_c1 = 0. * np.ones((2,2))
        ab_c2 = self.len_c * np.ones((2,2))

        ac_b1 = 0 * np.ones((2,2))
        ac_b2 = self.len_b * np.ones((2,2))

        bc_a1 = 0 * np.ones((2,2))
        bc_a2 = self.len_a * np.ones((2,2))



        ax.plot_surface(ab_a, ab_b, ab_c1, alpha=1, color="b")
        ax.plot_surface(ab_a, ab_b, ab_c2, alpha=1, color="b")

        ax.plot_surface(ac_a, ac_b1, ac_c, alpha=1, color="b")
        ax.plot_surface(ac_a, ac_b2, ac_c, alpha=1, color="b")

        ax.plot_surface(bc_a1, bc_b, bc_c, alpha=1, color="b")
        ax.plot_surface(bc_a2, bc_b, bc_c, alpha=1, color="b")


        ax.set_title('rectangular_prism plot')
        plt.show()


class hollow_rectangular_prism_thin_walls:

    def __init__(self):
        """Construct parameters for hollow_rectangular_prism_thin_walls"""

        self.len_a = 1       # Side a of cuboid
        self.b2a = 2         # Ratio sides b/a
        self.c2a = 2         # Ratio sides c/a



        # Discretization parallelepiped

        self.len_b = self.len_a * self.b2a  # side b of cuboid
        self.len_c = self.len_a * self.c2a  # side c of cuboid

        self.discr_len_a = np.linspace(0., self.len_a, 2)
        self.discr_len_b = np.linspace(0., self.len_b, 2)
        self.discr_len_c = np.linspace(0., self.len_c, 2)

    def plot(self):
        """Plot hollow_rectangular_prism_thin_walls"""

        fig = plt.figure()
        ax = plt.axes(projection='3d')
        ax.set_aspect('auto')
        ax.set_xlabel('A')
        ax.set_ylabel('A')
        ax.set_zlabel('A');
        # Set viev by elevation and azimuth angles
        # Eelevation above x-y plane and counter clockwiserotation on z axis
        ax.view_init(20, 45)

        # Plot sides

        ab_a, ab_b = np.meshgrid(self.discr_len_a, self.discr_len_b)

        ac_a, ac_c = np.meshgrid(self.discr_len_a, self.discr_len_c)

        bc_b, bc_c = np.meshgrid(self.discr_len_b, self.discr_len_c)

        ab_c1 = 0. * np.ones((2,2))
        ab_c2 = self.len_c * np.ones((2,2))

        ac_b1 = 0 * np.ones((2,2))
        ac_b2 = self.len_b * np.ones((2,2))

        bc_a1 = 0 * np.ones((2,2))
        bc_a2 = self.len_a * np.ones((2,2))

        alfa = 0.5

        ax.plot_surface(ab_a, ab_b, ab_c1, alpha=alfa, color="r")
        ax.plot_surface(ab_a, ab_b, ab_c2, alpha=alfa, color="r")

        ax.plot_surface(ac_a, ac_b1, ac_c, alpha=alfa, color="r")
        ax.plot_surface(ac_a, ac_b2, ac_c, alpha=alfa, color="r")

        ax.plot_surface(bc_a1, bc_b, bc_c, alpha=alfa, color="r")
        ax.plot_surface(bc_a2, bc_b, bc_c, alpha=alfa, color="r")


        ax.set_title('hollow_rectangular_prism_thin_walls plot')
        plt.show()


# parallelepiped().plot()
# core_shell_parallelepiped().plot()
# hollow_rectangular_prism().plot()
# rectangular_prism().plot()
# hollow_rectangular_prism_thin_walls().plot()
