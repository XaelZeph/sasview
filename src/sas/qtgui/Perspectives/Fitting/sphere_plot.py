import matplotlib
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np
import random

"""Plotfunctions for the sphere category"""

class sphere:

     def __init__(self, param):

        """Construct parameters for sphere"""

        # self.radius = 1

        self.radius = float(param['radius'][1])

        # Discretization sphere

        self.discr_phi = np.linspace(0., 2 * np.pi, 30)
        self.discr_teta = np.linspace(0.,np.pi,30)

     def plot(self):

        """Plot sphere"""

        fig = plt.figure()
        ax = plt.axes(projection = '3d')
        ax.set_aspect('auto')
        ax.set_xlabel('A')
        ax.set_ylabel('A')
        ax.set_zlabel('A');
        # Set viev by elevation and azimuth angles
        # Eelevation above x-y plane and counter clockwiserotation on z axis
        ax.view_init(20,45)

        phi, teta = np.meshgrid(self.discr_phi, self.discr_teta)
        x = self.radius * np.cos(phi) * np.sin(teta)
        y = self.radius * np.sin(phi) * np.sin(teta)
        z = self.radius * np.cos(teta)
        ax.plot_surface(x, y, z, alpha = 1, color = "b")
        ax.set_title('Sphere plot')
        plt.show()


class raspberry:

    def __init__(self):

        """Construct parameters for raspberry"""

        self.radius_lg = 150.     # large sphere
        self.radius_sm = 10.     # small sphere

        self.packing_flag = 'equi_fibbo' # Set to 'random' or 'equi_approx' or 'equi_fibbo' methods

        self.n_lg = 30  # Number of discretization elements large sphere
        self.n_sm = 6  # Number of discretization elements small sphere

        self.volfraction_lg = 0.01  # Volume fraction large spheres
        self.volfraction_sm = 0.005 # Volume fraction small spheres
        self.surface_fraction = 0.4 # Fraction of small spheres at the surface
        self.penetration = 0. # Fractional penetration depth of small spheres into large sphere

        self.Np =  int((self.surface_fraction *  self.volfraction_sm * self.radius_lg ** 3) / (
                    self.volfraction_lg * self.radius_sm ** 3)) # Number of small spheres per large sphere

        # Center to center distance spheres
        self.c_to_c = (self.radius_lg + self.radius_sm - self.penetration)
        self.c_to_c_arr = self.c_to_c * np.ones(self.Np)

        ## Generate parameters for randomly distributed "berries" on large sphere

        self.z_ran = np.linspace(-self.c_to_c,self.c_to_c,self.Np)
        self.phi_ran = np.linspace(0.,2. * np.pi,self.Np)
        random.shuffle(self.phi_ran)

        self.x_ran = np.sqrt(self.c_to_c_arr ** 2 - self.z_ran ** 2) * np.cos(self.phi_ran)
        self.y_ran = np.sqrt(self.c_to_c_arr ** 2 - self.z_ran ** 2) * np.sin(self.phi_ran)

        ## Generate parameters for equidistributed "berries" on large sphere with approx method

        self.a = (4 * np.pi * self.c_to_c ** 2) / self.Np
        self.d = np.sqrt(self.a)
        self.M_teta = int(2 * np.pi * self.c_to_c / self.d)
        self.d_teta = self.c_to_c * np.pi / self.M_teta
        self.d_phi = self.a / self.d_teta

        # Generate parameters for equidstributed "berries" on sphere Fibbonacci method

        self.gr = (1 + np.sqrt(5)) / 2

        self.x_sq = np.ones(self.Np)
        self.y_sq = np.ones(self.Np)


        # Discretization small and large spheres

        self.discr_phi_lg = np.linspace(0., 2 * np.pi,self.n_lg)
        self.discr_teta_lg = np.linspace(0., np.pi, self.n_lg)

        self.discr_phi_sm = np.linspace(0., 2 * np.pi, self.n_sm)
        self.discr_teta_sm = np.linspace(0., np.pi, self.n_sm)

    def plot(self):

        """Plot raspberry"""

        fig = plt.figure()
        ax = plt.axes(projection = '3d')
        ax.set_aspect('auto')
        ax.set_xlabel('A')
        ax.set_ylabel('A')
        ax.set_zlabel('A');
        ax.set_title('Raspberry plot')
        # Set viev by elevation and azimuth angles
        # Eelevation above x-y plane and counter clockwiserotation on z axis
        ax.view_init(20, 45)

        phi_sm, teta_sm = np.meshgrid(self.discr_phi_sm, self.discr_teta_sm)
        phi_lg, teta_lg = np.meshgrid(self.discr_phi_lg, self.discr_teta_lg)

        x_lg = self.radius_lg * np.cos(phi_lg) * np.sin(teta_lg)
        y_lg = self.radius_lg * np.sin(phi_lg) * np.sin(teta_lg)
        z_lg = self.radius_lg * np.cos(teta_lg)

        x_sm = self.radius_sm * np.cos(phi_sm) * np.sin(teta_sm)
        y_sm = self.radius_sm * np.sin(phi_sm) * np.sin(teta_sm)
        z_sm = self.radius_sm * np.cos(teta_sm)

        ax.plot_surface(x_lg, y_lg, z_lg, alpha = 0.5, color = "b")

        # ax.scatter(x_lg, y_lg, z_lg, alpha=1., color="b")
        # ax.plot_wireframe(x_lg, y_lg, z_lg, alpha=1., color="b")

        if self.packing_flag == 'random':

            for berry in range(0,self.Np):

                ax.plot_surface(x_sm + self.x_ran[berry] * np.ones((self.n_sm,self.n_sm)),
                                y_sm + self.y_ran[berry] * np.ones((self.n_sm,self.n_sm)),
                                z_sm + self.z_ran[berry] * np.ones((self.n_sm,self.n_sm)), alpha = 1., color = "r")


            plt.show()


        elif self.packing_flag == 'equi_approx':

            for m in range(0,self.M_teta):

                self.teta_eq = np.pi * (m + 0.5) / self.M_teta
                self.M_phi = int(2 * self.c_to_c * np.pi * np.sin(self.teta_eq) / self.d_phi)



                for n in range(0,self.M_phi):

                    self.phi_eq = 2 * np.pi * n / self.M_phi

                    x_eq = self.c_to_c * np.cos(self.phi_eq) * np.sin(self.teta_eq)
                    y_eq = self.c_to_c * np.sin(self.phi_eq) * np.sin(self.teta_eq)
                    z_eq = self.c_to_c * np.cos(self.teta_eq)




                    ax.plot_surface(x_sm + x_eq * np.ones((self.n_sm, self.n_sm)),
                                    y_sm + y_eq * np.ones((self.n_sm, self.n_sm)),
                                    z_sm + z_eq * np.ones((self.n_sm, self.n_sm)), alpha = 1, color = "r")

        elif self.packing_flag == 'equi_fibbo':

            for j in range(0, self.Np):

                self.x_sq[j] = ((j / self.gr) - np.floor(j / self.gr))
                self.y_sq[j] = j / self.Np


                self.theta_sph = 2 * np.pi * self.x_sq[j]
                self.phi_sph = np.arccos(1 - 2 * self.y_sq[j])

                x_sph = self.c_to_c * np.cos(self.theta_sph) * np.sin(self.phi_sph);
                y_sph = self.c_to_c * np.sin(self.theta_sph) * np.sin(self.phi_sph);
                z_sph = self.c_to_c * np.cos(self.phi_sph);

                ax.plot_surface(x_sm + x_sph * np.ones((self.n_sm, self.n_sm)),
                                y_sm + y_sph * np.ones((self.n_sm, self.n_sm)),
                                z_sm + z_sph * np.ones((self.n_sm, self.n_sm)), alpha=1, color="r")




        plt.show()


class spherical_sld:

    def __init__(self):

        """Construct parameters for spherical_sld"""


        self.n_shells = 10   # Number of shells
        self.w_shell = 1.   # Shell thickness
        self.w_inf = 1.     # Interface thickness


        # Discretization spheres

        self.discr_phi = np.linspace( 2 * np.pi /3, 2 * np.pi, 30)
        self.discr_teta = np.linspace(0., np.pi, 30)



    def plot(self):

        """Plot spherical_sld"""

        fig = plt.figure()
        ax = plt.axes(projection = '3d')
        ax.set_aspect('auto')
        ax.set_xlabel('A')
        ax.set_ylabel('A')
        ax.set_zlabel('A');
        # Set viev by elevation and azimuth angles
        # Eelevation above x-y plane and counter clockwiserotation on z axis
        ax.view_init(25, 45)

        phi, teta = np.meshgrid(self.discr_phi, self.discr_teta)

        for n in range(1,self.n_shells + 1):

            xs = (n * self.w_shell + (n-1) * self.w_inf) * np.cos(phi) * np.sin(teta)
            ys = (n * self.w_shell + (n-1) * self.w_inf) * np.sin(phi) * np.sin(teta)
            zs = (n * self.w_shell + (n-1) * self.w_inf) * np.cos(teta)

            xi = (n * self.w_shell + n * self.w_inf) * np.cos(phi) * np.sin(teta)
            yi = (n * self.w_shell + n * self.w_inf) * np.sin(phi) * np.sin(teta)
            zi = (n * self.w_shell + n * self.w_inf) * np.cos(teta)


            self.discr_rs = np.linspace(((n-1) * self.w_shell + (n-1) * self.w_inf),
                                         (n * self.w_shell + (n-1) * self.w_inf),2)

            self.discr_ri = np.linspace((n * self.w_shell + (n - 1) * self.w_inf),
                                        (n * self.w_shell + n * self.w_inf),2)


            rs_s12, teta_s12 = np.meshgrid(self.discr_rs, self.discr_teta)

            ri_s12, teta_s12 = np.meshgrid(self.discr_ri, self.discr_teta)

            phi_s1 = 2 * np.pi / 3

            xs_s1 = rs_s12 * np.cos(phi_s1) * np.sin(teta_s12)
            ys_s1 = rs_s12 * np.sin(phi_s1) * np.sin(teta_s12)
            zs_s1 = rs_s12 * np.cos(teta_s12)

            xi_s1 = ri_s12 * np.cos(phi_s1) * np.sin(teta_s12)
            yi_s1 = ri_s12 * np.sin(phi_s1) * np.sin(teta_s12)
            zi_s1 = ri_s12 * np.cos(teta_s12)

            phi_s2 = 2 * np.pi

            xs_s2 = rs_s12 * np.cos(phi_s2) * np.sin(teta_s12)
            ys_s2 = rs_s12 * np.sin(phi_s2) * np.sin(teta_s12)
            zs_s2 = rs_s12 * np.cos(teta_s12)

            xi_s2 = ri_s12 * np.cos(phi_s2) * np.sin(teta_s12)
            yi_s2 = ri_s12 * np.sin(phi_s2) * np.sin(teta_s12)
            zi_s2 = ri_s12 * np.cos(teta_s12)


            ax.plot_surface(xs_s1, ys_s1, zs_s1, alpha = 1., color = "b")
            ax.plot_surface(xs_s2, ys_s2, zs_s2, alpha = 1., color ="b")
            ax.plot_surface(xi_s1, yi_s1, zi_s1, alpha = 1., color = "r")
            ax.plot_surface(xi_s2, yi_s2, zi_s2, alpha = 1., color = "r")
            ax.plot_surface(xs, ys, zs, alpha = 0.1, color = "b")
            ax.plot_surface(xi, yi, zi, alpha = 0.1, color = "r")


        ax.set_title('Spherical_sld plot')

        plt.show()


class onion:

    def __init__(self):

        """Construct parameters for onion"""

        self.rad_core = 10    # Core radius
        self.n_shells = 5   # Number of shells
        self.w_shell = 1.   # Shell thickness



        # Discretization spheres

        self.discr_phi = np.linspace( 2 * np.pi /3, 2 * np.pi, 30)
        self.discr_teta = np.linspace(0., np.pi, 30)



    def plot(self):

        """Plot onion"""

        fig = plt.figure()
        ax = plt.axes(projection = '3d')
        ax.set_aspect('auto')
        ax.set_xlabel('A')
        ax.set_ylabel('A')
        ax.set_zlabel('A');
        # Set viev by elevation and azimuth angles
        # Eelevation above x-y plane and counter clockwiserotation on z axis
        ax.view_init(25, 45)

        phi, teta = np.meshgrid(self.discr_phi, self.discr_teta)

        xc = self.rad_core * np.cos(phi) * np.sin(teta)
        yc = self.rad_core * np.sin(phi) * np.sin(teta)
        zc = self.rad_core * np.cos(teta)


        self.discr_rc = np.linspace(0,self.rad_core, 2)

        rc_s12, teta_s12 = np.meshgrid(self.discr_rc, self.discr_teta)

        phi_s1 = 2 * np.pi / 3

        xc_s1 = rc_s12 * np.cos(phi_s1) * np.sin(teta_s12)
        yc_s1 = rc_s12 * np.sin(phi_s1) * np.sin(teta_s12)
        zc_s1 = rc_s12 * np.cos(teta_s12)

        phi_s2 = 2 * np.pi

        xc_s2 = rc_s12 * np.cos(phi_s2) * np.sin(teta_s12)
        yc_s2 = rc_s12 * np.sin(phi_s2) * np.sin(teta_s12)
        zc_s2 = rc_s12 * np.cos(teta_s12)

        ax.plot_surface(xc_s1, yc_s1, zc_s1, alpha=1., color="white")
        ax.plot_surface(xc_s2, yc_s2, zc_s2, alpha=1., color="white")

        ax.plot_surface(xc, yc, zc, alpha=0.1, color="white")



        for n in range(1,self.n_shells + 1):

            xs = (n * self.w_shell + self.rad_core) * np.cos(phi) * np.sin(teta)
            ys = (n * self.w_shell + self.rad_core) * np.sin(phi) * np.sin(teta)
            zs = (n * self.w_shell + self.rad_core) * np.cos(teta)


            self.discr_rs = np.linspace(((n-1) * self.w_shell + self.rad_core),
                                         (n * self.w_shell + self.rad_core),2)


            rs_s12, teta_s12 = np.meshgrid(self.discr_rs, self.discr_teta)


            phi_s1 = 2 * np.pi / 3

            xs_s1 = rs_s12 * np.cos(phi_s1) * np.sin(teta_s12)
            ys_s1 = rs_s12 * np.sin(phi_s1) * np.sin(teta_s12)
            zs_s1 = rs_s12 * np.cos(teta_s12)

            phi_s2 = 2 * np.pi

            xs_s2 = rs_s12 * np.cos(phi_s2) * np.sin(teta_s12)
            ys_s2 = rs_s12 * np.sin(phi_s2) * np.sin(teta_s12)
            zs_s2 = rs_s12 * np.cos(teta_s12)

            if (n % 2) == 0:
                ax.plot_surface(xs_s1, ys_s1, zs_s1, alpha = 1., color = "b")
                ax.plot_surface(xs_s2, ys_s2, zs_s2, alpha = 1., color ="b")

                ax.plot_surface(xs, ys, zs, alpha = 0.1, color = "b")


            else:

                ax.plot_surface(xs_s1, ys_s1, zs_s1, alpha=1., color="r")
                ax.plot_surface(xs_s2, ys_s2, zs_s2, alpha=1., color="r")

                ax.plot_surface(xs, ys, zs, alpha=0.1, color="r")



        ax.set_title('onion plot')

        plt.show()


class vesicle:

    def __init__(self):

        """Construct parameters for vesicle"""

        self.rad_core = 10    # Core radius
        self.w_shell = 2   # Shell thickness



        # Discretization spheres

        self.discr_phi = np.linspace( 2 * np.pi /3, 2 * np.pi, 30)
        self.discr_teta = np.linspace(0., np.pi, 30)



    def plot(self):

        """Plot vesicle"""

        fig = plt.figure()
        ax = plt.axes(projection = '3d')
        ax.set_aspect('auto')
        ax.set_xlabel('A')
        ax.set_ylabel('A')
        ax.set_zlabel('A');
        # Set viev by elevation and azimuth angles
        # Eelevation above x-y plane and counter clockwiserotation on z axis
        ax.view_init(25, 45)

        phi, teta = np.meshgrid(self.discr_phi, self.discr_teta)

        xc = self.rad_core * np.cos(phi) * np.sin(teta)
        yc = self.rad_core * np.sin(phi) * np.sin(teta)
        zc = self.rad_core * np.cos(teta)

        ax.plot_surface(xc, yc, zc, alpha=0.05, color="b")


        xs = (self.w_shell + self.rad_core) * np.cos(phi) * np.sin(teta)
        ys = (self.w_shell + self.rad_core) * np.sin(phi) * np.sin(teta)
        zs = (self.w_shell + self.rad_core) * np.cos(teta)


        self.discr_rs = np.linspace(( self.rad_core),
                                     (self.w_shell + self.rad_core),2)


        rs_s12, teta_s12 = np.meshgrid(self.discr_rs, self.discr_teta)


        phi_s1 = 2 * np.pi / 3

        xs_s1 = rs_s12 * np.cos(phi_s1) * np.sin(teta_s12)
        ys_s1 = rs_s12 * np.sin(phi_s1) * np.sin(teta_s12)
        zs_s1 = rs_s12 * np.cos(teta_s12)

        phi_s2 = 2 * np.pi

        xs_s2 = rs_s12 * np.cos(phi_s2) * np.sin(teta_s12)
        ys_s2 = rs_s12 * np.sin(phi_s2) * np.sin(teta_s12)
        zs_s2 = rs_s12 * np.cos(teta_s12)

        ax.plot_surface(xs_s1, ys_s1, zs_s1, alpha=1., color="r")
        ax.plot_surface(xs_s2, ys_s2, zs_s2, alpha=1., color="r")

        ax.plot_surface(xs, ys, zs, alpha=0.05, color="r")



        ax.set_title('vesicle plot')

        plt.show()

class polymer_micelle:

    def __init__(self):

        """Construct parameters for polymer_micelle"""

        self.rad_core = 10    # Core radius
        self.rad_gyr = 1    # Radius of gyration of gaussian chains
        self.d_pen = 1 # Factor to mimic non-penetration of Gaussian chains


        self.w_shell = 2 * self.rad_gyr * self.d_pen    # Outer part of chains (shell thickness)



        # Discretization spheres

        self.discr_phi = np.linspace( 2 * np.pi /3, 2 * np.pi, 30)
        self.discr_teta = np.linspace(0., np.pi, 30)



    def plot(self):

        """Plot polymer_micelle"""

        fig = plt.figure()
        ax = plt.axes(projection = '3d')
        ax.set_aspect('auto')
        ax.set_xlabel('A')
        ax.set_ylabel('A')
        ax.set_zlabel('A');
        # Set viev by elevation and azimuth angles
        # Eelevation above x-y plane and counter clockwiserotation on z axis
        ax.view_init(25, 45)

        phi, teta = np.meshgrid(self.discr_phi, self.discr_teta)

        xc = self.rad_core * np.cos(phi) * np.sin(teta)
        yc = self.rad_core * np.sin(phi) * np.sin(teta)
        zc = self.rad_core * np.cos(teta)

        self.discr_rc = np.linspace(0,self.rad_core, 2)

        rc_s12, teta_s12 = np.meshgrid(self.discr_rc, self.discr_teta)

        phi_s1 = 2 * np.pi / 3

        xc_s1 = rc_s12 * np.cos(phi_s1) * np.sin(teta_s12)
        yc_s1 = rc_s12 * np.sin(phi_s1) * np.sin(teta_s12)
        zc_s1 = rc_s12 * np.cos(teta_s12)

        phi_s2 = 2 * np.pi

        xc_s2 = rc_s12 * np.cos(phi_s2) * np.sin(teta_s12)
        yc_s2 = rc_s12 * np.sin(phi_s2) * np.sin(teta_s12)
        zc_s2 = rc_s12 * np.cos(teta_s12)


        ax.plot_surface(xc, yc, zc, alpha=0.1, color="b")

        ax.plot_surface(xc_s1, yc_s1, zc_s1, alpha=1., color="b")
        ax.plot_surface(xc_s2, yc_s2, zc_s2, alpha=1., color="b")

        xs = (self.w_shell + self.rad_core) * np.cos(phi) * np.sin(teta)
        ys = (self.w_shell + self.rad_core) * np.sin(phi) * np.sin(teta)
        zs = (self.w_shell + self.rad_core) * np.cos(teta)


        self.discr_rs = np.linspace(( self.rad_core),
                                     (self.w_shell + self.rad_core),2)


        rs_s12, teta_s12 = np.meshgrid(self.discr_rs, self.discr_teta)


        phi_s1 = 2 * np.pi / 3

        xs_s1 = rs_s12 * np.cos(phi_s1) * np.sin(teta_s12)
        ys_s1 = rs_s12 * np.sin(phi_s1) * np.sin(teta_s12)
        zs_s1 = rs_s12 * np.cos(teta_s12)

        phi_s2 = 2 * np.pi

        xs_s2 = rs_s12 * np.cos(phi_s2) * np.sin(teta_s12)
        ys_s2 = rs_s12 * np.sin(phi_s2) * np.sin(teta_s12)
        zs_s2 = rs_s12 * np.cos(teta_s12)

        ax.plot_surface(xs_s1, ys_s1, zs_s1, alpha=1., color="r")
        ax.plot_surface(xs_s2, ys_s2, zs_s2, alpha=1., color="r")

        ax.plot_surface(xs, ys, zs, alpha=0.1, color="r")



        ax.set_title('polymer_micelle plot')

        plt.show()


class multilayer_vesicle:

    def __init__(self):

        """Construct parameters for multilayer_vesicle"""

        self.rad_core = 10   # Radius of solvent filled core
        self.n_shells = 10   # Number of pairs (shells + solvent)
        self.w_shell = 1.   # Shell thickness
        self.w_inf = 1.     # Solvent interface thickness


        # Discretization spheres

        self.discr_phi = np.linspace( 2 * np.pi /3, 2 * np.pi, 30)
        self.discr_teta = np.linspace(0., np.pi, 30)



    def plot(self):

        """Plot multilayer_vesicle"""

        fig = plt.figure()
        ax = plt.axes(projection = '3d')
        ax.set_aspect('auto')
        ax.set_xlabel('A')
        ax.set_ylabel('A')
        ax.set_zlabel('A');
        # Set viev by elevation and azimuth angles
        # Eelevation above x-y plane and counter clockwiserotation on z axis
        ax.view_init(25, 45)

        phi, teta = np.meshgrid(self.discr_phi, self.discr_teta)

        xc = self.rad_core * np.cos(phi) * np.sin(teta)
        yc = self.rad_core * np.sin(phi) * np.sin(teta)
        zc = self.rad_core * np.cos(teta)

        ax.plot_surface(xc, yc, zc, alpha=0.1, color="b")

        for n in range(1,self.n_shells + 1):

            xs = (n * self.w_shell + (n-1) * self.w_inf + self.rad_core) * np.cos(phi) * np.sin(teta)
            ys = (n * self.w_shell + (n-1) * self.w_inf + self.rad_core) * np.sin(phi) * np.sin(teta)
            zs = (n * self.w_shell + (n-1) * self.w_inf + self.rad_core) * np.cos(teta)

            xi = (n * self.w_shell + n * self.w_inf + self.rad_core) * np.cos(phi) * np.sin(teta)
            yi = (n * self.w_shell + n * self.w_inf + self.rad_core) * np.sin(phi) * np.sin(teta)
            zi = (n * self.w_shell + n * self.w_inf + self.rad_core) * np.cos(teta)


            self.discr_rs = np.linspace(((n-1) * self.w_shell + (n-1) * self.w_inf + self.rad_core),
                                         (n * self.w_shell + (n-1) * self.w_inf + self.rad_core),2)

            self.discr_ri = np.linspace((n * self.w_shell + (n - 1) * self.w_inf + self.rad_core),
                                        (n * self.w_shell + n * self.w_inf + self.rad_core),2)


            ri_s12, teta_s12 = np.meshgrid(self.discr_ri, self.discr_teta)

            phi_s1 = 2 * np.pi / 3


            xi_s1 = ri_s12 * np.cos(phi_s1) * np.sin(teta_s12)
            yi_s1 = ri_s12 * np.sin(phi_s1) * np.sin(teta_s12)
            zi_s1 = ri_s12 * np.cos(teta_s12)

            phi_s2 = 2 * np.pi


            xi_s2 = ri_s12 * np.cos(phi_s2) * np.sin(teta_s12)
            yi_s2 = ri_s12 * np.sin(phi_s2) * np.sin(teta_s12)
            zi_s2 = ri_s12 * np.cos(teta_s12)



            ax.plot_surface(xi_s1, yi_s1, zi_s1, alpha = 1., color = "r")
            ax.plot_surface(xi_s2, yi_s2, zi_s2, alpha = 1., color = "r")
            ax.plot_surface(xs, ys, zs, alpha = 0.1, color = "b")


        ax.set_title('multilayer_vesicle plot')

        plt.show()


class core_shell_sphere:

    def __init__(self):

        """Construct parameters for core_shell_sphere"""

        self.rad_core = 10    # Core radius
        self.w_shell = 1    # shell thickness



        # Discretization spheres

        self.discr_phi = np.linspace( 2 * np.pi /3, 2 * np.pi, 30)
        self.discr_teta = np.linspace(0., np.pi, 30)



    def plot(self):

        """Plot core_shell_sphere"""

        fig = plt.figure()
        ax = plt.axes(projection = '3d')
        ax.set_aspect('auto')
        ax.set_xlabel('A')
        ax.set_ylabel('A')
        ax.set_zlabel('A');
        # Set viev by elevation and azimuth angles
        # Eelevation above x-y plane and counter clockwiserotation on z axis
        ax.view_init(25, 45)

        phi, teta = np.meshgrid(self.discr_phi, self.discr_teta)

        xc = self.rad_core * np.cos(phi) * np.sin(teta)
        yc = self.rad_core * np.sin(phi) * np.sin(teta)
        zc = self.rad_core * np.cos(teta)

        self.discr_rc = np.linspace(0,self.rad_core, 2)

        rc_s12, teta_s12 = np.meshgrid(self.discr_rc, self.discr_teta)

        phi_s1 = 2 * np.pi / 3

        xc_s1 = rc_s12 * np.cos(phi_s1) * np.sin(teta_s12)
        yc_s1 = rc_s12 * np.sin(phi_s1) * np.sin(teta_s12)
        zc_s1 = rc_s12 * np.cos(teta_s12)

        phi_s2 = 2 * np.pi

        xc_s2 = rc_s12 * np.cos(phi_s2) * np.sin(teta_s12)
        yc_s2 = rc_s12 * np.sin(phi_s2) * np.sin(teta_s12)
        zc_s2 = rc_s12 * np.cos(teta_s12)


        ax.plot_surface(xc, yc, zc, alpha=0.1, color="b")

        ax.plot_surface(xc_s1, yc_s1, zc_s1, alpha=1., color="b")
        ax.plot_surface(xc_s2, yc_s2, zc_s2, alpha=1., color="b")

        xs = (self.w_shell + self.rad_core) * np.cos(phi) * np.sin(teta)
        ys = (self.w_shell + self.rad_core) * np.sin(phi) * np.sin(teta)
        zs = (self.w_shell + self.rad_core) * np.cos(teta)


        self.discr_rs = np.linspace(( self.rad_core),
                                     (self.w_shell + self.rad_core),2)


        rs_s12, teta_s12 = np.meshgrid(self.discr_rs, self.discr_teta)


        phi_s1 = 2 * np.pi / 3

        xs_s1 = rs_s12 * np.cos(phi_s1) * np.sin(teta_s12)
        ys_s1 = rs_s12 * np.sin(phi_s1) * np.sin(teta_s12)
        zs_s1 = rs_s12 * np.cos(teta_s12)

        phi_s2 = 2 * np.pi

        xs_s2 = rs_s12 * np.cos(phi_s2) * np.sin(teta_s12)
        ys_s2 = rs_s12 * np.sin(phi_s2) * np.sin(teta_s12)
        zs_s2 = rs_s12 * np.cos(teta_s12)

        ax.plot_surface(xs_s1, ys_s1, zs_s1, alpha=1., color="r")
        ax.plot_surface(xs_s2, ys_s2, zs_s2, alpha=1., color="r")

        ax.plot_surface(xs, ys, zs, alpha=0.1, color="r")



        ax.set_title('core_shell_sphere')

        plt.show()


class core_multi_shell:

    def __init__(self):

        """Construct parameters for core_multi_shell"""

        self.rad_core = 10    # Core radius
        self.n_shells = 5   # Number of shells
        self.w_shell = 1.   # Shell thickness



        # Discretization spheres

        self.discr_phi = np.linspace( 2 * np.pi /3, 2 * np.pi, 30)
        self.discr_teta = np.linspace(0., np.pi, 30)



    def plot(self):

        """Plot core_multi_shell"""

        fig = plt.figure()
        ax = plt.axes(projection = '3d')
        ax.set_aspect('auto')
        ax.set_xlabel('A')
        ax.set_ylabel('A')
        ax.set_zlabel('A');
        # Set viev by elevation and azimuth angles
        # Eelevation above x-y plane and counter clockwiserotation on z axis
        ax.view_init(25, 45)

        phi, teta = np.meshgrid(self.discr_phi, self.discr_teta)

        xc = self.rad_core * np.cos(phi) * np.sin(teta)
        yc = self.rad_core * np.sin(phi) * np.sin(teta)
        zc = self.rad_core * np.cos(teta)


        self.discr_rc = np.linspace(0,self.rad_core, 2)

        rc_s12, teta_s12 = np.meshgrid(self.discr_rc, self.discr_teta)

        phi_s1 = 2 * np.pi / 3

        xc_s1 = rc_s12 * np.cos(phi_s1) * np.sin(teta_s12)
        yc_s1 = rc_s12 * np.sin(phi_s1) * np.sin(teta_s12)
        zc_s1 = rc_s12 * np.cos(teta_s12)

        phi_s2 = 2 * np.pi

        xc_s2 = rc_s12 * np.cos(phi_s2) * np.sin(teta_s12)
        yc_s2 = rc_s12 * np.sin(phi_s2) * np.sin(teta_s12)
        zc_s2 = rc_s12 * np.cos(teta_s12)

        ax.plot_surface(xc_s1, yc_s1, zc_s1, alpha=1., color="white")
        ax.plot_surface(xc_s2, yc_s2, zc_s2, alpha=1., color="white")

        ax.plot_surface(xc, yc, zc, alpha=0.1, color="white")



        for n in range(1,self.n_shells + 1):

            xs = (n * self.w_shell + self.rad_core) * np.cos(phi) * np.sin(teta)
            ys = (n * self.w_shell + self.rad_core) * np.sin(phi) * np.sin(teta)
            zs = (n * self.w_shell + self.rad_core) * np.cos(teta)


            self.discr_rs = np.linspace(((n-1) * self.w_shell + self.rad_core),
                                         (n * self.w_shell + self.rad_core),2)


            rs_s12, teta_s12 = np.meshgrid(self.discr_rs, self.discr_teta)


            phi_s1 = 2 * np.pi / 3

            xs_s1 = rs_s12 * np.cos(phi_s1) * np.sin(teta_s12)
            ys_s1 = rs_s12 * np.sin(phi_s1) * np.sin(teta_s12)
            zs_s1 = rs_s12 * np.cos(teta_s12)

            phi_s2 = 2 * np.pi

            xs_s2 = rs_s12 * np.cos(phi_s2) * np.sin(teta_s12)
            ys_s2 = rs_s12 * np.sin(phi_s2) * np.sin(teta_s12)
            zs_s2 = rs_s12 * np.cos(teta_s12)

            if (n % 2) == 0:
                ax.plot_surface(xs_s1, ys_s1, zs_s1, alpha = 1., color = "b")
                ax.plot_surface(xs_s2, ys_s2, zs_s2, alpha = 1., color ="b")

                ax.plot_surface(xs, ys, zs, alpha = 0.1, color = "b")


            else:

                ax.plot_surface(xs_s1, ys_s1, zs_s1, alpha=1., color="r")
                ax.plot_surface(xs_s2, ys_s2, zs_s2, alpha=1., color="r")

                ax.plot_surface(xs, ys, zs, alpha=0.1, color="r")


        ax.set_title('core_multi_shell plot')

        plt.show()


class fuzzy_sphere:

    def __init__(self):

        """Construct parameters for fuzzy_sphere"""

        self.rad_core = 10    # Core radius
        self.fuzziness = 1    # Standard deviation of gaussian convolution (length)




        self.w_shell = self.fuzziness    # shell thickness



        # Discretization spheres

        self.discr_phi = np.linspace( 2 * np.pi /3, 2 * np.pi, 30)
        self.discr_teta = np.linspace(0., np.pi, 30)



    def plot(self):

        """Plot fuzzy_sphere"""

        fig = plt.figure()
        ax = plt.axes(projection = '3d')
        ax.set_aspect('auto')
        ax.set_xlabel('A')
        ax.set_ylabel('A')
        ax.set_zlabel('A');
        # Set viev by elevation and azimuth angles
        # Eelevation above x-y plane and counter clockwiserotation on z axis
        ax.view_init(25, 45)

        phi, teta = np.meshgrid(self.discr_phi, self.discr_teta)

        xc = self.rad_core * np.cos(phi) * np.sin(teta)
        yc = self.rad_core * np.sin(phi) * np.sin(teta)
        zc = self.rad_core * np.cos(teta)

        self.discr_rc = np.linspace(0,self.rad_core, 2)

        rc_s12, teta_s12 = np.meshgrid(self.discr_rc, self.discr_teta)

        phi_s1 = 2 * np.pi / 3

        xc_s1 = rc_s12 * np.cos(phi_s1) * np.sin(teta_s12)
        yc_s1 = rc_s12 * np.sin(phi_s1) * np.sin(teta_s12)
        zc_s1 = rc_s12 * np.cos(teta_s12)

        phi_s2 = 2 * np.pi

        xc_s2 = rc_s12 * np.cos(phi_s2) * np.sin(teta_s12)
        yc_s2 = rc_s12 * np.sin(phi_s2) * np.sin(teta_s12)
        zc_s2 = rc_s12 * np.cos(teta_s12)


        ax.plot_surface(xc, yc, zc, alpha=0.1, color="b")

        ax.plot_surface(xc_s1, yc_s1, zc_s1, alpha=1., color="b")
        ax.plot_surface(xc_s2, yc_s2, zc_s2, alpha=1., color="b")

        xs = (self.w_shell + self.rad_core) * np.cos(phi) * np.sin(teta)
        ys = (self.w_shell + self.rad_core) * np.sin(phi) * np.sin(teta)
        zs = (self.w_shell + self.rad_core) * np.cos(teta)


        self.discr_rs = np.linspace(( self.rad_core),
                                     (self.w_shell + self.rad_core),2)


        rs_s12, teta_s12 = np.meshgrid(self.discr_rs, self.discr_teta)


        phi_s1 = 2 * np.pi / 3

        xs_s1 = rs_s12 * np.cos(phi_s1) * np.sin(teta_s12)
        ys_s1 = rs_s12 * np.sin(phi_s1) * np.sin(teta_s12)
        zs_s1 = rs_s12 * np.cos(teta_s12)

        phi_s2 = 2 * np.pi

        xs_s2 = rs_s12 * np.cos(phi_s2) * np.sin(teta_s12)
        ys_s2 = rs_s12 * np.sin(phi_s2) * np.sin(teta_s12)
        zs_s2 = rs_s12 * np.cos(teta_s12)

        ax.plot_surface(xs_s1, ys_s1, zs_s1, alpha=0.3, color="b")
        ax.plot_surface(xs_s2, ys_s2, zs_s2, alpha=0.3, color="b")

        ax.plot_surface(xs, ys, zs, alpha=0.3, color="b")



        ax.set_title('fuzzy_sphere')

        plt.show()


class linear_pearls:

    def __init__(self):
        """Construct parameters for linear_pearls"""

        self.radius = 1   # Pearl radius
        self.edge_sep = 1   # Length of the string segment (surface to surface)
        self.n_pearls = 7  # Number of pearls

        # Discretization sphere

        self.discr_phi = np.linspace(0., 2 * np.pi, 30)
        self.discr_teta = np.linspace(0., np.pi, 30)

    def plot(self):
        """Plot linear_pearls"""

        fig = plt.figure()
        ax = plt.axes(projection='3d')
        ax.set_aspect('auto')
        ax.set_xlabel('A')
        ax.set_ylabel('A')
        ax.set_zlabel('A');
        # Set viev by elevation and azimuth angles
        # Eelevation above x-y plane and counter clockwiserotation on z axis
        ax.view_init(20, 90)

        phi, teta = np.meshgrid(self.discr_phi, self.discr_teta)

        for n in range(0,self.n_pearls):

            x = self.radius * np.cos(phi) * np.sin(teta) + n * (self.edge_sep + 2 * self.radius)
            y = self.radius * np.sin(phi) * np.sin(teta)
            z = self.radius * np.cos(teta)

            if n <= self.n_pearls-2:
                xm = n * (self.edge_sep + 2 * self.radius)
                x1 = xm + self.radius
                x2 = xm + self.radius + self.edge_sep
                xlin = np.linspace(x1, x2, 2)
                ax.plot3D(xlin, 0 * xlin, 0 * xlin, alpha=1, color="b")


            ax.plot_surface(x, y, z, alpha=1, color="b")

            ax.set_title('linear_pearls plot')
        plt.show()

class binary_hard_sphere:

    def __init__(self):
        """Construct parameters for binary_hard_sphere"""

        self.radius_lg = 20   # radius large sphere
        self.radius_sm = 1   # radius small sphere


        # Discretization sphere

        self.discr_phi = np.linspace(0., 2 * np.pi, 30)
        self.discr_teta = np.linspace(0., np.pi, 30)

    def plot(self):
        """Plot binary_hard_sphere"""

        fig = plt.figure()
        ax = plt.axes(projection='3d')
        ax.set_aspect('auto')
        ax.set_xlabel('A')
        ax.set_ylabel('A')
        ax.set_zlabel('A');
        # Set viev by elevation and azimuth angles
        # Eelevation above x-y plane and counter clockwiserotation on z axis
        ax.view_init(20, 70)

        phi, teta = np.meshgrid(self.discr_phi, self.discr_teta)


        x_lg = self.radius_lg * np.cos(phi) * np.sin(teta)
        y_lg = self.radius_lg * np.sin(phi) * np.sin(teta)
        z_lg = self.radius_lg * np.cos(teta)

        x_sm = self.radius_sm * np.cos(phi) * np.sin(teta) + 3 * self.radius_lg
        y_sm = self.radius_sm * np.sin(phi) * np.sin(teta)
        z_sm = self.radius_sm * np.cos(teta)

        ax.plot_surface(x_lg, y_lg, z_lg, alpha=1, color="b")

        ax.plot_surface(x_sm, y_sm, z_sm, alpha=1, color="r")

        ax.set_title('binary_hard_sphere plot')
        plt.show()

class adsorbed_layer:

    def __init__(self):

        """Construct parameters for adsorbed_layer"""

        self.rad_core = 500    # Core radius
        self.ad_am = 1.9      # Adsorbed amount mg / m^2 in shell
        self.denshell = 0.7     # Bulk density of polymer in the shell  g / cm^3




        self.w_shell = (self.ad_am * 10)/ self.denshell    # shell thickness



        # Discretization spheres

        self.discr_phi = np.linspace( 2 * np.pi /3, 2 * np.pi, 30)
        self.discr_teta = np.linspace(0., np.pi, 30)



    def plot(self):

        """Plot adsorbed_layer"""

        fig = plt.figure()
        ax = plt.axes(projection = '3d')
        ax.set_aspect('auto')
        ax.set_xlabel('A')
        ax.set_ylabel('A')
        ax.set_zlabel('A');
        # Set viev by elevation and azimuth angles
        # Eelevation above x-y plane and counter clockwiserotation on z axis
        ax.view_init(25, 45)

        phi, teta = np.meshgrid(self.discr_phi, self.discr_teta)

        xc = self.rad_core * np.cos(phi) * np.sin(teta)
        yc = self.rad_core * np.sin(phi) * np.sin(teta)
        zc = self.rad_core * np.cos(teta)

        self.discr_rc = np.linspace(0,self.rad_core, 2)

        rc_s12, teta_s12 = np.meshgrid(self.discr_rc, self.discr_teta)

        phi_s1 = 2 * np.pi / 3

        xc_s1 = rc_s12 * np.cos(phi_s1) * np.sin(teta_s12)
        yc_s1 = rc_s12 * np.sin(phi_s1) * np.sin(teta_s12)
        zc_s1 = rc_s12 * np.cos(teta_s12)

        phi_s2 = 2 * np.pi

        xc_s2 = rc_s12 * np.cos(phi_s2) * np.sin(teta_s12)
        yc_s2 = rc_s12 * np.sin(phi_s2) * np.sin(teta_s12)
        zc_s2 = rc_s12 * np.cos(teta_s12)


        ax.plot_surface(xc, yc, zc, alpha=0.1, color="b")

        ax.plot_surface(xc_s1, yc_s1, zc_s1, alpha=1., color="b")
        ax.plot_surface(xc_s2, yc_s2, zc_s2, alpha=1., color="b")

        xs = (self.w_shell + self.rad_core) * np.cos(phi) * np.sin(teta)
        ys = (self.w_shell + self.rad_core) * np.sin(phi) * np.sin(teta)
        zs = (self.w_shell + self.rad_core) * np.cos(teta)


        self.discr_rs = np.linspace(( self.rad_core),
                                     (self.w_shell + self.rad_core),2)


        rs_s12, teta_s12 = np.meshgrid(self.discr_rs, self.discr_teta)


        phi_s1 = 2 * np.pi / 3

        xs_s1 = rs_s12 * np.cos(phi_s1) * np.sin(teta_s12)
        ys_s1 = rs_s12 * np.sin(phi_s1) * np.sin(teta_s12)
        zs_s1 = rs_s12 * np.cos(teta_s12)

        phi_s2 = 2 * np.pi

        xs_s2 = rs_s12 * np.cos(phi_s2) * np.sin(teta_s12)
        ys_s2 = rs_s12 * np.sin(phi_s2) * np.sin(teta_s12)
        zs_s2 = rs_s12 * np.cos(teta_s12)

        ax.plot_surface(xs_s1, ys_s1, zs_s1, alpha=1., color="r")
        ax.plot_surface(xs_s2, ys_s2, zs_s2, alpha=1., color="r")

        ax.plot_surface(xs, ys, zs, alpha=0.1, color="r")



        ax.set_title('adsorbed_layer')

        plt.show()



# raspberry().plot()
# sphere().plot()
# spherical_sld().plot()
# onion().plot()
# vesicle().plot()
# polymer_micelle().plot()
# multilayer_vesicle().plot()
# core_shell_sphere().plot()
# core_multi_shell().plot()
# fuzzy_sphere().plot()
# linear_pearls().plot()
# binary_hard_sphere().plot()
# adsorbed_layer().plot()











































