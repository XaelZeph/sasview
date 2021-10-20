from math import radians
import numpy as np
import matplotlib
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

class plotModelSphere:

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

        if self.param['model_name'] != 'raspberry':

            if (self.param['model_name'] != 'sphere' and
                self.param['model_name'] != 'binary_hard_sphere' and
                self.param['model_name'] != 'ellipsoid' and
                self.param['model_name'] != 'triaxial_ellipsoid'):

                # Discretization sphere (cut)
                discr_phi = np.linspace(2 * np.pi /3, 2 * np.pi, 30)
                self.discr_teta = np.linspace(0., np.pi, 30)

            else:
                # Discretization sphere
                discr_phi = np.linspace(0., 2 * np.pi, 30)
                self.discr_teta = np.linspace(0.,np.pi,30)

            self.phi, self.teta = np.meshgrid(discr_phi, self.discr_teta)

        else:
            # Discretization small and large spheres


            self.n_lg = 30  # Number of discretization elements large sphere
            self.n_sm = 6  # Number of discretization elements small sphere
            
            discr_phi_lg = np.linspace(0., 2 * np.pi, self.n_lg)
            discr_teta_lg = np.linspace(0., np.pi, self.n_lg)

            discr_phi_sm = np.linspace(0., 2 * np.pi, self.n_sm)
            discr_teta_sm = np.linspace(0., np.pi, self.n_sm)


            self.phi_sm, self.teta_sm = np.meshgrid(discr_phi_sm, discr_teta_sm)
            self.phi_lg, self.teta_lg = np.meshgrid(discr_phi_lg, discr_teta_lg)


        if self.param['model_name'] == 'sphere':
            self.sphere()

        if self.param['model_name'] == 'raspberry':
            self.raspberry()

        if self.param['model_name'] == 'spherical_sld':
            self.spherical_sld()

        if self.param['model_name'] == 'onion':
            self.onion()

        if self.param['model_name'] == 'vesicle':
            self.vesicle()

        if self.param['model_name'] == 'polymer_micelle':
            self.polymer_micelle()

        if self.param['model_name'] == 'multilayer_vesicle':
            self.multilayer_vesicle()

        if self.param['model_name'] == 'core_shell_sphere':
            self.core_shell_sphere()

        if self.param['model_name'] == 'core_multi_shell':
            self.core_multi_shell()

        if self.param['model_name'] == 'fuzzy_sphere':
            self.fuzzy_sphere()

        if self.param['model_name'] == 'linear_pearls':
            self.linear_pearls()

        if self.param['model_name'] == 'binary_hard_sphere':
            self.binary_hard_sphere()

        if self.param['model_name'] == 'adsorbed_layer':
            self.adsorbed_layer()

        if self.param['model_name'] == 'ellipsoid':
            self.ellipsoid()

        if self.param['model_name'] == 'core_shell_ellipsoid':
            self.core_shell_ellipsoid()

        if self.param['model_name'] == 'triaxial_ellipsoid':
            self.triaxial_ellipsoid()

    def sphere(self):

        self.x = self.param['radius'] * np.cos(self.phi) * np.sin(self.teta)
        self.y = self.param['radius'] * np.sin(self.phi) * np.sin(self.teta)
        self.z = self.param['radius'] * np.cos(self.teta)

        self.fig = plt.figure('3d-model')
        self.ax = plt.axes(projection = '3d')
        self.ax.set_xlabel('A')
        self.ax.set_ylabel('A')
        self.ax.set_zlabel('A')
        self.ax.set_title('Sphere plot')
        self.ax.view_init(25, 45)

        self.surf = self.ax.plot_surface(self.x, self.y, self.z, color='b', antialiased=False, linewidth=0)

        self.fig.canvas.draw()
        self.fig.show()

    def raspberry(self):

        radius_lg = self.param['radius_lg']  # large sphere
        radius_sm = self.param['radius_sm']  # small sphere

        volfraction_lg = self.param['volfraction_lg']  # Volume fraction large spheres
        volfraction_sm = self.param['volfraction_sm']  # Volume fraction small spheres
        surface_fraction = self.param['surface_fraction']  # Fraction of small spheres at the surface
        penetration = self.param['penetration']  # Fractional penetration depth of small spheres into large sphere

        Np = int((surface_fraction * volfraction_sm * radius_lg ** 3) / (
                volfraction_lg * radius_sm ** 3))  # Number of small spheres per large sphere

        # Center to center distance spheres
        c_to_c = (radius_lg + radius_sm - penetration)

        # Generate parameters for equidstributed "berries" on sphere Fibbonacci method

        gr = (1 + np.sqrt(5)) / 2

        x_sq = np.ones(Np)
        y_sq = np.ones(Np)


        x_lg = radius_lg * np.cos(self.phi_lg) * np.sin(self.teta_lg)
        y_lg = radius_lg * np.sin(self.phi_lg) * np.sin(self.teta_lg)
        z_lg = radius_lg * np.cos(self.teta_lg)

        x_sm = radius_sm * np.cos(self.phi_sm) * np.sin(self.teta_sm)
        y_sm = radius_sm * np.sin(self.phi_sm) * np.sin(self.teta_sm)
        z_sm = radius_sm * np.cos(self.teta_sm)

        self.fig = plt.figure('3d-model')
        self.ax = plt.axes(projection='3d')
        self.ax.set_xlabel('A')
        self.ax.set_ylabel('A')
        self.ax.set_zlabel('A')
        self.ax.set_title('raspberry plot')
        self.ax.view_init(25, 45)

        self.surf = self.ax.plot_surface(x_lg, y_lg, z_lg, alpha=0.5, color="b")

        # Equidistribute berries with Fibonacci method

        for j in range(0, Np):
            x_sq[j] = ((j / gr) - np.floor(j / gr))
            y_sq[j] = j / Np

            self.theta_sph = 2 * np.pi * x_sq[j]
            self.phi_sph = np.arccos(1 - 2 * y_sq[j])

            x_sph = c_to_c * np.cos(self.theta_sph) * np.sin(self.phi_sph);
            y_sph = c_to_c * np.sin(self.theta_sph) * np.sin(self.phi_sph);
            z_sph = c_to_c * np.cos(self.phi_sph);

            self.surf = self.ax.plot_surface(x_sm + x_sph * np.ones((self.n_sm, self.n_sm)),
                            y_sm + y_sph * np.ones((self.n_sm, self.n_sm)),
                            z_sm + z_sph * np.ones((self.n_sm, self.n_sm)), alpha=1, color="r")


        self.fig.canvas.draw()
        self.fig.show()

    def spherical_sld(self):

        n_shells = int(self.param['n_shells'])  # Number of shells
        w_shell = self.param['thickness1']  # Shell thickness
        w_inf = self.param['interface1']  # Interface thickness

        self.fig = plt.figure('3d-model')
        self.ax = plt.axes(projection='3d')
        self.ax.set_xlabel('A')
        self.ax.set_ylabel('A')
        self.ax.set_zlabel('A')
        self.ax.set_title('spherical_sld plot')
        self.ax.view_init(25, 45)


        for n in range(1,n_shells + 2):      # + 2 To match n_shells = 1 gives a core and an interface.

            xs = (n * w_shell + (n-1) * w_inf) * np.cos(self.phi) * np.sin(self.teta)
            ys = (n * w_shell + (n-1) * w_inf) * np.sin(self.phi) * np.sin(self.teta)
            zs = (n * w_shell + (n-1) * w_inf) * np.cos(self.teta)

            xi = (n * w_shell + n * w_inf) * np.cos(self.phi) * np.sin(self.teta)
            yi = (n * w_shell + n * w_inf) * np.sin(self.phi) * np.sin(self.teta)
            zi = (n * w_shell + n * w_inf) * np.cos(self.teta)


            discr_rs = np.linspace(((n-1) * w_shell + (n-1) * w_inf),
                                         (n * w_shell + (n-1) * w_inf),2)

            discr_ri = np.linspace((n * w_shell + (n - 1) * w_inf),
                                        (n * w_shell + n * w_inf),2)


            rs_s12, teta_s12 = np.meshgrid(discr_rs, self.discr_teta)

            ri_s12, teta_s12 = np.meshgrid(discr_ri, self.discr_teta)

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


            self.surf = self.ax.plot_surface(xs_s1, ys_s1, zs_s1, alpha = 1., color = "b")
            self.surf = self.ax.plot_surface(xs_s2, ys_s2, zs_s2, alpha = 1., color ="b")
            self.surf = self.ax.plot_surface(xi_s1, yi_s1, zi_s1, alpha = 1., color = "r")
            self.surf = self.ax.plot_surface(xi_s2, yi_s2, zi_s2, alpha = 1., color = "r")
            self.surf = self.ax.plot_surface(xs, ys, zs, alpha = 0.1, color = "b")
            self.surf = self.ax.plot_surface(xi, yi, zi, alpha = 0.1, color = "r")

        self.fig.canvas.draw()
        self.fig.show()

    def onion(self):

        rad_core = self.param['radius_core']  # Core radius
        n_shells = int(self.param['n_shells'])  # Number of shells
        w_shell = self.param['thickness1']  # Shell thickness

        self.fig = plt.figure('3d-model')
        self.ax = plt.axes(projection='3d')
        self.ax.set_xlabel('A')
        self.ax.set_ylabel('A')
        self.ax.set_zlabel('A')
        self.ax.set_title('onion plot')
        self.ax.view_init(25, 45)

        xc = rad_core * np.cos(self.phi) * np.sin(self.teta)
        yc = rad_core * np.sin(self.phi) * np.sin(self.teta)
        zc = rad_core * np.cos(self.teta)

        discr_rc = np.linspace(0, rad_core, 2)

        rc_s12, teta_s12 = np.meshgrid(discr_rc, self.discr_teta)

        phi_s1 = 2 * np.pi / 3

        xc_s1 = rc_s12 * np.cos(phi_s1) * np.sin(teta_s12)
        yc_s1 = rc_s12 * np.sin(phi_s1) * np.sin(teta_s12)
        zc_s1 = rc_s12 * np.cos(teta_s12)

        phi_s2 = 2 * np.pi

        xc_s2 = rc_s12 * np.cos(phi_s2) * np.sin(teta_s12)
        yc_s2 = rc_s12 * np.sin(phi_s2) * np.sin(teta_s12)
        zc_s2 = rc_s12 * np.cos(teta_s12)



        for n in range(1,n_shells + 1):

            xs = (n * w_shell + rad_core) * np.cos(self.phi) * np.sin(self.teta)
            ys = (n * w_shell + rad_core) * np.sin(self.phi) * np.sin(self.teta)
            zs = (n * w_shell + rad_core) * np.cos(self.teta)


            discr_rs = np.linspace(((n-1) * w_shell + rad_core),
                                         (n * w_shell + rad_core),2)


            rs_s12, teta_s12 = np.meshgrid(discr_rs, self.discr_teta)


            phi_s1 = 2 * np.pi / 3

            xs_s1 = rs_s12 * np.cos(phi_s1) * np.sin(teta_s12)
            ys_s1 = rs_s12 * np.sin(phi_s1) * np.sin(teta_s12)
            zs_s1 = rs_s12 * np.cos(teta_s12)

            phi_s2 = 2 * np.pi

            xs_s2 = rs_s12 * np.cos(phi_s2) * np.sin(teta_s12)
            ys_s2 = rs_s12 * np.sin(phi_s2) * np.sin(teta_s12)
            zs_s2 = rs_s12 * np.cos(teta_s12)

            if (n % 2) == 0:
                self.surf = self.ax.plot_surface(xs_s1, ys_s1, zs_s1, alpha = 1., color = "b")
                self.surf = self.ax.plot_surface(xs_s2, ys_s2, zs_s2, alpha = 1., color ="b")

                self.surf = self.ax.plot_surface(xs, ys, zs, alpha = 0.1, color = "b")


            else:

                self.surf = self.ax.plot_surface(xs_s1, ys_s1, zs_s1, alpha=1., color="r")
                self.surf = self.ax.plot_surface(xs_s2, ys_s2, zs_s2, alpha=1., color="r")

                self.surf = self.ax.plot_surface(xs, ys, zs, alpha=0.1, color="r")


        self.surf = self.ax.plot_surface(xc_s1, yc_s1, zc_s1, alpha=1., color="white")
        self.surf = self.ax.plot_surface(xc_s2, yc_s2, zc_s2, alpha=1., color="white")
        self.surf = self.ax.plot_surface(xc, yc, zc, alpha=0.1, color="white")

        self.fig.canvas.draw()
        self.fig.show()

    def vesicle(self):

        rad_core = self.param['radius']  # Core radius
        w_shell = self.param['thickness']  # Shell thickness

        self.fig = plt.figure('3d-model')
        self.ax = plt.axes(projection='3d')
        self.ax.set_xlabel('A')
        self.ax.set_ylabel('A')
        self.ax.set_zlabel('A')
        self.ax.set_title('vesicle plot')
        self.ax.view_init(25, 45)


        xc = rad_core * np.cos(self.phi) * np.sin(self.teta)
        yc = rad_core * np.sin(self.phi) * np.sin(self.teta)
        zc = rad_core * np.cos(self.teta)



        xs = (w_shell + rad_core) * np.cos(self.phi) * np.sin(self.teta)
        ys = (w_shell + rad_core) * np.sin(self.phi) * np.sin(self.teta)
        zs = (w_shell + rad_core) * np.cos(self.teta)

        discr_rs = np.linspace((rad_core),
                                    (w_shell + rad_core), 2)

        rs_s12, teta_s12 = np.meshgrid(discr_rs, self.discr_teta)

        phi_s1 = 2 * np.pi / 3

        xs_s1 = rs_s12 * np.cos(phi_s1) * np.sin(teta_s12)
        ys_s1 = rs_s12 * np.sin(phi_s1) * np.sin(teta_s12)
        zs_s1 = rs_s12 * np.cos(teta_s12)

        phi_s2 = 2 * np.pi

        xs_s2 = rs_s12 * np.cos(phi_s2) * np.sin(teta_s12)
        ys_s2 = rs_s12 * np.sin(phi_s2) * np.sin(teta_s12)
        zs_s2 = rs_s12 * np.cos(teta_s12)

        self.surf = self.ax.plot_surface(xc, yc, zc, alpha=0.1, color="b")

        self.surf = self.ax.plot_surface(xs_s1, ys_s1, zs_s1, alpha=1., color="r")
        self.surf = self.ax.plot_surface(xs_s2, ys_s2, zs_s2, alpha=1., color="r")

        self.surf = self.ax.plot_surface(xs, ys, zs, alpha=0.1, color="r")

        self.fig.canvas.draw()
        self.fig.show()

    def polymer_micelle(self):

        rad_core = self.param['radius_core']  # Core radius
        rad_gyr = self.param['rg']  # Radius of gyration of gaussian chains
        d_pen = self.param['d_penetration']  # Factor to mimic non-penetration of Gaussian chains

        w_shell = 2 * rad_gyr * d_pen  # Outer part of chains (shell thickness)

        self.fig = plt.figure('3d-model')
        self.ax = plt.axes(projection='3d')
        self.ax.set_xlabel('A')
        self.ax.set_ylabel('A')
        self.ax.set_zlabel('A')
        self.ax.set_title('polymer_micelle plot')
        self.ax.view_init(25, 45)

        xc = rad_core * np.cos(self.phi) * np.sin(self.teta)
        yc = rad_core * np.sin(self.phi) * np.sin(self.teta)
        zc = rad_core * np.cos(self.teta)

        discr_rc = np.linspace(0, rad_core, 2)

        rc_s12, teta_s12 = np.meshgrid(discr_rc, self.discr_teta)

        phi_s1 = 2 * np.pi / 3

        xc_s1 = rc_s12 * np.cos(phi_s1) * np.sin(teta_s12)
        yc_s1 = rc_s12 * np.sin(phi_s1) * np.sin(teta_s12)
        zc_s1 = rc_s12 * np.cos(teta_s12)

        phi_s2 = 2 * np.pi

        xc_s2 = rc_s12 * np.cos(phi_s2) * np.sin(teta_s12)
        yc_s2 = rc_s12 * np.sin(phi_s2) * np.sin(teta_s12)
        zc_s2 = rc_s12 * np.cos(teta_s12)


        xs = (w_shell + rad_core) * np.cos(self.phi) * np.sin(self.teta)
        ys = (w_shell + rad_core) * np.sin(self.phi) * np.sin(self.teta)
        zs = (w_shell + rad_core) * np.cos(self.teta)

        discr_rs = np.linspace(rad_core,
                                    (w_shell + rad_core), 2)

        rs_s12, teta_s12 = np.meshgrid(discr_rs, self.discr_teta)

        phi_s1 = 2 * np.pi / 3

        xs_s1 = rs_s12 * np.cos(phi_s1) * np.sin(teta_s12)
        ys_s1 = rs_s12 * np.sin(phi_s1) * np.sin(teta_s12)
        zs_s1 = rs_s12 * np.cos(teta_s12)

        phi_s2 = 2 * np.pi

        xs_s2 = rs_s12 * np.cos(phi_s2) * np.sin(teta_s12)
        ys_s2 = rs_s12 * np.sin(phi_s2) * np.sin(teta_s12)
        zs_s2 = rs_s12 * np.cos(teta_s12)



        self.surf = self.ax.plot_surface(xc, yc, zc, alpha=0.1, color="b")

        self.surf = self.ax.plot_surface(xc_s1, yc_s1, zc_s1, alpha=1., color="b")
        self.surf = self.ax.plot_surface(xc_s2, yc_s2, zc_s2, alpha=1., color="b")

        self.surf = self.ax.plot_surface(xs_s1, ys_s1, zs_s1, alpha=1., color="r")
        self.surf = self.ax.plot_surface(xs_s2, ys_s2, zs_s2, alpha=1., color="r")

        self.surf = self.ax.plot_surface(xs, ys, zs, alpha=0.1, color="r")

        self.fig.canvas.draw()
        self.fig.show()

    def multilayer_vesicle(self):

        rad_core = self.param['radius']  # Radius of solvent filled core
        n_shells = int(self.param['n_shells'])  # Number of pairs (shells + solvent)
        w_shell = self.param['thick_shell']  # Shell thickness
        w_inf = self.param['thick_solvent']  # Solvent interface thickness

        self.fig = plt.figure('3d-model')
        self.ax = plt.axes(projection='3d')
        self.ax.set_xlabel('A')
        self.ax.set_ylabel('A')
        self.ax.set_zlabel('A')
        self.ax.set_title('multilayer_vesicle plot')
        self.ax.view_init(25, 45)

        xc = rad_core * np.cos(self.phi) * np.sin(self.teta)
        yc = rad_core * np.sin(self.phi) * np.sin(self.teta)
        zc = rad_core * np.cos(self.teta)


        for n in range(1, n_shells + 1):

            xs = ((n - 1) * w_shell + (n - 1) * w_inf + rad_core) * np.cos(self.phi) * np.sin(self.teta)
            ys = ((n - 1) * w_shell + (n - 1) * w_inf + rad_core) * np.sin(self.phi) * np.sin(self.teta)
            zs = ((n - 1) * w_shell + (n - 1) * w_inf + rad_core) * np.cos(self.teta)

            xi = (n * w_shell + (n - 1) * w_inf + rad_core) * np.cos(self.phi) * np.sin(self.teta)
            yi = (n * w_shell + (n - 1) * w_inf + rad_core) * np.sin(self.phi) * np.sin(self.teta)
            zi = (n * w_shell + (n - 1) * w_inf + rad_core) * np.cos(self.teta)

            discr_ri = np.linspace(((n - 1) * w_shell + (n - 1) * w_inf + rad_core),
                                        (n * w_shell + (n - 1) * w_inf + rad_core), 2)

            ri_s12, teta_s12 = np.meshgrid(discr_ri, self.discr_teta)

            phi_s1 = 2 * np.pi / 3

            xi_s1 = ri_s12 * np.cos(phi_s1) * np.sin(teta_s12)
            yi_s1 = ri_s12 * np.sin(phi_s1) * np.sin(teta_s12)
            zi_s1 = ri_s12 * np.cos(teta_s12)

            phi_s2 = 2 * np.pi

            xi_s2 = ri_s12 * np.cos(phi_s2) * np.sin(teta_s12)
            yi_s2 = ri_s12 * np.sin(phi_s2) * np.sin(teta_s12)
            zi_s2 = ri_s12 * np.cos(teta_s12)

            self.surf = self.ax.plot_surface(xc, yc, zc, alpha=0.1, color="b")

            self.surf = self.ax.plot_surface(xi_s1, yi_s1, zi_s1, alpha=1., color="r")
            self.surf = self.ax.plot_surface(xi_s2, yi_s2, zi_s2, alpha=1., color="r")
            self.surf = self.ax.plot_surface(xs, ys, zs, alpha=0.1, color="r")
            self.surf = self.ax.plot_surface(xi, yi, zi, alpha=0.1, color="r")

        self.fig.canvas.draw()
        self.fig.show()


    def core_shell_sphere(self):

        rad_core = self.param['radius']  # Core radius
        w_shell = self.param['thickness']  # shell thickness

        self.fig = plt.figure('3d-model')
        self.ax = plt.axes(projection='3d')
        self.ax.set_xlabel('A')
        self.ax.set_ylabel('A')
        self.ax.set_zlabel('A')
        self.ax.set_title('core_shell_sphere plot')
        self.ax.view_init(25, 45)

        xc = rad_core * np.cos(self.phi) * np.sin(self.teta)
        yc = rad_core * np.sin(self.phi) * np.sin(self.teta)
        zc = rad_core * np.cos(self.teta)

        discr_rc = np.linspace(0, rad_core, 2)

        rc_s12, teta_s12 = np.meshgrid(discr_rc, self.discr_teta)

        phi_s1 = 2 * np.pi / 3

        xc_s1 = rc_s12 * np.cos(phi_s1) * np.sin(teta_s12)
        yc_s1 = rc_s12 * np.sin(phi_s1) * np.sin(teta_s12)
        zc_s1 = rc_s12 * np.cos(teta_s12)

        phi_s2 = 2 * np.pi

        xc_s2 = rc_s12 * np.cos(phi_s2) * np.sin(teta_s12)
        yc_s2 = rc_s12 * np.sin(phi_s2) * np.sin(teta_s12)
        zc_s2 = rc_s12 * np.cos(teta_s12)


        xs = (w_shell + rad_core) * np.cos(self.phi) * np.sin(self.teta)
        ys = (w_shell + rad_core) * np.sin(self.phi) * np.sin(self.teta)
        zs = (w_shell + rad_core) * np.cos(self.teta)

        discr_rs = np.linspace((rad_core),(w_shell + rad_core), 2)

        rs_s12, teta_s12 = np.meshgrid(discr_rs, self.discr_teta)

        phi_s1 = 2 * np.pi / 3

        xs_s1 = rs_s12 * np.cos(phi_s1) * np.sin(teta_s12)
        ys_s1 = rs_s12 * np.sin(phi_s1) * np.sin(teta_s12)
        zs_s1 = rs_s12 * np.cos(teta_s12)

        phi_s2 = 2 * np.pi

        xs_s2 = rs_s12 * np.cos(phi_s2) * np.sin(teta_s12)
        ys_s2 = rs_s12 * np.sin(phi_s2) * np.sin(teta_s12)
        zs_s2 = rs_s12 * np.cos(teta_s12)


        self.surf = self.ax.plot_surface(xc, yc, zc, alpha=0.1, color="b")

        self.surf = self.ax.plot_surface(xc_s1, yc_s1, zc_s1, alpha=1., color="b")
        self.surf = self.ax.plot_surface(xc_s2, yc_s2, zc_s2, alpha=1., color="b")

        self.surf = self.ax.plot_surface(xs_s1, ys_s1, zs_s1, alpha=1., color="r")
        self.surf = self.ax.plot_surface(xs_s2, ys_s2, zs_s2, alpha=1., color="r")

        self.surf = self.ax.plot_surface(xs, ys, zs, alpha=0.1, color="r")



        self.fig.canvas.draw()
        self.fig.show()


    def core_multi_shell(self):

        rad_core = self.param['radius']   # Core radius
        n_shells = int(self.param['n'])   # Number of shells
        w_shell = self.param['thickness1']   # Shell thickness

        self.fig = plt.figure('3d-model')
        self.ax = plt.axes(projection='3d')
        self.ax.set_xlabel('A')
        self.ax.set_ylabel('A')
        self.ax.set_zlabel('A')
        self.ax.set_title('core_multi_shell plot')
        self.ax.view_init(25, 45)


        xc = rad_core * np.cos(self.phi) * np.sin(self.teta)
        yc = rad_core * np.sin(self.phi) * np.sin(self.teta)
        zc = rad_core * np.cos(self.teta)

        discr_rc = np.linspace(0, rad_core, 2)

        rc_s12, teta_s12 = np.meshgrid(discr_rc, self.discr_teta)

        phi_s1 = 2 * np.pi / 3

        xc_s1 = rc_s12 * np.cos(phi_s1) * np.sin(teta_s12)
        yc_s1 = rc_s12 * np.sin(phi_s1) * np.sin(teta_s12)
        zc_s1 = rc_s12 * np.cos(teta_s12)

        phi_s2 = 2 * np.pi

        xc_s2 = rc_s12 * np.cos(phi_s2) * np.sin(teta_s12)
        yc_s2 = rc_s12 * np.sin(phi_s2) * np.sin(teta_s12)
        zc_s2 = rc_s12 * np.cos(teta_s12)


        for n in range(1, n_shells + 1):

            xs = (n * w_shell + rad_core) * np.cos(self.phi) * np.sin(self.teta)
            ys = (n * w_shell + rad_core) * np.sin(self.phi) * np.sin(self.teta)
            zs = (n * w_shell + rad_core) * np.cos(self.teta)

            discr_rs = np.linspace(((n - 1) * w_shell + rad_core),
                                        (n * w_shell + rad_core), 2)

            rs_s12, teta_s12 = np.meshgrid(discr_rs, self.discr_teta)

            phi_s1 = 2 * np.pi / 3

            xs_s1 = rs_s12 * np.cos(phi_s1) * np.sin(teta_s12)
            ys_s1 = rs_s12 * np.sin(phi_s1) * np.sin(teta_s12)
            zs_s1 = rs_s12 * np.cos(teta_s12)

            phi_s2 = 2 * np.pi

            xs_s2 = rs_s12 * np.cos(phi_s2) * np.sin(teta_s12)
            ys_s2 = rs_s12 * np.sin(phi_s2) * np.sin(teta_s12)
            zs_s2 = rs_s12 * np.cos(teta_s12)

            if (n % 2) == 0:

                self.surf = self.ax.plot_surface(xs_s1, ys_s1, zs_s1, alpha=1., color="b")
                self.surf = self.ax.plot_surface(xs_s2, ys_s2, zs_s2, alpha=1., color="b")

                self.surf = self.ax.plot_surface(xs, ys, zs, alpha=0.1, color="b")


            else:

                self.surf = self.ax.plot_surface(xs_s1, ys_s1, zs_s1, alpha=1., color="r")
                self.surf = self.ax.plot_surface(xs_s2, ys_s2, zs_s2, alpha=1., color="r")

                self.surf = self.ax.plot_surface(xs, ys, zs, alpha=0.1, color="r")



        self.surf = self.ax.plot_surface(xc_s1, yc_s1, zc_s1, alpha=1., color="white")
        self.surf = self.ax.plot_surface(xc_s2, yc_s2, zc_s2, alpha=1., color="white")

        self.surf = self.ax.plot_surface(xc, yc, zc, alpha=0.1, color="white")

        self.fig.canvas.draw()
        self.fig.show()

    def fuzzy_sphere(self):

        rad_core = self.param['radius']  # Core radius
        w_shell = self.param['fuzziness']  # Standard deviation of gaussian convolution (length) (shell thickness)

        self.fig = plt.figure('3d-model')
        self.ax = plt.axes(projection='3d')
        self.ax.set_xlabel('A')
        self.ax.set_ylabel('A')
        self.ax.set_zlabel('A')
        self.ax.set_title('fuzzy_sphere plot')
        self.ax.view_init(25, 45)

        xc = rad_core * np.cos(self.phi) * np.sin(self.teta)
        yc = rad_core * np.sin(self.phi) * np.sin(self.teta)
        zc = rad_core * np.cos(self.teta)

        discr_rc = np.linspace(0, rad_core, 2)

        rc_s12, teta_s12 = np.meshgrid(discr_rc, self.discr_teta)

        phi_s1 = 2 * np.pi / 3

        xc_s1 = rc_s12 * np.cos(phi_s1) * np.sin(teta_s12)
        yc_s1 = rc_s12 * np.sin(phi_s1) * np.sin(teta_s12)
        zc_s1 = rc_s12 * np.cos(teta_s12)

        phi_s2 = 2 * np.pi

        xc_s2 = rc_s12 * np.cos(phi_s2) * np.sin(teta_s12)
        yc_s2 = rc_s12 * np.sin(phi_s2) * np.sin(teta_s12)
        zc_s2 = rc_s12 * np.cos(teta_s12)

        xs = (w_shell + rad_core) * np.cos(self.phi) * np.sin(self.teta)
        ys = (w_shell + rad_core) * np.sin(self.phi) * np.sin(self.teta)
        zs = (w_shell + rad_core) * np.cos(self.teta)

        discr_rs = np.linspace((rad_core),
                                    (w_shell + rad_core), 2)

        rs_s12, teta_s12 = np.meshgrid(discr_rs, self.discr_teta)

        phi_s1 = 2 * np.pi / 3

        xs_s1 = rs_s12 * np.cos(phi_s1) * np.sin(teta_s12)
        ys_s1 = rs_s12 * np.sin(phi_s1) * np.sin(teta_s12)
        zs_s1 = rs_s12 * np.cos(teta_s12)

        phi_s2 = 2 * np.pi

        xs_s2 = rs_s12 * np.cos(phi_s2) * np.sin(teta_s12)
        ys_s2 = rs_s12 * np.sin(phi_s2) * np.sin(teta_s12)
        zs_s2 = rs_s12 * np.cos(teta_s12)

        self.surf = self.ax.plot_surface(xc, yc, zc, alpha=0.1, color="b")

        self.surf = self.ax.plot_surface(xc_s1, yc_s1, zc_s1, alpha=1., color="b")
        self.surf = self.ax.plot_surface(xc_s2, yc_s2, zc_s2, alpha=1., color="b")

        self.surf = self.ax.plot_surface(xs_s1, ys_s1, zs_s1, alpha=0.3, color="b")
        self.surf = self.ax.plot_surface(xs_s2, ys_s2, zs_s2, alpha=0.3, color="b")

        self.surf = self.ax.plot_surface(xs, ys, zs, alpha=0.3, color="b")

        self.fig.canvas.draw()
        self.fig.show()


    def linear_pearls(self):

        radius = self.param['radius']   # Pearl radius
        edge_sep = self.param['edge_sep']   # Length of the string segment (surface to surface)
        n_pearls = int(self.param['num_pearls'])   # Number of pearls


        self.fig = plt.figure('3d-model')
        self.ax = plt.axes(projection='3d')
        self.ax.set_xlabel('A')
        self.ax.set_ylabel('A')
        self.ax.set_zlabel('A')
        self.ax.set_title('linear_pearls plot')
        self.ax.view_init(20, 90)

        for n in range(0,n_pearls):

            x = radius * np.cos(self.phi) * np.sin(self.teta) + n * (edge_sep + 2 * radius)
            y = radius * np.sin(self.phi) * np.sin(self.teta)
            z = radius * np.cos(self.teta)

            if n <= n_pearls-2:
                xm = n * (edge_sep + 2 * radius)
                x1 = xm + radius
                x2 = xm + radius + edge_sep
                xlin = np.linspace(x1, x2, 2)
                self.surf = self.ax.plot3D(xlin, 0 * xlin, 0 * xlin, alpha=1, color="b")


            self.surf = self.ax.plot_surface(x, y, z, alpha=1, color="b")

        self.fig.canvas.draw()
        self.fig.show()


    def binary_hard_sphere(self):

        radius_lg = self.param['radius_lg']   # radius large sphere
        radius_sm = self.param['radius_sm']   # radius small sphere

        self.fig = plt.figure('3d-model')
        self.ax = plt.axes(projection='3d')
        self.ax.set_xlabel('A')
        self.ax.set_ylabel('A')
        self.ax.set_zlabel('A')
        self.ax.set_title('binary_hard_sphere plot')
        self.ax.view_init(20, 70)

        x_lg = radius_lg * np.cos(self.phi) * np.sin(self.teta)
        y_lg = radius_lg * np.sin(self.phi) * np.sin(self.teta)
        z_lg = radius_lg * np.cos(self.teta)

        x_sm = radius_sm * np.cos(self.phi) * np.sin(self.teta) + 3 * max(radius_lg,radius_sm)
        y_sm = radius_sm * np.sin(self.phi) * np.sin(self.teta)
        z_sm = radius_sm * np.cos(self.teta)

        self.surf = self.ax.plot_surface(x_lg, y_lg, z_lg, alpha=1, color="b")

        self.surf = self.ax.plot_surface(x_sm, y_sm, z_sm, alpha=1, color="r")


        self.fig.canvas.draw()
        self.fig.show()

    def adsorbed_layer(self):

        rad_core = self.param['radius']   # Core radius
        ad_am = self.param['adsorbed_amount']   # Adsorbed amount mg / m^2 in shell
        denshell = self.param['density_shell']   # Bulk density of polymer in the shell  g / cm^3

        w_shell = (ad_am * 10) / denshell  # shell thickness

        self.fig = plt.figure('3d-model')
        self.ax = plt.axes(projection='3d')
        self.ax.set_xlabel('A')
        self.ax.set_ylabel('A')
        self.ax.set_zlabel('A')
        self.ax.set_title('adsorbed_layer plot')
        self.ax.view_init(25, 45)

        xc = rad_core * np.cos(self.phi) * np.sin(self.teta)
        yc = rad_core * np.sin(self.phi) * np.sin(self.teta)
        zc = rad_core * np.cos(self.teta)

        discr_rc = np.linspace(0, rad_core, 2)

        rc_s12, teta_s12 = np.meshgrid(discr_rc, self.discr_teta)

        phi_s1 = 2 * np.pi / 3

        xc_s1 = rc_s12 * np.cos(phi_s1) * np.sin(teta_s12)
        yc_s1 = rc_s12 * np.sin(phi_s1) * np.sin(teta_s12)
        zc_s1 = rc_s12 * np.cos(teta_s12)

        phi_s2 = 2 * np.pi

        xc_s2 = rc_s12 * np.cos(phi_s2) * np.sin(teta_s12)
        yc_s2 = rc_s12 * np.sin(phi_s2) * np.sin(teta_s12)
        zc_s2 = rc_s12 * np.cos(teta_s12)

        xs = (w_shell + rad_core) * np.cos(self.phi) * np.sin(self.teta)
        ys = (w_shell + rad_core) * np.sin(self.phi) * np.sin(self.teta)
        zs = (w_shell + rad_core) * np.cos(self.teta)

        discr_rs = np.linspace(rad_core, (w_shell + rad_core), 2)

        rs_s12, teta_s12 = np.meshgrid(discr_rs, self.discr_teta)

        phi_s1 = 2 * np.pi / 3

        xs_s1 = rs_s12 * np.cos(phi_s1) * np.sin(teta_s12)
        ys_s1 = rs_s12 * np.sin(phi_s1) * np.sin(teta_s12)
        zs_s1 = rs_s12 * np.cos(teta_s12)

        phi_s2 = 2 * np.pi

        xs_s2 = rs_s12 * np.cos(phi_s2) * np.sin(teta_s12)
        ys_s2 = rs_s12 * np.sin(phi_s2) * np.sin(teta_s12)
        zs_s2 = rs_s12 * np.cos(teta_s12)

        self.surf = self.ax.plot_surface(xc, yc, zc, alpha=0.1, color="b")

        self.surf = self.ax.plot_surface(xc_s1, yc_s1, zc_s1, alpha=1., color="b")
        self.surf = self.ax.plot_surface(xc_s2, yc_s2, zc_s2, alpha=1., color="b")

        self.surf = self.ax.plot_surface(xs_s1, ys_s1, zs_s1, alpha=1., color="r")
        self.surf = self.ax.plot_surface(xs_s2, ys_s2, zs_s2, alpha=1., color="r")

        self.surf = self.ax.plot_surface(xs, ys, zs, alpha=0.1, color="r")

        self.fig.canvas.draw()
        self.fig.show()

    def ellipsoid(self):

        rad_po = self.param['radius_polar']  # Polar radius
        rad_eq = self.param['radius_equatorial']  # Equatorial radius

        self.fig = plt.figure('3d-model')
        self.ax = plt.axes(projection='3d')
        self.ax.set_xlabel('A')
        self.ax.set_ylabel('A')
        self.ax.set_zlabel('A')
        self.ax.set_title('ellipsoid plot')
        self.ax.view_init(25, 45)

        x = rad_eq * np.cos(self.phi) * np.sin(self.teta)
        y = rad_eq * np.sin(self.phi) * np.sin(self.teta)
        z = rad_po * np.cos(self.teta)
        self.surf = self.ax.plot_surface(x, y, z, alpha=1, color="b")

        self.fig.canvas.draw()
        self.fig.show()

    def core_shell_ellipsoid(self):

        rad_core_eq = self.param['radius_equat_core'] # Equatorial radius
        x_core = self.param['x_core'] # Axial ratio of core (polar / equatorial)
        w_shell_eq = self.param['thick_shell']  # Shell thickness at equator
        x_pol_rat = self.param['x_polar_shell']  # Ratio of thickness of shell at pole to that at equator

        rad_core_pol = rad_core_eq * x_core
        w_shell_pol = w_shell_eq * x_pol_rat


        self.fig = plt.figure('3d-model')
        self.ax = plt.axes(projection='3d')
        self.ax.set_xlabel('A')
        self.ax.set_ylabel('A')
        self.ax.set_zlabel('A')
        self.ax.set_title('core_shell_ellipsoid plot')
        self.ax.view_init(25, 45)

        xc = rad_core_eq * np.cos(self.phi) * np.sin(self.teta)
        yc = rad_core_eq * np.sin(self.phi) * np.sin(self.teta)
        zc = rad_core_pol * np.cos(self.teta)

        discr_rc_eq = np.linspace(0, rad_core_eq, 2)
        discr_rc_pol = np.linspace(0, rad_core_pol, 2)

        rc_s12_eq, teta_s12 = np.meshgrid(discr_rc_eq, self.discr_teta)
        rc_s12_pol, teta_s12 = np.meshgrid(discr_rc_pol, self.discr_teta)

        phi_s1 = 2 * np.pi / 3

        xc_s1 = rc_s12_eq * np.cos(phi_s1) * np.sin(teta_s12)
        yc_s1 = rc_s12_eq * np.sin(phi_s1) * np.sin(teta_s12)
        zc_s1 = rc_s12_pol * np.cos(teta_s12)

        phi_s2 = 2 * np.pi

        xc_s2 = rc_s12_eq * np.cos(phi_s2) * np.sin(teta_s12)
        yc_s2 = rc_s12_eq * np.sin(phi_s2) * np.sin(teta_s12)
        zc_s2 = rc_s12_pol * np.cos(teta_s12)


        xs = (w_shell_eq + rad_core_eq) * np.cos(self.phi) * np.sin(self.teta)
        ys = (w_shell_eq + rad_core_eq) * np.sin(self.phi) * np.sin(self.teta)
        zs = (w_shell_pol + rad_core_pol) * np.cos(self.teta)

        discr_rs_eq = np.linspace(rad_core_eq, (w_shell_eq + rad_core_eq), 2)
        discr_rs_pol = np.linspace(rad_core_pol, (w_shell_pol + rad_core_pol), 2)

        rs_s12_eq, teta_s12 = np.meshgrid(discr_rs_eq, self.discr_teta)
        rs_s12_pol, teta_s12 = np.meshgrid(discr_rs_pol, self.discr_teta)

        phi_s1 = 2 * np.pi / 3

        xs_s1 = rs_s12_eq * np.cos(phi_s1) * np.sin(teta_s12)
        ys_s1 = rs_s12_eq * np.sin(phi_s1) * np.sin(teta_s12)
        zs_s1 = rs_s12_pol * np.cos(teta_s12)

        phi_s2 = 2 * np.pi

        xs_s2 = rs_s12_eq * np.cos(phi_s2) * np.sin(teta_s12)
        ys_s2 = rs_s12_eq * np.sin(phi_s2) * np.sin(teta_s12)
        zs_s2 = rs_s12_pol * np.cos(teta_s12)

        self.surf = self.ax.plot_surface(xc, yc, zc, alpha=0.1, color="b")

        self.surf = self.ax.plot_surface(xc_s1, yc_s1, zc_s1, alpha=1., color="b")
        self.surf = self.ax.plot_surface(xc_s2, yc_s2, zc_s2, alpha=1., color="b")

        self.surf = self.ax.plot_surface(xs_s1, ys_s1, zs_s1, alpha=1., color="r")
        self.surf = self.ax.plot_surface(xs_s2, ys_s2, zs_s2, alpha=1., color="r")

        self.surf = self.ax.plot_surface(xs, ys, zs, alpha=0.1, color="r")


        self.fig.canvas.draw()
        self.fig.show()

    def triaxial_ellipsoid(self):

        rad_po = self.param['radius_polar']  # Polar radius
        rad_eq_min = self.param['radius_equat_minor']  # Equatorial radius minor axis
        rad_eq_maj = self.param['radius_equat_major']  # Equatorial radius major axis

        self.fig = plt.figure('3d-model')
        self.ax = plt.axes(projection='3d')
        self.ax.set_xlabel('A')
        self.ax.set_ylabel('A')
        self.ax.set_zlabel('A')
        self.ax.set_title('triaxial_ellipsoid plot')
        self.ax.view_init(25, 45)

        x = rad_eq_min * np.cos(self.phi) * np.sin(self.teta)
        y = rad_eq_maj * np.sin(self.phi) * np.sin(self.teta)
        z = rad_po * np.cos(self.teta)

        self.surf = self.ax.plot_surface(x, y, z, alpha=1, color="b")

        self.fig.canvas.draw()
        self.fig.show()


#def plotSurface(param, model3d):

#    fig = plt.figure('3d-model')
#    ax = plt.axes(projection = '3d')
#    surf = ax.plot_surface(model3d.x, model3d.y, model3d.z, color='b', antialiased=False, linewidth=0)
#    fig.show()