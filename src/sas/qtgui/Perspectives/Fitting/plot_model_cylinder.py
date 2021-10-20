from math import radians
import numpy as np
import matplotlib
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

class plotModelCylinder:

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

        if (self.param['model_name'] == 'cylinder' or self.param['model_name'] == 'pearl_necklace' or
            self.param['model_name'] == 'flexible_cylinder' or
            self.param['model_name'] == 'flexible_cylinder_elliptical' or
            self.param['model_name'] == 'elliptical_cylinder' or
            self.param['model_name'] == 'stacked_disks' or
            self.param['model_name'] == 'capped_cylinder' or
            self.param['model_name'] == 'barbell'):

            self.discr_phi = np.linspace(0., 2 * np.pi, 30)

        if self.param['model_name'] == 'pringle':

            # Discretization cylinder mantle

            self.discr_phi = np.linspace(0., 2 * np.pi, 60)

        if (self.param['model_name'] == 'hollow_cylinder' or self.param['model_name'] == 'core_shell_cylinder' or
            self.param['model_name'] == 'core_shell_bicelle' or
            self.param['model_name'] == 'core_shell_bicelle_elliptical' or
            self.param['model_name'] == 'core_shell_bicelle_elliptical_belt_rough'):

            # Discretization cylinder mantle (core and shell)
            self.discr_phi = np.linspace(0., -4 * np.pi / 3, 30)

        if self.param['model_name'] == 'pearl_necklace':

            # Discretization sphere

            self.discr_teta = np.linspace(0., np.pi, 30)

            self.phi, self.teta = np.meshgrid(self.discr_phi, self.discr_teta)

        if (self.param['model_name'] == 'flexible_cylinder' or\
           self.param['model_name'] == 'flexible_cylinder_elliptical'):

            self.wig = np.pi / 24  # Wiggle angle (radians) in xy_plane for plotting only

            # Define wiggle (rotation) matrix

            self.r11p = np.cos(self.wig) * np.ones((3, 30))
            self.r12p = -np.sin(self.wig) * np.ones((3, 30))
            self.r21p = np.sin(self.wig) * np.ones((3, 30))
            self.r22p = np.cos(self.wig) * np.ones((3, 30))

            self.r11m = np.cos(-self.wig) * np.ones((3, 30))
            self.r12m = -np.sin(-self.wig) * np.ones((3, 30))
            self.r21m = np.sin(-self.wig) * np.ones((3, 30))
            self.r22m = np.cos(-self.wig) * np.ones((3, 30))



        if self.param['model_name'] == 'cylinder':
            self.cylinder()

        if self.param['model_name'] == 'hollow_cylinder':
            self.hollow_cylinder()

        if self.param['model_name'] == 'pearl_necklace':
            self.pearl_necklace()

        if self.param['model_name'] == 'pringle':
            self.pringle()

        if self.param['model_name'] == 'flexible_cylinder':
            self.flexible_cylinder()

        if self.param['model_name'] == 'flexible_cylinder_elliptical':
            self.flexible_cylinder_elliptical()

        if self.param['model_name'] == 'elliptical_cylinder':
            self.elliptical_cylinder()

        if self.param['model_name'] == 'stacked_disks':
            self.stacked_disks()

        if self.param['model_name'] == 'core_shell_cylinder':
            self.core_shell_cylinder()

        if self.param['model_name'] == 'capped_cylinder':
            self.capped_cylinder()

        if self.param['model_name'] == 'core_shell_bicelle':
            self.core_shell_bicelle()

        if self.param['model_name'] == 'core_shell_bicelle_elliptical':
            self.core_shell_bicelle_elliptical()

        if self.param['model_name'] == 'core_shell_bicelle_elliptical_belt_rough':
            self.core_shell_bicelle_elliptical_belt_rough()

        if self.param['model_name'] == 'barbell':
            self.barbell()



    def cylinder(self):

        radius = self.param['radius']   # cylinder radius
        length = self.param['length']   # cylinder length

        self.fig = plt.figure('3d-model')
        self.ax = plt.axes(projection='3d')
        self.ax.set_xlabel('A')
        self.ax.set_ylabel('A')
        self.ax.set_zlabel('A')
        self.ax.set_title('cylinder plot')
        self.ax.view_init(25, 45)


        # Discretization cylinder mantle

        discr_z = np.linspace(0., length, 2)

        # Discretization cylinder top and bottom

        discr_r = np.linspace(0., radius, 30)

        # Mantle

        phi, z = np.meshgrid(self.discr_phi, discr_z)

        x = radius * np.cos(phi)
        y = radius * np.sin(phi)

        # top and bottom

        phi, r = np.meshgrid(self.discr_phi, discr_r)

        xtb = r * np.cos(phi)
        ytb = r * np.sin(phi)

        zt = length * np.ones((30, 30))
        zb = 0. * np.ones((30, 30))

        self.surf = self.ax.plot_surface(x, y, z, alpha=1, color="b")
        self.surf = self.ax.plot_surface(xtb, ytb, zt, alpha=1, color="b")
        self.surf = self.ax.plot_surface(xtb, ytb, zb, alpha=1, color="b")

        self.fig.canvas.draw()
        self.fig.show()

    def hollow_cylinder(self):

        radius = self.param['radius']   # cylinder core radius
        tot_length = self.param['length']   # cylinder total length
        w_shell = self.param['thickness']   # shell thickness

        self.fig = plt.figure('3d-model')
        self.ax = plt.axes(projection='3d')
        self.ax.set_xlabel('A')
        self.ax.set_ylabel('A')
        self.ax.set_zlabel('A')
        self.ax.set_title('hollow_cylinder plot')
        self.ax.view_init(25, 45)

        length = tot_length - 2 * w_shell


        discr_zc = np.linspace(0., length, 2)
        discr_zs = np.linspace(-w_shell, length + w_shell, 2)

        # Discretization cylinder top and bottom (core and shell)

        discr_rc = np.linspace(0., radius, 30)
        discr_rs = np.linspace(0., radius + w_shell, 30)

        # Discretization of shell thickness in r and z directions

        discr_rsm_s12 = np.linspace(radius, radius + w_shell, 2)

        discr_zst_s12 = np.linspace(length, length + w_shell, 2)
        discr_zsb_s12 = np.linspace(-w_shell, 0., 2)

        # Mantle

        phi, z_c = np.meshgrid(self.discr_phi, discr_zc)

        phi, z_s = np.meshgrid(self.discr_phi, discr_zs)

        x_c = radius * np.cos(phi)
        y_c = radius * np.sin(phi)

        x_s = (radius + w_shell) * np.cos(phi)
        y_s = (radius + w_shell) * np.sin(phi)

        # top and bottom

        phi, r_c = np.meshgrid(self.discr_phi, discr_rc)

        xtb_c = r_c * np.cos(phi)
        ytb_c = r_c * np.sin(phi)

        zt_c = length * np.ones((30, 30))
        zb_c = 0. * np.ones((30, 30))

        phi, r_s = np.meshgrid(self.discr_phi, discr_rs)

        xtb_s = r_s * np.cos(phi)
        ytb_s = r_s * np.sin(phi)

        zt_s = (length + w_shell) * np.ones((30, 30))
        zb_s = -w_shell * np.ones((30, 30))

        # Shell

        # Mantle

        rsm, zsm_s12 = np.meshgrid(discr_rsm_s12, discr_zs)

        phi = 0.

        xsm_s1 = rsm * np.cos(phi)
        ysm_s1 = rsm * np.sin(phi)

        phi = -4 * np.pi / 3

        xsm_s2 = rsm * np.cos(phi)
        ysm_s2 = rsm * np.sin(phi)

        # Top bottom

        rst_s12, zst = np.meshgrid(discr_rc, discr_zst_s12)

        phi = 0.

        xst_s1 = rst_s12 * np.cos(phi)
        yst_s1 = rst_s12 * np.sin(phi)

        phi = -4 * np.pi / 3

        xst_s2 = rst_s12 * np.cos(phi)
        yst_s2 = rst_s12 * np.sin(phi)

        rsb_s12, zsb = np.meshgrid(discr_rc, discr_zsb_s12)

        phi = 0.

        xsb_s1 = rsb_s12 * np.cos(phi)
        ysb_s1 = rsb_s12 * np.sin(phi)

        phi = -4 * np.pi / 3

        xsb_s2 = rsb_s12 * np.cos(phi)
        ysb_s2 = rsb_s12 * np.sin(phi)

        self.surf = self.ax.plot_surface(x_c, y_c, z_c, alpha=0.1, color="b")
        self.surf = self.ax.plot_surface(xtb_c, ytb_c, zt_c, alpha=0.1, color="b")
        self.surf = self.ax.plot_surface(xtb_c, ytb_c, zb_c, alpha=0.1, color="b")

        self.surf = self.ax.plot_surface(x_s, y_s, z_s, alpha=0.1, color="r")
        self.surf = self.ax.plot_surface(xtb_s, ytb_s, zt_s, alpha=0.1, color="r")
        self.surf = self.ax.plot_surface(xtb_s, ytb_s, zb_s, alpha=0.1, color="r")

        self.surf = self.ax.plot_surface(xsm_s1, ysm_s1, zsm_s12, alpha=1, color="r")
        self.surf = self.ax.plot_surface(xsm_s2, ysm_s2, zsm_s12, alpha=1, color="r")

        self.surf = self.ax.plot_surface(xst_s1, yst_s1, zst, alpha=1, color="r")
        self.surf = self.ax.plot_surface(xst_s2, yst_s2, zst, alpha=1, color="r")

        self.surf = self.ax.plot_surface(xsb_s1, ysb_s1, zsb, alpha=1, color="r")
        self.surf = self.ax.plot_surface(xsb_s2, ysb_s2, zsb, alpha=1, color="r")


        self.fig.canvas.draw()
        self.fig.show()


    def pearl_necklace(self):

        radius = self.param['radius']   # Pearl radius
        edge_sep = self.param['edge_sep']   # Length of the string segment (surface to surface)
        n_pearls = int(self.param['num_pearls'])   # Number of pearls
        w_string = self.param['thick_string']   # Thickness of string segment

        self.fig = plt.figure('3d-model')
        self.ax = plt.axes(projection='3d')
        self.ax.set_xlabel('A')
        self.ax.set_ylabel('A')
        self.ax.set_zlabel('A')
        self.ax.set_title('pearl_necklace plot')
        self.ax.view_init(20, 90)

        # Discretization rod

        discr_xr = np.linspace(0, edge_sep, 3)

        phirm, xrm = np.meshgrid(self.discr_phi, discr_xr)

        for n in range(0,n_pearls):

            x = radius * np.cos(self.phi) * np.sin(self.teta) + n * (edge_sep + 2 * radius)
            y = radius * np.sin(self.phi) * np.sin(self.teta)
            z = radius * np.cos(self.teta)

            if n <= n_pearls-2:

                zr = w_string * np.cos(phirm)
                yr = w_string * np.sin(phirm)
                xr = xrm + n * (edge_sep + 2 * radius) * np.ones((3,30)) + radius * np.ones((3,30))

                self.surf = self.ax.plot_surface(xr, yr, zr, alpha=1, color="b")

            self.surf = self.ax.plot_surface(x, y, z, alpha=1, color="b")


        self.fig.canvas.draw()
        self.fig.show()


    def pringle(self):

        radius = self.param['radius']  # radius of pringle
        length = self.param['thickness']  # thickness of pringle
        alpha = self.param['alpha']  # curvature parameter alpha
        beta = self.param['beta']  # cuvature parameter beta

        self.fig = plt.figure('3d-model')
        self.ax = plt.axes(projection='3d')
        self.ax.set_xlabel('A')
        self.ax.set_ylabel('A')
        self.ax.set_zlabel('A')
        self.ax.set_title('pringle plot')
        self.ax.view_init(25, 45)


        discr_z = np.linspace(0., length, 60)

        # Discretization cylinder top and bottom

        discr_r = np.linspace(0., radius, 60)

        # Mantle

        phi, z = np.meshgrid(self.discr_phi, discr_z)

        x = radius * np.cos(phi)
        y = radius * np.sin(phi)
        z = radius ** 2 * (alpha * np.cos(phi) ** 2 - beta * np.sin(phi) ** 2) + z

        # top and bottom

        phi, r = np.meshgrid(self.discr_phi, discr_r)

        xtb = r * np.cos(phi)
        ytb = r * np.sin(phi)

        zt = radius ** 2 * (alpha * np.cos(phi) ** 2 - beta * np.sin(phi) ** 2) \
             + length * np.ones((60, 60))
        zb = radius ** 2 * (alpha * np.cos(phi) ** 2 - beta * np.sin(phi) ** 2)

        self.surf = self.ax.plot_surface(x, y, z, alpha=1, color="r")
        self.surf = self.ax.plot_surface(xtb, ytb, zt, alpha=0.3, color="r")
        self.surf = self.ax.plot_surface(xtb, ytb, zb, alpha=0.3, color="b")

        self.fig.canvas.draw()
        self.fig.show()

    def flexible_cylinder(self):

        k_length = self.param['kuhn_length']  # Kuhn length of the cylinder segment
        length = self.param['length']  # Total length of flexible cylinder
        radius = self.param['radius']  # Radius of flexible cylinder

        self.fig = plt.figure('3d-model')
        self.ax = plt.axes(projection='3d')
        self.ax.set_xlabel('A')
        self.ax.set_ylabel('A')
        self.ax.set_zlabel('A')
        self.ax.set_title('flexible_cylinder plot')
        self.ax.view_init(20, 90)

        n_cyl = int(length / k_length)

        # Discretization cylinder

        discr_xr = np.linspace(0, k_length, 3)

        phirm, xrm = np.meshgrid(self.discr_phi, discr_xr)


        zr = radius * np.cos(phirm)
        yr = radius * np.sin(phirm)
        xr = xrm

        # Segment Rotated up

        xr_rp = xr * self.r11p + yr * self.r12p
        yr_rp = xr * self.r12p + yr * self.r22p

        # Segment Rotated down

        xr_rm = xr * self.r11m + yr * self.r12m
        yr_rm = xr * self.r12m + yr * self.r22m

        for n in range(0, 10):

            if (n % 2) == 0:
                x = xr_rp + n * k_length * np.cos(self.wig) * np.ones((3, 30))
                y = yr_rp
                z = zr

                self.surf = self.ax.plot_surface(x, y, z, alpha=0.5, color="b")
            else:
                x = xr_rm + n * k_length * np.cos(-self.wig) * np.ones((3, 30))
                y = yr_rm + k_length * np.sin(-self.wig) * np.ones((3, 30))
                z = zr

                self.surf = self.ax.plot_surface(x, y, z, alpha=0.5, color="r")

        self.fig.canvas.draw()
        self.fig.show()

    def flexible_cylinder_elliptical(self):

        k_length = self.param['kuhn_length']  # Kuhn length of the cylinder segment
        length = self.param['length']  # Total length of flexible cylinder
        radius = self.param['radius']  # Radius of flexible cylinder
        ax_rat = self.param['axis_ratio']  # Ratio major/ minor radius of cylinder

        self.fig = plt.figure('3d-model')
        self.ax = plt.axes(projection='3d')
        self.ax.set_xlabel('A')
        self.ax.set_ylabel('A')
        self.ax.set_zlabel('A')
        self.ax.set_title('flexible_cylinder_elliptical plot')
        self.ax.view_init(20, 90)

        n_cyl = int(length / k_length)

        discr_xr = np.linspace(0, k_length, 3)

        phirm, xrm = np.meshgrid(self.discr_phi, discr_xr)

        zr = ax_rat * radius * np.cos(phirm)
        yr = radius * np.sin(phirm)
        xr = xrm

        # Segment Rotated up

        xr_rp = xr * self.r11p + yr * self.r12p
        yr_rp = xr * self.r12p + yr * self.r22p

        # Segment Rotated down

        xr_rm = xr * self.r11m + yr * self.r12m
        yr_rm = xr * self.r12m + yr * self.r22m

        for n in range(0, 10):

            if (n % 2) == 0:
                x = xr_rp + n * k_length * np.cos(self.wig) * np.ones((3, 30))
                y = yr_rp
                z = zr

                self.surf = self.ax.plot_surface(x, y, z, alpha=0.5, color="b")
            else:
                x = xr_rm + n * k_length * np.cos(-self.wig) * np.ones((3, 30))
                y = yr_rm + k_length * np.sin(-self.wig) * np.ones((3, 30))
                z = zr

                self.surf = self.ax.plot_surface(x, y, z, alpha=0.5, color="r")

        self.fig.canvas.draw()
        self.fig.show()

    def elliptical_cylinder(self):

        radius = self.param['radius_minor']     # Cylinder radius
        length = self.param['length']          # Cylinder length
        ax_rat = self.param['axis_ratio']     # Cylinder axis ratio (major/minor)

        self.fig = plt.figure('3d-model')
        self.ax = plt.axes(projection='3d')
        self.ax.set_xlabel('A')
        self.ax.set_ylabel('A')
        self.ax.set_zlabel('A')
        self.ax.set_title('elliptical_cylinder plot')
        self.ax.view_init(25, 45)

        # Discretization cylinder mantle

        discr_z = np.linspace(0., length, 2)

        # Discretization cylinder top and bottom

        discr_r = np.linspace(0., radius, 30)


        # Mantle

        phi, z = np.meshgrid(self.discr_phi, discr_z)

        x = ax_rat * radius * np.cos(phi)
        y = radius * np.sin(phi)

        # top and bottom

        phi, r = np.meshgrid(self.discr_phi, discr_r)

        xtb = ax_rat * r * np.cos(phi)
        ytb = r * np.sin(phi)

        zt = length * np.ones((30, 30))
        zb = 0. * np.ones((30, 30))

        self.surf = self.ax.plot_surface(x, y, z, alpha=1, color="b")
        self.surf = self.ax.plot_surface(xtb, ytb, zt, alpha=1, color="b")
        self.surf = self.ax.plot_surface(xtb, ytb, zb, alpha=1, color="b")


        self.fig.canvas.draw()
        self.fig.show()


    def stacked_disks(self):

        radius = self.param['radius']  # Radius of discs
        w_core = self.param['thick_core']  # Thickness of core disc
        w_lay = self.param['thick_layer']  # Thickness of the 2 layers around each core disc
        n_lay = int(self.param['n_stacking'])  # Number of stacked layer/core/layer discs

        self.fig = plt.figure('3d-model')
        self.ax = plt.axes(projection='3d')
        self.ax.set_xlabel('A')
        self.ax.set_ylabel('A')
        self.ax.set_zlabel('A')
        self.ax.set_title('stacked_disks plot')
        self.ax.view_init(25, 45)

        # Discretization cylinder mantle

        discr_zc = np.linspace(0., w_core, 2)
        discr_zl = np.linspace(0., w_lay, 2)

        # Discretization cylinder top and bottom

        discr_r = np.linspace(0., radius, 30)

        phi_c, zc = np.meshgrid(self.discr_phi, discr_zc)

        phi_l, zl = np.meshgrid(self.discr_phi, discr_zl)

        phi, r = np.meshgrid(self.discr_phi, discr_r)

        # Mantle

        xc = radius * np.cos(phi_c)
        yc = radius * np.sin(phi_c)

        xl = radius * np.cos(phi_l)
        yl = radius * np.sin(phi_l)

        # top and bottom

        xtb = r * np.cos(phi)
        ytb = r * np.sin(phi)

        for n in range(0, n_lay):
            # Layer

            zt_l1 = (w_lay + n * (2 * w_lay + w_core)) * np.ones((30, 30))
            zb_l1 = (n * (2 * w_lay + w_core)) * np.ones((30, 30))

            zm_l1 = zl + n * (2 * w_lay + w_core) * np.ones((2, 30))

            # Core

            zt_c = (w_lay + w_core + n * (2 * w_lay + w_core)) * np.ones((30, 30))
            zb_c = (w_lay + n * (2 * w_lay + w_core)) * np.ones((30, 30))

            zm_c = zc + (w_lay + n * (2 * w_lay + w_core)) * np.ones((2, 30))

            # Layer

            zt_l2 = (2 * w_lay + w_core + n * (2 * w_lay + w_core)) * np.ones((30, 30))
            zb_l2 = (w_lay + w_core + n * (2 * w_lay + w_core)) * np.ones((30, 30))

            zm_l2 = zl + (w_lay + w_core + n * (2 * w_lay + w_core)) * np.ones((2, 30))

            self.surf = self.ax.plot_surface(xl, yl, zm_l1, alpha=0.5, color="b")
            self.surf = self.ax.plot_surface(xtb, ytb, zt_l1, alpha=0.5, color="b")
            self.surf = self.ax.plot_surface(xtb, ytb, zb_l1, alpha=0.5, color="b")

            self.surf = self.ax.plot_surface(xc, yc, zm_c, alpha=1, color="r")
            self.surf = self.ax.plot_surface(xtb, ytb, zt_c, alpha=1, color="r")
            self.surf = self.ax.plot_surface(xtb, ytb, zb_c, alpha=1, color="r")

            self.surf = self.ax.plot_surface(xl, yl, zm_l2, alpha=1, color="b")
            self.surf = self.ax.plot_surface(xtb, ytb, zt_l2, alpha=1, color="b")
            self.surf = self.ax.plot_surface(xtb, ytb, zb_l2, alpha=1, color="b")


        self.fig.canvas.draw()
        self.fig.show()


    def core_shell_cylinder(self):

        radius = self.param['radius']  # core radius
        length = self.param['length']  # cylinder total length
        w_shell = self.param['thickness']  # shell thickness


        self.fig = plt.figure('3d-model')
        self.ax = plt.axes(projection='3d')
        self.ax.set_xlabel('A')
        self.ax.set_ylabel('A')
        self.ax.set_zlabel('A')
        self.ax.set_title('stacked_disks plot')
        self.ax.view_init(25, 45)


        # Discretization cylinder mantle (core and shell)

        discr_zc = np.linspace(0., length, 2)
        discr_zs = np.linspace(-w_shell, length + w_shell, 2)

        # Discretization cylinder top and bottom (core and shell)

        discr_rc = np.linspace(0., radius, 30)
        discr_rs = np.linspace(0., radius + w_shell, 30)

        # Discretization of shell thickness in r and z directions

        discr_rsm_s12 = np.linspace(radius, radius + w_shell, 2)

        discr_zst_s12 = np.linspace(length, length + w_shell, 2)
        discr_zsb_s12 = np.linspace(-w_shell, 0., 2)

        # Mantle

        phi, z_c = np.meshgrid(self.discr_phi, discr_zc)

        phi, z_s = np.meshgrid(self.discr_phi, discr_zs)

        x_c = radius * np.cos(phi)
        y_c = radius * np.sin(phi)

        x_s = (radius + w_shell) * np.cos(phi)
        y_s = (radius + w_shell) * np.sin(phi)

        # top and bottom

        phi, r_c = np.meshgrid(self.discr_phi, discr_rc)

        xtb_c = r_c * np.cos(phi)
        ytb_c = r_c * np.sin(phi)

        zt_c = length * np.ones((30, 30))
        zb_c = 0. * np.ones((30, 30))

        phi, r_s = np.meshgrid(self.discr_phi, discr_rs)

        xtb_s = r_s * np.cos(phi)
        ytb_s = r_s * np.sin(phi)

        zt_s = (length + w_shell) * np.ones((30, 30))
        zb_s = -w_shell * np.ones((30, 30))

        # Shell

        # Mantle

        rsm, zsm_s12 = np.meshgrid(discr_rsm_s12, discr_zs)

        rcm, zcm_s12 = np.meshgrid(discr_rc, discr_zc)

        phi = 0.

        xsm_s1 = rsm * np.cos(phi)
        ysm_s1 = rsm * np.sin(phi)

        xcm_s1 = rcm * np.cos(phi)
        ycm_s1 = rcm * np.sin(phi)

        phi = -4 * np.pi / 3

        xsm_s2 = rsm * np.cos(phi)
        ysm_s2 = rsm * np.sin(phi)

        xcm_s2 = rcm * np.cos(phi)
        ycm_s2 = rcm * np.sin(phi)

        # Top bottom

        rst_s12, zst = np.meshgrid(discr_rc, discr_zst_s12)

        phi = 0.

        xst_s1 = rst_s12 * np.cos(phi)
        yst_s1 = rst_s12 * np.sin(phi)

        phi = -4 * np.pi / 3

        xst_s2 = rst_s12 * np.cos(phi)
        yst_s2 = rst_s12 * np.sin(phi)

        rsb_s12, zsb = np.meshgrid(discr_rc, discr_zsb_s12)

        phi = 0.

        xsb_s1 = rsb_s12 * np.cos(phi)
        ysb_s1 = rsb_s12 * np.sin(phi)

        phi = -4 * np.pi / 3

        xsb_s2 = rsb_s12 * np.cos(phi)
        ysb_s2 = rsb_s12 * np.sin(phi)

        self.surf = self.ax.plot_surface(x_c, y_c, z_c, alpha=0.2, color="b")
        self.surf = self.ax.plot_surface(xtb_c, ytb_c, zt_c, alpha=0.2, color="b")
        self.surf = self.ax.plot_surface(xtb_c, ytb_c, zb_c, alpha=0.2, color="b")

        self.surf = self.ax.plot_surface(x_s, y_s, z_s, alpha=0.2, color="r")
        self.surf = self.ax.plot_surface(xtb_s, ytb_s, zt_s, alpha=0.2, color="r")
        self.surf = self.ax.plot_surface(xtb_s, ytb_s, zb_s, alpha=0.2, color="r")

        self.surf = self.ax.plot_surface(xsm_s1, ysm_s1, zsm_s12, alpha=1, color="r")
        self.surf = self.ax.plot_surface(xsm_s2, ysm_s2, zsm_s12, alpha=1, color="r")

        self.surf = self.ax.plot_surface(xcm_s1, ycm_s1, zcm_s12, alpha=1, color="b")
        self.surf = self.ax.plot_surface(xcm_s2, ycm_s2, zcm_s12, alpha=1, color="b")

        self.surf = self.ax.plot_surface(xst_s1, yst_s1, zst, alpha=1, color="r")
        self.surf = self.ax.plot_surface(xst_s2, yst_s2, zst, alpha=1, color="r")

        self.surf = self.ax.plot_surface(xsb_s1, ysb_s1, zsb, alpha=1, color="r")
        self.surf = self.ax.plot_surface(xsb_s2, ysb_s2, zsb, alpha=1, color="r")



        self.fig.canvas.draw()
        self.fig.show()


    def capped_cylinder(self):

        radius = self.param['radius']  # radius cylinder
        length = self.param['length']  # length of cylinder
        radius_cap = self.param['radius_cap']  # Radius of cap   > radius

        self.fig = plt.figure('3d-model')
        self.ax = plt.axes(projection='3d')
        self.ax.set_xlabel('A')
        self.ax.set_ylabel('A')
        self.ax.set_zlabel('A')
        self.ax.set_title('capped_cylinder plot')
        self.ax.view_init(25, 45)

        h_in = np.sqrt(radius_cap ** 2 - radius ** 2)

        # top

        teta_t = np.arcsin(radius / radius_cap)

        discr_tetat = np.linspace(0., teta_t, 30)

        # bottom

        teta_b = np.pi - teta_t

        discr_tetab = np.linspace(teta_b, np.pi, 30)

        # Discretization cylinder mantle

        discr_z = np.linspace(0., length, 2)

        # Caps

        phi, teta_top = np.meshgrid(self.discr_phi, discr_tetat)

        x_top = radius_cap * np.cos(phi) * np.sin(teta_top)
        y_top = radius_cap * np.sin(phi) * np.sin(teta_top)
        z_top = radius_cap * np.cos(teta_top) + (length - h_in) * np.ones((30, 30))

        phi, teta_bot = np.meshgrid(self.discr_phi, discr_tetab)

        x_bot = radius_cap * np.cos(phi) * np.sin(teta_bot)
        y_bot = radius_cap * np.sin(phi) * np.sin(teta_bot)
        z_bot = radius_cap * np.cos(teta_bot) + h_in * np.ones((30, 30))

        # Mantle

        phi, z = np.meshgrid(self.discr_phi, discr_z)

        x = radius * np.cos(phi)
        y = radius * np.sin(phi)

        self.surf = self.ax.plot_surface(x, y, z, alpha=1, color="b")

        self.surf = self.ax.plot_surface(x_top, y_top, z_top, alpha=1, color="b")
        self.surf = self.ax.plot_surface(x_bot, y_bot, z_bot, alpha=1, color="b")


        self.fig.canvas.draw()
        self.fig.show()

    def core_shell_bicelle(self):

        radius = self.param['radius']  # core radius
        length = self.param['length']  # cylinder total length
        w_face = self.param['thick_face']  # shell thickness face
        w_rim = self.param['thick_rim']  # shell thickness rim

        self.fig = plt.figure('3d-model')
        self.ax = plt.axes(projection='3d')
        self.ax.set_xlabel('A')
        self.ax.set_ylabel('A')
        self.ax.set_zlabel('A')
        self.ax.set_title('core_shell_bicelle plot')
        self.ax.view_init(25, 45)


        # Discretization cylinder mantle (core and shell)

        discr_zc = np.linspace(0., length, 2)
        discr_zs = np.linspace(-w_face, length + w_face, 2)

        # Discretization cylinder top and bottom (core and shell)

        discr_rc = np.linspace(0., radius, 30)
        discr_rs = np.linspace(0., radius + w_rim, 30)

        # Discretization of shell thickness in r and z directions

        discr_rsm_s12 = np.linspace(radius, radius + w_rim, 2)

        discr_zst_s12 = np.linspace(length, length + w_face, 2)
        discr_zsb_s12 = np.linspace(-w_face, 0., 2)

        phi, z_c = np.meshgrid(self.discr_phi, discr_zc)

        phi, z_s = np.meshgrid(self.discr_phi, discr_zs)

        x_c = radius * np.cos(phi)
        y_c = radius * np.sin(phi)

        x_s = (radius + w_rim) * np.cos(phi)
        y_s = (radius + w_rim) * np.sin(phi)

        # top and bottom

        phi, r_c = np.meshgrid(self.discr_phi, discr_rc)

        xtb_c = r_c * np.cos(phi)
        ytb_c = r_c * np.sin(phi)

        zt_c = length * np.ones((30, 30))
        zb_c = 0. * np.ones((30, 30))

        phi, r_s = np.meshgrid(self.discr_phi, discr_rs)

        xtb_s = r_s * np.cos(phi)
        ytb_s = r_s * np.sin(phi)

        zt_s = (length + w_face) * np.ones((30, 30))
        zb_s = -w_face * np.ones((30, 30))

        # Shell

        # Mantle

        rsm, zsm_s12 = np.meshgrid(discr_rsm_s12, discr_zs)

        rcm, zcm_s12 = np.meshgrid(discr_rc, discr_zc)

        phi = 0.

        xsm_s1 = rsm * np.cos(phi)
        ysm_s1 = rsm * np.sin(phi)

        xcm_s1 = rcm * np.cos(phi)
        ycm_s1 = rcm * np.sin(phi)

        phi = -4 * np.pi / 3

        xsm_s2 = rsm * np.cos(phi)
        ysm_s2 = rsm * np.sin(phi)

        xcm_s2 = rcm * np.cos(phi)
        ycm_s2 = rcm * np.sin(phi)

        # Top bottom

        rst_s12, zst = np.meshgrid(discr_rc, discr_zst_s12)

        phi = 0.

        xst_s1 = rst_s12 * np.cos(phi)
        yst_s1 = rst_s12 * np.sin(phi)

        phi = -4 * np.pi / 3

        xst_s2 = rst_s12 * np.cos(phi)
        yst_s2 = rst_s12 * np.sin(phi)

        rsb_s12, zsb = np.meshgrid(discr_rc, discr_zsb_s12)

        phi = 0.

        xsb_s1 = rsb_s12 * np.cos(phi)
        ysb_s1 = rsb_s12 * np.sin(phi)

        phi = -4 * np.pi / 3

        xsb_s2 = rsb_s12 * np.cos(phi)
        ysb_s2 = rsb_s12 * np.sin(phi)


        self.surf = self.ax.plot_surface(x_c, y_c, z_c, alpha=0.2, color="b")
        self.surf = self.ax.plot_surface(xtb_c, ytb_c, zt_c, alpha=0.2, color="b")
        self.surf = self.ax.plot_surface(xtb_c, ytb_c, zb_c, alpha=0.2, color="b")

        self.surf = self.ax.plot_surface(x_s, y_s, z_s, alpha=0.2, color="r")
        self.surf = self.ax.plot_surface(xtb_s, ytb_s, zt_s, alpha=0.2, color="r")
        self.surf = self.ax.plot_surface(xtb_s, ytb_s, zb_s, alpha=0.2, color="r")

        self.surf = self.ax.plot_surface(xsm_s1, ysm_s1, zsm_s12, alpha=1, color="r")
        self.surf = self.ax.plot_surface(xsm_s2, ysm_s2, zsm_s12, alpha=1, color="r")

        self.surf = self.ax.plot_surface(xcm_s1, ycm_s1, zcm_s12, alpha=1, color="b")
        self.surf = self.ax.plot_surface(xcm_s2, ycm_s2, zcm_s12, alpha=1, color="b")

        self.surf = self.ax.plot_surface(xst_s1, yst_s1, zst, alpha=1, color="r")
        self.surf = self.ax.plot_surface(xst_s2, yst_s2, zst, alpha=1, color="r")

        self.surf = self.ax.plot_surface(xsb_s1, ysb_s1, zsb, alpha=1, color="r")
        self.surf = self.ax.plot_surface(xsb_s2, ysb_s2, zsb, alpha=1, color="r")


        self.fig.canvas.draw()
        self.fig.show()

    def core_shell_bicelle_elliptical(self):

        radius = self.param['radius']  # core radius
        length = self.param['length']  # cylinder total length
        x_core = self.param['x_core']  # Axial ratio of core (Major/Minor)
        w_face = self.param['thick_face']  # shell thickness face
        w_rim = self.param['thick_rim']  # shell thickness rim


        self.fig = plt.figure('3d-model')
        self.ax = plt.axes(projection='3d')
        self.ax.set_xlabel('A')
        self.ax.set_ylabel('A')
        self.ax.set_zlabel('A')
        self.ax.set_title('core_shell_bicelle_elliptical plot')
        self.ax.view_init(25, 45)


        # Discretization cylinder mantle (core and shell)

        discr_zc = np.linspace(0., length, 2)
        discr_zs = np.linspace(-w_face, length + w_face, 2)

        # Discretization cylinder top and bottom (core and shell)

        discr_rc_min = np.linspace(0., radius, 30)
        discr_rc_max = np.linspace(0., x_core * radius, 30)

        discr_rs_min = np.linspace(0., radius + w_rim, 30)
        discr_rs_max = np.linspace(0., x_core * radius + w_rim, 30)

        # Discretization of shell thickness in r and z directions

        discr_rsm_s12_min = np.linspace(radius, radius + w_rim, 2)
        discr_rsm_s12_max = np.linspace(x_core * radius, x_core * radius + w_rim, 2)

        discr_zst_s12 = np.linspace(length, length + w_face, 2)
        discr_zsb_s12 = np.linspace(-w_face, 0., 2)

        # Mantle

        phi, z_c = np.meshgrid(self.discr_phi, discr_zc)

        phi, z_s = np.meshgrid(self.discr_phi, discr_zs)

        x_c = x_core * radius * np.cos(phi)
        y_c = radius * np.sin(phi)

        x_s = (x_core * radius + w_rim) * np.cos(phi)
        y_s = (radius + w_rim) * np.sin(phi)

        # top and bottom

        phi, r_c_min = np.meshgrid(self.discr_phi, discr_rc_min)
        phi, r_c_max = np.meshgrid(self.discr_phi, discr_rc_max)

        xtb_c = r_c_max * np.cos(phi)
        ytb_c = r_c_min * np.sin(phi)

        zt_c = length * np.ones((30, 30))
        zb_c = 0. * np.ones((30, 30))

        phi, r_s_min = np.meshgrid(self.discr_phi, discr_rs_min)
        phi, r_s_max = np.meshgrid(self.discr_phi, discr_rs_max)

        xtb_s = r_s_max * np.cos(phi)
        ytb_s = r_s_min * np.sin(phi)

        zt_s = (length + w_face) * np.ones((30, 30))
        zb_s = -w_face * np.ones((30, 30))

        # Shell

        # Mantle

        rsm_min, zsm_s12 = np.meshgrid(discr_rsm_s12_min, discr_zs)
        rsm_max, zsm_s12 = np.meshgrid(discr_rsm_s12_max, discr_zs)

        rcm_min, zcm_s12 = np.meshgrid(discr_rc_min, discr_zc)
        rcm_max, zcm_s12 = np.meshgrid(discr_rc_max, discr_zc)

        phi = 0.

        xsm_s1 = rsm_max * np.cos(phi)
        ysm_s1 = rsm_min * np.sin(phi)

        xcm_s1 = rcm_max * np.cos(phi)
        ycm_s1 = rcm_min * np.sin(phi)

        phi = -4 * np.pi / 3

        xsm_s2 = rsm_max * np.cos(phi)
        ysm_s2 = rsm_min * np.sin(phi)

        xcm_s2 = rcm_max * np.cos(phi)
        ycm_s2 = rcm_min * np.sin(phi)

        # Top bottom

        rst_s12_min, zst = np.meshgrid(discr_rc_min, discr_zst_s12)
        rst_s12_max, zst = np.meshgrid(discr_rc_max, discr_zst_s12)

        phi = 0.

        xst_s1 = rst_s12_max * np.cos(phi)
        yst_s1 = rst_s12_min * np.sin(phi)

        phi = -4 * np.pi / 3

        xst_s2 = rst_s12_max * np.cos(phi)
        yst_s2 = rst_s12_min * np.sin(phi)

        rsb_s12_min, zsb = np.meshgrid(discr_rc_min, discr_zsb_s12)
        rsb_s12_max, zsb = np.meshgrid(discr_rc_max, discr_zsb_s12)

        phi = 0.

        xsb_s1 = rsb_s12_max * np.cos(phi)
        ysb_s1 = rsb_s12_min * np.sin(phi)

        phi = -4 * np.pi / 3

        xsb_s2 = rsb_s12_max * np.cos(phi)
        ysb_s2 = rsb_s12_min * np.sin(phi)


        self.surf = self.ax.plot_surface(x_c, y_c, z_c, alpha=0.2, color="b")
        self.surf = self.ax.plot_surface(xtb_c, ytb_c, zt_c, alpha=0.2, color="b")
        self.surf = self.ax.plot_surface(xtb_c, ytb_c, zb_c, alpha=0.2, color="b")

        self.surf = self.ax.plot_surface(x_s, y_s, z_s, alpha=0.2, color="r")
        self.surf = self.ax.plot_surface(xtb_s, ytb_s, zt_s, alpha=0.2, color="r")
        self.surf = self.ax.plot_surface(xtb_s, ytb_s, zb_s, alpha=0.2, color="r")

        self.surf = self.ax.plot_surface(xsm_s1, ysm_s1, zsm_s12, alpha=1, color="r")
        self.surf = self.ax.plot_surface(xsm_s2, ysm_s2, zsm_s12, alpha=1, color="r")

        self.surf = self.ax.plot_surface(xcm_s1, ycm_s1, zcm_s12, alpha=1, color="b")
        self.surf = self.ax.plot_surface(xcm_s2, ycm_s2, zcm_s12, alpha=1, color="b")

        self.surf = self.ax.plot_surface(xst_s1, yst_s1, zst, alpha=1, color="r")
        self.surf = self.ax.plot_surface(xst_s2, yst_s2, zst, alpha=1, color="r")

        self.surf = self.ax.plot_surface(xsb_s1, ysb_s1, zsb, alpha=1, color="r")
        self.surf = self.ax.plot_surface(xsb_s2, ysb_s2, zsb, alpha=1, color="r")


        self.fig.canvas.draw()
        self.fig.show()

    def core_shell_bicelle_elliptical_belt_rough(self):

        radius = self.param['radius']  # core radius
        length = self.param['length']  # cylinder total length
        x_core = self.param['x_core']  # Axial ratio of core (Major/Minor)
        w_face = self.param['thick_face']  # shell thickness face
        w_rim = self.param['thick_rim'] # shell thickness rim

        self.fig = plt.figure('3d-model')
        self.ax = plt.axes(projection='3d')
        self.ax.set_xlabel('A')
        self.ax.set_ylabel('A')
        self.ax.set_zlabel('A')
        self.ax.set_title('core_shell_bicelle_elliptical_belt_rough plot')
        self.ax.view_init(25, 45)


        # Discretization cylinder mantle (core and shell)

        discr_zc = np.linspace(0., length, 2)
        discr_zs = np.linspace(0 * -w_face, length + 0 * w_face, 2)

        # Discretization cylinder top and bottom (core and shell)

        discr_rc_min = np.linspace(0., radius, 30)
        discr_rc_max = np.linspace(0., x_core * radius, 30)

        discr_rs_min = np.linspace(0., radius + 0 * w_rim, 30)
        discr_rs_max = np.linspace(0., x_core * radius + 0 * w_rim, 30)

        # Discretization of shell thickness in r and z directions

        discr_rsm_s12_min = np.linspace(radius, radius + w_rim, 2)
        discr_rsm_s12_max = np.linspace(x_core * radius, x_core * radius + w_rim, 2)

        discr_zst_s12 = np.linspace(length, length + w_face, 2)
        discr_zsb_s12 = np.linspace(-w_face, 0., 2)

        # Mantle

        phi, z_c = np.meshgrid(self.discr_phi, discr_zc)

        phi, z_s = np.meshgrid(self.discr_phi, discr_zs)

        x_c = x_core * radius * np.cos(phi)
        y_c = radius * np.sin(phi)

        x_s = (x_core * radius + w_rim) * np.cos(phi)
        y_s = (radius + w_rim) * np.sin(phi)

        phi, z_c_t = np.meshgrid(self.discr_phi, discr_zst_s12)

        phi, z_c_b = np.meshgrid(self.discr_phi, discr_zsb_s12)

        x_c_tb = x_core * radius * np.cos(phi)
        y_c_tb = radius * np.sin(phi)

        # top and bottom

        phi, r_c_min = np.meshgrid(self.discr_phi, discr_rc_min)
        phi, r_c_max = np.meshgrid(self.discr_phi, discr_rc_max)

        xtb_c = r_c_max * np.cos(phi)
        ytb_c = r_c_min * np.sin(phi)

        zt_c = length * np.ones((30, 30))
        zb_c = 0. * np.ones((30, 30))

        phi, r_s_min = np.meshgrid(self.discr_phi, discr_rs_min)
        phi, r_s_max = np.meshgrid(self.discr_phi, discr_rs_max)

        xtb_s = r_s_max * np.cos(phi)
        ytb_s = r_s_min * np.sin(phi)

        zt_s = (length + w_face) * np.ones((30, 30))
        zb_s = -w_face * np.ones((30, 30))

        # Shell

        # Mantle

        rsm_min, zsm_s12 = np.meshgrid(discr_rsm_s12_min, discr_zs)
        rsm_max, zsm_s12 = np.meshgrid(discr_rsm_s12_max, discr_zs)

        rcm_min, zcm_s12 = np.meshgrid(discr_rc_min, discr_zc)
        rcm_max, zcm_s12 = np.meshgrid(discr_rc_max, discr_zc)

        phi = 0.

        xsm_s1 = rsm_max * np.cos(phi)
        ysm_s1 = rsm_min * np.sin(phi)

        xcm_s1 = rcm_max * np.cos(phi)
        ycm_s1 = rcm_min * np.sin(phi)

        phi = -4 * np.pi / 3

        xsm_s2 = rsm_max * np.cos(phi)
        ysm_s2 = rsm_min * np.sin(phi)

        xcm_s2 = rcm_max * np.cos(phi)
        ycm_s2 = rcm_min * np.sin(phi)

        # Top bottom

        rst_s12_min, zst = np.meshgrid(discr_rc_min, discr_zst_s12)
        rst_s12_max, zst = np.meshgrid(discr_rc_max, discr_zst_s12)

        phi = 0.

        xst_s1 = rst_s12_max * np.cos(phi)
        yst_s1 = rst_s12_min * np.sin(phi)

        phi = -4 * np.pi / 3

        xst_s2 = rst_s12_max * np.cos(phi)
        yst_s2 = rst_s12_min * np.sin(phi)

        rsb_s12_min, zsb = np.meshgrid(discr_rc_min, discr_zsb_s12)
        rsb_s12_max, zsb = np.meshgrid(discr_rc_max, discr_zsb_s12)

        phi = 0.

        xsb_s1 = rsb_s12_max * np.cos(phi)
        ysb_s1 = rsb_s12_min * np.sin(phi)

        phi = -4 * np.pi / 3

        xsb_s2 = rsb_s12_max * np.cos(phi)
        ysb_s2 = rsb_s12_min * np.sin(phi)

        self.surf = self.ax.plot_surface(x_c, y_c, z_c, alpha=0.2, color="b")
        self.surf = self.ax.plot_surface(xtb_c, ytb_c, zt_c, alpha=0.2, color="b")
        self.surf = self.ax.plot_surface(xtb_c, ytb_c, zb_c, alpha=0.2, color="b")

        self.surf = self.ax.plot_surface(x_s, y_s, z_s, alpha=0.2, color="r")
        self.surf = self.ax.plot_surface(xtb_s, ytb_s, zt_s, alpha=0.2, color="r")
        self.surf = self.ax.plot_surface(xtb_s, ytb_s, zb_s, alpha=0.2, color="r")

        self.surf = self.ax.plot_surface(xsm_s1, ysm_s1, zsm_s12, alpha=1, color="r")
        self.surf = self.ax.plot_surface(xsm_s2, ysm_s2, zsm_s12, alpha=1, color="r")

        self.surf = self.ax.plot_surface(xcm_s1, ycm_s1, zcm_s12, alpha=1, color="b")
        self.surf = self.ax.plot_surface(xcm_s2, ycm_s2, zcm_s12, alpha=1, color="b")

        self.surf = self.ax.plot_surface(xst_s1, yst_s1, zst, alpha=1, color="r")
        self.surf = self.ax.plot_surface(xst_s2, yst_s2, zst, alpha=1, color="r")

        self.surf = self.ax.plot_surface(xsb_s1, ysb_s1, zsb, alpha=1, color="r")
        self.surf = self.ax.plot_surface(xsb_s2, ysb_s2, zsb, alpha=1, color="r")

        self.surf = self.ax.plot_surface(x_c_tb, y_c_tb, z_c_t, alpha=0.2, color="r")
        self.surf = self.ax.plot_surface(x_c_tb, y_c_tb, z_c_b, alpha=0.2, color="r")

        self.fig.canvas.draw()
        self.fig.show()

    def barbell(self):

        radius = self.param['radius']  # radius cylinder
        length = self.param['length']  # length of cylinder bar
        radius_bell = self.param['radius_bell']  # Radius of bell   > radius

        self.fig = plt.figure('3d-model')
        self.ax = plt.axes(projection='3d')
        self.ax.set_xlabel('A')
        self.ax.set_ylabel('A')
        self.ax.set_zlabel('A')
        self.ax.set_title('barbell plot')
        self.ax.view_init(25, 45)

        h_in = np.sqrt(radius_bell ** 2 - radius ** 2)

        # Discretization caps

        # top

        teta_t = np.pi - np.arcsin(radius / radius_bell)

        discr_tetat = np.linspace(0., teta_t, 30)

        # bottom

        teta_b = np.arcsin(radius / radius_bell)

        discr_tetab = np.linspace(teta_b, np.pi, 30)

        # Discretization cylinder mantle

        discr_z = np.linspace(0., length, 2)

        # Caps

        phi, teta_top = np.meshgrid(self.discr_phi, discr_tetat)

        x_top = radius_bell * np.cos(phi) * np.sin(teta_top)
        y_top = radius_bell * np.sin(phi) * np.sin(teta_top)
        z_top = radius_bell * np.cos(teta_top) + (length + h_in) * np.ones((30, 30))

        phi, teta_bot = np.meshgrid(self.discr_phi, discr_tetab)

        x_bot = radius_bell * np.cos(phi) * np.sin(teta_bot)
        y_bot = radius_bell * np.sin(phi) * np.sin(teta_bot)
        z_bot = radius_bell * np.cos(teta_bot) - h_in * np.ones((30, 30))

        # Mantle

        phi, z = np.meshgrid(self.discr_phi, discr_z)

        x = radius * np.cos(phi)
        y = radius * np.sin(phi)

        self.surf = self.ax.plot_surface(x, y, z, alpha=1, color="b")

        self.surf = self.ax.plot_surface(x_top, y_top, z_top, alpha=1, color="b")
        self.surf = self.ax.plot_surface(x_bot, y_bot, z_bot, alpha=1, color="b")


        self.fig.canvas.draw()
        self.fig.show()

