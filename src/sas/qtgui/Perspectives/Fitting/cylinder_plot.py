import matplotlib
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np


"""Plotfunctions for the cylinder category"""


class cylinder:

    def __init__(self):
        """Construct parameters for cylinder"""

        self.radius = 1
        self.length = 2



        # Discretization cylinder mantle

        self.discr_phi = np.linspace(0., 2 * np.pi, 30)
        self.discr_z = np.linspace(0.,self.length,2)

        # Discretization cylinder top and bottom

        self.discr_r = np.linspace(0.,self.radius,30)


    def plot(self):
        """Plot cylinder"""

        fig = plt.figure()
        ax = plt.axes(projection='3d')
        ax.set_aspect('auto')
        ax.set_xlabel('A')
        ax.set_ylabel('A')
        ax.set_zlabel('A');
        # Set viev by elevation and azimuth angles
        # Eelevation above x-y plane and counter clockwiserotation on z axis
        ax.view_init(20, 45)

        # Mantle

        phi, z = np.meshgrid(self.discr_phi, self.discr_z)

        x = self.radius * np.cos(phi)
        y = self.radius * np.sin(phi)

        # top and bottom

        phi, r = np.meshgrid(self.discr_phi, self.discr_r)

        xtb = r * np.cos(phi)
        ytb = r * np.sin(phi)

        zt = self.length * np.ones((30,30))
        zb = 0. * np.ones((30,30))

        ax.plot_surface(x, y, z, alpha=1, color="b")
        ax.plot_surface(xtb, ytb, zt, alpha=1, color="b")
        ax.plot_surface(xtb, ytb, zb, alpha=1, color="b")


        ax.set_title('cylinder plot')
        plt.show()


class hollow_cylinder:

    def __init__(self):
        """Construct parameters for hollow_cylinder"""

        self.radius = 10       # cylinder core radius
        self.tot_length = 10   # cylinder total length
        self.w_shell = 2       # shell thickness


        self.length = self.tot_length - 2 * self.w_shell

        # Discretization cylinder mantle (core and shell)

        self.discr_phi = np.linspace(0., -4 * np.pi / 3, 30)

        self.discr_zc = np.linspace(0.,self.length,2)
        self.discr_zs = np.linspace(-self.w_shell, self.length + self.w_shell, 2)

        # Discretization cylinder top and bottom (core and shell)

        self.discr_rc = np.linspace(0.,self.radius,30)
        self.discr_rs = np.linspace(0., self.radius + self.w_shell, 30)

        # Discretization of shell thickness in r and z directions

        self.discr_rsm_s12 = np.linspace(self.radius, self.radius + self.w_shell, 2)

        self.discr_zst_s12 = np.linspace(self.length, self.length + self.w_shell, 2)
        self.discr_zsb_s12 = np.linspace(-self.w_shell, 0., 2)

    def plot(self):
        """Plot hollow_cylinder"""

        fig = plt.figure()
        ax = plt.axes(projection='3d')
        ax.set_aspect('auto')
        ax.set_xlabel('A')
        ax.set_ylabel('A')
        ax.set_zlabel('A');
        # Set viev by elevation and azimuth angles
        # Eelevation above x-y plane and counter clockwiserotation on z axis
        ax.view_init(20, 45)

        # Mantle

        phi, z_c = np.meshgrid(self.discr_phi, self.discr_zc)

        phi, z_s = np.meshgrid(self.discr_phi, self.discr_zs)


        x_c = self.radius * np.cos(phi)
        y_c = self.radius * np.sin(phi)

        x_s = (self.radius + self.w_shell) * np.cos(phi)
        y_s = (self.radius + self.w_shell) * np.sin(phi)


        # top and bottom

        phi, r_c = np.meshgrid(self.discr_phi, self.discr_rc)

        xtb_c = r_c * np.cos(phi)
        ytb_c = r_c * np.sin(phi)

        zt_c = self.length * np.ones((30,30))
        zb_c = 0. * np.ones((30,30))

        phi, r_s = np.meshgrid(self.discr_phi, self.discr_rs)

        xtb_s = r_s * np.cos(phi)
        ytb_s = r_s * np.sin(phi)

        zt_s = (self.length+ self.w_shell) * np.ones((30, 30))
        zb_s = -self.w_shell * np.ones((30, 30))


        # Shell

        # Mantle

        rsm, zsm_s12 = np.meshgrid(self.discr_rsm_s12, self.discr_zs)

        phi = 0.

        xsm_s1 = rsm * np.cos(phi)
        ysm_s1 = rsm * np.sin(phi)

        phi = -4 * np.pi / 3

        xsm_s2 = rsm * np.cos(phi)
        ysm_s2 = rsm * np.sin(phi)

        # Top bottom

        rst_s12, zst = np.meshgrid(self.discr_rc, self.discr_zst_s12)

        phi = 0.

        xst_s1 = rst_s12 * np.cos(phi)
        yst_s1 = rst_s12 * np.sin(phi)

        phi = -4 * np.pi / 3

        xst_s2 = rst_s12 * np.cos(phi)
        yst_s2 = rst_s12 * np.sin(phi)

        rsb_s12, zsb = np.meshgrid(self.discr_rc, self.discr_zsb_s12)

        phi = 0.

        xsb_s1 = rsb_s12 * np.cos(phi)
        ysb_s1 = rsb_s12 * np.sin(phi)

        phi = -4 * np.pi / 3

        xsb_s2 = rsb_s12 * np.cos(phi)
        ysb_s2 = rsb_s12 * np.sin(phi)

        ax.plot_surface(x_c, y_c, z_c, alpha=0.1, color="b")
        ax.plot_surface(xtb_c, ytb_c, zt_c, alpha=0.1, color="b")
        ax.plot_surface(xtb_c, ytb_c, zb_c, alpha=0.1, color="b")

        ax.plot_surface(x_s, y_s, z_s, alpha=0.1, color="r")
        ax.plot_surface(xtb_s, ytb_s, zt_s, alpha=0.1, color="r")
        ax.plot_surface(xtb_s, ytb_s, zb_s, alpha=0.1, color="r")

        ax.plot_surface(xsm_s1, ysm_s1, zsm_s12, alpha=1, color="r")
        ax.plot_surface(xsm_s2, ysm_s2, zsm_s12, alpha=1, color="r")

        ax.plot_surface(xst_s1, yst_s1, zst, alpha=1, color="r")
        ax.plot_surface(xst_s2, yst_s2, zst, alpha=1, color="r")

        ax.plot_surface(xsb_s1, ysb_s1, zsb, alpha=1, color="r")
        ax.plot_surface(xsb_s2, ysb_s2, zsb, alpha=1, color="r")

        ax.set_title('hollow_cylinder plot')
        plt.show()


class pearl_necklace:

    def __init__(self):
        """Construct parameters for pearl_necklace"""

        self.radius = 1   # Pearl radius
        self.edge_sep = 1   # Length of the string segment (surface to surface)
        self.n_pearls = 7  # Number of pearls
        self.w_string = 0.2  # Thickness of string segment

        # Discretization sphere

        self.discr_phi = np.linspace(0., 2 * np.pi, 30)
        self.discr_teta = np.linspace(0., np.pi, 30)

        # Discretization rod

        self.discr_xr = np.linspace(0,self.edge_sep,3)

    def plot(self):
        """Plot pearl_necklace"""

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

        phirm, xrm = np.meshgrid(self.discr_phi,self.discr_xr)

        for n in range(0,self.n_pearls):

            x = self.radius * np.cos(phi) * np.sin(teta) + n * (self.edge_sep + 2 * self.radius)
            y = self.radius * np.sin(phi) * np.sin(teta)
            z = self.radius * np.cos(teta)

            if n <= self.n_pearls-2:

                zr = self.w_string * np.cos(phirm)
                yr = self.w_string * np.sin(phirm)
                xr = xrm + n * (self.edge_sep + 2 * self.radius) * np.ones((3,30)) + self.radius * np.ones((3,30))
                ax.plot_surface(xr, yr, zr, alpha=1, color="b")

            ax.plot_surface(x, y, z, alpha=1, color="b")

            ax.set_title('pearl_necklace plot')
        plt.show()


class pringle:

    def __init__(self):
        """Construct parameters for pringle"""

        self.radius = 60   # radius of pringle
        self.length = 20   # thickness of pringle
        self.alpha = 0.02 # curvature parameter alpha
        self.beta = 0.02   # cuvature parameter beta



        # Discretization cylinder mantle

        self.discr_phi = np.linspace(0., 2 * np.pi, 60)

        self.discr_z = np.linspace(0.,self.length,60)

        # Discretization cylinder top and bottom

        self.discr_r = np.linspace(0.,self.radius,60)


    def plot(self):
        """Plot pringle"""

        fig = plt.figure()
        ax = plt.axes(projection='3d')
        ax.set_aspect('auto')
        ax.set_xlabel('A')
        ax.set_ylabel('A')
        ax.set_zlabel('A');
        # Set viev by elevation and azimuth angles
        # Eelevation above x-y plane and counter clockwiserotation on z axis
        ax.view_init(20, 45)

        # Mantle

        phi, z = np.meshgrid(self.discr_phi, self.discr_z)

        x = self.radius * np.cos(phi)
        y = self.radius * np.sin(phi)
        z = self.radius ** 2 * (self.alpha * np.cos(phi) ** 2 - self.beta * np.sin(phi) ** 2) + z

        # top and bottom

        phi, r = np.meshgrid(self.discr_phi, self.discr_r)

        xtb = r * np.cos(phi)
        ytb = r * np.sin(phi)

        zt = self.radius ** 2 * (self.alpha * np.cos(phi) ** 2 - self.beta * np.sin(phi) ** 2) \
                                 + self.length * np.ones((60,60))
        zb = self.radius ** 2 * (self.alpha * np.cos(phi) ** 2 - self.beta * np.sin(phi) ** 2)

        ax.plot_surface(x, y, z, alpha=1, color="r")
        ax.plot_surface(xtb, ytb, zt, alpha=0.3, color="r")
        ax.plot_surface(xtb, ytb, zb, alpha=0.3, color="b")


        ax.set_title('pringle plot')
        plt.show()


class flexible_cylinder:

    def __init__(self):
        """Construct parameters for flexible_cylinder"""

        self.k_length = 1   # Kuhn length of the cylinder segment
        self.length = 10   # Total length of flexible cylinder
        self.radius = 0.2  # Radius of flexible cylinder




        self.wig = np.pi / 24   # Wiggle angle in xy_plane for plotting only

        self.n_cyl = int(self.length / self.k_length)


        # Discretization cylinder

        self.discr_phi = np.linspace(0., 2 * np.pi, 30)

        self.discr_xr = np.linspace(0,self.k_length,3)

    def plot(self):
        """Plot flexible_cylinder"""

        fig = plt.figure()
        ax = plt.axes(projection='3d')
        ax.set_aspect('auto')
        ax.set_xlabel('A')
        ax.set_ylabel('A')
        ax.set_zlabel('A');
        # Set viev by elevation and azimuth angles
        # Eelevation above x-y plane and counter clockwiserotation on z axis
        ax.view_init(20, 90)


        phirm, xrm = np.meshgrid(self.discr_phi,self.discr_xr)

        # Define wiggle (rotation) matrix

        r11p = np.cos(self.wig) * np.ones((3,30))
        r12p = -np.sin(self.wig) * np.ones((3,30))
        r21p = np.sin(self.wig) * np.ones((3, 30))
        r22p = np.cos(self.wig) * np.ones((3, 30))

        r11m = np.cos(-self.wig) * np.ones((3, 30))
        r12m = -np.sin(-self.wig) * np.ones((3, 30))
        r21m = np.sin(-self.wig) * np.ones((3, 30))
        r22m = np.cos(-self.wig) * np.ones((3, 30))

        zr = self.radius * np.cos(phirm)
        yr = self.radius * np.sin(phirm)
        xr = xrm

        # Segment Rotated up

        xr_rp = xr * r11p + yr * r12p
        yr_rp = xr * r12p + yr * r22p

        # Segment Rotated down

        xr_rm = xr * r11m + yr * r12m
        yr_rm = xr * r12m + yr * r22m


        for n in range(0,10):

            if (n % 2) == 0:
                x = xr_rp + n * self.k_length * np.cos(self.wig) * np.ones((3,30))
                y = yr_rp
                z = zr

                ax.plot_surface(x, y, z, alpha=0.5, color="b")
            else:
                x = xr_rm + n * self.k_length * np.cos(-self.wig) * np.ones((3,30))
                y = yr_rm + self.k_length * np.sin(-self.wig) * np.ones((3,30))
                z = zr

                ax.plot_surface(x, y, z, alpha=0.5, color="r")




        ax.set_title('flexible_cylinder plot')
        plt.show()


class flexible_cylinder_elliptical:

    def __init__(self):
        """Construct parameters for flexible_cylinder_elliptical"""

        self.k_length = 1   # Kuhn length of the cylinder segment
        self.length = 10   # Total length of flexible cylinder
        self.radius = 0.2  # Radius of flexible cylinder
        self.ax_rat = 2    # Ratio major/ minor radius of cylinder




        self.wig = np.pi / 24   # Wiggle angle in xy_plane for plotting only

        self.n_cyl = int(self.length / self.k_length)


        # Discretization cylinder

        self.discr_phi = np.linspace(0., 2 * np.pi, 30)

        self.discr_xr = np.linspace(0,self.k_length,3)


    def plot(self):
        """Plot flexible_cylinder_elliptical"""

        fig = plt.figure()
        ax = plt.axes(projection='3d')
        ax.set_aspect('auto')
        ax.set_xlabel('A')
        ax.set_ylabel('A')
        ax.set_zlabel('A');
        # Set viev by elevation and azimuth angles
        # Eelevation above x-y plane and counter clockwiserotation on z axis
        ax.view_init(20, 90)


        phirm, xrm = np.meshgrid(self.discr_phi,self.discr_xr)

        # Define wiggle (rotation) matrix

        r11p = np.cos(self.wig) * np.ones((3,30))
        r12p = -np.sin(self.wig) * np.ones((3,30))
        r21p = np.sin(self.wig) * np.ones((3, 30))
        r22p = np.cos(self.wig) * np.ones((3, 30))

        r11m = np.cos(-self.wig) * np.ones((3, 30))
        r12m = -np.sin(-self.wig) * np.ones((3, 30))
        r21m = np.sin(-self.wig) * np.ones((3, 30))
        r22m = np.cos(-self.wig) * np.ones((3, 30))

        zr = self.ax_rat * self.radius * np.cos(phirm)
        yr = self.radius * np.sin(phirm)
        xr = xrm

        # Segment Rotated up

        xr_rp = xr * r11p + yr * r12p
        yr_rp = xr * r12p + yr * r22p

        # Segment Rotated down

        xr_rm = xr * r11m + yr * r12m
        yr_rm = xr * r12m + yr * r22m


        for n in range(0,10):

            if (n % 2) == 0:
                x = xr_rp + n * self.k_length * np.cos(self.wig) * np.ones((3,30))
                y = yr_rp
                z = zr

                ax.plot_surface(x, y, z, alpha=0.5, color="b")
            else:
                x = xr_rm + n * self.k_length * np.cos(-self.wig) * np.ones((3,30))
                y = yr_rm + self.k_length * np.sin(-self.wig) * np.ones((3,30))
                z = zr

                ax.plot_surface(x, y, z, alpha=0.5, color="r")


        ax.set_title('flexible_cylinder_elliptical plot')
        plt.show()


class elliptical_cylinder:

    def __init__(self):
        """Construct parameters for elliptical_cylinder"""

        self.radius = 1
        self.length = 2
        self.ax_rat = 2


        # Discretization cylinder mantle

        self.discr_phi = np.linspace(0., 2 * np.pi, 30)
        self.discr_z = np.linspace(0.,self.length,2)

        # Discretization cylinder top and bottom

        self.discr_r = np.linspace(0.,self.radius,30)


    def plot(self):
        """Plot elliptical_cylinder"""

        fig = plt.figure()
        ax = plt.axes(projection='3d')
        ax.set_aspect('auto')
        ax.set_xlabel('A')
        ax.set_ylabel('A')
        ax.set_zlabel('A');
        # Set viev by elevation and azimuth angles
        # Eelevation above x-y plane and counter clockwiserotation on z axis
        ax.view_init(20, 45)

        # Mantle

        phi, z = np.meshgrid(self.discr_phi, self.discr_z)

        x = self.ax_rat * self.radius * np.cos(phi)
        y = self.radius * np.sin(phi)

        # top and bottom

        phi, r = np.meshgrid(self.discr_phi, self.discr_r)

        xtb = self.ax_rat * r * np.cos(phi)
        ytb = r * np.sin(phi)

        zt = self.length * np.ones((30,30))
        zb = 0. * np.ones((30,30))

        ax.plot_surface(x, y, z, alpha=1, color="b")
        ax.plot_surface(xtb, ytb, zt, alpha=1, color="b")
        ax.plot_surface(xtb, ytb, zb, alpha=1, color="b")


        ax.set_title('elliptical_cylinder plot')
        plt.show()


class stacked_discs:

    def __init__(self):
        """Construct parameters for stacked_discs"""

        self.radius = 1    # Radius of discs
        self.w_core = 1    # Thickness of core disc
        self.w_lay = 1     # Thickness of the 2 layers around each core disc
        self.n_lay = 2     # Number of stacked layer/core/layer discs




        # Discretization cylinder mantle

        self.discr_phi = np.linspace(0., 2 * np.pi, 30)

        self.discr_zc = np.linspace(0.,self.w_core,2)
        self.discr_zl = np.linspace(0., self.w_lay, 2)

        # Discretization cylinder top and bottom

        self.discr_r = np.linspace(0.,self.radius,30)


    def plot(self):
        """Plot stacked_discs"""

        fig = plt.figure()
        ax = plt.axes(projection='3d')
        ax.set_aspect('auto')
        ax.set_xlabel('A')
        ax.set_ylabel('A')
        ax.set_zlabel('A');
        # Set viev by elevation and azimuth angles
        # Eelevation above x-y plane and counter clockwiserotation on z axis
        ax.view_init(20, 45)

        phi_c, zc = np.meshgrid(self.discr_phi, self.discr_zc)

        phi_l, zl = np.meshgrid(self.discr_phi, self.discr_zl)

        phi, r = np.meshgrid(self.discr_phi, self.discr_r)

        # Mantle

        xc = self.radius * np.cos(phi_c)
        yc = self.radius * np.sin(phi_c)

        xl = self.radius * np.cos(phi_l)
        yl = self.radius * np.sin(phi_l)


        # top and bottom

        xtb = r * np.cos(phi)
        ytb = r * np.sin(phi)


        for n in range(0,self.n_lay):

            # Layer

            zt_l1 = (self.w_lay + n * (2*self.w_lay + self.w_core)) * np.ones((30,30))
            zb_l1 = (n * (2*self.w_lay + self.w_core)) * np.ones((30,30))

            zm_l1 = zl + n * (2*self.w_lay + self.w_core) * np.ones((2, 30))

            # Core

            zt_c = (self.w_lay + self.w_core + n * (2*self.w_lay + self.w_core)) * np.ones((30, 30))
            zb_c = (self.w_lay + n * (2*self.w_lay + self.w_core)) * np.ones((30, 30))

            zm_c = zc + (self.w_lay + n * (2*self.w_lay + self.w_core)) * np.ones((2, 30))

            # Layer

            zt_l2 = (2 * self.w_lay + self.w_core + n * (2*self.w_lay + self.w_core)) * np.ones((30, 30))
            zb_l2 = (self.w_lay + self.w_core + n * (2*self.w_lay + self.w_core)) * np.ones((30, 30))

            zm_l2 = zl + (self.w_lay + self.w_core + n * (2*self.w_lay + self.w_core)) * np.ones((2, 30))



            ax.plot_surface(xl, yl, zm_l1, alpha=0.5, color="b")
            ax.plot_surface(xtb, ytb, zt_l1, alpha=0.5, color="b")
            ax.plot_surface(xtb, ytb, zb_l1, alpha=0.5, color="b")

            ax.plot_surface(xc, yc, zm_c, alpha=1, color="r")
            ax.plot_surface(xtb, ytb, zt_c, alpha=1, color="r")
            ax.plot_surface(xtb, ytb, zb_c, alpha=1, color="r")

            ax.plot_surface(xl, yl, zm_l2, alpha=1, color="b")
            ax.plot_surface(xtb, ytb, zt_l2, alpha=1, color="b")
            ax.plot_surface(xtb, ytb, zb_l2, alpha=1, color="b")



        ax.set_title('stacked_discs plot')
        plt.show()


class core_shell_cylinder:

    def __init__(self):
        """Construct parameters for core_shell_cylinder"""

        self.radius = 10       # core radius
        self.tot_length = 10   # cylinder total length
        self.w_shell = 2       # shell thickness


        self.length = self.tot_length - 2 * self.w_shell

        # Discretization cylinder mantle (core and shell)

        self.discr_phi = np.linspace(0., -4 * np.pi / 3, 30)

        self.discr_zc = np.linspace(0.,self.length,2)
        self.discr_zs = np.linspace(-self.w_shell, self.length + self.w_shell, 2)

        # Discretization cylinder top and bottom (core and shell)

        self.discr_rc = np.linspace(0.,self.radius,30)
        self.discr_rs = np.linspace(0., self.radius + self.w_shell, 30)

        # Discretization of shell thickness in r and z directions

        self.discr_rsm_s12 = np.linspace(self.radius, self.radius + self.w_shell, 2)

        self.discr_zst_s12 = np.linspace(self.length, self.length + self.w_shell, 2)
        self.discr_zsb_s12 = np.linspace(-self.w_shell, 0., 2)

    def plot(self):
        """Plot core_shell_cylinder"""

        fig = plt.figure()
        ax = plt.axes(projection='3d')
        ax.set_aspect('auto')
        ax.set_xlabel('A')
        ax.set_ylabel('A')
        ax.set_zlabel('A');
        # Set viev by elevation and azimuth angles
        # Eelevation above x-y plane and counter clockwiserotation on z axis
        ax.view_init(20, 45)

        # Mantle

        phi, z_c = np.meshgrid(self.discr_phi, self.discr_zc)

        phi, z_s = np.meshgrid(self.discr_phi, self.discr_zs)


        x_c = self.radius * np.cos(phi)
        y_c = self.radius * np.sin(phi)

        x_s = (self.radius + self.w_shell) * np.cos(phi)
        y_s = (self.radius + self.w_shell) * np.sin(phi)


        # top and bottom

        phi, r_c = np.meshgrid(self.discr_phi, self.discr_rc)

        xtb_c = r_c * np.cos(phi)
        ytb_c = r_c * np.sin(phi)

        zt_c = self.length * np.ones((30,30))
        zb_c = 0. * np.ones((30,30))

        phi, r_s = np.meshgrid(self.discr_phi, self.discr_rs)

        xtb_s = r_s * np.cos(phi)
        ytb_s = r_s * np.sin(phi)

        zt_s = (self.length+ self.w_shell) * np.ones((30, 30))
        zb_s = -self.w_shell * np.ones((30, 30))


        # Shell

        # Mantle

        rsm, zsm_s12 = np.meshgrid(self.discr_rsm_s12, self.discr_zs)

        rcm, zcm_s12 = np.meshgrid(self.discr_rc, self.discr_zc)

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

        rst_s12, zst = np.meshgrid(self.discr_rc, self.discr_zst_s12)

        phi = 0.

        xst_s1 = rst_s12 * np.cos(phi)
        yst_s1 = rst_s12 * np.sin(phi)

        phi = -4 * np.pi / 3

        xst_s2 = rst_s12 * np.cos(phi)
        yst_s2 = rst_s12 * np.sin(phi)

        rsb_s12, zsb = np.meshgrid(self.discr_rc, self.discr_zsb_s12)

        phi = 0.

        xsb_s1 = rsb_s12 * np.cos(phi)
        ysb_s1 = rsb_s12 * np.sin(phi)

        phi = -4 * np.pi / 3

        xsb_s2 = rsb_s12 * np.cos(phi)
        ysb_s2 = rsb_s12 * np.sin(phi)

        ax.plot_surface(x_c, y_c, z_c, alpha=0.2, color="b")
        ax.plot_surface(xtb_c, ytb_c, zt_c, alpha=0.2, color="b")
        ax.plot_surface(xtb_c, ytb_c, zb_c, alpha=0.2, color="b")

        ax.plot_surface(x_s, y_s, z_s, alpha=0.2, color="r")
        ax.plot_surface(xtb_s, ytb_s, zt_s, alpha=0.2, color="r")
        ax.plot_surface(xtb_s, ytb_s, zb_s, alpha=0.2, color="r")

        ax.plot_surface(xsm_s1, ysm_s1, zsm_s12, alpha=1, color="r")
        ax.plot_surface(xsm_s2, ysm_s2, zsm_s12, alpha=1, color="r")

        ax.plot_surface(xcm_s1, ycm_s1, zcm_s12, alpha=1, color="b")
        ax.plot_surface(xcm_s2, ycm_s2, zcm_s12, alpha=1, color="b")

        ax.plot_surface(xst_s1, yst_s1, zst, alpha=1, color="r")
        ax.plot_surface(xst_s2, yst_s2, zst, alpha=1, color="r")

        ax.plot_surface(xsb_s1, ysb_s1, zsb, alpha=1, color="r")
        ax.plot_surface(xsb_s2, ysb_s2, zsb, alpha=1, color="r")

        ax.set_title('core_shell_cylinder plot')
        plt.show()


class capped_cylinder:

    def __init__(self):
        """Construct parameters for capped_cylinder"""

        self.radius = 1       # radius cylinder
        self.length = 2       # length of cylinder
        self.radius_cap = 2   # Radius of cap   > radius


        self.h_in = np.sqrt(self.radius_cap ** 2 - self.radius ** 2)

        # Discretization caps


        # top

        self.teta_t = np.arcsin(self.radius/self.radius_cap)

        self.discr_tetat = np.linspace(0., self.teta_t, 30)

        # bottom

        self.teta_b = np.pi - self.teta_t

        self.discr_tetab = np.linspace(self.teta_b, np.pi, 30)

        # Discretization cylinder mantle

        self.discr_phi = np.linspace(0., 2 * np.pi, 30)
        self.discr_z = np.linspace(0.,self.length,2)


    def plot(self):
        """Plot capped_cylinder"""

        fig = plt.figure()
        ax = plt.axes(projection='3d')
        ax.set_aspect('auto')
        ax.set_xlabel('A')
        ax.set_ylabel('A')
        ax.set_zlabel('A');
        # Set viev by elevation and azimuth angles
        # Eelevation above x-y plane and counter clockwiserotation on z axis
        ax.view_init(10, 45)

        # Caps

        phi, teta_top = np.meshgrid(self.discr_phi, self.discr_tetat)
        x_top = self.radius_cap * np.cos(phi) * np.sin(teta_top)
        y_top = self.radius_cap * np.sin(phi) * np.sin(teta_top)
        z_top = self.radius_cap * np.cos(teta_top) + (self.length - self.h_in) * np.ones((30,30))

        phi, teta_bot = np.meshgrid(self.discr_phi, self.discr_tetab)
        x_bot = self.radius_cap * np.cos(phi) * np.sin(teta_bot)
        y_bot = self.radius_cap * np.sin(phi) * np.sin(teta_bot)
        z_bot = self.radius_cap * np.cos(teta_bot) + self.h_in * np.ones((30,30))

        # Mantle

        phi, z = np.meshgrid(self.discr_phi, self.discr_z)

        x = self.radius * np.cos(phi)
        y = self.radius * np.sin(phi)



        ax.plot_surface(x, y, z, alpha=1, color="b")

        ax.plot_surface(x_top, y_top, z_top, alpha=1, color="b")
        ax.plot_surface(x_bot, y_bot, z_bot, alpha=1, color="b")

        ax.set_title('capped_cylinder plot')
        plt.show()

class core_shell_bicelle:

    def __init__(self):
        """Construct parameters for core_shell_bicelle"""

        self.radius = 10       # core radius
        self.tot_length = 10   # cylinder total length
        self.w_face = 2       # shell thickness face
        self.w_rim = 2        # shell thickness rim

        self.length = self.tot_length - 2 * self.w_face

        # Discretization cylinder mantle (core and shell)

        self.discr_phi = np.linspace(0., -4 * np.pi / 3, 30)

        self.discr_zc = np.linspace(0.,self.length,2)
        self.discr_zs = np.linspace(-self.w_face, self.length + self.w_face, 2)

        # Discretization cylinder top and bottom (core and shell)

        self.discr_rc = np.linspace(0.,self.radius,30)
        self.discr_rs = np.linspace(0., self.radius + self.w_rim, 30)

        # Discretization of shell thickness in r and z directions

        self.discr_rsm_s12 = np.linspace(self.radius, self.radius + self.w_rim, 2)

        self.discr_zst_s12 = np.linspace(self.length, self.length + self.w_face, 2)
        self.discr_zsb_s12 = np.linspace(-self.w_face, 0., 2)

    def plot(self):
        """Plot core_shell_bicelle"""

        fig = plt.figure()
        ax = plt.axes(projection='3d')
        ax.set_aspect('auto')
        ax.set_xlabel('A')
        ax.set_ylabel('A')
        ax.set_zlabel('A');
        # Set viev by elevation and azimuth angles
        # Eelevation above x-y plane and counter clockwiserotation on z axis
        ax.view_init(20, 45)

        # Mantle

        phi, z_c = np.meshgrid(self.discr_phi, self.discr_zc)

        phi, z_s = np.meshgrid(self.discr_phi, self.discr_zs)


        x_c = self.radius * np.cos(phi)
        y_c = self.radius * np.sin(phi)

        x_s = (self.radius + self.w_rim) * np.cos(phi)
        y_s = (self.radius + self.w_rim) * np.sin(phi)


        # top and bottom

        phi, r_c = np.meshgrid(self.discr_phi, self.discr_rc)

        xtb_c = r_c * np.cos(phi)
        ytb_c = r_c * np.sin(phi)

        zt_c = self.length * np.ones((30,30))
        zb_c = 0. * np.ones((30,30))

        phi, r_s = np.meshgrid(self.discr_phi, self.discr_rs)

        xtb_s = r_s * np.cos(phi)
        ytb_s = r_s * np.sin(phi)

        zt_s = (self.length+ self.w_face) * np.ones((30, 30))
        zb_s = -self.w_face * np.ones((30, 30))


        # Shell

        # Mantle

        rsm, zsm_s12 = np.meshgrid(self.discr_rsm_s12, self.discr_zs)

        rcm, zcm_s12 = np.meshgrid(self.discr_rc, self.discr_zc)

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

        rst_s12, zst = np.meshgrid(self.discr_rc, self.discr_zst_s12)

        phi = 0.

        xst_s1 = rst_s12 * np.cos(phi)
        yst_s1 = rst_s12 * np.sin(phi)

        phi = -4 * np.pi / 3

        xst_s2 = rst_s12 * np.cos(phi)
        yst_s2 = rst_s12 * np.sin(phi)

        rsb_s12, zsb = np.meshgrid(self.discr_rc, self.discr_zsb_s12)

        phi = 0.

        xsb_s1 = rsb_s12 * np.cos(phi)
        ysb_s1 = rsb_s12 * np.sin(phi)

        phi = -4 * np.pi / 3

        xsb_s2 = rsb_s12 * np.cos(phi)
        ysb_s2 = rsb_s12 * np.sin(phi)

        ax.plot_surface(x_c, y_c, z_c, alpha=0.2, color="b")
        ax.plot_surface(xtb_c, ytb_c, zt_c, alpha=0.2, color="b")
        ax.plot_surface(xtb_c, ytb_c, zb_c, alpha=0.2, color="b")

        ax.plot_surface(x_s, y_s, z_s, alpha=0.2, color="r")
        ax.plot_surface(xtb_s, ytb_s, zt_s, alpha=0.2, color="r")
        ax.plot_surface(xtb_s, ytb_s, zb_s, alpha=0.2, color="r")

        ax.plot_surface(xsm_s1, ysm_s1, zsm_s12, alpha=1, color="r")
        ax.plot_surface(xsm_s2, ysm_s2, zsm_s12, alpha=1, color="r")

        ax.plot_surface(xcm_s1, ycm_s1, zcm_s12, alpha=1, color="b")
        ax.plot_surface(xcm_s2, ycm_s2, zcm_s12, alpha=1, color="b")

        ax.plot_surface(xst_s1, yst_s1, zst, alpha=1, color="r")
        ax.plot_surface(xst_s2, yst_s2, zst, alpha=1, color="r")

        ax.plot_surface(xsb_s1, ysb_s1, zsb, alpha=1, color="r")
        ax.plot_surface(xsb_s2, ysb_s2, zsb, alpha=1, color="r")

        ax.set_title('core_shell_bicelle plot')
        plt.show()



class core_shell_bicelle_elliptical:

    def __init__(self):
        """Construct parameters for core_shell_bicelle_elliptical"""

        self.radius = 10       # core radius
        self.tot_length = 10   # cylinder total length
        self.x_core = 3       # Axial ratio of core (Major/Minor)
        self.w_face = 2       # shell thickness face
        self.w_rim = 2        # shell thickness rim

        self.length = self.tot_length - 2 * self.w_face

        # Discretization cylinder mantle (core and shell)

        self.discr_phi = np.linspace(0., -4 * np.pi / 3, 30)

        self.discr_zc = np.linspace(0.,self.length,2)
        self.discr_zs = np.linspace(-self.w_face, self.length + self.w_face, 2)

        # Discretization cylinder top and bottom (core and shell)

        self.discr_rc_min = np.linspace(0.,self.radius,30)
        self.discr_rc_max = np.linspace(0.,self.x_core * self.radius,30)

        self.discr_rs_min = np.linspace(0., self.radius + self.w_rim, 30)
        self.discr_rs_max = np.linspace(0., self.x_core * self.radius + self.w_rim, 30)

        # Discretization of shell thickness in r and z directions

        self.discr_rsm_s12_min = np.linspace(self.radius, self.radius + self.w_rim, 2)
        self.discr_rsm_s12_max = np.linspace(self.x_core * self.radius, self.x_core * self.radius + self.w_rim, 2)


        self.discr_zst_s12 = np.linspace(self.length, self.length + self.w_face, 2)
        self.discr_zsb_s12 = np.linspace(-self.w_face, 0., 2)

    def plot(self):
        """Plot core_shell_bicelle_elliptical"""

        fig = plt.figure()
        ax = plt.axes(projection='3d')
        ax.set_aspect('auto')
        ax.set_xlabel('A')
        ax.set_ylabel('A')
        ax.set_zlabel('A');
        # Set viev by elevation and azimuth angles
        # Eelevation above x-y plane and counter clockwiserotation on z axis
        ax.view_init(20, 45)

        # Mantle

        phi, z_c = np.meshgrid(self.discr_phi, self.discr_zc)

        phi, z_s = np.meshgrid(self.discr_phi, self.discr_zs)


        x_c = self.x_core * self.radius * np.cos(phi)
        y_c = self.radius * np.sin(phi)

        x_s = (self.x_core * self.radius + self.w_rim) * np.cos(phi)
        y_s = (self.radius + self.w_rim) * np.sin(phi)


        # top and bottom

        phi, r_c_min = np.meshgrid(self.discr_phi, self.discr_rc_min)
        phi, r_c_max = np.meshgrid(self.discr_phi, self.discr_rc_max)

        xtb_c = r_c_max * np.cos(phi)
        ytb_c = r_c_min * np.sin(phi)

        zt_c = self.length * np.ones((30,30))
        zb_c = 0. * np.ones((30,30))

        phi, r_s_min = np.meshgrid(self.discr_phi, self.discr_rs_min)
        phi, r_s_max = np.meshgrid(self.discr_phi, self.discr_rs_max)

        xtb_s = r_s_max * np.cos(phi)
        ytb_s = r_s_min * np.sin(phi)

        zt_s = (self.length+ self.w_face) * np.ones((30, 30))
        zb_s = -self.w_face * np.ones((30, 30))


        # Shell

        # Mantle

        rsm_min, zsm_s12 = np.meshgrid(self.discr_rsm_s12_min, self.discr_zs)
        rsm_max, zsm_s12 = np.meshgrid(self.discr_rsm_s12_max, self.discr_zs)

        rcm_min, zcm_s12 = np.meshgrid(self.discr_rc_min, self.discr_zc)
        rcm_max, zcm_s12 = np.meshgrid(self.discr_rc_max, self.discr_zc)

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

        rst_s12_min, zst = np.meshgrid(self.discr_rc_min, self.discr_zst_s12)
        rst_s12_max, zst = np.meshgrid(self.discr_rc_max, self.discr_zst_s12)

        phi = 0.

        xst_s1 = rst_s12_max * np.cos(phi)
        yst_s1 = rst_s12_min * np.sin(phi)

        phi = -4 * np.pi / 3

        xst_s2 = rst_s12_max * np.cos(phi)
        yst_s2 = rst_s12_min * np.sin(phi)

        rsb_s12_min, zsb = np.meshgrid(self.discr_rc_min, self.discr_zsb_s12)
        rsb_s12_max, zsb = np.meshgrid(self.discr_rc_max, self.discr_zsb_s12)

        phi = 0.

        xsb_s1 = rsb_s12_max * np.cos(phi)
        ysb_s1 = rsb_s12_min * np.sin(phi)

        phi = -4 * np.pi / 3

        xsb_s2 = rsb_s12_max * np.cos(phi)
        ysb_s2 = rsb_s12_min * np.sin(phi)

        ax.plot_surface(x_c, y_c, z_c, alpha=0.2, color="b")
        ax.plot_surface(xtb_c, ytb_c, zt_c, alpha=0.2, color="b")
        ax.plot_surface(xtb_c, ytb_c, zb_c, alpha=0.2, color="b")

        ax.plot_surface(x_s, y_s, z_s, alpha=0.2, color="r")
        ax.plot_surface(xtb_s, ytb_s, zt_s, alpha=0.2, color="r")
        ax.plot_surface(xtb_s, ytb_s, zb_s, alpha=0.2, color="r")

        ax.plot_surface(xsm_s1, ysm_s1, zsm_s12, alpha=1, color="r")
        ax.plot_surface(xsm_s2, ysm_s2, zsm_s12, alpha=1, color="r")

        ax.plot_surface(xcm_s1, ycm_s1, zcm_s12, alpha=1, color="b")
        ax.plot_surface(xcm_s2, ycm_s2, zcm_s12, alpha=1, color="b")

        ax.plot_surface(xst_s1, yst_s1, zst, alpha=1, color="r")
        ax.plot_surface(xst_s2, yst_s2, zst, alpha=1, color="r")

        ax.plot_surface(xsb_s1, ysb_s1, zsb, alpha=1, color="r")
        ax.plot_surface(xsb_s2, ysb_s2, zsb, alpha=1, color="r")

        ax.set_title('core_shell_bicelle_elliptical plot')
        plt.show()



class core_shell_bicelle_elliptical_belt_rough:

    def __init__(self):
        """Construct parameters for core_shell_bicelle_elliptical_belt_rough"""

        self.radius = 10       # core radius
        self.tot_length = 10   # cylinder total length
        self.x_core = 3       # Axial ratio of core (Major/Minor)
        self.w_face = 2       # shell thickness face
        self.w_rim = 2        # shell thickness rim

        self.length = self.tot_length - 2 * self.w_face

        # Discretization cylinder mantle (core and shell)

        self.discr_phi = np.linspace(0., -4 * np.pi / 3, 30)

        self.discr_zc = np.linspace(0.,self.length,2)
        self.discr_zs = np.linspace(0 * -self.w_face, self.length + 0 * self.w_face, 2)

        # Discretization cylinder top and bottom (core and shell)

        self.discr_rc_min = np.linspace(0.,self.radius,30)
        self.discr_rc_max = np.linspace(0.,self.x_core * self.radius,30)

        self.discr_rs_min = np.linspace(0., self.radius + 0 * self.w_rim, 30)
        self.discr_rs_max = np.linspace(0., self.x_core * self.radius + 0 * self.w_rim, 30)

        # Discretization of shell thickness in r and z directions

        self.discr_rsm_s12_min = np.linspace(self.radius, self.radius + self.w_rim, 2)
        self.discr_rsm_s12_max = np.linspace(self.x_core * self.radius, self.x_core * self.radius + self.w_rim, 2)


        self.discr_zst_s12 = np.linspace(self.length, self.length + self.w_face, 2)
        self.discr_zsb_s12 = np.linspace(-self.w_face, 0., 2)

    def plot(self):
        """Plot core_shell_bicelle_elliptical_belt_rough"""

        fig = plt.figure()
        ax = plt.axes(projection='3d')
        ax.set_aspect('auto')
        ax.set_xlabel('A')
        ax.set_ylabel('A')
        ax.set_zlabel('A');
        # Set viev by elevation and azimuth angles
        # Eelevation above x-y plane and counter clockwiserotation on z axis
        ax.view_init(20, 45)

        # Mantle

        phi, z_c = np.meshgrid(self.discr_phi, self.discr_zc)

        phi, z_s = np.meshgrid(self.discr_phi, self.discr_zs)




        x_c = self.x_core * self.radius * np.cos(phi)
        y_c = self.radius * np.sin(phi)

        x_s = (self.x_core * self.radius + self.w_rim) * np.cos(phi)
        y_s = (self.radius + self.w_rim) * np.sin(phi)

        phi, z_c_t = np.meshgrid(self.discr_phi, self.discr_zst_s12)

        phi, z_c_b = np.meshgrid(self.discr_phi, self.discr_zsb_s12)

        x_c_tb = self.x_core * self.radius * np.cos(phi)
        y_c_tb = self.radius * np.sin(phi)


        # top and bottom

        phi, r_c_min = np.meshgrid(self.discr_phi, self.discr_rc_min)
        phi, r_c_max = np.meshgrid(self.discr_phi, self.discr_rc_max)

        xtb_c = r_c_max * np.cos(phi)
        ytb_c = r_c_min * np.sin(phi)

        zt_c = self.length * np.ones((30,30))
        zb_c = 0. * np.ones((30,30))

        phi, r_s_min = np.meshgrid(self.discr_phi, self.discr_rs_min)
        phi, r_s_max = np.meshgrid(self.discr_phi, self.discr_rs_max)

        xtb_s = r_s_max * np.cos(phi)
        ytb_s = r_s_min * np.sin(phi)

        zt_s = (self.length+ self.w_face) * np.ones((30, 30))
        zb_s = -self.w_face * np.ones((30, 30))


        # Shell

        # Mantle

        rsm_min, zsm_s12 = np.meshgrid(self.discr_rsm_s12_min, self.discr_zs)
        rsm_max, zsm_s12 = np.meshgrid(self.discr_rsm_s12_max, self.discr_zs)

        rcm_min, zcm_s12 = np.meshgrid(self.discr_rc_min, self.discr_zc)
        rcm_max, zcm_s12 = np.meshgrid(self.discr_rc_max, self.discr_zc)

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

        rst_s12_min, zst = np.meshgrid(self.discr_rc_min, self.discr_zst_s12)
        rst_s12_max, zst = np.meshgrid(self.discr_rc_max, self.discr_zst_s12)

        phi = 0.

        xst_s1 = rst_s12_max * np.cos(phi)
        yst_s1 = rst_s12_min * np.sin(phi)

        phi = -4 * np.pi / 3

        xst_s2 = rst_s12_max * np.cos(phi)
        yst_s2 = rst_s12_min * np.sin(phi)

        rsb_s12_min, zsb = np.meshgrid(self.discr_rc_min, self.discr_zsb_s12)
        rsb_s12_max, zsb = np.meshgrid(self.discr_rc_max, self.discr_zsb_s12)

        phi = 0.

        xsb_s1 = rsb_s12_max * np.cos(phi)
        ysb_s1 = rsb_s12_min * np.sin(phi)

        phi = -4 * np.pi / 3

        xsb_s2 = rsb_s12_max * np.cos(phi)
        ysb_s2 = rsb_s12_min * np.sin(phi)

        ax.plot_surface(x_c, y_c, z_c, alpha=0.2, color="b")
        ax.plot_surface(xtb_c, ytb_c, zt_c, alpha=0.2, color="b")
        ax.plot_surface(xtb_c, ytb_c, zb_c, alpha=0.2, color="b")

        ax.plot_surface(x_s, y_s, z_s, alpha=0.2, color="r")
        ax.plot_surface(xtb_s, ytb_s, zt_s, alpha=0.2, color="r")
        ax.plot_surface(xtb_s, ytb_s, zb_s, alpha=0.2, color="r")

        ax.plot_surface(xsm_s1, ysm_s1, zsm_s12, alpha=1, color="r")
        ax.plot_surface(xsm_s2, ysm_s2, zsm_s12, alpha=1, color="r")

        ax.plot_surface(xcm_s1, ycm_s1, zcm_s12, alpha=1, color="b")
        ax.plot_surface(xcm_s2, ycm_s2, zcm_s12, alpha=1, color="b")

        ax.plot_surface(xst_s1, yst_s1, zst, alpha=1, color="r")
        ax.plot_surface(xst_s2, yst_s2, zst, alpha=1, color="r")

        ax.plot_surface(xsb_s1, ysb_s1, zsb, alpha=1, color="r")
        ax.plot_surface(xsb_s2, ysb_s2, zsb, alpha=1, color="r")

        ax.plot_surface(x_c_tb, y_c_tb, z_c_t, alpha=0.2, color="r")
        ax.plot_surface(x_c_tb, y_c_tb, z_c_b, alpha=0.2, color="r")

        ax.set_title('core_shell_bicelle_elliptical_belt_rough plot')
        plt.show()


class barbell:

    def __init__(self):
        """Construct parameters for barbell"""

        self.radius = 1       # radius cylinder
        self.length = 1       # length of cylinder
        self.radius_bell = 2   # Radius of bell   > radius


        self.h_in = np.sqrt(self.radius_bell ** 2 - self.radius ** 2)

        # Discretization caps


        # top

        self.teta_t = np.pi - np.arcsin(self.radius/self.radius_bell)

        self.discr_tetat = np.linspace(0., self.teta_t, 30)

        # bottom

        self.teta_b = np.arcsin(self.radius/self.radius_bell)

        self.discr_tetab = np.linspace(self.teta_b, np.pi, 30)

        # Discretization cylinder mantle

        self.discr_phi = np.linspace(0., 2 * np.pi, 30)
        self.discr_z = np.linspace(0.,self.length,2)


    def plot(self):
        """Plot barbell"""

        fig = plt.figure()
        ax = plt.axes(projection='3d')
        ax.set_aspect('auto')
        ax.set_xlabel('A')
        ax.set_ylabel('A')
        ax.set_zlabel('A');
        # Set viev by elevation and azimuth angles
        # Eelevation above x-y plane and counter clockwiserotation on z axis
        ax.view_init(10, 45)

        # Caps

        phi, teta_top = np.meshgrid(self.discr_phi, self.discr_tetat)
        x_top = self.radius_bell * np.cos(phi) * np.sin(teta_top)
        y_top = self.radius_bell * np.sin(phi) * np.sin(teta_top)
        z_top = self.radius_bell * np.cos(teta_top) + (self.length + self.h_in) * np.ones((30,30))

        phi, teta_bot = np.meshgrid(self.discr_phi, self.discr_tetab)
        x_bot = self.radius_bell * np.cos(phi) * np.sin(teta_bot)
        y_bot = self.radius_bell * np.sin(phi) * np.sin(teta_bot)
        z_bot = self.radius_bell * np.cos(teta_bot) - self.h_in * np.ones((30,30))

        # Mantle

        phi, z = np.meshgrid(self.discr_phi, self.discr_z)

        x = self.radius * np.cos(phi)
        y = self.radius * np.sin(phi)



        ax.plot_surface(x, y, z, alpha=1, color="b")

        ax.plot_surface(x_top, y_top, z_top, alpha=1, color="b")
        ax.plot_surface(x_bot, y_bot, z_bot, alpha=1, color="b")

        ax.set_title('barbell plot')
        plt.show()




# cylinder().plot()
# hollow_cylinder().plot()
# pearl_necklace().plot()
# pringle().plot()
# flexible_cylinder().plot()
# flexible_cylinder_elliptical().plot()
# elliptical_cylinder().plot()
# stacked_discs().plot()
# core_shell_cylinder().plot()
# capped_cylinder().plot()
# core_shell_bicelle().plot()
# core_shell_bicelle_elliptical().plot()
# core_shell_bicelle_elliptical_belt_rough().plot()
# barbell().plot()