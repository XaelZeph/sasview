import matplotlib
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np



"""Plotfunctions for the ellipsoid category"""

class ellipsoid:

     def __init__(self):

        """Construct parameters for ellipsoid"""

        self.rad_po = 1    # Polar radius
        self.rad_eq = 2    # Equatorial radius

        # Discretization sphere

        self.discr_phi = np.linspace(0., 2 * np.pi, 30)
        self.discr_teta = np.linspace(0.,np.pi,30)

     def plot(self):

        """Plot ellipsoid"""

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
        x = self.rad_eq * np.cos(phi) * np.sin(teta)
        y = self.rad_eq * np.sin(phi) * np.sin(teta)
        z = self.rad_po * np.cos(teta)
        ax.plot_surface(x, y, z, alpha = 1, color = "b")
        ax.set_title('ellipsoid plot')
        plt.show()


class core_shell_ellipsoid:

   def __init__(self):
      """Construct parameters for core_shell_ellipsoid"""

      self.rad_core_eq = 10  # Equatorial radius
      self.x_core = 0.5        # Axial ratio of core (polar / equatorial)
      self.w_shell_eq = 1  # Shell thickness at equator
      self.x_pol_rat = 1  # Ratio of thickness of shell at pole to that at equator


      self.rad_core_pol = self.rad_core_eq * self.x_core
      self.w_shell_pol = self.w_shell_eq * self. x_pol_rat


      # Discretization spheres

      self.discr_phi = np.linspace(2 * np.pi / 3, 2 * np.pi, 30)
      self.discr_teta = np.linspace(0., np.pi, 30)

   def plot(self):
      """Plot core_shell_ellipsoid"""

      fig = plt.figure()
      ax = plt.axes(projection='3d')
      ax.set_aspect('auto')
      ax.set_xlabel('A')
      ax.set_ylabel('A')
      ax.set_zlabel('A');
      # Set viev by elevation and azimuth angles
      # Eelevation above x-y plane and counter clockwiserotation on z axis
      ax.view_init(25, 45)

      phi, teta = np.meshgrid(self.discr_phi, self.discr_teta)

      xc = self.rad_core_eq * np.cos(phi) * np.sin(teta)
      yc = self.rad_core_eq * np.sin(phi) * np.sin(teta)
      zc = self.rad_core_pol * np.cos(teta)

      self.discr_rc_eq = np.linspace(0, self.rad_core_eq, 2)
      self.discr_rc_pol = np.linspace(0,self.rad_core_pol,2)

      rc_s12_eq, teta_s12 = np.meshgrid(self.discr_rc_eq, self.discr_teta)
      rc_s12_pol, teta_s12 = np.meshgrid(self.discr_rc_pol, self.discr_teta)

      phi_s1 = 2 * np.pi / 3

      xc_s1 = rc_s12_eq * np.cos(phi_s1) * np.sin(teta_s12)
      yc_s1 = rc_s12_eq * np.sin(phi_s1) * np.sin(teta_s12)
      zc_s1 = rc_s12_pol * np.cos(teta_s12)

      phi_s2 = 2 * np.pi

      xc_s2 = rc_s12_eq * np.cos(phi_s2) * np.sin(teta_s12)
      yc_s2 = rc_s12_eq * np.sin(phi_s2) * np.sin(teta_s12)
      zc_s2 = rc_s12_pol * np.cos(teta_s12)

      ax.plot_surface(xc, yc, zc, alpha=0.1, color="b")

      ax.plot_surface(xc_s1, yc_s1, zc_s1, alpha=1., color="b")
      ax.plot_surface(xc_s2, yc_s2, zc_s2, alpha=1., color="b")

      xs = (self.w_shell_eq + self.rad_core_eq) * np.cos(phi) * np.sin(teta)
      ys = (self.w_shell_eq + self.rad_core_eq) * np.sin(phi) * np.sin(teta)
      zs = (self.w_shell_pol + self.rad_core_pol) * np.cos(teta)

      self.discr_rs_eq = np.linspace((self.rad_core_eq),(self.w_shell_eq + self.rad_core_eq), 2)
      self.discr_rs_pol = np.linspace((self.rad_core_pol), (self.w_shell_pol + self.rad_core_pol), 2)

      rs_s12_eq, teta_s12 = np.meshgrid(self.discr_rs_eq, self.discr_teta)
      rs_s12_pol, teta_s12 = np.meshgrid(self.discr_rs_pol, self.discr_teta)

      phi_s1 = 2 * np.pi / 3

      xs_s1 = rs_s12_eq * np.cos(phi_s1) * np.sin(teta_s12)
      ys_s1 = rs_s12_eq * np.sin(phi_s1) * np.sin(teta_s12)
      zs_s1 = rs_s12_pol * np.cos(teta_s12)

      phi_s2 = 2 * np.pi

      xs_s2 = rs_s12_eq * np.cos(phi_s2) * np.sin(teta_s12)
      ys_s2 = rs_s12_eq * np.sin(phi_s2) * np.sin(teta_s12)
      zs_s2 = rs_s12_pol * np.cos(teta_s12)

      ax.plot_surface(xs_s1, ys_s1, zs_s1, alpha=1., color="r")
      ax.plot_surface(xs_s2, ys_s2, zs_s2, alpha=1., color="r")

      ax.plot_surface(xs, ys, zs, alpha=0.1, color="r")

      ax.set_title('core_shell_ellipsoid')

      plt.show()


class triaxial_ellipsoid:

   def __init__(self):
      """Construct parameters for triaxial_ellipsoid"""

      self.rad_po = 1  # Polar radius
      self.rad_eq_min = 2  # Equatorial radius minor axis
      self.rad_eq_maj = 3  # Equatorial radius major axis

      # Discretization sphere

      self.discr_phi = np.linspace(0., 2 * np.pi, 30)
      self.discr_teta = np.linspace(0., np.pi, 30)

   def plot(self):
      """Plot triaxial_ellipsoid"""

      fig = plt.figure()
      ax = plt.axes(projection='3d')
      ax.set_aspect('auto')
      ax.set_xlabel('A')
      ax.set_ylabel('A')
      ax.set_zlabel('A');
      # Set viev by elevation and azimuth angles
      # Eelevation above x-y plane and counter clockwiserotation on z axis
      ax.view_init(20, 45)

      phi, teta = np.meshgrid(self.discr_phi, self.discr_teta)
      x = self.rad_eq_min * np.cos(phi) * np.sin(teta)
      y = self.rad_eq_maj * np.sin(phi) * np.sin(teta)
      z = self.rad_po * np.cos(teta)
      ax.plot_surface(x, y, z, alpha=1, color="b")
      ax.set_title('triaxial_ellipsoid plot')
      plt.show()

# ellipsoid().plot()
# core_shell_ellipsoid().plot()
# triaxial_ellipsoid().plot()