#******************************************************************************#
#
# This program generates a list of stars with the following conditions:
#
# - Mass distribution follows Kroupa(2001) IMF
#   The mass-giving procedure goes as follows:
#   (1) draw a cdf(cumulative distribution function) from the given IMF
#   (2) pick a random number u from [0,1)
#   (3) equate cdf(m) = u
#   (4) invert m = cdf^-1(u)
#
# - uniform spatial distribution within the given cluster size, i.e. constant density
#   (1) draw a cdf from the given pdf; pdf_r ~ r^2, pdf_theta ~ sin(theta)
#   (2) pick a random number u,u' from [0,1)
#   (3) equate cdf(r) = u, cdf(theta) = u'
#   (4) invert r = cdf^-1(u), theta = cdf^-1(u')
#
# - Gaussian velocity distribution peaked at Keplerian velocity;
#   |v| = sqrt(GM_enc/r)
#   v_tangential = Gaussian(mean = |v|, sigma = |v|/sqrt(2))
#   v_radial = Gaussian(mean = 0, sigma = |v|/sqrt(2))
#   The standard deviation is set so that the total velocity dispersion is |v|
#
# << Units >>
# Conditions(ClusterMass,ClusterSize) are given in 'solar mass', 'pc'
# Results(mass,x,y,z,vx,vy,vz) are given in 'solar mass', 'kpc', 'km/s'
#
# Created: 20 Jan 2021
# Modified:
#******************************************************************************#

import numpy as np
from sympy import *
from scipy import stats
from timeit import default_timer as timer

start = timer()

# Given conditions
ClusterMass = 10**4 # solar mass
ClusterSize = 3.0   # pc

# Initialize
MassOfGeneratedStars = 0.0
NumberOfGeneratedStars = 0

# Physical constants and unit conversion constants
G = 6.674 * 10**(-11)           # gravitational constant in SI
MassUnit = 1                    # solarmass to solarmass
LengthUnit = 10**(-3)           # pc to kpc
VelocityUnit = 8.0276 * 10**3   # sqrt((solarmass in kg)/(pc in m))*(1/1000)


# Precalculations --------------------------------------------------------------

# I. cdf from the Kroupa(2001) IMF

# m stands for mass; k is the normalization const.
m,k = symbols('m k')

# Kroupa IMF (not normalized)
xi_1 = (m/0.08)**(-0.3)                      # 0.01 <= m <= 0.08
xi_2 = (m/0.08)**(-1.3)                      # 0.08 < m <= 0.5
xi_3 = (0.5/0.08)**(-1.3) * (m/0.5)**(-2.3)  # 0.5 < m

# cdf calculation; note that we integrate with m in the range (0.01,inf)
cdf_1 = integrate(xi_1,m) - integrate(xi_1,m).subs(m,0.01)
cdf_2 = cdf_1.subs(m,0.08) + integrate(xi_2,m) - integrate(xi_2,m).subs(m,0.08)
cdf_3 = cdf_2.subs(m,0.5) + integrate(xi_3,m) - integrate(xi_3,m).subs(m,0.5)

# piecewise integrals
integral_1 = cdf_1.subs(m,0.08)
integral_2 = cdf_2.subs(m,0.5)
integral_3 = cdf_3.subs(m,oo) # total integral of the pdf

# k as a normalization constant
k = 1/integral_3

# the normalized IMF would be k * xi_1,2,3
# the normalized cdf would be k * cdf_1,2,3
# ...and so on.



# II. cdf from the uniform spatial distributions

# r,theta is radial coordinate; l,n are normalization constants
r,theta,l,n = symbols('r theta l n')

# uniform density pdf (not normalized)
pdf_r = r**2

# cdf calculation
cdf_r = integrate(pdf_r,r) - integrate(pdf_r,r).subs(r,0)

# total integral of the pdf
integral_r = cdf_r.subs(r,ClusterSize)

# l as a normalizaton constant
l = 1/integral_r


# spherically symmetric pdf (not normalized)
pdf_theta = sin(theta)

# cdf calculation
cdf_theta = integrate(pdf_theta,theta) - integrate(pdf_theta,theta).subs(theta,0)

# total integral of the pdf
integral_theta = cdf_theta.subs(theta,np.pi)

# l as a normalizaton constant
n = 1/integral_theta


# End of precalculations -------------------------------------------------------


f = open("demo.txt", "w")

# star-generating loop
while MassOfGeneratedStars < ClusterMass * MassUnit :

    # MASS

    # pick a random number and solve u = cdf_i(m) for m
    # since the Kroupa IMF is a piecewise function, we need to break it to invert it.
    # the explicit form of cdfs are calculated from the IMF by hand.
    u = np.random.random()

    if u <= k*integral_1:
        mass = (u/k * 0.7/0.08**0.3 + 0.01**0.7)**(1/0.7) * MassUnit

    elif k*integral_1 < u <= k*integral_2:
        mass = ((k*integral_1-u)*0.3/k/0.08**1.3 + 0.08**(-0.3))**(-1/0.3) * MassUnit

    else:
        mass = ((k*integral_2-u)*1.3/k/0.5/0.08**1.3 + 0.5**(-1.3))**(-1/1.3) * MassUnit



    # POSITION

    # r
    # pick a random number and solve u = cdf_r(r) for r
    u = np.random.random()
    r0 = (3*u/l)**(1/3)

    # theta
    # pick a random number and solve u = cdf_theta(theta) for theta
    u = np.random.random()
    theta0 = acos(1-u/n)

    # phi
    # is uniform
    phi = np.random.uniform(low=0.0,high=2*np.pi)

    # transform into Cartesian coordinates
    x = r0*sin(theta0)*cos(phi) * LengthUnit
    y = r0*sin(theta0)*sin(phi) * LengthUnit
    z = r0*cos(theta0) * LengthUnit



    # VELOCITY

    #also assume uniform density
    rho = ClusterMass/((4*np.pi/3)*ClusterSize**3)
    M_enc = rho * (4/3*np.pi)*r0**3

    # Gaussian distribution peaked at Keplerian velocity
    v = sqrt(G * M_enc/r0)
    vr = np.random.normal(0,v/sqrt(2))   # radial velocity
    vt = np.random.normal(v,v/sqrt(2))   # tangential velocity

    # choose the direction of tangential velocity
    psi = np.random.uniform(low=0.0,high=2*np.pi)

    vx = (vr*sin(theta0)*cos(phi) + vt*(cos(psi)*cos(theta0)*cos(phi) - sin(psi)*sin(phi))) * VelocityUnit
    vy = (vr*sin(theta0)*sin(phi) + vt*(cos(psi)*cos(theta0)*sin(phi) + sin(psi)*cos(phi))) * VelocityUnit
    vz = (vr*cos(theta0) - vt*cos(psi)*sin(theta0)) * VelocityUnit

    # write the star to the file
    star = '%f %f %f %f %f %f %f' % (mass, x, y, z, vx, vy, vz)
    f.write(star + '\n')

    NumberOfGeneratedStars += 1
    MassOfGeneratedStars = MassOfGeneratedStars + mass


f.close()

end = timer()


print('Done!/n')
print('running time: ' + str(end - start) + ' seconds\n')
print('number of generated stars: ' + str(NumberOfGeneratedStars) + '\n')
print('total mass: ' + str(MassOfGeneratedStars) + ' solar mass\n')
