import numpy as np
import pdb
import matplotlib.pyplot as plt
import scipy.stats as stats
import scipy.interpolate as interp
from scipy.constants import c
from lisatools.utils.constants import *
from few.waveform import GenerateEMRIWaveform, FastSchwarzschildEccentricFlux
from functools import partial
import copy
import lisaorbits
import lisaconstants
from lisatools.sources.emri import EMRITDIWaveform
from fastlisaresponse import ResponseWrapper
from lisatools.detector import EqualArmlengthOrbits

# Implement Time Delay Interferometry (TDI) for LISA. This module should provide the 
# necessary functions and objects to generate TDI waveforms given source parameters. In its
# current version it relies on the FEW's FastSchwarzschildEccentricFlux, a kind of 
# kludge waveform generator. At the end the the TDI waveform is checked with the 
# EMRITDIWaveform from the LISA Analysis Tools (LAT) package.
# Relevant equations are taken from arXiv:2204.06633

# Constants
YRSID_SI = 31558149.763545603 
SEC_TO_SOLAR_M = 4.925491025873693e-06 
MPC_TO_SOLAR_M = 2.089682521268661e+19
GPC_TO_SOLAR_M = 2.089682521268661e+22
M_TO_SOLAR_M = 1476.6250615036158
MPC_TO_M = 3.0856775814913674e+22
GPC_TO_M = 3.0856775814913674e+25
# We are working in the equal arm limit effectively meaning that our arms are stationary
proplen = 2.5e9 # constant arm length in m divide by c to get seconds

# First we import the satellite positions using lisa orbits and consider a 0.1 year 
# observation period with sampling frequency of 15 seconds 

orbits = lisaorbits.KeplerianOrbits()

def make_galactic(orbits, T=3.156e6, N=210388):
    tm = np.linspace(0, T, num=N, endpoint=False)

    # spacecraft positions (km) - SC x T x XYZ
    pos = np.array([orbits.compute_position(orbits.t0 + tm, [sc]).reshape(len(tm),3)
                     for sc in orbits.SC]) / lisaconstants.SPEED_OF_LIGHT
    
    return tm, pos

tm, pos = make_galactic(orbits)

# Now we want a way of of obtaining a particular spacecrafts (x, y, z)

class SpacecraftPosition:
    def __init__(self, pos):
        
        '''Instantiate with the position given by lisaorbits which is a 3 x T x 3 array
        where the first index is the gives the spacecraft, the second index gives the 
        position at a particular time and the third index gives whether that position is
        in the x, y or z direction. Call with u to get the relevant T x 3 array.'''
        
        self.pos = pos

    def __call__(self, u):
        # specify desired spacecraft with u and desired x,y,z motion with v (or do we need xyz?)
        return self.pos[u, :, :]

positions = SpacecraftPosition(pos)

# Next we define the link unit vectors nhat which describe between which arms of the
# of the constellation we are traveling

class LinkUnitVector:
    def __init__(self):
        
        '''Link unit vectors unitary in the corresponding positions for arm traveled
        i.e. nhat(1, 2) is [1, 1, 0] and describes traveling between spacecraft 1 and 2 
        while nhat(2, 1) is [-1, -1, 0] and describes traveling between spacecraft 2 and 1.
        call with i and j to get the corresponding vector.'''

        self.n_dict = {
            (1, 2): np.array([1, 1, 0]),
            (2, 1): np.array([-1, -1, 0]),
            (1, 3): np.array([1, 0, 1]),
            (3, 1): np.array([-1, 0, -1]),
            (2, 3): np.array([0, 1, 1]),
            (3, 2): np.array([0, -1, -1])
        }

    def __call__(self, i, j):
        return self.n_dict.get((i, j))
    
nhat = LinkUnitVector()

# Now we may define the solar system barycenter (SSB) coordinates which gives the 
# the spacecraft positions relative to the center of mass which the earth and sun orbit

class SSBReferenceFrame:
    def __init__(self, beta, lam):
        
        ''' instantiate with beta and lambda, the ecliptic latitude and longitde where 
        beta = pi/2 - theta and lambda = phi and (theta, phi) constitude the SSB spherical
        coordinates. (er, etheta, ephi) constitute the orthonormal basis of the spherical 
        SSB frame. Finally the polarization vectors u and v and the propgation vector k
        are constructed from the basis vectors and are used in the rest of the 
        TDI calculation. (u, v, k) also create an orhtonormal basis. calling returns
        all of these vectors.'''

        self.beta = beta
        self.lam = lam

        self.er = np.array([np.cos(self.beta)*np.cos(self.lam), np.cos(self.beta)*np.sin(self.lam), np.sin(self.beta)])
        self.etheta = np.array([np.sin(self.beta)*np.cos(self.lam), np.sin(self.beta)*np.sin(self.lam), -np.cos(self.beta)])
        self.ephi = np.array([-np.sin(self.lam), np.cos(self.lam), 0])

        self.khat = -self.er
        self.uhat = -self.etheta
        self.vhat = -self.ephi

    def __call__(self):
        return self.er, self.etheta, self.ephi, self.khat, self.uhat, self.vhat
    
frame = SSBReferenceFrame(beta = 8, lam = 7)

er, etheta, ephi, khat, uhat, vhat = frame()

# We may now construct the antenna pattern functions which are used to create the 
# induced strain in the LISA arms
 
class AntennaPatternFunctions:
    def __init__(self, frame):
        '''Instantiate with a particular SSB frame. Call in the same way we
        did LinkUnitVector to get the antenna pattern functions corresponding to the 
        plus and cross polarizations.'''

        self.frame = frame

    def __call__(self, i, j):
        er, etheta, ephi, khat, uhat, vhat = self.frame() 
        nhat = LinkUnitVector()
        
        xiplus = np.dot(uhat, nhat(i, j))**2 - np.dot(vhat, nhat(i, j))**2
        xicross = 2*np.dot(uhat, nhat(i, j))*np.dot(vhat, nhat(i, j))

        return xiplus, xicross 

antennta_pattern = AntennaPatternFunctions(frame)

# Before moving to the induced strain we first need a waveform. We use the 
# FastSchwarzschildEccentricFlux from FEW. 

gen_wave = GenerateEMRIWaveform("FastSchwarzschildEccentricFlux")

# parameters
T = 0.1             # years (total length of waveform)
dt = 15.0           # seconds (space between points)
M = 1e6             # mass of the central black hole
a = 0.1             # spin of central black hole (ignored for Schwarzschild)
mu = 1e1            # mass of the compact object
p0 = 12.0           # initial semi-latus rectum
e0 = 0.2            # initial eccentricity
x0 = 1.0            # initial cosine of the inclination (ignored for Schwarzschild)
qK = 0.2            # polar spin angle in ellicptic coordinates
phiK = 0.2          # azimuthal viewing angle (azimuthal spin angle in ellicptic coordinates)
qS = 0.3            # polar sky angle (sky location polar angle in ellipctic coordinates)
phiS = 0.3          # azimuthal viewing angle (sky location azimuthal angle in ellipctic coordinates)
dist = 1.0          # Luminosity distance in Gpc
Phi_phi0 = 1.0      # initial phase for Phi_phi
Phi_theta0 = 2.0    # initial phase for Phi_theta
Phi_r0 = 3.0        # initial phase for Phi_r

h = gen_wave(
    M,
    mu,
    a,
    p0,
    e0,
    x0,
    dist,
    qS,
    phiS,
    qK,
    phiK,
    Phi_phi0,
    Phi_theta0,
    Phi_r0,
    T=T,
    dt=dt,
)

# Split h into its plus and cross components
hplus = np.array(h.real)
hcross = np.array(h.imag)


# Now we may construct induced strain

class InducedDeformation:
    def __init__(self, hplus, hcross, frame, dt):

        '''Instantiate with the plus and cross components of the waveform SSB frame
        and with the sampling frequency dt. Sampling frequency used in the interpolation 
        step turning the strain arrays into actual functions of time. 
        Call with t an ARRAY of time values which you will see labeled as observation time in the 
        code completing the goal of interpolation. i and j are called in order to reference 
        antenna pattern functions - physically, down which arm the signal travels. 
        Note also that interp1d is used without extrapolation so any value below 
        zero comes back as zero.'''

        self.hplus = hplus
        self.hcross = hcross
        self.antenna_pattern = AntennaPatternFunctions(frame)
        self.dt = dt
    

    def __call__(self, t, i, j):
        xiplus, xicross = self.antenna_pattern(i, j)
        
        t_values = np.arange(len(self.hplus)) * self.dt # account for IF signals merges within T
        
        hplus_interp = interp.interp1d(t_values, self.hplus, bounds_error=False, fill_value=0.0)
        hcross_interp = interp.interp1d(t_values, self.hcross, bounds_error=False, fill_value=0.0)
    
        cap_h = hplus_interp(t) * xiplus + hcross_interp(t) * xicross
        
        return cap_h 
    
cap_h = InducedDeformation(hplus, hcross, frame, dt)

# the following is the array we would use in calling cap_h - the total time in seconds spaced by dt
obs_time = np.arange(0.0, T * YRSID_SI, dt) 
obs_length = T * 3.154e7 / dt

# Now we can construct the frequency shift again under the equal arm approximation which 
# in turn assumes that the spacecraft moves slowly compared to the propagation timescale
# note that this assumption could become problematic for high frequency waves

class FrequencyShift:
    def __init__(self, hplus, hcross, frame, dt):

        '''Instantiate with the SSB frame, the plus and cross components of the waveform and dt.
        Call with the observation time array t and the arm traveled (i, j) as in LinkUnitVector 
        as well as the the positions of the ith and jth spacecraft from SpacecraftPosition with 
        (u, v). i and j and u and v will not be same numbers since LinkUnitVector is 1-indexed 
        and SpacecraftPosition is 0-indexed. Note again that interp1d is used as it is in 
        InducedDeformation. Spacecraft positions also need to be interpolated too and is done
        in the same way as InducedStrain.'''
        
        self.hplus = hplus
        self.hcross = hcross
        self.frame = frame
        self.dt = dt
        self.induced_deformation = InducedDeformation(hplus, hcross, frame, dt)

    def __call__(self, t, i, j, u, v):
        er, etheta, ephi, khat, uhat, vhat = self.frame()
        positionu = positions(u)
        positionv = positions(v)
        cap_h = self.induced_deformation

        t_values = np.arange(len(positionu)) * self.dt

        positionu_interp = [interp.interp1d(t_values, positionu[:, k], bounds_error=False, fill_value=0.0)(t) for k in range(3)]
        positionv_interp = [interp.interp1d(t_values, positionv[:, k], bounds_error=False, fill_value=0.0)(t) for k in range(3)]

        positionu_interp = np.array(positionu_interp)
        positionv_interp = np.array(positionv_interp)

        t0 = t - proplen / c - (np.dot(khat, positionv_interp)) /c
        t00 = t - (np.dot(khat, positionu_interp)) / c

        cap_h0 = cap_h(t0, i, j)
        cap_h00 = cap_h(t00, i, j)
        
        
        y_ij = 1 / (2 * (1 - np.dot(khat, nhat(i,j)))) * (cap_h0 - cap_h00)

        return y_ij

y_ij = FrequencyShift(hplus, hcross, frame, dt)

# Next we must create the michelson combinations which we need to construct
# the TDI variables. These combinations are based on the frequency shift and 
# the delay operator. 

class DelayOperator:    
    def __init__(self, hplus, hcross, frame, dt):

        '''Instantiate with the SSB frame and plus and cross components of the waveform and sampling frequency dt.
        Call in the same way as FrequencyShift with the addition of the number of indices
        n. Note that this is valid in the equal arm approximation where the delay 
        operator amounts to the subtracting L/c from the time a number of times 
        which is equal to n - 1.'''

        self.hplus = hplus
        self.hcross = hcross
        self.frame = frame
        self.dt = dt
        self.freqshift = FrequencyShift(hplus, hcross, frame, dt)

    def __call__(self, t, i, j, u, v, n):
        tdelay = t - proplen / c * (n - 1)
        y_ijdelay = self.freqshift(tdelay, i, j, u, v)
        
        return y_ijdelay

cap_d = DelayOperator(hplus, hcross, frame, dt)

# Now construct X1, the first generation Michelson combination. These are then used to 
# create the desired uncorrelated TDI variables A, E and T - the combinations of X, Y, Z 
# that are used to cancel laser noise in the equal arm approximation. 

class MichelsonCombinationGen1:
    def __init__(self, hplus, hcross, frame, dt):

        '''Instantiate in the same way as FrequencyShift. Three rotations creates 
        X, Y, Z which are then used to create A, E and T. Call with the time observation time array t.'''

        self.freqshift = FrequencyShift(hplus, hcross, frame, dt)
        self.delay = DelayOperator(hplus, hcross, frame, dt)
    
    def __call__(self, t):
        y_ij = self.freqshift
        y_ijdelay = self.delay
        
        X_1 = y_ij(t, 1, 3, 0, 2) + y_ijdelay(t, 3, 1, 2, 0, 2) + y_ijdelay(t, 1, 2, 0, 1, 3) 
        + y_ijdelay(t, 2, 1, 1, 0, 4) - (y_ij(t, 1, 2, 0, 1) + y_ijdelay(t, 2, 1, 1, 0, 2)
                                         + y_ijdelay(t, 1, 3, 0, 2, 3)
                                         + y_ijdelay(t, 3, 1, 2, 0, 4))
        
        Y_1 = y_ij(t, 2, 1, 1, 0) + y_ijdelay(t, 1, 2, 0, 1, 2) + y_ijdelay(t, 2, 3, 1, 2, 3)
        + y_ijdelay(t, 3, 2, 2, 1, 4) - (y_ij(t, 2, 3, 1, 2) + y_ijdelay(t, 3, 2, 2, 1, 2)
                                         + y_ijdelay(t, 2, 1, 1, 0, 3)
                                         + y_ijdelay(t, 1, 2, 0, 1, 4))
        
        Z_1 = y_ij(t, 3, 2, 2, 1) + y_ijdelay(t, 2, 3, 1, 2, 2) + y_ijdelay(t, 3, 1, 2, 0, 3)
        + y_ijdelay(t, 1, 3, 0, 2, 4) - (y_ij(t, 3, 1, 2, 0) + y_ijdelay(t, 1, 3, 0, 2, 2)
                                         + y_ijdelay(t, 3, 2, 2, 1, 3)
                                         + y_ijdelay(t, 2, 3, 1, 2, 4))
        
        A_1 = 1 / np.sqrt(2) * (Z_1 - X_1)
        E_1 = 1 / np.sqrt(6) * (X_1 - 2 * Y_1 + Z_1)
        T_1 = 1 / np.sqrt(3) * (X_1 + Y_1 + Z_1)

        return X_1, Y_1, Z_1, A_1, E_1, T_1


uncorrelated_tdi_gen1 = MichelsonCombinationGen1(hplus, hcross, frame, dt)

# Now for the second generation Michelson combinations. 

class MichelsonCombinationGen2:
    def __init__(self, hplus, hcross, frame, dt):

        '''Instantiate in the same way as the first generation Michelson combinations.
        Calling with t retuns the second generation versions of the AET variables.'''

        '''01/16/25 added dt to the class which is necessary for 1st gen michelson combination'''

        self.freqshift = FrequencyShift(hplus, hcross, frame, dt)
        self.delay = DelayOperator(hplus, hcross, frame, dt)
        self.uncorrelated_tdi_gen1 = MichelsonCombinationGen1(hplus, hcross, frame, dt)
    
    def __call__(self, t):
        y_ij = self.freqshift
        y_ijdelay = self.delay
        X_1, Y_1, Z_1, A_1, E_1, T_1 = self.uncorrelated_tdi_gen1(t)
        
        X_2 = X_1 + y_ijdelay(t, 1, 2, 0, 1, 5) + y_ijdelay(t, 2, 1, 1, 0, 6) 
        + y_ijdelay(t, 1, 3, 0, 2, 7) + y_ijdelay(t, 3, 1, 2, 0, 8) - (y_ijdelay(t, 1, 3, 0, 2, 5)
                                                        + y_ijdelay(t, 3, 1, 2, 0, 6)
                                                        + y_ijdelay(t, 2, 1, 0, 1, 7)
                                                        + y_ijdelay(t, 2, 1, 1, 0, 8))
        
        Y_2 = Y_1 + y_ijdelay(t, 2, 3, 1, 2, 5) + y_ijdelay(t, 3, 2, 2, 1, 6) 
        + y_ijdelay(t, 2, 1, 1, 0, 7) + y_ijdelay(t, 1, 2, 0, 1, 8) - (y_ijdelay(t, 2, 1, 1, 0, 5)
                                                        + y_ijdelay(t, 1, 2, 0, 1, 6)
                                                        + y_ijdelay(t, 2, 3, 1, 2, 7)
                                                        + y_ijdelay(t, 3, 2, 2, 1, 8))
        
        Z_2 = Z_1 + y_ijdelay(t, 3, 1, 2, 0, 5) + y_ijdelay(t, 1, 3, 0, 2, 6) 
        + y_ijdelay(t, 3, 2, 2, 1, 7) + y_ijdelay(t, 2, 3, 1, 2, 8) - (y_ijdelay(t, 3, 2, 2, 1, 5)
                                                        + y_ijdelay(t, 2, 3, 1, 2, 6)
                                                        + y_ijdelay(t, 3, 1, 2, 0, 7)
                                                        + y_ijdelay(t, 1, 3, 0, 2, 8))
        
        A_2 = 1 / np.sqrt(2) * (Z_2 - X_2)
        E_2 = 1 / np.sqrt(6) * (X_2 - 2 * Y_2 + Z_2)
        T_2 = 1 / np.sqrt(3) * (X_2 + Y_2 + Z_2)

        return X_2, Y_2, Z_2, A_2, E_2, T_2
    

uncorrelated_tdi_gen2 = MichelsonCombinationGen2(hplus, hcross, frame, dt)
X_2, Y_2, Z_2, A_2, E_2, T_2 = uncorrelated_tdi_gen2(obs_time)

# Finally, check the results with LAT's EMRITDIWaveform


#### LAT's EMRITDIWaveform ####
# sampling_frequency = 1 / dt
# t0 = 20000.0
# # order of the langrangian interpolation
# order = 25

# orbit = EqualArmlengthOrbits()

# # 1st or 2nd or custom (see docs for custom)
# tdi_gen = "2nd generation"

# index_lambda = 8
# index_beta = 7

# tdi_kwargs = dict(
#     order=order, tdi=tdi_gen, tdi_chan="AET",
# )

# emri_lisa = ResponseWrapper(
#     gen_wave,
#     T,
#     dt,
#     index_lambda,
#     index_beta,
#     t0=t0,
#     flip_hx=True,  # set to True if waveform is h+ - ihx
#     remove_sky_coords=False,  # True if the waveform generator does not take sky coordinates
#     is_ecliptic_latitude=False,  # False if using polar angle (theta)
#     remove_garbage=True,  # removes the beginning of the signal that has bad information
#     orbits=orbit,
#     **tdi_kwargs,
# )

# # compute
# AET = emri_lisa(
#     M,
#     mu,
#     a,
#     p0,
#     e0,
#     x0,
#     dist,
#     qS,
#     phiS,
#     qK,
#     phiK,
#     Phi_phi0,
#     Phi_theta0,
#     Phi_r0,
# )

# fewA = AET[0]; fewE = AET[1]; fewT = AET[2]


# Plot our A channel against that produced by FEW. Note that Discrepancy in length of the two A channels 
# is due to the t0 response kwarg whose default is 2000 points with 15s sampling frequency this means 30000 
# seconds are taken off the beginning and end of the few signal 
# plt.plot(A_2[1951:2500], label='My A Channel', marker='o')
# plt.plot(8.0*fewA[:550], label='few A Channel', marker='x')

# plt.xlabel('Index')
# plt.ylabel('A Channel')
# plt.legend(loc='lower right')

# plt.show()

