
from astropy import units as u
from astropy.time import Time, TimeDelta
from astropy.coordinates import solar_system_ephemeris

from poliastro.twobody.propagation import propagate, cowell
from poliastro.ephem import build_ephem_interpolant
from poliastro.core.elements import rv2coe

from poliastro.constants import rho0_earth, H0_earth
from poliastro.core.perturbations import atmospheric_drag_exponential, atmospheric_drag, \
    third_body, J2_perturbation
from poliastro.bodies import Earth, Moon
from poliastro.core.util import jit
import numpy as np



# parameters of the atmosphere
H0       = H0_earth.to(u.km).value
rho0     = rho0_earth.to(u.kg / u.km ** 3).value  # kg/km^3

# parameters of the body
C_D      = 2.07  # dimentionless (any value would do)
A        = ((4000) * (u.m ** 2)).to(u.km ** 2).value  # km^2
m        = 450000 * u.kg # kg
A_over_m = A/m # km^2/kg
B        = C_D * A_over_m

# Params for J2
J2_params = {
    'J2'       : Earth.J2.value,
    'R'        : Earth.R.to(u.km).value,
}

# Params for atmospheric_drag_exponential
atm_drag_exp_params = {
    'C_D'      : C_D,
    'A_over_m' : A_over_m,
    'H0'       : H0,
    'rho0'     : rho0
}

#-----------------------------------------------------------------------------
# Boilerplate non keplerian perturbation
@jit
def a_d0(t0, state, k):
    return 0, 0, 0

def accel(t0, state, k):
    """Constant acceleration aligned with the velocity. """
    v_vec = state[3:]
    norm_v = (v_vec * v_vec).sum() ** 0.5
    return 1e-5 * v_vec / norm_v

#-----------------------------------------------------------------------------
from poliastro.core.propagation import func_twobody
@jit
def a_d1(t0, state, k):
    du_kep = func_twobody(t0, state, k)
    # ax, ay, az = accel(t0, state, k)
    # du_ad = np.array([0, 0, 0, ax, ay, az])
    du_ad = np.array([0, 0, 0, 0, 0, 0])
    return du_kep + du_ad

    # return J2_perturbation(t0, state, k, Earth.J2.value, Earth.R.to(u.km).value)


#-----------------------------------------------------------------------------
@jit
def a_d2(t0, state, k):
    
    return J2_perturbation(t0, state, k, J2_params["J2"], J2_params["R"]) + \
           atmospheric_drag_exponential(\
                t0, state, k, \
                atm_drag_exp_params["C_D"], \
                atm_drag_exp_params["A_over_m"], \
                atm_drag_exp_params["H0"], \
                atm_drag_exp_params["rho0"])

#-----------------------------------------------------------------------------
# @jit
# def a_d3(t0, state, k, J2, R, C_D, A, m, H0, rho0):
#     return J2_perturbation(t0, state, k, J2, R) + \
#            atmospheric_drag(t0, state, k, R, C_D, A, m, H0, rho0)


# # database keeping positions of bodies in Solar system over time
# solar_system_ephemeris.set("de432s")

# epoch = Time(
#     2454283.0, format="jd", scale="tdb"
# )  # setting the exact event date is important

# # create interpolant of 3rd body coordinates (calling in on every iteration will be just too slow)
# body_r = build_ephem_interpolant(
#     Moon, 28 * u.day, (epoch.value * u.day, epoch.value * u.day + 60 * u.day), rtol=1e-2
# )

#-----------------------------------------------------------------------------
#
#-----------------------------------------------------------------------------
a_d = a_d1