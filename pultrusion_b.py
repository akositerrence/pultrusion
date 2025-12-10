import numpy as np
from scipy.integrate import solve_bvp
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = ["Times New Roman"]

effective_composite_density = 1800
effective_resin_density = 1200
cp_effective = 900

# KINETIC PARAMETERS
kinetic_heating_rate = 175e3
# kinetic_heating_rate = 0
a0 = 1e4
e = 60e3
r_gas = 8.314
kinetic_exponent_n = 1.2

# PROBLEM GEOMETRY
side = 0.05
cross_sectional_area = side**2
perimeter = side*4
steel_thickness = 0.125
steel_k = 20.0
thermal_contact_resistance = 0.01
pull_speed = 0.01667

# HEATING ZONES (m)
L1 = 0.25
L2 = 0.25
L3 = 0.25
L4 = 0.25
T_platen_2 = 2 + 273.15
L_total = L1 + L2 + L3 + L4
steel_as = side * steel_thickness

# INLET CONDITIONS
t_in = 50 + 273.15
alpha_in = 0.01

# ENVIRONMENT CONDITIONS
t_inf = 20 + 273.15
hi, hl = 10.0, 10.0
a_end = side * steel_thickness

# TOTAL THERMAL RESISTANCE
r_tot = (steel_thickness/steel_k) + thermal_contact_resistance

def q_heater(x):
    q1 = 1000.0
    q2 = 1500.0
    q3 = 1000.0
    q4 = 500.0
    
    x = np.asarray(x)
    out = np.empty_like(x)
    for i, xi in enumerate(x):
        if xi < L1:
            out[i] = q1
        elif xi < L1 + L2:
            out[i] = q2
        elif xi < L1 + L2 + L3:
            out[i] = q3
        else:
            out[i] = q4
            
    return out
    
def f_alpha(alpha):
    # KINETIC MODEL
    alpha = np.clip(alpha, 0.0, 0.999999)
    y = (1.0-alpha)**kinetic_exponent_n
    return y

def ode_right(x, y):
    T = y[0, :]
    alpha = y[1, :]
    Tint = y[2, :]
    q = y[3, :]
    
    # KINETICS
    da_dx = ( (a0/pull_speed)* np.exp(-e/(r_gas*T)) ) * (f_alpha(alpha))
    
    # COMPOSITE 
    q_ppp_wall = (perimeter/ cross_sectional_area) * (Tint - T) / r_tot
    q_ppp_reaction = (effective_resin_density*kinetic_heating_rate*pull_speed*da_dx)
    dt_dx = (q_ppp_wall + q_ppp_reaction) / (effective_composite_density * cp_effective * pull_speed)
    
    # WALL
    dTint_dx = q
    dq_dx = ((perimeter/r_tot) * (Tint - T) - q_heater(x) ) / (steel_k * steel_as)
    
    return np.vstack([dt_dx, da_dx, dTint_dx, dq_dx])

def bc(y_i, y_l):
    T_i, a_i, Tint_i, q_i = y_i
    T_l, a_l, Tint_l, q_l = y_l
    bc1 = T_i - t_in
    bc2 = a_i - alpha_in
    bc3 = -steel_k * steel_as * q_i - hi * a_end * (Tint_i - t_inf)
    bc4 = -steel_k * steel_as * q_l - hl * a_end * (Tint_l - t_inf)
    
    return np.array([bc1, bc2, bc3, bc4])

x_stuff = np.linspace(0.0, L_total, 200)
y_guess = np.zeros((4, x_stuff.size))
y_guess[0, :] = t_in
y_guess[1, :] = alpha_in
y_guess[2, :] = T_platen_2
y_guess[3, :] = 0.0

solution = solve_bvp(ode_right, bc, x_stuff, y_guess)
x = solution.x
temperatures = solution.y[0, :]
alphas = solution.y[1, :]
temperatures_interface = solution.y[2, :]

plt.figure()
plt.plot(x, temperatures, label="Composite Temperature")
plt.plot(x, temperatures_interface, label="Die-Composite Interface Temperature")
plt.grid(True)
plt.xlabel("x (m)")
plt.ylabel("T (K)")
plt.legend()
plt.show()

plt.figure()
plt.plot(x, alphas, label="Degree of Cure α")
plt.grid(True)
plt.xlabel("x (m)")
plt.ylabel("α")
plt.legend()
plt.show()