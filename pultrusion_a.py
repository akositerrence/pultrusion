import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = ["Times New Roman"]

effective_composite_density = 1800
effective_resin_density = 1200
cp_effective = 900

# KINETIC PARAMETERS
kinetic_heating_rate = 175e3
a0 = 1e4
e = 6e4
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
T_platen_1 = 150 + 273.15
T_platen_2 = 200 + 273.15
T_platen_3 = 150 + 273.15
T_platen_4 = 40 + 273.15
L_total = L1 + L2 + L3 + L4

# INLET CONDITIONS
t_in = 50 + 273.15
alpha_in = 0.01

# TOTAL THERMAL RESISTANCE
r_tot = (steel_thickness/steel_k) + thermal_contact_resistance

def platen_temperatures(x):
    if x < L1:
        return T_platen_1
    if x < (L1 + L2):
        return T_platen_2
    if x < (L1 + L2 + L3):
        return T_platen_3
    else:
        return T_platen_4
    
def f_alpha(alpha):
    # KINETIC MODEL
    alpha = np.clip(alpha, 0.0, 0.999999)
    y = (1.0-alpha)**kinetic_exponent_n
    return y

def ode_rhs(x, y):
    T, alpha = y
    # KINETICS
    da_dx = ( (a0/pull_speed)* np.exp(-e/(r_gas*T)) ) * (f_alpha(alpha))
    
    # PLATEN HEAT GENERATION
    temperature_platen = platen_temperatures(x)
    q_ppp_platen = (perimeter/cross_sectional_area) * (temperature_platen - T) / r_tot
    
    # REACTION HEAT GENERATION
    q_ppp_reaction = (effective_resin_density*kinetic_heating_rate*pull_speed*da_dx)
    
    # TEMPERATURE ROC
    dt_dx = (q_ppp_platen + q_ppp_reaction) / (effective_composite_density * cp_effective * pull_speed)
    
    return [dt_dx, da_dx]

x_span = (0.0, L_total)
y_0 = [t_in, alpha_in]
x_eval = np.linspace(0, L_total, 500)

solution = solve_ivp(ode_rhs, x_span, y_0, t_eval=x_eval, max_step=0.001)

x = solution.t
temperatures = solution.y[0, :]
alphas = solution.y[1, :]

plt.figure()
plt.plot(x, temperatures)
plt.grid(True)
plt.xlabel("x (m)")
plt.ylabel("T (K)")
plt.show()

plt.figure()
plt.plot(x, alphas)
plt.grid(True)
plt.xlabel("x (m)")
plt.ylabel("α")
plt.show()