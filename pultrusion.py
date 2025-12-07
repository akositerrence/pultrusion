import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

effective_composite_density = 0.0
effective_resin_density = 0.0
cp_effective = 0.0

# KINETIC PARAMETERS
kinetic_heating_rate = 0.0
a0 = 0.0
e = 0.0
r_gas = 8.314
kinetic_exponent_n = 0.0

# PROBLEM GEOMETRY
cross_sectional_area = 0.0
perimeter = 0.0
steel_thickness = 0.0
steel_k = 0.0
thermal_contact_resistance = 0.0
pull_speed = 0.0

# HEATING ZONES (m)
L1 = 0.5
L2 = 0.5
L3 = 0.5
T_platen_1 = 0.0
T_platen_2 = 0.0
T_platen_3 = 0.0
L_total = L1 + L2 + L3

# INLET CONDITIONS
t_in = 0.0
alpha_in = 0.0

# TOTAL THERMAL RESISTANCE
r_tot = (steel_thickness/steel_k) + thermal_contact_resistance

def platen_temperatures(x):
    if x < L1:
        return T_platen_1
    if x < (L1 + L2):
        return T_platen_2
    else:
        return T_platen_3
    
def f_alpha(alpha):
    # KINETIC MODEL
    y = (1.0-alpha)**kinetic_exponent_n
    return y

def ode_rhs(T, alpha):
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

solution = solve_ivp(ode_rhs, x_span, y_0, t_eval=x_eval, max_step=0.1)

x = solution.t
temperatures = solution.y[0, :]
alphas = solution.y[1, :]

plt.figure
plt.plot(x, temperatures)
plt.xlabel("x")
plt.ylabel("T")
plt.show()