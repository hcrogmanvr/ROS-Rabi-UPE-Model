
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

def model(t, y, params):
    P, R, u, v, w = y

    r = params["r"]
    Pmax = params["Pmax"]
    eta_clear = params["eta_clear"]
    kR = params["kR"]
    gammaR = params["gammaR"]
    alpha = params["alpha"]
    KR = params["KR"]
    hill_n = params["hill_n"]
    Delta = params["Delta"]
    Gamma1 = params["Gamma1"]
    Gamma2 = params["Gamma2"]

    dPdt = r * P * (1 - P / Pmax) - eta_clear * P
    dRdt = kR * P - gammaR * R

    Omega = alpha * (R**hill_n) / (KR**hill_n + R**hill_n + 1e-12)

    dudt = -Delta * v - Gamma2 * u
    dvdt = Delta * u - Omega * w - Gamma2 * v
    dwdt = Omega * v - Gamma1 * (w + 1)

    return [dPdt, dRdt, dudt, dvdt, dwdt]

params = {
    "r": 0.9,
    "Pmax": 1.0,
    "eta_clear": 0.25,
    "kR": 1.4,
    "gammaR": 0.55,
    "alpha": 2.5,
    "KR": 0.35,
    "hill_n": 2.0,
    "Delta": 0.0,
    "Gamma1": 0.35,
    "Gamma2": 0.45,
    "eta_det": 1.0
}

y0 = [0.02, 0.0, 0.0, 0.0, -1.0]
t_span = (0, 25)
t_eval = np.linspace(t_span[0], t_span[1], 1500)

sol = solve_ivp(
    fun=lambda t, y: model(t, y, params),
    t_span=t_span,
    y0=y0,
    t_eval=t_eval,
    method="RK45",
    rtol=1e-8,
    atol=1e-10
)

t = sol.t
P, R, u, v, w = sol.y

rho_ee = (1 + w) / 2
I = params["eta_det"] * params["Gamma1"] * rho_ee
Omega = params["alpha"] * (R**params["hill_n"]) / (
    params["KR"]**params["hill_n"] + R**params["hill_n"] + 1e-12
)

plt.figure(figsize=(7, 4.5))
plt.plot(t, P, label="Infection stimulus, P(t)")
plt.plot(t, R, label="ROS concentration, R(t)")
plt.xlabel("Time")
plt.ylabel("Normalized amplitude")
plt.title("Infection-driven ROS dynamics")
plt.legend()
plt.tight_layout()
plt.show()

plt.figure(figsize=(7, 4.5))
plt.plot(t, Omega, label="ROS-driven excitation, Ω(t)")
plt.plot(t, I, label="Photon emission, I(t)")
plt.xlabel("Time")
plt.ylabel("Normalized amplitude")
plt.title("ROS-mediated quantum excitation and photon emission")
plt.legend()
plt.tight_layout()
plt.show()

plt.figure(figsize=(7, 4.5))
plt.plot(t, rho_ee, label="Excited-state population, ρee(t)")
plt.plot(t, u, label="Coherence variable, u(t)")
plt.plot(t, v, label="Coherence variable, v(t)")
plt.xlabel("Time")
plt.ylabel("Quantum-state variables")
plt.title("Open quantum dynamics of the emitting subsystem")
plt.legend()
plt.tight_layout()
plt.show()

plt.figure(figsize=(6, 4.5))
plt.plot(R, I)
plt.xlabel("ROS concentration, R(t)")
plt.ylabel("Photon emission, I(t)")
plt.title("Nonlinear ROS-to-photon-emission mapping")
plt.tight_layout()
plt.show()
