import numpy as np
import matplotlib.pyplot as plt
import niceplots

plt.style.use(niceplots.get_style())
plt.rcParams["font.family"] = "DejaVu Sans"

current = [
    20,
    25,
    30,
    35,
    40,
    45,
    50,
    55,
    60,
]

ansys_20kW_torque = [
    11.360,
    14.186,
    16.992,
    19.817,
    22.590,
    25.328,
    28.032,
    30.691,
    33.301,
]

ansys_20kW_eff = [
    94.283,
    93.945,
    93.475,
    92.886,
    92.265,
    91.610,
    90.933,
    90.236,
    89.520,
]

fea_20kW_torque = [
    11.6558,
    14.5729,
    17.4891,
    20.4044,
    23.3189,
    26.2325,
    29.1453,
    32.0571,
    34.9681,
]

fea_20kW_eff = [
    95.1189,
    94.7318,
    94.2352,
    93.6800,
    93.0921,
    92.4855,
    91.8689,
    91.2474,
    90.6247,
]

analytic_20kW_torque = [
    9.54,
    11.87,
    14.14,
    16.41,
    18.88,
    20.94,
    23.08,
    25.18,
    27.23,
]

analytic_20kW_eff = [
    95.00,
    94.58,
    94.00,
    93.35,
    92.62,
    91.89,
    91.15,
    90.40,
    89.63,
]

fig, ax = plt.subplots(ncols=2, nrows=1, figsize=(10, 5))
ms = 8
# Torque
ax[0].plot(
    current,
    ansys_20kW_torque,
    "o-",
    color="#f26f6f",
    label="MotorCAD",
    markersize=ms,
)
ax[0].plot(
    current,
    fea_20kW_torque,
    "o-",
    color="#b96ff2",
    label="FEA",
    markersize=ms,
)
ax[0].plot(
    current,
    analytic_20kW_torque,
    "o-",
    color="#6faaf2",
    label="Analytic",
    markersize=ms,
)
ax[0].set_ylabel(r"$\tau$ (N.m)", fontweight="normal")
ax[0].set_xlabel(r"Current (A)", fontweight="normal")
ax[0].set_xlim(18, 62)
ax[0].set_ylim(8, 36)


# Efficiency
ax[1].plot(
    current,
    ansys_20kW_eff,
    "o-",
    color="#f26f6f",
    label="MotorCAD",
    markersize=ms,
)
ax[1].plot(
    current,
    fea_20kW_eff,
    "o-",
    color="#b96ff2",
    label="FEA",
    markersize=ms,
)
ax[1].plot(
    current,
    analytic_20kW_eff,
    "o-",
    color="#6faaf2",
    label="Analytic",
    markersize=ms,
)
ax[1].set_ylabel(r"$\eta$ (%)", fontweight="normal")
ax[1].set_xlabel(r"Current (A)", fontweight="normal")
ax[1].set_xlim(18, 62)
ax[1].set_ylim(87, 100)
ax[1].legend()

fig.tight_layout()
plt.savefig("20kW_validation.png")
plt.show()
