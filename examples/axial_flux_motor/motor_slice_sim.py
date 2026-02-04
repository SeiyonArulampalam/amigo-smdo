import subprocess
import utils
import numpy as np
import matplotlib.pyplot as plt


slide_numbers = [0, 50, 80, 100, 120]
n = len(slide_numbers)
force_vals = np.zeros(n)
iron_loss_vals = np.zeros(n)
load_loss_vals = np.zeros(n)
cu_loss_vals = np.zeros(n)
torque_vals = np.zeros(n)
efficiency_vals = np.zeros(n)
Pout_vals = np.zeros(n)

# Initialize mesh
subprocess.run(
    [
        "python",
        "model.py",
        "--mesh",
        "--build",
        "--slide_number",
        str(0),
        "--radial_slice_mean_diam",
    ],
)

# Loop through each slide number
for i, slide in enumerate(slide_numbers):
    print(f"\nIteration: {i}")
    result = subprocess.run(
        [
            "python",
            "model.py",
            "--mesh",
            "--build",
            "--slide_number",
            str(slide),
            # "--plot_potential",
            # "--plot_flux",
            # "--plot_loss",
            "--radial_slice_mean_diam",
        ],
        capture_output=True,  # capture stdout and stderr
        text=True,  # decode bytes to string
    )

    # The full printed output from model.py is in result.stdout
    for line in result.stdout.splitlines():
        if line.startswith("Force (N):"):
            force = abs(float(line.split(":")[1]))
            print(f"Force for slide {slide}: {force}")
            force_vals[i] = force
        elif line.startswith("Total Iron Loss (W):"):
            loss = float(line.split(":")[1])
            print(f"Total Iron Loss (W): {loss}")
            iron_loss_vals[i] = loss
        elif line.startswith("Torque (N.m):"):
            torque = float(line.split(":")[1])
            print(f"Torque (N.m): {torque}")
            torque_vals[i] = np.abs(torque)
        elif line.startswith("Efficiency (%):"):
            eta = float(line.split(":")[1])
            print(f"Efficiency (%): {eta}")
            efficiency_vals[i] = eta
        elif line.startswith("Total Load Loss (W):"):
            load_loss = float(line.split(":")[1])
            print(f"Total Load Loss (W): {load_loss}")
            load_loss_vals[i] = load_loss
        elif line.startswith("Copper Loss (W):"):
            cu_loss = float(line.split(":")[1])
            print(f"Copper Loss (W): {cu_loss}")
            cu_loss_vals[i] = cu_loss
        elif line.startswith("Output Power (W):"):
            Pout_loss = float(line.split(":")[1])
            print(f"Output Power (W): {Pout_loss}")
            Pout_vals[i] = Pout_loss

np.save("Figures/force_data_slice.npy", force_vals)
np.save("Figures/torque_data_slice.npy", torque_vals)
np.save("Figures/efficiency_data_slice.npy", efficiency_vals)

forces = np.load("Figures/force_data_slice.npy")
torques = np.load("Figures/torque_data_slice.npy")
eta = np.load("Figures/efficiency_data_slice.npy")

avg_force = np.average(forces)
avg_torque = np.average(torques)
avg_eta = np.average(eta)
avg_iron_loss = np.average(iron_loss_vals)
avg_total_load_loss = np.average(load_loss_vals)
avg_cu_loss = np.average(cu_loss_vals)
avg_Pout = np.average(Pout_vals)

print()
print(f"Avg. Force (N): {avg_force:.4f}")
print(f"Avg. Torque (N.m): {avg_torque:.4f}")
print(f"Avg. Iron Loss (W): {avg_iron_loss:.4f}")
print(f"Avg. Cu Loss (W): {avg_cu_loss:.4f}")
print(f"Avg. Total Load Loss (W): {avg_total_load_loss:.4f}")
print(f"Avg. Pout (W): {avg_Pout:.4f}")
print(f"Avg. Efficiency (%): {avg_eta:.4f}")

# utils.plot_data(
#     x=slide_numbers,
#     y=forces,
#     title=f"Avg. Force: {avg_force:.4f}",
#     xlabel="Slide Number",
#     ylabel=r"Force (N)",
#     color="red",
#     save_loc="Figures/force_slice.png",
# )
# utils.plot_data(
#     x=slide_numbers,
#     y=torques,
#     title=f"Avg. Torque: {avg_torque:.4f}",
#     xlabel="Slide Number",
#     ylabel=r"$\tau$ (N.m)",
#     color="blue",
#     save_loc="Figures/torque_slice.png",
# )

# utils.create_gif(
#     folder="/Users/seiyonarulampalam/git/amigo/examples/axial_flux_motor/Figures/Potential",
#     gif_name="potential.gif",
#     duration=0.5,
#     slice_start=0,
#     slice_end=niters,
# )

# utils.create_gif(
#     folder="/Users/seiyonarulampalam/git/amigo/examples/axial_flux_motor/Figures/Flux",
#     gif_name="flux.gif",
#     duration=0.5,
#     slice_start=0,
#     slice_end=niters,
# )

# utils.create_gif(
#     folder="/Users/seiyonarulampalam/git/amigo/examples/axial_flux_motor/Figures/Loss_Density",
#     gif_name="loss.gif",
#     duration=0.5,
#     slice_start=0,
#     slice_end=niters,
# )

plt.show()
