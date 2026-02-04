import subprocess
import utils
import numpy as np
import matplotlib.pyplot as plt

N = 25
radial_slices = np.linspace(0.01, 0.99, N)
rot_nums = [0, 10, 20, 30, 40, 50, 60]
force_vals = np.zeros(N)
loss_vals = np.zeros(N)
cu_loss_vals = np.zeros(N)
for j, slice in enumerate(radial_slices):
    # Loop through each slice
    print(f"\nRadial Slice: {round(j, 4)}")
    force_slice = np.zeros_like(rot_nums)
    loss_slice = np.zeros_like(rot_nums)
    cu_loss_slice = np.zeros_like(rot_nums)
    for i, slide_num in enumerate(rot_nums):
        # Loop through each slide number
        result = subprocess.run(
            [
                "python",
                "model.py",
                "--build",
                "--mesh",
                "--slide_number",
                str(slide_num),
                "--radial_slice",
                str(round(slice, 4)),
            ],
            capture_output=True,  # capture stdout and stderr
            text=True,  # decode bytes to string
        )
        # print(result)

        # The full printed output from model.py is in result.stdout
        for line in result.stdout.splitlines():
            if line.startswith("Force:"):
                force = abs(float(line.split(":")[1]))
                print(f"\nForce for slide {slide_num}: {force}")
                force_slice[i] = force
            elif line.startswith("Total Iron Loss (W/m):"):
                loss = float(line.split(":")[1])
                print(f"\nTotal Iron Loss (W/m) for slide {slide_num}: {loss}")
                loss_slice[i] = loss
            elif line.startswith("Copper Loss (W/m):"):
                cu_loss = float(line.split(":")[1])
                print(f"\nCopper Loss (W/m) for slide {slide_num}: {cu_loss}")
                cu_loss_slice[i] = cu_loss

    # Compute the average torque for a slice and store it
    force_vals[j] = np.average(force_slice)
    loss_vals[j] = np.average(loss_slice)
    cu_loss_vals[j] = np.average(cu_loss_slice)
    print(f"Average Force in Slice {round(slice, 4)}: {force_vals[j]:.6f} N")
    print(f"Average Loss in Slice {round(slice, 4)}: {loss_vals[j]:.6f} W/m")
    print(f"Average Cu Loss in Slice {round(slice, 4)}: {cu_loss_vals[j]:.6f} W/m")

# Save the data
np.save("Figures/force_data.npy", force_vals)
np.save("Figures/loss_data.npy", loss_vals)
np.save("Figures/cu_loss_data.npy", cu_loss_vals)

# Load the data
force_data = np.load("Figures/force_data.npy")
loss_data = np.load("Figures/loss_data.npy")
cu_loss_data = np.load("Figures/cu_loss_data.npy")

# Plot the force
Ri = 80e-3  # mm
Ro = 150e-3  # mm
x_data = (Ro - Ri) * radial_slices + Ri
torque = np.trapezoid(y=force_data, x=x_data)
loss = np.trapezoid(y=loss_data, x=x_data)
cu_loss = np.trapezoid(y=cu_loss_data, x=x_data)

# Compute the efficiency of the design
omega = 1000 * np.pi / 30.0
P_out = torque * omega
P_total = P_out + cu_loss + loss
efficiency = (P_out / P_total) * 100
print(f"P_out: {P_out:.4f} W")
print(f"P_total: {P_total:.4f} W")
print(f"Efficiency: {efficiency:.4f} %")


fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(20, 5))
fig.suptitle(f"Motor Efficiency: {efficiency:.2f}%", fontsize=16, fontweight="bold")
utils.plot_data(
    fig,
    ax[0],
    x=x_data,
    y=force_data,
    title=f"Torque {torque:.2f} N.m",
    xlabel="Radial Location (m)",
    ylabel=r"Force (N)",
    color="#5266ff",
)

utils.plot_data(
    fig,
    ax[1],
    x=x_data,
    y=loss_data,
    title=f"Iron Loss {loss:.2f} W",
    xlabel="Radial Location (m)",
    ylabel=r"Loss (W/m)",
    color="#ff7252",
)

utils.plot_data(
    fig,
    ax[2],
    x=x_data,
    y=cu_loss_data,
    title=f"Copper Loss {cu_loss:.2f} W",
    xlabel="Radial Location (m)",
    ylabel=r"Cu Loss (W/m)",
    color="#a052ff",
)
plt.savefig(f"Figures/data_plot_{N}.png", dpi=800)
print(f"Torque: {torque:.4f} N.m")
print(f"Iron Loss: {loss:.4f} W")
print(f"Cu Loss: {cu_loss:.4f} W")


plt.show()
