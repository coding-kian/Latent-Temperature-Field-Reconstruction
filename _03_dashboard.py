from _01_simulation import run_sim, reconstruct_temperature_field
from _02_sensors import place_sensors, create_sensor_outputs
import numpy as np, matplotlib.pyplot as plt


def sensor_overlay(ax, sensor_xy, active_mask_t):
    sx, sy = sensor_xy[:, 0], sensor_xy[:, 1]
    ax.scatter(sx[active_mask_t], sy[active_mask_t], s=25, marker="o", edgecolors="k", facecolors="none") # can see temperatures underneath the sensors
    ax.scatter(sx[~active_mask_t], sy[~active_mask_t], s=40, marker="x") # ~tilde, just selects all of the false values from the masks


def dashboard(true_temp, recon_temp, sensor_xy, sensor_mask, target_temp, time_step):
    final_index = recon_temp.shape[0] -1
    frame_index = range(0,final_index+1) # adjust this if using time multiplier

    above = recon_temp > target_temp
    ramp_max = np.abs(np.diff(recon_temp, axis=0)).reshape(final_index,-1).max(axis=1)/time_step
    metrics = {"max_temp": recon_temp.reshape(final_index+1, -1).max(axis=1), # maximum overall tempertaure
        "hot_area_frac": above.mean(axis=(1, 2))*100, # average temperature above target temperature
        "time_above_target_temp": above.sum(axis=0), # total steps above the target temperature
        "max_ramp_rate": np.concatenate(([0.0], ramp_max)), # maximum rate of change of temperature
        "sensor_active_rate": sensor_mask.mean(axis=1)*100, # average percentage of actors receiving infomation
        "rmse": np.sqrt(np.mean((true_temp-recon_temp)**2, axis=(1, 2))) }# difference between real and reconstructed temperatures
    
    fig, axes = plt.subplots(3, 3, figsize=(16, 9))
    axes = axes.ravel()

    heatmaps = [(true_temp[final_index], "True heat"),
        (recon_temp[final_index], "Reconstructed Readings"),
        (metrics["time_above_target_temp"], "Steps Above Target Temperature")]
    
    time_series = [(metrics["max_temp"], "Hotspot max temp", "°C"),
        (metrics["hot_area_frac"], f"Area > {target_temp}°C", "%"),
        (metrics["max_ramp_rate"], "Ramp rate", "°C/second"),
        (metrics["rmse"], "RMSE over time", "RMSE"),
        (metrics["sensor_active_rate"], "Sensor active rate", "% active")]

    summary_lines = f"""
    Average RMSE: {metrics["rmse"].mean():.3f}
    Final RMSE: {metrics["rmse"][final_index]:.3f}
    Final Maximum Temperature: {metrics['max_temp'][final_index]:.2f} °C
    Final Area above {target_temp}°C: {metrics['hot_area_frac'][final_index]:.2f}%"""
    
    for ax, (vals, t) in zip(axes[:3], heatmaps):
        plt.colorbar(ax.imshow(vals, origin="upper"), ax=ax)
        ax.set_title(t); ax.axis("off")
    
    for ax, (vals, t, y) in zip(axes[3:8], time_series):
        ax.plot(frame_index, vals)
        ax.set_title(t); ax.set_xlabel("Step"); ax.set_ylabel(y); ax.grid(True)

    sensor_overlay(axes[1], sensor_xy, sensor_mask[final_index])
    axes[8].axis("off"); axes[8].text(0, 0.95, summary_lines, va="top")

    plt.suptitle("Latent Temperature Reconstruction")
    plt.tight_layout()
    

if __name__ == "__main__":
    rng = np.random.default_rng(10)
    # since this is a toy simulation, you have to define your own cell_size ratio IRL and time_step ratio IRL
    cell_size = 1 # careful with  exploding gradient in PDE if cellsze <1
    time_step = 1 
    height, width = 64, 64
    total_sensors = 50
    num_steps = 200
    ambient_temp = 20 # temperature outside of the environment
    target_temp = 20 # target tempature inside the enivonrment
    ambient_grid = np.full((height, width), ambient_temp, dtype=np.float64)
    diffusion_rate = 0.1 # how quick heat spreads into cooler regions
    relaxation_rate = 0.01 # how quick heat moves back to ambient temperature, so how quick heat passes past the boundary to open environment
    strength_rate = 0.02 # how quickly the heat source pushes tarwards the environment target temp, like relation but inside environment
    trust_rate = 0.7 # 30% towards PDE prediction and 70% towards sensor reading. 
    hotspots = [(20, 20, 60, 6), (45, 40, 45.0, 10)] # locations of temperature source, x_coords, y_coords, target_temp, radius
    noise_std, sensor_invalid, spike_probabilty, spike_scale = 0.5, 0.15, 0.03, 6

    true_temp = run_sim(ambient_grid, num_steps, ambient_temp, diffusion_rate, relaxation_rate, strength_rate, hotspots, cell_size, time_step) 
    sensor_location = place_sensors(rng, height, width, total_sensors)
    sensor_vals, sensor_mask = create_sensor_outputs(true_temp, sensor_location, (rng, noise_std, sensor_invalid, spike_probabilty, spike_scale))
    recon_temp = reconstruct_temperature_field(ambient_grid, sensor_location, sensor_vals, sensor_mask,
        ambient_temp, diffusion_rate, relaxation_rate, trust_rate, cell_size, time_step)

    dashboard(true_temp, recon_temp, sensor_location, sensor_mask, target_temp, time_step)
    plt.show()
