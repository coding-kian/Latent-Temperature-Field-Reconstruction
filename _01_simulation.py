import numpy as np

def add_heat_sources(temperature, hotspots, strength_rate): # re adjusts heat source every step. 
    h, w = temperature.shape
    yy, xx = np.ogrid[:h, :w] # open grid, so creates circular mask on vertorized single line. coordinate grid, rowindex then colindex
    for x_coords, y_coords, target_temp, radius in hotspots: # 
        mask = (yy - y_coords)**2 + (xx - x_coords)**2 <= radius**2
        # temperature[mask] += strength_rate*(target_temp - temperature[mask]) # move masked region toward target each step
        temperature[mask] = strength_rate*target_temp + (1-strength_rate)*temperature[mask]
    return temperature


def pde_temperature_prediction(temperature, ambient_temp, diffusion_rate, relaxation_rate, cell_size, time_step):
    """
    pde for the temperature diffusion. heat equation in respect to time: d(temperature)/d(time)
    Since ambient boundary set on edge cells, use pde prediction on the next one in edge cells. So just PDE on interior
    Physics based prediction step for the temperature instead of using neural networks uses PDE reconstruction. laplacian.
    
    padded edge temperature as a ghost boarder. to simulate external environment
    used Dirichlet method, we assume outside environment is a constant ambient temperature. allowing environment to move towards ambient temperature
    so it uses numerical methods (finite difference simulation) as it is not a closed form simulation
    If neumann method then inside environment would be isolated, meaning it would trap heat.
    """
    temperature = np.pad(temperature, 1, mode="constant", constant_values=ambient_temp)
    center, up, down, right, left = temperature[1:-1, 1:-1], temperature[:-2, 1:-1], temperature[2:,  1:-1], temperature[1:-1, 2:], temperature[1:-1, :-2]
    laplacian = (up+down+right+left-4*center)/cell_size**2 # laplacian, second derivative of temperature change, 5 point laplacian, 4 neigbhours

    return center+time_step*(diffusion_rate*laplacian - relaxation_rate*(center-ambient_temp)) # relaxation_rate is rate back to ambient


def run_sim(temperature, num_steps, ambient_temp, diffusion_rate, relaxation_rate, strength_rate, hotspots, cell_size, time_step):
    frames = [temperature.copy()]
    for _ in range(num_steps):
        temperature = pde_temperature_prediction(temperature, ambient_temp, diffusion_rate, relaxation_rate, cell_size, time_step)
        temperature = add_heat_sources(temperature, hotspots, strength_rate)
        frames.append(temperature.copy())
    return np.stack(frames, axis=0)


def reconstruct_temperature_field(temperature, sensor_location, sensor_vals, sensor_mask, ambient_temp,
    diffusion_rate, relaxation_rate, trust_rate, cell_size, time_step):
    # trust_rate = 0 ignores the sensor use PDE, if 1 ovewrites predicted PDE value with sensor value
    num_steps = sensor_vals.shape[0]
    frames = [temperature.copy()]
    sensor_x, sensor_y = sensor_location[:, 0], sensor_location[:, 1]

    for step in range(1, num_steps):
        temperature = pde_temperature_prediction(temperature, ambient_temp, diffusion_rate, relaxation_rate, cell_size, time_step)
        read_temperature = sensor_mask[step]
        # below uses trust_rate as likelihood of correctness
        residual = sensor_vals[step, read_temperature] - temperature[sensor_y[read_temperature], sensor_x[read_temperature]] # remaining temperature
        mad = np.median(np.abs(residual - np.median(residual))) + 1e-6 # median absolute deviation, std=1.4826×MAD, MAD better for outliers
        kept = np.where(read_temperature)[0][np.abs(residual) < 2.9652*mad] # decides if sensor or pde
        temperature[sensor_y[kept], sensor_x[kept]] = ((1-trust_rate)*temperature[sensor_y[kept], sensor_x[kept]]+trust_rate*sensor_vals[step, kept])

        frames.append(temperature.copy())

    return np.stack(frames, axis=0)