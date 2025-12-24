import numpy as np

def place_sensors(rng, h, w, total_sensors):
    return np.stack([rng.integers(0, w, size=total_sensors), rng.integers(0, h, size=total_sensors)], axis=1) 

def fuzzy_sensors(current_temp, fuzzy_variables):
    """
    68% of readings will be within 1std if noise_std = 0.5 then ±0.5 °C.5
    noise_std means the range between the sensor is allowed to be incorrect
    sensor_invalid means probability the sensor fails to read
    spike_probabilty means chance of very bad reading
    spike_scale - the amount of std from mean the spike will be from mean, this makes MAD work well
    """
    rng, noise_std, sensor_invalid, spike_probabilty, spike_scale = fuzzy_variables
    total_sensors = current_temp.shape[0]
    mask = rng.random(total_sensors) > sensor_invalid
    spikes = rng.random(total_sensors) < spike_probabilty
    augmented_temp = current_temp + rng.normal(0.0, noise_std, size=total_sensors) # observed temperature after noise and curruption
    augmented_temp[spikes] = augmented_temp[spikes] + rng.normal(0.0, spike_scale*noise_std, size=spikes.sum())

    return augmented_temp*mask, mask


def create_sensor_outputs(true_temp, sensor_location, fuzzy_variables):
    sensor_vals, sensor_masks = [], []
    for step in range(true_temp.shape[0]):
        current_temp = true_temp[step][sensor_location[:,1], sensor_location[:,0]] # find real temperature at coords
        augmented_temp, mask = fuzzy_sensors(current_temp, fuzzy_variables)
        sensor_vals.append(augmented_temp); sensor_masks.append(mask) # mask is just a boolean with the same index as sensorvals
    return np.stack(sensor_vals,0), np.stack(sensor_masks,0)
