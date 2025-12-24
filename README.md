*Published 24/12/2025*
# Latent Temperature Field Reconstruction
The is a personal indepedent project exploring 2D temperature diffusion using the heat equation and reconstruction from noisy (fuzzy), unreliable sensor readings using a physics based PDE model instead of modern machine learning.

## Purpose
The purpose was to understand diffusion (heat spread), handle numerical boundaries and estimate accurate values with unreliable senor readings.

## Objective
Simulates heat diffusion on a 2D grid using the heat equation, while applying a continuous heat sources which inject heat at each time step. 
Uses Dirichlet boundary to model the outside environment as a fixed ambient temperature, done so via np.pad

## Sensor simulation
Randomly places chosen amoutn of sensors on grid, uses gaussian noise to misread (similar to real life) temperatures, and occasional outlier spikes

## Temperature Reconstruction
PDE estimation using physics prior value to reconstruct temperature fields, using statistics (MAD instead of STD) to reject which is the least reliable between PDE and sensor readings

## Dashboard output
True Heat: Actual abient temperature and the hotspots
Reconsutrcted Temperature: Where the sensors are placed and if they were active (o) or inactive (x) with the temperature shown behind active sensors
Steps above target temperature: Heatmap showing a boolean summation of the areas of the map which were above the target temperature of the environment
Hotspot max temp: Of the final index the maximum temperature Hottest point 
Area > {target_temp} : percentage of the map at each step above the target temperature
Ramp rate: rate of change in the environment over each step
RMSE over time: Random mean square error, this is the difference between the predicted PDE + sensor readings (reconstructed heat) and the true heat. Overtime it converges as the temperature also does
Sensor active rate: the toal amount of sensors which were active at each step
Report: Text output of RMSE Max & Mean, The final maximum temperature reading, and the final area above target_temp at the end of the simulation

## Files & Functoins
#### simulation.py
add_heat_sources: Places continuous hotspot to temperature map
pde_temperature_prediction: Uses the heat equation from prior temperature to estimate next.
run_sim: Generates groudn truth for the temperature field, ambient and hotspots.
reconstruct_temperature_field: Uses PDE & sensors to estimate temperature

#### sensors.py
place_sensors: places sensors randomly.
fuzzy_sensor: creates stochastic readings from the sensors.
create_sensor_outputs: Uses the fuzzy_sensor function to create sensor data over time

#### dashboard.py
sensor_overlay: Displays sensor locations and status (x & o)
dashboard: Generates plots and summary report

#### running.ipynb
Runs the full simulation and dashboard.py displaying the report.


## Motivations
Robustness of components in server rooms is one of the biggest challenges of modern life with the IOT and blockchain rapidly growing, the reliability and downtime of equiptment needs to be optmised, this is the reason I created this thought expierment. To show the potential way which known physics can be used to expeirment with noisy/unreliable sensors instead of using an AI black box solution. 

#### Benefits to field.
Some benefits of implemnting this idea into real world data centers include the improvement in longevity of components, by notifying serverroom technicians when components are in a potentially dangerous temperature zone, which overall can increase performance, and reduce failure rate. Also the improvement in effective sensor placement 

#### Potential Future Works
**V2**: Add Advection to simulate airflow, such as HVAC.
**V3**: Add Humidity and condensation during temperature changes on surfaces.
