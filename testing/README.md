There are 11 scenarios: from 100 to 110. For each scenario, all three weathers are tested. For each weather, 10 degrees of intensity are tested, from 0 to 9 (param variable). The specific parameters are set as below:
```python
if weather == "shade":
    param *= 0.15

elif weather == "rain":
    param *= 150
    
elif weather == "haze":
    param *= 25
```

- `paramsweep_results_lead_slowdown_weather_baseline`: Only the images are modified. There should be plenty of accidents. Serves as a baseline.

- `paramsweep_results_lead_slowdown_weather_maha` Not only the images are modified. The vehicle also acts more cautiously if OOD is detected using maha distance. 
  - `control.throttle *= 0.7`: throttle is eased
  - `control.brake = float(desired_speed < 0.4 * 1.3 or (speed / desired_speed) > 1.05)`: makes braking more frequent
  - There should be less accidents. 

- There is another OOD metric that is being generated.

Need to compare some metrics to show that our OOD detection and fallback actually works. Compare the 
1. Accident rate
   - In results_summary, "outside_route_lanes" should contain something like "Agent collided against object with type=vehicle.tesla.model3 and id=201 at (x=191.44, y=-68.778, z=0.035)"
2. The number of times vehicle is outside of lanes
   - In results_summary, "collisions_vehicle" should contain something like "Agent went outside its route lanes for about 9.0 meters (17.27% of the completed route)"

        
            