There are 11 scenarios: from #100 to #110. For each scenario, all three weathers are tested. For each weather, 10 degrees of intensity are tested, from 0 to 9 (param variable). The specific parameters are set as below:
```python
if weather == "shade":
    param *= 0.15

elif weather == "rain":
    param *= 150
    
elif weather == "haze":
    param *= 25
```

- `paramsweep_results_lead_slowdown_weather_baseline`: Only the images are modified, no fallback mode activated.

- `paramsweep_results_lead_slowdown_weather_maha_mitigated` Not only the images are modified, the vehicle also acts more cautiously if OOD is detected using maha distance. 
  - `control.throttle *= 0.7`: throttle is eased
  - `control.brake = float(desired_speed < 0.4 * 1.3 or (speed / desired_speed) > 1.05)`: easier to trigger braking
            