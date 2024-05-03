import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import pandas as pd
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.max_colwidth', None)

MAHA = 'maha'
BASELINE = 'baseline'
# reading csv file 
df = pd.read_csv("maharesults.csv")
# print(df.head())
csv_rows = []
csv_rows.append(",".join(["scenario", "scenario_type_intensity", "route_change", "collisions_vehicle", "collisions_layout"]))
scenarios = ['haze', 'shade', 'rain']
for scenario in scenarios:
    for scenario_intensity in range(0, 10):
        route_change = df.loc[(df['scenario_type'] == scenario) & (df['scenario_type_intensity'] == scenario_intensity) & (pd.notna(df['outside_route_lanes']))].describe().loc['count'][0]
        collisions_vehicle = df.loc[(df['scenario_type'] == scenario) & (df['scenario_type_intensity'] == scenario_intensity) & (pd.notna(df['collisions_vehicle']))].describe().loc['count'][0]
        collisions_layout = df.loc[(df['scenario_type'] == scenario) & (df['scenario_type_intensity'] == scenario_intensity) & (pd.notna(df['collisions_layout']))].describe().loc['count'][0]
        csv_rows.append(",".join([str(scenario), str(scenario_intensity), str(route_change/0.11), str(collisions_vehicle/0.11), str(collisions_layout/0.11)]))
        # print(scenario, scenario_intensity, route_change/0.11, collisions_vehicle/0.11, collisions_layout/0.11)
    #print(df.loc[:,['scenario_num', 'scenario_type_intensity', 'collisions_vehicle', 'collisions_layout', 'outside_route_lanes']]) 

with open(BASELINE + 'aggregate.csv', 'w') as result_file:
    result_file.write("\n".join(csv_rows))