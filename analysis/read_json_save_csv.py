import os
import json
import re
# This file tells us the number of values contained in the json at max
PREFIX = 'paramsweep_results_lead_slowdown_weather_'
BASELINE = 'baseline'
MAHA = 'maha_mitigated'
TEST_FILE_COMMON_NAME = "/results_summary.json"
TEST_FILE_ROOT = "../testing/"
PERCENT_SEARCH = re.compile(r'\(.*%')

def get_csv_file_string(test_results_dir):
    path = TEST_FILE_ROOT + PREFIX + test_results_dir
    dir_list = os.listdir(path)
    labels = ["scenario_num",\
                "scenario_type",\
                "scenario_type_intensity",\
                "collisions_vehicle",\
                "collisions_layout",\
                "outside_route_lanes",\
                "outside_route_lanes_average",\
                "meta_duration_game",\
                "meta_duration_system",\
                "meta_route_length",\
                "scores_score_composed",\
                "scores_score_penalty",\
                "scores_score_route",\
                "status"]
    csv_rows = []
    csv_rows.append(",".join(labels))
    scenarios = set()
    for file_name in dir_list:
        file_name_split = file_name.split('_')
        scenarios.add("_".join(file_name_split[:5]))
        with open(path + "/" + file_name + TEST_FILE_COMMON_NAME, 'r') as json_raw:
            json_content = json.load(json_raw)
            record = json_content['_checkpoint']['records'][0]
            infractions = record['infractions']
            collisions_vehicle = infractions['collisions_vehicle'][0] if len(infractions['collisions_vehicle']) > 0 else ""
            collisions_vehicle = collisions_vehicle.replace(",", " ")
            collisions_layout = infractions['collisions_layout'][0] if len(infractions['collisions_layout']) > 0 else ""
            collisions_layout = collisions_layout.replace(",", " ")
            outside_route_lanes = infractions['outside_route_lanes'][0] if len(infractions['outside_route_lanes']) > 0 else ""
            outside_route_lanes = outside_route_lanes.replace(",", " ")
            outside_route_lanes_average = '0.0'
            if (outside_route_lanes.strip()):
                outside_route_lanes_average = PERCENT_SEARCH.search(outside_route_lanes).group()[1:-1]
            scenario_num = file_name_split[5][2:]
            scenario_type = file_name_split[6]
            scenario_type_intesnsity = file_name_split[7][2]
            meta_duration_game = str(record["meta"]["duration_game"])
            meta_duration_system = str(record["meta"]["duration_system"])
            meta_route_length = str(record["meta"]["route_length"])
            scores_score_composed = str(record["scores"]["score_composed"])
            scores_score_penalty = str(record["scores"]["score_penalty"])
            scores_score_route = str(record["scores"]["score_route"])
            status = record["status"]
            cur = [scenario_num,\
                scenario_type,\
                scenario_type_intesnsity,\
                collisions_vehicle,\
                collisions_layout,\
                outside_route_lanes,\
                outside_route_lanes_average,\
                meta_duration_game,\
                meta_duration_system,\
                meta_route_length,\
                scores_score_composed,\
                scores_score_penalty,\
                scores_score_route,\
                status
            ]
            csv_rows.append(",".join(cur))
    return "\n".join(csv_rows)

result_type = BASELINE
with open(result_type + 'results.csv', 'w') as result_file:
    result_file.write(get_csv_file_string(result_type))