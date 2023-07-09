import json


with open('tvm_RandomTuner.json', 'r') as handle:
    json_data = [json.loads(line) for line in handle]



for record in json_data:
    # print(record['config']['entity'][0][2])
    # print(record['result'][0][0])
    print(len(record['config']['entity']))
    