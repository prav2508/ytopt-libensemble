import json
import csv
import matplotlib.pyplot as plt
import numpy as np

RandomData = {}
GAData = {}
XGBData = {}
GridSearchData = {}
YTOPTData = {}



def processTVMResults(filename):
    data = {}
    with open(filename, 'r') as handle:
        json_data = [json.loads(line) for line in handle]


    a = 1
    startTime = round(json_data[0]['result'][3],2)
    for record in json_data:

        # RandomData[(record['config']['entity'][0][2],record['config']['entity'][1][2])] = round(record['result'][0][0],2)
        object = {}
        object['tile_y'] = record['config']['entity'][0][2]
        object['tile_x'] = record['config']['entity'][1][2]
        object['runtime'] = round(record['result'][0][0],2)
        object['elapsed'] = round(round(record['result'][3],2) - startTime,2)

        data[a] = object
        a += 1
    return data


def processYtoptResults(filename):
    data = {}
    with open(filename, 'r') as file:
        reader = csv.reader(file)
        header = next(reader) 
        i=1
        for row in reader:
            object = {}
            runtime = round(float(row[2]),2) 
            elapsed = round(float(row[3]),2)
            object['tile_y'] = row[0]
            object['tile_x'] = row[1]
            object['runtime'] = runtime
            object['elapsed'] = elapsed

            data[i] = object
            i += 1
    return data



RandomData = processTVMResults('results/tvm_RandomTuner.json')
GAData = processTVMResults('results/tvm_GATuner.json')
XGBData = processTVMResults('results/tvm_XGBTuner.json')
GridSearchData = processTVMResults('results/tvm_GridSearchTuner.json')

YTOPTData = processYtoptResults('../tvm_matMul/results.csv')



plt.plot([d['elapsed'] for d in GAData.values()], [d['runtime'] for d in GAData.values()],marker='o',alpha=0.7)
plt.plot([d['elapsed'] for d in RandomData.values()], [d['runtime'] for d in RandomData.values()],marker='o',alpha=0.7)
plt.plot([d['elapsed'] for d in GridSearchData.values()], [d['runtime'] for d in GridSearchData.values()],marker='o',alpha=0.7)
plt.plot([d['elapsed'] for d in XGBData.values()], [d['runtime'] for d in XGBData.values()],marker='o',alpha=0.7)

plt.plot([d['elapsed'] for d in YTOPTData.values()], [d['runtime'] for d in YTOPTData.values()],marker='o',alpha=0.7)

# plt.axhline(min([d['runtime'] for d in GAData.values()]), color='blue',alpha=0.7)
# plt.axhline(min([np.log2(d['runtime']) for d in RandomData.values()]), color='orange',alpha=0.7)
# plt.axhline(min([np.log2(d['runtime']) for d in YTOPTData.values()]), color='green',alpha=0.7)
# print(min([np.log2(d['runtime']) for d in GAData.values()]))
# print(min([d['runtime'] for d in RandomData.values()]))
# print(min([d['runtime'] for d in YTOPTData.values()]))


plt.legend(['AutoTVM - GA','AutoTVM - Random','AutoTVM - GridSearch','AutoTVM - XGB','YTOPT Tuner'], loc='upper right')
plt.xlabel("Cummulative time(secs)")
plt.ylabel("Runtime(secs)")
plt.title("Performance of Matrix Multiplication (AutoTVM vs YTOPT)")
plt.savefig('Eval_performance.png')