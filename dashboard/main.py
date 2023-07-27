import streamlit as st
import json
import csv
import matplotlib.pyplot as plt


paths = {
    "Gems (Matrix Multiplication)" : {
        "tvm" : "../matMul_TVM/",
        "ytopt":"../matMul_Ytopt/"
    },
    "3MM": {
        "tvm" : "../3mm_TVM/",
        "ytopt":"../3mm_Ytopt/"
    },
    "Cholesky" : {
        "tvm" : "../cholesky_TVM/",
        "ytopt":"../cholesky_Ytopt/"
    }

}

configPath = {
    "Large":"results/large/",
    "Extra Large": "results/extraLarge/"
    }

fileNameTVM = ["tvmGATuner.json","tvmGridSearchTuner.json","tvmRandomTuner.json","tvmXGBTuner.json"]


errorVal = 1000000000

st.title("Autotuning Apache TVM Applications Using ytopt (Bayesian Optimization)")

option = st.selectbox("Select Experiment".upper(),("Gems (Matrix Multiplication)","3MM","Cholesky"))

config = st.radio(
    "Select Configuration",
    ('Large', 'Extra Large'),
    horizontal=True
    )
data = {}
ytopData = {}

for file in fileNameTVM:
    with open(paths[option]["tvm"]+configPath[config]+file, 'r') as handle:
        data[file[3:-5]] = [json.loads(line) for line in handle]

with open(paths[option]["ytopt"]+configPath[config]+"results.csv", 'r') as file:
        reader = csv.reader(file)
        header = next(reader) 
        i=1
        if option == "Gems (Matrix Multiplication)" or option == "Cholesky":
             
            for row in reader:
                object = {}
                runtime = round(float(row[2]),2) 
                elapsed = round(float(row[3]),2)
                # object['tile_y'] = row[0]
                # object['tile_x'] = row[1]
                object['runtime'] = runtime
                object['elapsed'] = elapsed

                ytopData[i] = object
                i += 1
        else:
              for row in reader:
                object = {}
                runtime = round(float(row[6]),2) 
                elapsed = round(float(row[7]),2)

                object['runtime'] = runtime
                object['elapsed'] = elapsed

                ytopData[i] = object
                i += 1
             

RTstartTime = round(data["RandomTuner"][0]['result'][3],2)
RandomTunerResults = {
     "runtime" : [d["result"][0][0] for d in data["RandomTuner"]],
     "elapsed" : [round(round(d['result'][3],2) - RTstartTime,2) for d in data["RandomTuner"]]
     
} 
GATstartTime = round(data["GATuner"][0]['result'][3],2)
GATunerResults = {
    "runtime" : [d["result"][0][0] for d in data["GATuner"]],
    "elapsed" : [round(round(d['result'][3],2) - GATstartTime,2) for d in data["GATuner"]]
   
} 
GRTstartTime = round(data["GridSearchTuner"][0]['result'][3],2)
GridSearchTunerResults = {
    "runtime" : [d["result"][0][0] for d in data["GridSearchTuner"]],
    "elapsed" : [round(round(d['result'][3],2) - GRTstartTime,2) for d in data["GridSearchTuner"]]
   
} 
XTstartTime = round(data["XGBTuner"][0]['result'][3],2)
XGBTunerResults = {
    "runtime" : [d["result"][0][0] for d in data["XGBTuner"]],
    "elapsed" : [round(round(d['result'][3],2) - XTstartTime,2) for d in data["XGBTuner"]]
   
} 
# XGBTunerResults =  [d["result"][0][0] for d in data["XGBTuner"]]
# GridSearchTunerResults =  [d["result"][0][0] for d in data["GridSearchTuner"]]
YtoptResults = {
     "runtime":[ytopData[d]["runtime"] for d in ytopData.keys()],
     "elapsed":[ytopData[d]["elapsed"] for d in ytopData.keys()]
}


# st.write(GridSearchTunerResults)


plt.plot(GATunerResults["elapsed"], GATunerResults["runtime"],marker='o',alpha=0.7)
plt.plot(RandomTunerResults["elapsed"], RandomTunerResults["runtime"],marker='o',alpha=0.7)
plt.plot(GATunerResults["elapsed"], GATunerResults["runtime"],marker='o',alpha=0.7)
plt.plot(XGBTunerResults["elapsed"], XGBTunerResults["runtime"],marker='o',alpha=0.7)
plt.plot(YtoptResults["elapsed"], YtoptResults["runtime"],marker='o',alpha=0.7)



plt.legend(['AutoTVM - GA','AutoTVM - Random','AutoTVM - GridSearch','AutoTVM - XGB','YTOPT Tuner'], loc='upper right')
plt.xlabel("Cummulative time(secs)")
plt.ylabel("Runtime(secs)")
plt.title("Performance of Matrix Multiplication (AutoTVM vs YTOPT)")


st.pyplot(plt)
