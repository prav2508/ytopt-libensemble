import streamlit as st
import json
import csv
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from pathlib import Path

st.set_page_config(
    page_title="Autotuning Apache TVM Applications Using ytopt",
    page_icon="🧊",
    layout="centered",
    initial_sidebar_state="expanded",
    menu_items={
        'Get help': 'https://github.com/prav2508/ytopt-libensemble/tree/main/dashboard',
        'About': "https://github.com/prav2508/ytopt"
    }
)

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

try:
    for file in fileNameTVM:
        file_path = Path(paths[option]["tvm"]+configPath[config]+file)
    
        # data[file[3:-5]] = json.loads(file_path.read_bytes())
    # print(data)
    # for file in fileNameTVM:
        with open(file_path, 'r') as handle:
            data[file[3:-5]] = [json.loads(line) for line in handle]
            
    file_path_ytopt = Path(paths[option]["ytopt"]+configPath[config]+"results.csv")
    with open(file_path_ytopt, mode='r') as file:
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
except FileNotFoundError:
    print(f"The file '{file_path}' does not exist.")
except Exception as e:
    print(f"An error occurred while reading the file: {e}")
                

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

def find_min_excluding_zero(list):
  min_value = float("inf")
  for value in list:
    if value != 0 and value < min_value:
      min_value = value
  return min_value
# st.write(GridSearchTunerResults)


plt.plot(GATunerResults["elapsed"], GATunerResults["runtime"],marker='o',alpha=0.9,)
plt.plot(RandomTunerResults["elapsed"], RandomTunerResults["runtime"],marker='o',alpha=0.7)
plt.plot(GridSearchTunerResults["elapsed"], GridSearchTunerResults["runtime"],marker='o',alpha=0.7)
plt.plot(XGBTunerResults["elapsed"], XGBTunerResults["runtime"],marker='o',alpha=0.7)
plt.plot(YtoptResults["elapsed"], YtoptResults["runtime"],marker='o',alpha=0.7)



plt.legend(['AutoTVM - GA','AutoTVM - Random','AutoTVM - GridSearch','AutoTVM - XGB','YTOPT Tuner'], loc='upper right')
plt.xlabel("Cummulative time(secs)")
plt.ylabel("Runtime(secs)")
plt.title("Performance of {} (AutoTVM vs YTOPT)".format(option))


st.pyplot(plt)

# df = pd.DataFrame(
#    [find_min_excluding_zero(GATunerResults["runtime"]),find_min_excluding_zero(RandomTunerResults["runtime"]),find_min_excluding_zero(GRTstartTime["runtime"]),find_min_excluding_zero(XGBTunerResults["runtime"]),find_min_excluding_zero(YtoptResults["runtime"])],
#    columns=['GA','Random','GridSearch','XGB','YTOPT'])


st.header("Minimum Runtime")

fig, ax = plt.subplots()
ax.set_title("Runtime of tuners")
ax.bar(['GA','Random','GridSearch','XGB','YTOPT'], [find_min_excluding_zero(GATunerResults["runtime"]),find_min_excluding_zero(RandomTunerResults["runtime"]),find_min_excluding_zero(GridSearchTunerResults["runtime"]),find_min_excluding_zero(XGBTunerResults["runtime"]),find_min_excluding_zero(YtoptResults["runtime"])],color=['#d47b85', '#c8fac9', '#7cb2e6', '#dee08b', '#6fe1f2'])
ax.set_ylabel("Runtime(secs)")
st.pyplot(fig)


