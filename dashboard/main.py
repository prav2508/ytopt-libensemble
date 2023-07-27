import streamlit as st

paths = {
    "Gems (Matrix Multiplication)" : {
        "tvm" : "ytopt-libensemble/matMul_TVM",
        "ytopt":"ytopt-libensemble/matMul_Ytopt"
    },
    "3MM": {
        "tvm" : "ytopt-libensemble/3mm_TVM",
        "ytopt":"ytopt-libensemble/3mm_Ytopt"
    },
    "Cholesky" : {
        "tvm" : "ytopt-libensemble/cholesky_TVM",
        "ytopt":"ytopt-libensemble/cholesky_Ytopt"
    }

}

st.title("Autotuning Apache TVM Applications Using ytopt (Bayesian Optimization)")

option = st.selectbox("Select Experiment".upper(),("Gems (Matrix Multiplication)","3MM","Cholesky"))

config = st.radio(
    "Select Configuration",
    ('Large', 'Extra Large'),
    horizontal=True
    )


