import joblib
from embedding_data import embed_data
import streamlit as st
import matplotlib.pyplot as plt
from sklearn import metrics, svm
import numpy as np

def model_SVR_rbf_2(xtrain, ytrain, xtest, ytest):
    st.subheader("Experiment 2 with rbf kernel")

    m = svm.SVR(kernel='rbf', C=1000, gamma=0.85)
    # m.fit(xtrain, ytrain)
    # filename_2 = 'model_2_c_1000.sav'
    # joblib.dump(m, filename_2)
    loaded_model_x_1000 = joblib.load('saved_models/model_2_c_1000.sav')
    st.write("Mean squared error Support Vector Regression with rbf kernel and parameters as c=1000 and gamma=0.85:")
    st.info(metrics.mean_squared_error(loaded_model_x_1000.predict(xtrain), ytrain))
    st.write(loaded_model_x_1000)
    st.write("Next plot fits on the training data.")
    f3 = plt.figure()
    ax3 = f3.add_subplot(111)
    ax3.plot(loaded_model_x_1000.predict(xtrain), 'b-', np.array(ytrain), 'r-')
    st.pyplot(f3)

    st.write("Predictions for next 30 days")
    f4 = plt.figure()
    ax4 = f4.add_subplot(111)
    ax4.plot(loaded_model_x_1000.predict(xtest), 'b-', label="Prediction")
    ax4.plot(np.array(ytest), 'r-', label="Actual values")
    ax4.legend()
    st.pyplot(f4)