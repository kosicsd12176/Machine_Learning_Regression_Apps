import joblib
import streamlit as st
import matplotlib.pyplot as plt
from sklearn import metrics
import numpy as np

def model_RFR(xtrain, ytrain, xtest, ytest):
    # r_f = RandomForestRegressor()
    # r_f.fit(xtrain,ytrain)
    # filename_4 = 'model_4.sav'
    # joblib.dump(r_f, filename_4)
    loaded_model_4 = joblib.load('saved_models/model_4.sav')
    y = loaded_model_4.predict(xtest)
    st.write(loaded_model_4)
    st.write("Next plot fits on the training data.")
    f6 = plt.figure()
    ax6 = f6.add_subplot(111)
    ax6.plot(loaded_model_4.predict(xtrain), 'b-', np.array(ytrain), 'r-')
    st.pyplot(f6)

    st.write("Mean squared error for random forest regressor:")
    st.info(metrics.mean_squared_error(loaded_model_4.predict(xtrain), ytrain))

    st.write("Predictions for next 30 days")
    f7 = plt.figure()
    ax7 = f7.add_subplot(111)
    ax7.plot(loaded_model_4.predict(xtest), 'b-', label="Prediction")
    ax7.plot(np.array(ytest), 'r-', label="Actual values")
    ax7.legend()
    st.pyplot(f7)
