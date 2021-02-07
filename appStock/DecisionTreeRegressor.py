import joblib
import streamlit as st
import matplotlib.pyplot as plt
from sklearn import metrics
import numpy as np

def model_DTR(xtrain, ytrain, xtest, ytest):
    # regr_1 = DecisionTreeRegressor()
    # regr_1.fit(xtrain,ytrain)
    # filename_3 = 'model_3.sav'
    # joblib.dump(regr_1, filename_3)
    loaded_model_3 = joblib.load('saved_models/model_3.sav')
    y = loaded_model_3.predict(xtest)
    st.write(loaded_model_3)
    st.write("Next plot fits on the training data.")
    f5 = plt.figure()
    ax5 = f5.add_subplot(111)
    ax5.plot(loaded_model_3.predict(xtrain), 'b-', np.array(ytrain), 'r-')
    st.pyplot(f5)

    st.write("Mean squared error for decision tree regressor:")
    st.info(metrics.mean_squared_error(loaded_model_3.predict(xtrain), ytrain))

    st.write("Predictions for next 30 days")
    f5 = plt.figure()
    ax5 = f5.add_subplot(111)
    ax5.plot(loaded_model_3.predict(xtest), 'b-', label="Prediction")
    ax5.plot(np.array(ytest), 'r-', label="Actual values")
    ax5.legend()
    st.pyplot(f5)