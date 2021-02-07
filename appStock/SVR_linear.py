import joblib
from embedding_data import embed_data
import streamlit as st
import matplotlib.pyplot as plt
from sklearn import metrics

def model_SVR_linear(df):
    st.subheader("Train a support Vector Regression")

    x, y = embed_data(df, 5)
    #model = svm.SVR(kernel='linear', C=1)
    # model.fit(x, y)
    # filename = 'model_1.sav'
    # joblib.dump(model, filename)
    loaded_model = joblib.load('saved_models/model_1.sav')
    st.write("This model trained in entire dataset")
    st.write(loaded_model)

    plt.plot(loaded_model.predict(x), y, '.')
    f2 = plt.figure()
    ax2 = f2.add_subplot(111)
    ax2.plot(loaded_model.predict(x), y, '.')
    st.write(
        "Let us now plot the predicted values vs. the true values. If the prediction were perfect, we would see a perfect diagonal line, but due to noise, we will often see something else. Also we can notice if the range of the variance is different in some points of the range.")
    st.pyplot(f2)

    st.write("to evaluate the model we use mean squared error:")
    st.info(metrics.mean_squared_error(loaded_model.predict(x), y))
    st.write(
        "RMSE is a quadratic scoring rule that also measures the average magnitude of the error. It’s the square root of the average of squared differences between prediction and actual observation.")
