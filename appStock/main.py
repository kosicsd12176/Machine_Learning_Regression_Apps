import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm

from embedding_data import embed_data
from SVR_linear import model_SVR_linear
from SVR_rbf import model_SVR_rbf
from SVR_rbf_2 import model_SVR_rbf_2
from DecisionTreeRegressor import model_DTR
from RandomForestRegressor import model_RFR


import warnings
warnings.filterwarnings("ignore")


#---------------------------------------#
st.set_page_config(page_title="Stocks predictions", layout='wide')
#---------------------------------------#
def main():
    st.title("Time Series analysis")
    menu = ["Home"]
    submenu = ["Support Vector Regression with linear kernel",
               "Support Vector Regression with rbf kernel",
               "Decision Tree Regressor",
               "Random Forest Regressor"]

    x = pd.read_csv("https://www.dropbox.com/s/5o6rueo7eq7d1xo/ibm.us.txt?dl=1",  header=0, parse_dates=[0], index_col=0)


    st.write("Below we introduce our data from IBM stocks where index column are the timeslots of the stocks. Also is represented the Open value of the stock who is the price stock started on that day. Next we notice High value where is the maximum price of ibm's stock with the same way the Low price represents the minimum. The values we will keep is the Close column where stock is closing each day. The Volume column represents the number of trades that made each day and the last column we aren't gonna use it because it has no efficient on our problem where all values are zero.", x)
    st.write("This part of  data processing and feature engineering is very significant because we will normalize our data and transform them to use in our algorithm")
    st.write("The downside arrays represents some results to outcome some conlusions about our dataset. First we export some value like the maximum and minimum values of each column, the standard deviation, the mean and some other statistics. In this array valuable information we notice in Volume column where is obvious there isn't some correlation between these values. Last but not least we notice that there are no empty rows in any colums NaN values.", x.describe())
    st.write("Next we check about the types of the columns.",x.dtypes)
    st.write("We have our first plot with trending IBM's stock. We notice that we have two high values in the start an after we se the correlation between the points.")

    st.set_option('deprecation.showPyplotGlobalUse', False)
    plt.title("IBM's stock Closing price")
    plt.plot(x['Close'])
    st.pyplot()

    l = x['Close'][-150:]
    x.index = pd.DatetimeIndex(x.index).to_period('D')
    f = plt.figure()
    ax2 = f.add_subplot(2,1,2); _ = sm.graphics.tsa.plot_pacf(l, lags=50, ax=ax2)
    st.write("In the following figure we visualize the partial AutoCorrelation for the last 150 days of IBM's stock")
    st.pyplot(f)
    st.write("We have to transform the time series prediction into a supervised learning problem. For this, the previous values used fro prediction are extracted for each time step. So we have to create a process to pass one dimensional data and outputs the prediction based on the steps we have set.")

    code = '''def embed_data(x, steps):
        n = len(x)
        xout = np.zeros((n - steps, steps))
        yout = x[steps:]
        for i in np.arange(steps, n):
            xout[i - steps] = x[i-steps:i]
        return xout, yout
    '''
    st.code(code, language='python')
    df = x['Close']

    train = df[:-30]
    test = df[-30:]
    print(type(train))
    xtrain, ytrain = embed_data(train, 3)
    xtest, ytest = embed_data(test, 3)

    choice = st.sidebar.selectbox("Choose Algorithm", submenu)
    if choice == "Support Vector Regression with linear kernel":
        st.subheader("Support Vector Regression with linear kernel")
        model_SVR_linear(df=df)
    elif choice == "Support Vector Regression with rbf kernel":
        st.subheader("Support Vector Regression with rbf kernel")
        model_SVR_rbf(xtrain, ytrain, xtest, ytest)
        model_SVR_rbf_2(xtrain, ytrain, xtest, ytest)
    elif choice == "Decision Tree Regressor":
        st.subheader("Decision Tree Regressor")
        model_DTR(xtrain, ytrain, xtest, ytest)
    elif choice == "Random Forest Regressor":
        st.subheader("Random Forest Regressor")
        model_RFR(xtrain, ytrain, xtest, ytest)








if __name__ == '__main__':
    main()