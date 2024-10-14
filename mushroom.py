# building a machine learning web app using streamlit for mushroom data whether they are poisonous or edible
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import ConfusionMatrixDisplay, RocCurveDisplay, PrecisionRecallDisplay
from sklearn.metrics import roc_curve, precision_recall_curve

def main():
    st.title("Mushroom Classifier Web App")
    st.sidebar.title("Mushroom Classifier Web App")
    st.markdown("Are your mushrooms edible or poisonous? 🍄")
    st.sidebar.markdown("Are your mushrooms edible or poisonous? 🍄")

    @st.cache_data(persist=True) # caching the data so that it is not loaded again and again
    def load_data():
        data = pd.read_csv("mushrooms.csv") # load the data
        label = LabelEncoder() # label encoding for the data to be used in the model which helps in converting the categorical data into numerical data
        for col in data.columns:
            data[col] = label.fit_transform(data[col])
        return data

    @st.cache_data(persist=True) # caching the data so that it is not loaded again and again
    def split(df): # split the data into training and testing data
        y = df['type'] # target variable
        x = df.drop(columns=['type']) # features
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=0) # split the data into training and testing data with test size of 30% of the data
        return x_train, x_test, y_train, y_test

    def plot_metrics(metrics_list):
        if 'Confusion Matrix' in metrics_list:
            if hasattr(model, "predict_proba"):
                y_scores = model.predict_proba(x_test)[:, 1]
            else:
                y_scores = model.decision_function(x_test)
            st.subheader("Confusion Matrix")
            cm = confusion_matrix(y_test, y_pred)
            fig, ax = plt.subplots()
            disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
            disp.plot(ax=ax)
            st.pyplot(fig)

        if 'ROC Curve' in metrics_list:
            if hasattr(model, "predict_proba"):
                y_scores = model.predict_proba(x_test)[:, 1]
            else:
                y_scores = model.decision_function(x_test)
            st.subheader("ROC Curve")
            fpr, tpr, thresholds = roc_curve(y_test, y_scores)
            fig, ax = plt.subplots()
            disp = RocCurveDisplay(fpr=fpr, tpr=tpr)
            disp.plot(ax=ax)
            st.pyplot(fig)

        if 'Precision-Recall Curve' in metrics_list:
            if hasattr(model, "predict_proba"):
                y_scores = model.predict_proba(x_test)[:, 1]
            else:
                y_scores = model.decision_function(x_test)
            st.subheader("Precision-Recall Curve")
            precision, recall, thresholds = precision_recall_curve(y_test, y_scores)
            fig, ax = plt.subplots()
            disp = PrecisionRecallDisplay(precision=precision, recall=recall)
            disp.plot(ax=ax)
            st.pyplot(fig)


    df = load_data() # load the data
    x_train, x_test, y_train, y_test = split(df) # split the data
    class_names = ['edible', 'poisonous']
    st.sidebar.subheader("Choose Classifier")
    classifier = st.sidebar.selectbox("Classifier", ("Support Vector Machine (SVM)", "Random Forest", "Logistic Regression"))

    # if the classifier is Support Vector Machine (SVM)
    if classifier == "Support Vector Machine (SVM)":
        st.sidebar.subheader("Model Hyperparameters")
        C = st.sidebar.number_input("C (Regularization parameter)", 0.01, 10.0, step=0.01, key='C')
        kernel = st.sidebar.radio("Kernel", ("rbf", "linear"), key='kernel')
        gamma = st.sidebar.radio("Gamma (Kernel Coefficient)", ("scale", "auto"), key='gamma')
        metrics = st.sidebar.multiselect("What metrics to plot?", ("Confusion Matrix", "ROC Curve", "Precision-Recall Curve"))
        if st.sidebar.button("Classify", key='classify'):
            st.subheader("Support Vector Machine (SVM) Results")
            model = SVC(C=C, kernel=kernel, gamma=gamma)
            model.fit(x_train, y_train)
            y_pred = model.predict(x_test)
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred)
            recall = recall_score(y_test, y_pred)
            st.write("Accuracy: ", round(accuracy, 2))
            st.write("Precision: ", round(precision,2))
            st.write("Recall: ", round(recall,2))
            plot_metrics(metrics)

    # if the classifier is Random Forest
    if classifier == "Random Forest":
        st.sidebar.subheader("Model Hyperparameters")
        n_estimators = st.sidebar.number_input("Number of Trees", 100, 5000, step=10, key='n_estimators')
        max_depth = st.sidebar.number_input("Maximum Depth of the Tree", 1, 20, step=1, key='max_depth')
        bootstrap = st.sidebar.radio("Bootstrap samples when building trees", (True, False), key='bootstrap') # bootstrap samples when building trees which means that the model will randomly select samples from the training data with replacement where true means that the model will use bootstrap samples and false means that the model will use the entire training data
        metrics = st.sidebar.multiselect("What metrics to plot?", ("Confusion Matrix", "ROC Curve", "Precision-Recall Curve"))
        if st.sidebar.button("Classify", key='classify'):
            st.subheader("Random Forest Results")
            model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, bootstrap=bootstrap, n_jobs=-1)
            model.fit(x_train, y_train)
            y_pred = model.predict(x_test)
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred)
            recall = recall_score(y_test, y_pred)
            st.write("Accuracy: ", round(accuracy, 2))
            st.write("Precision: ", round(precision,2))
            st.write("Recall: ", round(recall,2))
            plot_metrics(metrics)

    if classifier == "Logistic Regression":
        st.sidebar.subheader("Model Hyperparameters")
        C = st.sidebar.number_input("C (Regularization parameter)", 0.01, 10.0, step=0.01, key='C')
        max_iter = st.sidebar.slider("Maximum number of iterations", 100, 500, key='max_iter')
        metrics = st.sidebar.multiselect("What metrics to plot?", ("Confusion Matrix", "ROC Curve", "Precision-Recall Curve"))
        if st.sidebar.button("Classify", key='classify'):
            st.subheader("Logistic Regression Results")
            model = LogisticRegression(C=C, max_iter=max_iter)
            model.fit(x_train, y_train)
            y_pred = model.predict(x_test)
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred)
            recall = recall_score(y_test, y_pred)
            st.write("Accuracy: ", round(accuracy, 2))
            st.write("Precision: ", round(precision,2))
            st.write("Recall: ", round(recall,2))
            plot_metrics(metrics)





    if st.sidebar.checkbox("Show raw data", False):
        st.subheader("Mushroom Data Set")
        st.write(df)



if __name__ == '__main__':
    main()