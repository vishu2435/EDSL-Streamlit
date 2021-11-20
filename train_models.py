import streamlit as st
import pandas as pd
from sklearn.ensemble import *
from sklearn.linear_model import LogisticRegression
from sklearn.tree import *
from sklearn.neighbors import *
from sklearn.metrics import classification_report
import pickle
from DeepSuperLearner import DeepSuperLearnerModified
import numpy as np
import os

def app():

    st.title("EDSL( Emoji Based Deep Super Learner )")

    file_names={}
    def file_selector(folder_path='.',type="training"):
        filenames = os.listdir(folder_path)
        selected_filename = st.selectbox('Select '+type+' file', filenames)
        return os.path.join(folder_path, selected_filename)

    file_names["training"] = file_selector('./preprocessed/training',"training")
    st.write('You selected `%s`' % file_names["training"])

    file_names["validation"] = file_selector('./preprocessed/validation',"validation")
    st.write('You selected `%s`' % file_names["validation"])

    file_names["testing"] = file_selector('./preprocessed/testing',"testing")
    st.write('You selected `%s`' % file_names["testing"])

    classifiers = {
        'Extra trees Classifer' : ExtraTreesClassifier(n_estimators=200, max_depth=None, max_features="auto"),
        "Logistic Regression":LogisticRegression(max_iter=200),
        'Random Forest Classifier': RandomForestClassifier(),
        'Decision Tree': DecisionTreeClassifier(),
        'K nearest neighbour': KNeighborsClassifier(),
        'Gradient Boosting Classifier': GradientBoostingClassifier(),

    }
    st.sidebar.header("Select classifiers to be used ")
    count = 0
    checkbox_trigger = []
    for i in classifiers.keys():
        t = st.sidebar.checkbox(i)
        checkbox_trigger.append((t,i))
        # if t:
        #     if count==0:
        #         st.header("Selected Classifiers")
        #         count+=1
        #     st.write(i)
    name = st.text_input("Enter model name to be saved ","")
    st.write(name)
    run = st.button("Run")
    Base_learners = {}


    if run and name:
        st.header("Base Learners")
        for i in checkbox_trigger:
            if i[0]:
                Base_learners[i[1]] = classifiers[i[1]]
                st.write(i[1])
        DSL_learner = DeepSuperLearnerModified(Base_learners,K=5)
        training_data = pickle.load(open(file_names['training'],'rb'))
        validation_data = pickle.load(open(file_names['validation'],'rb'))
        testing_data = pickle.load(open(file_names['testing'],'rb'))
        st.header("training Data")
        st.write(len(training_data['sentences_with_emoji']))
        st.header("testing Data")
        st.write(len(testing_data['sentences_with_emoji']))
        st.header("Validation Data")
        st.write(len(validation_data['sentences_with_emoji']))
        history,obj = DSL_learner.fit(training_data["sentences_with_emoji"],
                                    training_data["sentiment"],
                                    validation_data["sentences_with_emoji"],
                                    validation_data["sentiment"],
                                    max_iterations=20,sample_weight=None)
        # st.write(obj)
        history_data = pd.DataFrame({
            'iteration':history['iteration'],
            'loss' : history['loss'],
            "time" : history['time'],
            'Validation Accuracy': history['val_accuracy']
        })
        weights_data = {}
        learners = list(obj['BL'].keys())
        
        for j in history['weights']:
            for i,k in enumerate(j):
                if(weights_data.get(learners[i])):
                    weights_data[learners[i]].append(k)
                else:
                    weights_data[learners[i]] = [k]
        weights_data = pd.DataFrame(weights_data)
        st.write(history_data)
        st.header("Weights Data")
        st.write(weights_data)


        
        all_probs = DSL_learner.predict(testing_data["sentences_with_emoji"],return_base_learners_probs=True)
        Dsl_proba = all_probs[0]
        Base_learners_proba = all_probs[1]
        y_pred_dsl = [i for i in range(len(Dsl_proba))]
        for i,j in enumerate(Dsl_proba):
            y_pred_dsl[i] = np.argmax(j)

        base_learners_predictions = []


        for i in range(len(obj["BL"].keys())):
            base_learners_predictions.append([j for j in range(len(Base_learners_proba))])
        for k in range(len(obj["BL"].keys())):
            for i,j in enumerate(Base_learners_proba):
                base_learners_predictions[k][i] = np.argmax(j[k])
                # base_learners_predictions[1][i] = np.argmax(j[1])
                # base_learners_predictions[2][i] = np.argmax(j[2])
                # base_learners_predictions[3][i] = np.argmax(j[3])
        report_edsl = {}
        report_edsl['Deep superlearner'] = classification_report(testing_data['sentiment'],y_pred_dsl,output_dict=True)
        for i,j in enumerate(obj["BL"].keys()):
            report_edsl[j] = classification_report(testing_data['sentiment'],base_learners_predictions[i],output_dict=True)
        dataframes_array = {}
        for i in report_edsl.keys():
            dataframes_array[i] = pd.DataFrame(report_edsl[i])
        for k in dataframes_array.keys():
            st.write(k)
            st.write(dataframes_array[k])
        # name='airline_dataser'
        
        
        pickle.dump(obj,open("./saved/Saved_EDSL_"+name+".p",'wb'))
