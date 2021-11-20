from json import load
from operator import le
import pandas as pd
import numpy as np
import pickle as pk
import numpy as np
from scipy.sparse import data
import twitter_sentiment_dataset as tsd
import phrase2vec as p2v
from sklearn.metrics import classification_report
from DeepSuperLearner import DeepSuperLearnerModified
import os
import streamlit as st
def file_selector(folder_path='.',type="training"):
    filenames = os.listdir(folder_path)
    selected_filename = st.selectbox('Select '+type+' file', filenames)
    return os.path.join(folder_path, selected_filename)
def app():
    @st.cache
    def load_files(model_name,embedding_name):
        return pk.load(open(model_name,'rb')),pk.load(open(embedding_name,'rb'))

    st.title("Test The Model")
    file_names={}
        
    file_names["model"] = file_selector('./saved',"Model")
    st.write('You selected `%s`' % file_names["model"])
    mapping_file_name = str.lower(file_names['model'].split('\\')[-1].split('_')[0])
    st.write(mapping_file_name)
    mapping_dictionary = pk.load(open('./dicts/'+mapping_file_name+'_dict.p','rb'))
    st.write(mapping_dictionary)

    # test_data = pk.load(open('./preprocessed/testing/sandars_test.p','rb'))

    model,p2v_emoji = load_files(file_names['model'],'./phrase2vec.p')
    learner = DeepSuperLearnerModified(model["BL"],K=model["Kfolds"],classes=len(mapping_dictionary.keys()))
    learner.weights_per_iteration = model["weights_per_iteration"]
    learner.fitted_learners_per_iteration = model["fitted_learners_per_iteration"]

    # all_probs = learner.predict(test_data["sentences_with_emoji"],return_base_learners_probs=True)
    # Dsl_proba = all_probs[0]
    # Base_learners_proba = all_probs[1]
    # y_pred_dsl = [i for i in range(len(Dsl_proba))]
    # for i,j in enumerate(Dsl_proba):
    #     y_pred_dsl[i] = np.argmax(j)
    # print(classification_report(test_data['sentiment'],y_pred_dsl))

    # p2v_emoji = pk.load()
    sentence = st.text_input("Enter a sentence : ","")

    # print(os.listdir('./'))
    st.write(sentence)
    if sentence:
        sentence = str.lower(sentence)
        st.write('Sentence is : ',sentence)
        sentence = tsd.prepare_tweet_vector_averages(sentence,p2v_emoji)
        # sentence = sentence.reshape(1,-1)
        # print(sentence.shape)
        probs = learner.predict(sentence)
        data_frame = {}
        st.write(probs[0])
        st.write('Sentence is : ',mapping_dictionary[str(np.argmax(probs[0]))])
        
        # data_frame = pd.DataFrame(data_frame)
        # st.write(data_frame)


    # print(probs)
    # all_probs = learner.predict(sentence,return_base_learners_probs=True)