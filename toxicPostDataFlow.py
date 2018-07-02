# -*- coding: latin-1 -*-
import apache_beam as beam
from apache_beam.options.pipeline_options import PipelineOptions
import logging
# Create list and instances of a class
import argparse
import pickle
import pandas as pd
import models
import numpy as np
import re
import json
from keras.preprocessing.sequence import pad_sequences
import apache_beam.io
import tensorflow as tf
import time

# tokenizer = models.load_tokenizer()
model = models.load_keras()
graph = tf.get_default_graph()
loaded_dictionary = models.load_dictionary()


# import data 
#data = pd.read_gbq("SELECT ni, content FROM (SELECT JSON_EXTRACT(json,'$.ni') as ni, JSON_EXTRACT(json,'$.ni.content') as content FROM [pioneering-tome-782:raw.gnip_twitter_decahose_20180621] WHERE JSON_EXTRACT_SCALAR(json,'$.ni.language')=='en')  LIMIT 10000", project_id="cloud9-analytics")
#data = data.iloc[:,1].tolist()

def text_to_dictionary(X):
    x_clean = X["content"].apply(clean_text)
    word_dict = x_clean.apply(lambda sent: [[loaded_dictionary.get(word) for word in sent.split() if
                                             loaded_dictionary.get(word) is not None]])
    
    word_dict = word_dict.apply(lambda vector: pad_sequences(vector, maxlen=200))
    return word_dict

def clean_text(X):
    special_character_removal = re.compile(r'[^A-Za-z\.\-\?\!\,\#\@\% ]',re.IGNORECASE)
    x_ascii = unidecode(str(X))
    x_clean = special_character_removal.sub('', x_ascii)
    return x_clean

def predictText(X,model):
    #global model
    #global graph
    token = X['content']
    # Predict probabilities for labels
    global graph
    with graph.as_default():
        prediction = model.predict(np.array(token))

    prediction = prediction[0].tolist()
    classify_results = {
        "toxic":prediction[0],
        "severe_toxic":prediction[1],
        "obscene":prediction[2],
        "threat":prediction[3],
        "insult":prediction[4],
        "identity_hate":prediction[5]
    }
    X['content'] = classify_results
    return json.dumps(X)


def giveMeResults(X):
    info = X['ni']
    token = X['content']

    # Predict each label and jsonify it
    prediction = predictText(token)[0].tolist()

    classify_results = {
        "toxic":prediction[0],
        "severe_toxic":prediction[1],
        "obscene":prediction[2],
        "threat":prediction[3],
        "insult":prediction[4],
        "identity_hate":prediction[5]
    }

    # Format results into data to be returned by the function
    result = {}
    result['ni'] = info
    result['classify_results'] = classify_results

    return json.dumps({'result':result})



def main():
    tokenizer = models.load_tokenizer()
    model = models.load_keras()
    graph = tf.get_default_graph()
    start = time.time()
    parser = argparse.ArgumentParser() 
    parser.add_argument('--output',dest='output',required=True,help='Output file to write results to.')
    known_args, pipeline_args = parser.parse_known_args(None)
    gcquery = "SELECT ni, content FROM (SELECT JSON_EXTRACT(json,'$.ni') as ni, JSON_EXTRACT(json,'$.ni.content') as content FROM [pioneering-tome-782:raw.gnip_twitter_decahose_20180621] WHERE JSON_EXTRACT_SCALAR(json,'$.ni.language')=='en')  LIMIT 100"
        # We use the save_main_session option because one or more DoFn's in this
        # workflow rely on global context (e.g., a module imported at module level).
    pipeline_options = PipelineOptions(pipeline_args)
        # Apply the function and print results. Can we get it to write to the bucket? :D
    with beam.Pipeline(options=pipeline_options) as p:
        #list_of_json =  p | beam.Create(data) 

        query_results = p | beam.io.Read(beam.io.BigQuerySource(query=gcquery))
        tokenized_data = query_results | beam.Map(text_to_dictionary,tokenizer)
        predictions = tokenized_data | beam.Map(predictText,model)
        predictions | beam.io.WriteToText("gs://zach-test/output.txt")

    end = time.time()
    final_time = end - start
    print("Total time: " + str(final_time))
if __name__== '__main__':
	logging.getLogger().setLevel(logging.INFO)
	main()
