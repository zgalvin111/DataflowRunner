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
import time
# tokenizer = models.load_tokenizer()
loaded_dictionary = models.load_dict()

# import data 
#data = pd.read_gbq("SELECT ni, content FROM (SELECT JSON_EXTRACT(json,'$.ni') as ni, JSON_EXTRACT(json,'$.ni.content') as content FROM [pioneering-tome-782:raw.gnip_twitter_decahose_20180621] WHERE JSON_EXTRACT_SCALAR(json,'$.ni.language')=='en')  LIMIT 10000", project_id="cloud9-analytics")
#data = data.iloc[:,1].tolist()

def add_model(X):
    model = models.load_keras()
    return {"ni":ni_id,"vector":vector,"model":model}


def text_to_dictionary(item):
    ni_id = item.get('ni')
    content = item.get('content')
    assert content is not None
    assert ni_id is not None
    
    sent = clean_text(content)
    word_dict_indices = [[loaded_dictionary.get(word) for word in sent.split() if
                                             loaded_dictionary.get(word) is not None]]
    
    vector = pad_sequences(word_dict_indices, maxlen=200)
    return {"ni":ni_id,"vector":vector}

def clean_text(X):
    special_character_removal = re.compile(r'[^A-Za-z\.\-\?\!\,\#\@\% ]',re.IGNORECASE)
    x_ascii = X.encode("ascii","ignore").decode("ascii")
    x_clean = special_character_removal.sub('', x_ascii)
    return x_clean

def predictText(item):

    model = models.load_keras()
    ni_id = item.get('ni')
    vector = item.get('vector')
    assert ni_id is not None
    assert vector is not None
    
    # Predict probabilities for labels
    prediction = model.predict(np.array(vector))
    
    prediction = prediction[0].tolist()
    
    results = {}
    classify_results = {
        "toxic":prediction[0],
        "severe_toxic":prediction[1],
        "obscene":prediction[2],
        "threat":prediction[3],
        "insult":prediction[4],
        "identity_hate":prediction[5]
    }
    results['content'] = classify_results
    results['ni'] = ni_id
    return json.dumps(results)


def main():
    
    parser = argparse.ArgumentParser() 
    parser.add_argument('--output',dest='output',required=True,help='Output file to write results to.')
    known_args, pipeline_args = parser.parse_known_args(None)
    gcquery = "SELECT ni, content FROM (SELECT JSON_EXTRACT(json,'$.ni') as ni, JSON_EXTRACT(json,'$.ni.content') as content FROM [pioneering-tome-782:raw.gnip_twitter_decahose_20180621] WHERE JSON_EXTRACT_SCALAR(json,'$.ni.language')=='en')  LIMIT 5"
        # We use the save_main_session option because one or more DoFn's in this
        # workflow rely on global context (e.g., a module imported at module level).
    pipeline_options = PipelineOptions(pipeline_args)
        # Apply the function and print results. Can we get it to write to the bucket? :D
    with beam.Pipeline(options=pipeline_options) as p:
        #list_of_json =  p | beam.Create(data) 
        
        query_results = p | beam.io.Read(beam.io.BigQuerySource(query=gcquery))
        tokenized_data = query_results | beam.Map(text_to_dictionary)
        predictions = tokenized_data | beam.Map(predictText)
        predictions | beam.io.WriteToText("gs://zach-test/output.txt")
        
if __name__== '__main__':
	logging.getLogger().setLevel(logging.INFO)
	main()
