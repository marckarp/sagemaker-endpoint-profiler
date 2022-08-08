
import json
import time
import numpy as np
import logging
import boto3

client = boto3.client('sagemaker-featurestore-runtime')


def retrieve_latest_features(data,record_id):
    print("retrieve_latest_features")

    client = boto3.client('sagemaker-featurestore-runtime')

    response = client.get_record(
    FeatureGroupName='my-features',
    RecordIdentifierValueAsString=str(record_id),
    )
    
    return response["Record"][0]["ValueAsString"]
    
    
def retrieve_latest_features(data,record_id):
    print("retrieve_latest_features")
    response = client.get_record(
    FeatureGroupName='my-features',
    RecordIdentifierValueAsString=str(record_id),
    )
    
    return response["Record"][0]["ValueAsString"]
    

def input_handler(data, context):
    """ Pre-process request input before it is sent to TensorFlow Serving REST API
    Args:
        data (obj): the request data, in format of dict or string
        context (Context): an object containing request and configuration details
    Returns:
        (dict): a JSON-serializable dict that contains request body and headers
    """

    if context.request_content_type == 'application/json':

        d = data.read().decode('utf-8')
      
        input_data = json.loads(d)

        assert input_data["data"] == json.loads(retrieve_latest_features(input_data["data"],input_data["id"]))
      
        return json.dumps({"inputs" : input_data["data"]})


    raise ValueError('{{"error": "unsupported content type {}"}}'.format(
        context.request_content_type or "unknown"))


def output_handler(data, context):
    """Post-process TensorFlow Serving output before it is returned to the client.
    Args:
        data (obj): the TensorFlow serving response
        context (Context): an object containing request and configuration details
    Returns:
        (bytes, string): data to return to client, response content type
    """
    if data.status_code != 200:
        raise ValueError(data.content.decode('utf-8'))
    response_content_type = context.accept_header
    prediction = data.content
    return prediction, response_content_type
