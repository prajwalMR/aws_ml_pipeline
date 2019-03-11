import pandas as pd
import boto3
from time import gmtime, strftime

sm = boto3.client('sagemaker')
model_name = <name of the model that was created on sagemaker during training phase>

def predict(prep_data_path, prediction_output_path):
    
    batch_job_name = 'Prediction-Batch-Transform-Job-' + strftime("%Y-%m-%d-%H-%M-%S", gmtime())
    
    request = \
    {
        "TransformJobName": batch_job_name,
        "ModelName": model_name,
        "MaxConcurrentTransforms": 4,
        "MaxPayloadInMB": 6,
        "BatchStrategy": "MultiRecord",
        "TransformOutput": {
            "S3OutputPath": prediction_output_path
        },
        "TransformInput": {
            "DataSource": {
                "S3DataSource": {
                    "S3DataType": "S3Prefix",
                    "S3Uri": prep_data_path
                }
            },
            "ContentType": "text/csv",
            "SplitType": "Line",
            "CompressionType": "None"
        },
        "TransformResources": {
                "InstanceType": "ml.c4.xlarge",
                "InstanceCount": 1
        }
    }

    sm.create_transform_job(**request)
    
    print("Created Transform job with name: ", batch_job_name)
    
    print(batch_job_name)
    
    ### Wait until job completion
    while(True):
        response = sm.describe_transform_job(TransformJobName=batch_job_name)
        status = response['TransformJobStatus']
        if  status == 'Completed':
            print("Transform job ended with status: " + status)
            break
        if status == 'Failed':
            message = response['FailureReason']
            print('Transform failed with the following error: {}'.format(message))
            raise Exception('Transform job failed') 
        print("Transform job is in status: " + status)    
        time.sleep(30)
    
    print("Prediction job Completed")