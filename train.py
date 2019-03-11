import boto3
import sagemaker
from sagemaker.amazon.amazon_estimator import get_image_uri

sess = sagemaker.Session()
region = boto3.Session().region_name
container = get_image_uri(region, 'xgboost')

bucket = 'machine-learning'                     
prefix = 'demo/train/'
role = sagemaker.get_execution_role()

s3_input_train = sagemaker.s3_input(s3_data='s3://{}/{}/train.csv'.format(bucket, prefix), content_type='csv')
s3_input_validation = sagemaker.s3_input(s3_data='s3://{}/{}/validate.csv'.format(bucket, prefix), content_type='csv')
s3_data = {'train': s3_input_train, 'validation': s3_input_validation}

xgb = sagemaker.estimator.Estimator(container,
                                    role, 
                                    train_instance_count=1, 
                                    train_instance_type='ml.m4.2xlarge',
                                    input_mode="File",
                                    output_path='s3://{}/{}/output'.format(bucket, prefix),
                                    sagemaker_session=sess)

xgb.set_hyperparameters(objective='multi:softmax', num_round=100 , num_class=2, max_depth=3, eta=0.05, silent=0, min_child_weight=2)

xgb.fit(s3_data)
