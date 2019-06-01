import boto3

class s3_file_count(object):
    """
        Class to access spectogram data from s3 and Pre-train classifier.
    """
    def __init__(self, aws_access_key_id, aws_secret_access_key, bucket_name):
        self.aws_access_key_id = aws_access_key_id
        self.aws_secret_access_key = aws_secret_access_key
        self.bucket_name = bucket_name
