import boto3
import os
import pandas as pd

class s3_file_count(object):
    """
        Class to access spectogram data from s3 and Pre-train classifier.
    """
    def __init__(self, aws_access_key_id, aws_secret_access_key, bucket_name, prefix):
        self.aws_access_key_id = aws_access_key_id
        self.aws_secret_access_key = aws_secret_access_key
        self.bucket_name = bucket_name
        self.prefix = prefix

    def file_counter(self):
        s3 = boto3.resource('s3',aws_access_key_id= self.aws_access_key_id,
                aws_secret_access_key= self.aws_secret_access_key)
        bucket = s3.Bucket(self.bucket_name)
        folder_name = set()
        for obj in bucket.objects.filter(Delimiter='', Prefix= self.prefix):
            if obj.key.endswith('.jpg') or obj.key.endswith('.wav'):
                 folder_name.add(obj.key.split('/')[1] + '/' +obj.key.split('/')[2])    
        df = pd.DataFrame()
        for files in folder_name:
            spectrogram_files = 0
            sound_files = 0
            for obj in bucket.objects.filter(Delimiter='', Prefix= os.path.join(self.prefix, files)):
                if obj.key.endswith('.jpg'):
                    spectrogram_files += 1
                elif obj.key.endswith('.wav'):
                    sound_files += 1
            tempDF = pd.DataFrame([[files, spectrogram_files, sound_files]],columns=['Name','Spectrogram_files', 'Sound_files'])
            df = pd.concat([df,tempDF])
        cols = df.columns
        df_markdown = pd.DataFrame([['---',]*len(cols)], columns=cols)
        final_df = pd.concat([df_markdown, df])
        final_df.to_csv('test.md', sep="|", index=False)

if __name__ == '__main__':
    aws_access_key_id = ''
    aws_secret_access_key = ''
    bucket_name = 'whaledr'
    prefix = 'megaptera'    
    s3_file_count = s3_file_count(aws_access_key_id, aws_secret_access_key, bucket_name, prefix)
    s3_file_count.file_counter()
