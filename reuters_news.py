import pandas as pd
import numpy as np
from google.oauth2 import service_account
import pandas_gbq
import json
import seaborn as sns
import warnings
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import dates
from elasticsearch import Elasticsearch
from IPython.core.display import display, HTML

warnings.filterwarnings('ignore')
plt.style.use('seaborn')
pd.options.display.max_columns = None
pd.set_option('display.max_colwidth', -1)
display(HTML("<style>.container { width:100% !important; }</style>"))

personal_cred = json.loads('{"type":"service_account","project_id":"tr-data-workbench","private_key_id":"ea7f32de4aab54d0fd0f1559df32ab605a336000","private_key":"-----BEGIN PRIVATE KEY-----\\nMIIEvgIBADANBgkqhkiG9w0BAQEFAASCBKgwggSkAgEAAoIBAQDEcaEi6k50GrZ2\\n9VeOUDIRtJsOE2uCSdgoDFjfwysThqgmjum9A5V+T5YLhtKgXRW87a5KjhaFnZ1w\\n4p8FHmKYQWyT6mBn75YkBjquBPbVU4dWtIAS21cH6rfSY6mt/yAKyhZNgO19JwZQ\\nhOVbjXSPWANdt9gaMRbK7MxWEJR9pYA7DKQoOYD2v/+2UDGHnvxCN3Q7u4RBToLu\\nX+FeTgwo/5X1cRQ5rUV0xmT69AjeYL5piygTvT+ZTE9kmAmvlGfjNJGhZFbOhUT+\\nqbUA7gomSpnLzcJJfFhIwXctBM7+aQ03RFOE0pVM5X4XBZyTcoKpoPkjYv+2vLk8\\nUKd8xlGPAgMBAAECggEAHE3NoceQtjnYaxtpGiAulqPRvh1ziA+LuCQkn1jyLptg\\nRENNz5C3TBO7fWhXlgjTcP1DTOAGxAm0Uud+x1tcFAPug6wUEpBQBFtsE6wIxIPc\\neX8C/7SKvaFGtbxBCR478lIGafRW7jQJPOC+YIqTw3OISKCMinmNhyockZSh+y2f\\nJfJ3CKUGlGvIN66l+lbG2WPjN44g9BufxE0B/5YUU7LbkHaFEcOYBTREJyTsQWZC\\n7YWsSELAlwvYvIoBVxY6q2PEKiuVFT31iQ6hrY29vb2DGIM1kaD8EWc22PM2Bhbw\\nvdTtvdIkX7vEfnySqJNroVVH9ZEQesDbz36f7ns3wQKBgQD2bHqWZyUWq+cidlYK\\nz6dFJc19WZs1RDl2LbsegYCXL2OxOvcjw1t5Tq5N13t7auA/tUOSZZNm/ReyMB20\\nO34KkwMMHjx7G3H5cdnpgArYGMiSeLqGvSYeeQ/cM4uuCS/06tf2W0SjYrS1Grqi\\nhQ54qsWBsNS2diHz6uSnoiDL5QKBgQDME+5QHCKIsJNoG9wjukv664o6Yv5QzsLR\\nCb/9lj1WWHu940Rc0320fIQi1JEdAQ+9kkjDyfOHX4BQWAuiRkNt+EBtNFwl/1Zh\\n/QvxfW7asniuBS+f6I2qvsKEDV78RMIzIrMLdzQYCUYkc3rgwfQnzBSwoPEq7dyZ\\nWj+Nj3IYYwKBgGXTk6BcxNWmeR8jeRCEmCEZqt0WTu1m4Lu1z/GeH83ZVj+hFwif\\nLEhnO92Mokjwc/O/akmvUYaoJMeL3GpifwXNk9/JCsLwJ1ulMR+suk5rh04HN0SA\\nGqVS7zvVzO1AfKGe/ViehDFjkzqW7fJEOyOfpXee43b2P7d0I8IlW5oFAoGBALes\\nGIixScOMaJ8CtBbTzR6Ab2AtSIaiEGTjSZCKb5oO+Q/mQ9jn3/NZPfO7LP/VvLz5\\ns9PDJxNnuKyXEaFza6zGCfY/cdAhBzYZ8W3fXq4mgsbclAhv3VEXbyo6foq9t6se\\nR9bB6njXn/GenbE73usSmmx97ZCdlDXRK/HUcPPRAoGBAMx+c7wwr3YMZN9DRQMA\\n/hTyvK3k77wtBQWbN3iWbGmQiPqEhEL2tZ9fxGqcp0PaCIGyckvAcaSLFHJGGdkj\\na9qy+f4X6tmgzNNkRbh+aTr8C4MRrdh6RyI5LN/DLLiQyoGiCQGJ8/9+bDTLAgkJ\\n3YXu2OixxafkFwWNun29H4hQ\\n-----END PRIVATE KEY-----\\n","client_email":"saj57p2ict19fn8cv1shmj744a4b0e@tr-data-workbench.iam.gserviceaccount.com","client_id":"115271726142601077489","auth_uri":"https://accounts.google.com/o/oauth2/auth","token_uri":"https://oauth2.googleapis.com/token","auth_provider_x509_cert_url":"https://www.googleapis.com/oauth2/v1/certs","client_x509_cert_url":"https://www.googleapis.com/robot/v1/metadata/x509/saj57p2ict19fn8cv1shmj744a4b0e%40tr-data-workbench.iam.gserviceaccount.com"}') # your personal key for Tick History on BigQuery
credentials = service_account.Credentials.from_service_account_info(personal_cred)

ELASTICSEARCH_API_KEY = 'WjdutlDA8C8RcIB7PJEAt8XncWS5bU1X5mFnMSvS' # your personal key for Data Science Accelerator access to News on Elasticsearch
ELASTICSEARCH_HOST = 'dsa-stg-newsarchive-api.fr-nonprod.aws.thomsonreuters.com'

# Creating Elasticsearch connection through api gateway
es = Elasticsearch(
    host=ELASTICSEARCH_HOST,
    port=443,
    headers={'X-api-key': ELASTICSEARCH_API_KEY}, 
    use_ssl=True, 
    timeout=30
)

index_name = 'newsarchive'

sql_qry = """
    SELECT 
        * 
    FROM 
        TRNA.news 
    WHERE 
        id = 'tr:L2N1RH1C6_1804046gbF7v5pWAYkV91d03Vvjmsye7cwgu5B6H5qlv-4297297477'
"""
trna_news_df_single_row = pandas_gbq.read_gbq(sql_qry, project_id="tr-data-workbench", credentials=credentials, dialect='standard')

print('NEWS HEADLINE: {0}'.format(trna_news_df_single_row.headline.values[0]))
display(trna_news_df_single_row)

