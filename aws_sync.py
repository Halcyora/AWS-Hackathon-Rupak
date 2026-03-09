import boto3
from botocore.exceptions import NoCredentialsError
import os

# Replace these with your actual keys and bucket name
AWS_ACCESS_KEY = ""
AWS_SECRET_KEY = ""
BUCKET_NAME = "fractallens-edge-sync-rupak"

def sync_db_to_aws():
    """Uploads the local SQLite database to Amazon S3."""
    local_file = 'fractallens_edge.db'
    s3_file_name = f"sync_backups/fractallens_edge.db"
    
    if not os.path.exists(local_file):
        return False, "Local database not found."

    s3 = boto3.client('s3', aws_access_key_id=AWS_ACCESS_KEY, aws_secret_access_key=AWS_SECRET_KEY)
    
    try:
        s3.upload_file(local_file, BUCKET_NAME, s3_file_name)
        return True, "Successfully synced to AWS S3!"
    except NoCredentialsError:
        return False, "AWS credentials not available."
    except Exception as e:
        return False, str(e)