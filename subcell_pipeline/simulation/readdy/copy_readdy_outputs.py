import boto3


def copy_readdy_outputs():
    """
    Copy ReaDDy outputs from where they were saved from running
    https://github.com/simularium/readdy-models to have the same
    AWS S3 file structure as for Cytosim.
    """
    s3_client = boto3.client("s3")
    bucket = "readdy-working-bucket"
    src_name = "outputs/actin_compression_velocity="
    dest_name = "ACTIN_COMPRESSION_VELOCITY/outputs/ACTIN_COMPRESSION_VELOCITY"
    src_condition_keys = ["4.7", "15", "47", "150"]
    dest_condition_keys = ["0047", "0150", "0470", "1500"]
    n_replicates = 5
    
    for cond_ix in range(len(src_condition_keys)):
        for rep_ix in range(n_replicates):
            
            src_cond = src_condition_keys[cond_ix]
            src_path = f"{bucket}/{src_name}{src_cond}_{rep_ix}.h5"
            
            dest_cond = dest_condition_keys[cond_ix]
            dest_key = f"{dest_name}_{dest_cond}_{rep_ix}.h5"
            
            s3_client.copy_object(
                Bucket=bucket, 
                CopySource=src_path, 
                Key=dest_key,
            )
            
            print(f"copied {src_path} to {bucket}/{dest_key}")


if __name__ == "__main__":
    copy_readdy_outputs()