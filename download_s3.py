import os
import io
from itertools import repeat
from multiprocessing.pool import ThreadPool
from tarfile import TarFile, is_tarfile
from zipfile import ZipFile, is_zipfile
import boto3
import requests
from tqdm import tqdm

# Initialize S3 client
s3_client = boto3.client('s3', endpoint_url='https://s3-haosu.nrp-nautilus.io',
                         aws_access_key_id='OQXE5KU6208C9ULM9CZL',
                         aws_secret_access_key='tgzqkeWxfcwzRil8uI5lvrBxvD8qg0xsbQ1Leo3n')


def s3_file_exists(bucket_name, object_key):
    try:
        s3_client.head_object(Bucket=bucket_name, Key=object_key)
        print(f"Checked existence: s3://{bucket_name}/{object_key} exists.")
        return True
    except Exception as e:
        print(f"Checked existence: s3://{bucket_name}/{object_key} does not exist or error occurred: {e}")
        return False


def upload_to_s3(bucket_name, object_key, file_obj):
    print(f"Preparing to upload to s3://{bucket_name}/{object_key}...")

    if s3_file_exists(bucket_name, object_key):
        print(f"File s3://{bucket_name}/{object_key} already exists, skipping upload.")
        return

    try:
        s3_client.upload_fileobj(file_obj, bucket_name, object_key)
        print(f"Uploaded to s3://{bucket_name}/{object_key}")
    except Exception as e:
        print(f"Failed to upload to s3://{bucket_name}/{object_key}: {e}")


def process_zip_file(file_obj, s3_bucket, s3_prefix, exclude=(".DS_Store", "__MACOSX")):
    print(f"Processing ZIP file for upload to {s3_bucket}/{s3_prefix}")
    with ZipFile(file_obj) as zipObj:
        for f in zipObj.namelist():
            if all(x not in f for x in exclude):
                s3_object_key = f"{s3_prefix}/{f}"
                print(f"Processing file {f} from ZIP for upload to s3://{s3_bucket}/{s3_object_key}")
                if s3_file_exists(s3_bucket, s3_object_key):
                    print(f"File s3://{s3_bucket}/{s3_object_key} already exists, skipping.")
                    continue

                with zipObj.open(f) as extracted_file:
                    upload_to_s3(s3_bucket, s3_object_key, extracted_file)


def process_tar_file(file_obj, s3_bucket, s3_prefix):
    print(f"Processing TAR file for upload to {s3_bucket}/{s3_prefix}")
    with TarFile.open(fileobj=file_obj) as tarObj:
        for member in tarObj.getmembers():
            if member.isfile():
                s3_object_key = f"{s3_prefix}/{member.name}"
                print(f"Processing file {member.name} from TAR for upload to s3://{s3_bucket}/{s3_object_key}")
                if s3_file_exists(s3_bucket, s3_object_key):
                    print(f"File s3://{s3_bucket}/{s3_object_key} already exists, skipping.")
                    continue

                extracted_file = tarObj.extractfile(member)
                if extracted_file:
                    upload_to_s3(s3_bucket, s3_object_key, extracted_file)


def download_and_upload(urls, s3_bucket, s3_prefix, unzip=True, curl=True, threads=1, retry=3):
    def download_one(url, s3_bucket, s3_prefix):
        file_name = url.split('/')[-1]
        s3_object_key = f"{s3_prefix}/{file_name}"

        if s3_file_exists(s3_bucket, s3_object_key):
            print(f"File s3://{s3_bucket}/{s3_object_key} already exists, skipping download.")
            return

        success = False
        print(f"Downloading {url}...")
        for i in range(retry + 1):
            try:
                response = requests.get(url, stream=True)
                response.raise_for_status()
                total_size = int(response.headers.get('content-length', 0))
                file_obj = io.BytesIO()

                with tqdm(total=total_size, unit='B', unit_scale=True, desc=file_name) as pbar:
                    for chunk in response.iter_content(1024):
                        if chunk:
                            file_obj.write(chunk)
                            pbar.update(len(chunk))

                success = True
                break
            except Exception as e:
                print(f"Failed to download {url}: {e}")
                success = False

            if not success and i < retry:
                print(f"Retry {i + 1}/{retry} for {url}")
            elif not success and i == retry:
                print(f"Failed to download {url} after {retry} attempts.")

        if success:
            file_obj.seek(0)
            if unzip and (is_zipfile(file_obj) or is_tarfile(file_obj)):
                print(f"Processing {file_name} after download.")
                if is_zipfile(file_obj):
                    process_zip_file(file_obj, s3_bucket, s3_prefix)
                elif is_tarfile(file_obj):
                    process_tar_file(file_obj, s3_bucket, s3_prefix)
            else:
                upload_to_s3(s3_bucket, s3_object_key, file_obj)

    if threads > 1:
        pool = ThreadPool(threads)
        for _ in tqdm(pool.imap(lambda x: download_one(*x), zip(urls, repeat(s3_bucket), repeat(s3_prefix))), total=len(urls)):
            pass
        pool.close()
        pool.join()
    else:
        for url in tqdm(urls, desc="Downloading"):
            download_one(url, s3_bucket, s3_prefix)


# Example usage
s3_bucket = 'zhanling-vlm'
s3_prefix = 'object365'

for split, patches in [("train", 50 + 1), ("val", 43 + 1)]:
    print(f"Processing {split} in {patches} patches ...")
    base_url = f"https://dorc.ks3-cn-beijing.ksyun.com/data-set/2020Objects365%E6%95%B0%E6%8D%AE%E9%9B%86/{split}/"

    if split == "train":
        download_and_upload(
            [f"{base_url}zhiyuan_objv2_{split}.tar.gz"], s3_bucket=s3_bucket, s3_prefix=s3_prefix
        )
        patch_urls = [f"{base_url}patch{i}.tar.gz" for i in range(patches)]
    #     download_and_upload(
    #         patch_urls, s3_bucket=s3_bucket, s3_prefix=f"{s3_prefix}/images/train", curl=True, threads=64
    #     )
    # elif split == "val":
    #     download_and_upload(
    #         [f"{base_url}zhiyuan_objv2_{split}.json"], s3_bucket=s3_bucket, s3_prefix=s3_prefix
    #     )
    #     patch_urls_v1 = [f"{base_url}images/v1/patch{i}.tar.gz" for i in range(15 + 1)]
    #     download_and_upload(
    #         patch_urls_v1, s3_bucket=s3_bucket, s3_prefix=f"{s3_prefix}/images/val", curl=True, threads=64
    #     )
    #     patch_urls_v2 = [f"{base_url}images/v2/patch{i}.tar.gz" for i in range(16, patches)]
    #     download_and_upload(
    #         patch_urls_v2, s3_bucket=s3_bucket, s3_prefix=f"{s3_prefix}/images/val", curl=True, threads=64
        )
