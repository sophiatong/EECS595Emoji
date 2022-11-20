import pandas as pd
import numpy as np
import internetarchive as ia
import tarfile
import os
import bz2
import time
import concurrent.futures
import multiprocessing as mp
from itertools import repeat
import argparse
import json


def download_tweet_archive(year_month, month_year_dl_dict, download_path, download_mode):

    download_path_ym = f"{download_path}/archive_{year_month}"

    if download_mode == 'download':
        dry_run = False
    elif download_mode == 'test':
        dry_run = True

    archive_year = year_month[:4]
    archive_month = year_month[5:]
    dl_type = month_year_dl_dict[year_month]
    std_YYYY_MM = ['archiveteam-twitter-stream-YYYY-MM.tar']
    std_YYYY_MM_DD = [
        'twitter-YYYY-MM-DD.tar',
        'twitter-stream-YYYY-MM-DD.tar',
        'twitter_stream_YYYY_MM_DD.tar'
    ]
    overlap_months = [
        'in2017-11/twitter-stream-YYYY-MM-DD.tar',
        'in2019-08/twitter_stream_YYYY_MM_DD.tar'
    ]


    if dl_type in std_YYYY_MM:
        # create internet archive item
        archive_ym_str = f"archiveteam-twitter-stream-{archive_year}-{archive_month}"
        archive_ym = ia.get_item(archive_ym_str)
        # download tar file
        archive_ym_tar_filename = f"archiveteam-twitter-stream-{archive_year}-{archive_month}.tar"
        archive_ym.download(
            archive_ym_tar_filename, 
            destdir = download_path_ym,
            dry_run = dry_run
        )
    elif dl_type in std_YYYY_MM_DD:
        # create internet archive item
        archive_ym_str = f"archiveteam-twitter-stream-{archive_year}-{archive_month}"
        archive_ym = ia.get_item(archive_ym_str)
        # get indexes of tars
        tar_idx = []
        for i in range(len(archive_ym.files)):
            if archive_ym.files[i]['format'] == 'TAR':
                tar_idx.append(i)
        # get filenames of valid tars
        dl_tars = []
        for idx in tar_idx:
            tar_name = archive_ym.files[idx]['name']
            valid_tar = False
            valid_suffix = '.tar'
            if dl_type == 'twitter-YYYY-MM-DD.tar':
                valid_prefix = f"twitter-{archive_year}-{archive_month}-"
                if tar_name[:16] == valid_prefix and tar_name[18:] == valid_suffix:
                    valid_tar = True
            elif dl_type == 'twitter-stream-YYYY-MM-DD.tar':
                valid_prefix = f"twitter-stream-{archive_year}-{archive_month}-"
                if tar_name[:23] == valid_prefix and tar_name[25:] == valid_suffix:
                    valid_tar = True
            elif dl_type == 'twitter_stream_YYYY_MM_DD.tar':
                valid_prefix = f"twitter_stream_{archive_year}_{archive_month}_"
                if tar_name[:23] == valid_prefix and tar_name[25:] == valid_suffix:
                    valid_tar = True
            if valid_tar:
                dl_tars.append(tar_name)
        # download each tar 
        for filename in dl_tars:
            archive_ym.download(
                filename,
                destdir = download_path_ym,
                dry_run = dry_run
            )
    elif dl_type in overlap_months:
        if dl_type[:9] == 'in2017-11':
            # create internet archive item
            archive_ym_str = 'archiveteam-twitter-stream-2017-11'
            archive_ym = ia.get_item(archive_ym_str)
            # get indexes of tars
            tar_idx = []
            for i in range(len(archive_ym.files)):
                if archive_ym.files[i]['format'] == 'TAR':
                    tar_idx.append(i)
            # get filenames of valid tars
            dl_tars = []
            for idx in tar_idx:
                tar_name = archive_ym.files[idx]['name']
                valid_suffix = '.tar'
                valid_prefix = f"twitter-stream-{archive_year}-{archive_month}-"
                if tar_name[:23] == valid_prefix and tar_name[25:] == valid_suffix:
                    dl_tars.append(tar_name)
        elif dl_type[:9] == 'in2019-08':
            # create internet archive item
            archive_ym_str = 'archiveteam-twitter-stream-2019-08'
            archive_ym = ia.get_item(archive_ym_str)
            # get indexes of tars
            tar_idx = []
            for i in range(len(archive_ym.files)):
                if archive_ym.files[i]['format'] == 'TAR':
                    tar_idx.append(i)
            # get filenames of valid tars
            dl_tars = []
            for idx in tar_idx:
                tar_name = archive_ym.files[idx]['name']
                valid_suffix = '.tar'
                valid_prefix = f"twitter_stream_{archive_year}_{archive_month}_"
                if tar_name[:23] == valid_prefix and tar_name[25:] == valid_suffix:
                    dl_tars.append(tar_name)
        # download list of valid tars
        for filename in dl_tars:
            archive_ym.download(
                filename,
                destdir = download_path_ym,
                dry_run = dry_run
            )
    
    
    return f"DONE downloading {year_month} archive"


def main(params):
    # TM-Senti file dir and names
    en_data_file = 'en-ids.tsv'
    tm_senti_path = params.tm_senti_path

    # read in english tsv
    print("Start reading TSV...")
    t1 = time.perf_counter()
    data = pd.read_csv(f"{tm_senti_path}/{en_data_file}",sep='\t', header=None)
    print("Finished reading TSV")
    t2 = time.perf_counter()
    print(f"Took {round((t2 - t1) / 60, 2)} minutes to read TSV")

    # format data
    print("Start formatting TSV")
    t1 = time.perf_counter()
    data = data.drop_duplicates()
    data = data.rename(columns={0:'tweet_date',
                                    1:'tweet_id',
                                    2:'sentiment',
                                    3:'emojis'})
    data['tweet_date'] = pd.to_datetime(data['tweet_date'], format='%Y-%m-%d')
    data['tweet_day'] = data['tweet_date'].dt.day
    data['tweet_month'] = data['tweet_date'].dt.month
    data['tweet_year'] = data['tweet_date'].dt.year
    data['tweet_year_month'] = data['tweet_date'].dt.to_period('M')
    data['sentiment'] = data.sentiment.map({'pos':1, 'neg':0})
    data['tweet_text'] = ''
    print("Finished formatting TSV")
    t2 = time.perf_counter()
    print(f"Took {round((t2 - t1) / 60, 2)} minutes to format TSV")

    # read in dictionary on download info
    with open('month_year_dl.json') as f:
        month_year_dl_dict = json.load(f)

    # use multithreading to run download_tweet_archive
    print("Starting download...")
    t1 = time.perf_counter()
    year_month_list = [str(x) for x in data['tweet_year_month'].unique()]
    with concurrent.futures.ThreadPoolExecutor() as executor:
        out_list = executor.map(download_tweet_archive,
                                year_month_list,
                                repeat(month_year_dl_dict),
                                repeat(params.download_path),
                                repeat(params.download_mode)
        )
        for out in out_list:
            print(out)
    t2 = time.perf_counter()
    print(f"Took {round((t2 - t1) / 60, 2)} minutes to download")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Rehydrate tweets")
    parser.add_argument("--tm_senti_path", type=str, default="/home/emanwong/eecs595/final_proj/tm_senti_dataset")
    parser.add_argument("--download_path", type=str, default="/scratch/eecs595f22_class_root/eecs595f22_class/emanwong/archive_downloads")
    parser.add_argument("--download_mode", type=str, default="download")

    params, unknown = parser.parse_known_args()
    main(params)