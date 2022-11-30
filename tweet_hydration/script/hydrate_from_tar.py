import pandas as pd
import numpy as np
import internetarchive as ia
import tarfile
import os
import bz2
import time
import concurrent.futures
from itertools import repeat
import argparse
import json
import shutil
import multiprocessing as mp
import functools
print = functools.partial(print, flush=True)


def pickle_day(day, year_month, df_ym, month_year_dl_dict, extract_path, pkl_path):
    dl_type = month_year_dl_dict[year_month]
    archive_year = year_month[:4]
    archive_month = year_month[5:]
    archive_day = str(day).zfill(2)
    std_YYYY_MM = ['archiveteam-twitter-stream-YYYY-MM.tar']
    print(f"Starting pickling for {year_month}-{archive_day}")
    if dl_type in std_YYYY_MM:
        data_day = df_ym.loc[df_ym['tweet_day'] == day]
        for hour in range(24):
            # get bz2 filepath
            archive_hour = str(hour).zfill(2)
            archive_hour_path = f"{extract_path}/archive_{year_month}/{archive_month}/{archive_day}/{archive_hour}"
            try:
                # get filenames to loop
                bz2_filenames = os.listdir(archive_hour_path)
                for bz2_file in bz2_filenames:
                    # open bz2 file and save to df
                    bz2_file_path = f"{archive_hour_path}/{bz2_file}"
                    try:
                        with bz2.open(bz2_file_path, 'rt', encoding = 'utf-8') as f:
                            tweet_json_df = pd.read_json(f, lines = True)
                        tweet_json_df = tweet_json_df[['id', 'text']]
                        # extract relevant tweets
                        data_day = pd.merge(data_day, tweet_json_df, left_on='tweet_id', right_on='id', how ='left').drop('id', axis=1)
                        data_day['tweet_text'].update(data_day['text'])
                        data_day = data_day.drop('text', axis = 1)
                    except EOFError:
                        print(f"EOFError for {year_month}-{archive_day}-{str(hour).zfill(2)}")
            except FileNotFoundError:
                pass
        # select only tweets where rehydration was successful
        data_day = data_day[data_day['tweet_text'] != '']
        # create dir for rehydrated tweets
        rehydrated_dir = f"{pkl_path}/{archive_year}/{archive_month}"
        try:
            os.makedirs(rehydrated_dir)
        except FileExistsError:
            pass
        # save rehydrated data by day
        data_day.to_pickle(f"{rehydrated_dir}/en_{archive_year}_{archive_month}_{archive_day}.pkl")
        return f"DONE {archive_year}_{archive_month}_{archive_day}"
    else:
        # subset df by day
        data_day = df_ym.loc[df_ym['tweet_day'] == day]
        # get list of bz2 files to check
        bz2_path = f"{extract_path}/archive_{year_month}/{archive_day}"
        bz2_list = []
        for root, _, files in os.walk(bz2_path):
            for file in files:
                bz2_file_path = os.path.join(root, file)
                bz2_list.append(bz2_file_path)
        # loop through all bz2 files
        for bz2_file in bz2_list:
            try:
                # open bz2 file and save to df
                with bz2.open(bz2_file, 'rt', encoding = 'utf-8') as f:
                    tweet_json_df = pd.read_json(f, lines = True)
                tweet_json_df = tweet_json_df[['id', 'text']]
                #extract relevant tweets
                data_day = pd.merge(data_day, tweet_json_df, left_on='tweet_id', right_on='id', how ='left').drop('id', axis=1)
                data_day['tweet_text'].update(data_day['text'])
                data_day = data_day.drop('text', axis = 1)
            except EOFError:
                print(f"EOFError for {year_month}-{archive_day}-{str(hour).zfill(2)}")
        # select only tweets where rehydration was successful
        data_day = data_day[data_day['tweet_text'] != '']
        # create dir for rehydrated tweets
        rehydrated_dir = f"{pkl_path}/{archive_year}/{archive_month}"
        try:
            os.makedirs(rehydrated_dir)
        except FileExistsError:
            pass
        # save rehydrated data by day
        data_day.to_pickle(f"{rehydrated_dir}/en_{archive_year}_{archive_month}_{archive_day}.pkl")
        return f"DONE {archive_year}_{archive_month}_{archive_day}"


def unzip_tars(year_month, month_year_dl_dict, download_path, extract_path, pkl_path):
    dl_type = month_year_dl_dict[year_month]
    archive_year = year_month[:4]
    archive_month = year_month[5:]
    archive_ym_str = f"archiveteam-twitter-stream-{year_month}"
    # tar file formats
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
    # extract files
    print(f"Extracting files for {year_month}...")
    if dl_type in std_YYYY_MM:
        tar_file = f"{download_path}/archive_{year_month}/{archive_ym_str}/archiveteam-twitter-stream-{year_month}.tar"
        archive_tar = tarfile.open(tar_file, 'r')
        archive_tar.extractall(path=f"{extract_path}/archive_{year_month}")
    elif dl_type in std_YYYY_MM_DD:
        tar_path = f"{download_path}/archive_{year_month}/{archive_ym_str}"
        tars_list = os.listdir(tar_path)
        for tar_file in tars_list:
            archive_day = tar_file[-6:-4]
            tar_file_full_path = f"{download_path}/archive_{year_month}/{archive_ym_str}/{tar_file}"
            archive_tar = tarfile.open(tar_file_full_path, 'r')
            archive_tar.extractall(path=f"{extract_path}/archive_{year_month}/{archive_day}")
    elif dl_type in overlap_months:
        if dl_type[:9] == 'in2017-11':
            archive_ym_str = 'archiveteam-twitter-stream-2017-11'
            tar_path = f"{download_path}/archive_{year_month}/{archive_ym_str}"
            tars_list = os.listdir(tar_path)
            for tar_file in tars_list:
                archive_day = tar_file[-6:-4]
                tar_file_full_path = f"{download_path}/archive_{year_month}/{archive_ym_str}/{tar_file}"
                archive_tar = tarfile.open(tar_file_full_path, 'r')
                archive_tar.extractall(path=f"{extract_path}/archive_{year_month}/{archive_day}")
        elif dl_type[:9] == 'in2019-08':
            archive_ym_str = 'archiveteam-twitter-stream-2019-08'
            tar_path = f"{download_path}/archive_{year_month}/{archive_ym_str}"
            tars_list = os.listdir(tar_path)
            for tar_file in tars_list:
                archive_day = tar_file[-6:-4]
                tar_file_full_path = f"{download_path}/archive_{year_month}/{archive_ym_str}/{tar_file}"
                archive_tar = tarfile.open(tar_file_full_path, 'r')
                archive_tar.extractall(path=f"{extract_path}/archive_{year_month}/{archive_day}")
    return f"DONE EXTRACTING {archive_year}_{archive_month}"    


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

    # extraction task
    print("Starting extraction...")
    t1 = time.perf_counter()
    download_path = params.download_path
    extract_path = params.extract_path
    pkl_path = params.pkl_path
    year_month_list = [str(x) for x in data['tweet_year_month'].unique()][:21]
    with concurrent.futures.ThreadPoolExecutor() as executor:
        out_list = executor.map(
            unzip_tars,
            year_month_list,
            repeat(month_year_dl_dict),
            repeat(download_path),
            repeat(extract_path),
            repeat(pkl_path)
        )
        for out in out_list:
            print(out)
    t2 = time.perf_counter()
    print(f"Took {round((t2 - t1) / 60, 2)} minutes for extraction task")

    # extracted tweets to pickle
    print("Starting pickle process...")
    t1 = time.perf_counter()
    for ym in year_month_list:
        t3 = time.perf_counter()
        df_ym = data.loc[data['tweet_year_month'] == ym]
        target_days = list(df_ym['tweet_day'].unique())
        with concurrent.futures.ProcessPoolExecutor(mp_context=mp.get_context('fork')) as executor:
            out_list = executor.map(
                pickle_day,
                target_days,
                repeat(ym),
                repeat(df_ym),
                repeat(month_year_dl_dict),
                repeat(extract_path),
                repeat(pkl_path)
            )
            for out in out_list:
                print(out)
        t4 = time.perf_counter()
        print(f"Took {round((t4 - t3) / 60, 2)} minutes to pickle {ym}")
    t2 = time.perf_counter()
    print(f"Took {round((t2 - t1) / 60, 2)} minutes for pickle task")
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Rehydrate tweets")
    parser.add_argument("--tm_senti_path", type=str, default="/home/emanwong/eecs595/final_proj/tm_senti_dataset")
    parser.add_argument("--download_path", type=str, default="/scratch/eecs595f22_class_root/eecs595f22_class/emanwong/archive_downloads")
    parser.add_argument("--extract_path", type=str, default="/scratch/eecs595f22_class_root/eecs595f22_class/emanwong/archive_extracted")
    parser.add_argument("--pkl_path", type=str, default="/home/emanwong/eecs595/final_proj/rehydrated_tweets")

    params, unknown = parser.parse_known_args()
    main(params)