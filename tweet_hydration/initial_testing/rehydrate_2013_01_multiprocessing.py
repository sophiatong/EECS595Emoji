import pandas as pd
import numpy as np
import internetarchive as ia
import tarfile
import os
import bz2
import shutil
import json
import re
from tqdm import tqdm
import time
import concurrent.futures
import multiprocessing as mp
from itertools import repeat
import argparse


def pickle_rehydrated_day(day, data_ym, archive_ym_str, archive_year, archive_month, rehydrated_dir):
    #print(f"processing DAY {day}")

    # subset to df by day
    data_day = data_ym.loc[data_ym['tweet_day'] == day]

    archive_day = str(day).zfill(2)

    for hour in range(24):
        # get bz2 file path
        archive_hour = str(hour).zfill(2)
        archive_hour_path = f"{archive_ym_str}/{archive_month}/{archive_day}/{archive_hour}"
        try:
            # get filenames to loop
            bz2_filenames = os.listdir(archive_hour_path)

            #print(f"processing HOUR {archive_hour}")
            for bz2_file in bz2_filenames:
                # open bz2 file and save to df
                bz2_file_path = f"{archive_hour_path}/{bz2_file}"
                with bz2.open(bz2_file_path, 'rt', encoding='utf-8') as f:
                    tweet_json_df = pd.read_json(f, lines = True)
                tweet_json_df = tweet_json_df[['id', 'text']]

                # extract relevant tweets
                data_day = pd.merge(data_day, tweet_json_df, left_on='tweet_id', right_on='id', how ='left').drop('id', axis=1)
                data_day['tweet_text'].update(data_day['text'])
                data_day = data_day.drop('text', axis = 1)

        except FileNotFoundError:
            #print(f"No data for HOUR {archive_hour}")
            pass

    # select only tweets where rehydration was successful
    data_day = data_day[data_day['tweet_text'] != '']
    # save rehydrated data by day
    data_day.to_pickle(f"{rehydrated_dir}/en_{archive_year}_{archive_month}_{archive_day}.pkl")
    return f"DONE {archive_year}_{archive_month}_{archive_day}"


def main():
    # TM-Senti file dir and names
    data_dir = 'tm_senti_dataset'
    en_data_file = 'en-ids.tsv'
    tm_senti_path = params.tm_senti_path

    # read in english tsv
    print("Start reading TSV...")
    t1 = time.perf_counter()
    data = pd.read_csv(f"{tm_senti_path}/{data_dir}/{en_data_file}",sep='\t', header=None)
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

    # subset tweets by year month
    ym = '2013-01' # hard coded for this notebook
    data_ym = data.loc[data['tweet_year_month'] == ym]

    # get year and month to get retrieve
    archive_year = pd.to_datetime(str(ym)).year
    archive_month = str(pd.to_datetime(str(ym)).month).zfill(2)

    # create archive item 
    archive_ym_str = f"archiveteam-twitter-stream-{archive_year}-{archive_month}"
    archive_ym = ia.get_item(archive_ym_str)

    # download archive
    print("Starting archive download...")
    t1 = time.perf_counter()
    archive_ym_tar_filename = f"archiveteam-twitter-stream-{archive_year}-{archive_month}.tar"
    archive_ym.download(archive_ym_tar_filename)
    print("Finished archive download!")
    t2 = time.perf_counter()
    print(f"Took {round((t2 - t1) / 60, 2)} minutes to download archive")

    # open tar and extract
    print("Starting tar extraction...")
    t1 = time.perf_counter()
    extract_path = params.extract_path
    tar_file = f"{archive_ym_str}/{archive_ym_tar_filename}"
    archive_tar = tarfile.open(tar_file, 'r')
    archive_tar.extractall(path=f"{extract_path}/{archive_ym_str}")
    print("Finished extracting tar!")
    t2 = time.perf_counter()
    print(f"Took {round((t2 - t1) / 60, 2)} minutes to extract tar")

    # create dir for rehydrated tweets
    rehydrated_dir = f"./rehydrated_tweets/{archive_year}/{archive_month}"
    try:
        os.makedirs(rehydrated_dir)
    except FileExistsError:
        pass

    # use multiprocessing to run pickle_rehydrated_day
    print("Starting multiprocessing...")
    t1 = time.perf_counter()
    target_days = list(data_ym['tweet_day'].unique())
    with concurrent.futures.ProcessPoolExecutor(mp_context=mp.get_context('fork')) as executor:
        out_list = executor.map(pickle_rehydrated_day, 
                                target_days,
                                repeat(data_ym),
                                repeat(archive_ym_str),
                                repeat(archive_year), 
                                repeat(archive_month), 
                                repeat(rehydrated_dir)
        )
        for out in out_list:
            print(out)
    t2 = time.perf_counter()
    print(f"Took {round((t2 - t1) / 60, 2)} minutes for multiprocessing task")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Rehydrate tweets")
    parser.add_argument("--tm_senti_path", type=str, default="/home/emanwong/eecs595/final_proj")
    parser.add_argument("--extract_path", type=str, default="/scratch/eecs595f22_class_root/eecs595f22_class/emanwong")

    params, unknown = parser.parse_known_args()
    main(params)