"""
Here the retention time data is fetched from RepoRT and a .tsv file is build from it. Not only t is more suitable than fetching it as
a request every time we want to train models (MPNN) with it, but also, it would be easier if we wanted to get mol descriptors or column conditions.
"""


# 0. Import modules

import pandas as pd
import numpy as np

# 1.Define some input to the function.
seed_url = 'https://raw.githubusercontent.com/michaelwitting/RepoRT/refs/heads/master/processed_data/'
save_dir = "C:/Users/leonz/Github_reps/TFG_git/RepoRT_trials/RepoRT_data/RepoRT_data2.tsv"

# 2. Define get data function (canonical + isomeric SMILES and the retention time)
def get_raw_data(num_repos, save_dir=save_dir, seed_url=seed_url):
    """
    Input: num_repos to fetch, and the seed url for RepoRT.
    Output: a table containing the data fetched from RepoRT
    """
    final_dataframe = pd.DataFrame()
    for num in range (1, num_repos):
        seed_num = str (num)
        while len (seed_num) < 4:
            seed_num = "0" + seed_num
        can_url = seed_url + seed_num + "/" + seed_num + "_rtdata_canonical_success.tsv"
        iso_url = seed_url + seed_num + "/" + seed_num + "_rtdata_isomeric_success.tsv"
        print (f"Fetching repo nº{seed_num}...")
        try:
            temp_dataframe_can = pd.read_csv(can_url, sep="\t", encoding="utf-8")
            temp_dataframe_iso = pd.read_csv(iso_url, sep="\t", encoding = "utf-8")
            final_dataframe = pd.concat ([final_dataframe, temp_dataframe_can, temp_dataframe_iso], ignore_index=True)
            print (f"Data successfully fetched from {seed_num}")
        except:
            print (f"The repo {seed_num} does not exist ")
    del temp_dataframe_can, temp_dataframe_iso
    final_dataframe.drop(columns=["comment"], axis=1)
    final_dataframe.to_csv(save_dir, sep="\t", index=False)

get_raw_data (439)

df = pd.read_csv (save_dir, sep="\t", encoding="utf-8")
dir_id_array = [ idmol.split ("_") [0] for idmol in df ["id"]]
dir_id_array = np.array(dir_id_array)
df.insert (loc=1, column="dir_id", value=dir_id_array, ) #For later to insert metadata.

df.to_csv (save_dir, sep="\t", index=False)