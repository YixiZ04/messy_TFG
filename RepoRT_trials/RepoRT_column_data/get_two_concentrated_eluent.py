"""
The main objective of this Python Script is to do the final touch for column modeling. Here the gradient information will be incorporated into metadata.
The main information will be the initial time for a step and its flowrate.
"""


import pandas as pd
import numpy as np


filename = "/home/user/YixiTFG/TFG_Yixi/RepoRT_trials/RepoRT_column_data/RepoRT_column_units_changed.tsv"
df = pd.read_csv (filename, sep = '\t')


def get_index_by_number (number):
    index = str(number)
    while len(index) < 4:
        index = "0" + index
    return index

base_url = "https://raw.githubusercontent.com/michaelwitting/RepoRT/refs/heads/master/processed_data/"
def fetch_grad_data (index, base_url=base_url):
    url = f"{base_url}{index}/{index}_gradient.tsv"
    temp_df = pd.read_csv (url, sep = '\t')
    return temp_df


new_df = df.loc[:, df.columns.str.contains("eluent.A", regex=False)]

def get_most_concentrated_two_components (serie):
    sorted_serie = serie.sort_values(ascending=False)
    return sorted_serie [:2]

def get_total_eluent_composition (df, eluent):
    eluent_type = "eluent." + str(eluent)
    new_df = pd.concat([df["id"],df.loc[:, df.columns.str.contains (eluent_type, regex=False)]], axis = 1)
    return new_df

eluent_datas = {
    "A":get_total_eluent_composition(df, "A"),
    "B":get_total_eluent_composition(df, "B"),
    "C":get_total_eluent_composition(df, "C"),
    "D":get_total_eluent_composition(df, "D"),
}

def get_eluent_composition_by_index (df, index):
    return df[df["id"] == index].iloc [:,1:]

def add_gradient_data (num_repos, eluent_datas):
    final_df = pd.DataFrame ()
    for index in range(1, num_repos):
        report_index = get_index_by_number(index) #index
        grad_df = fetch_grad_data(report_index) #Get df of grad
        temp_df2 = pd.DataFrame ()
        for grad_index, grad_row in grad_df.iterrows():
            info_serie = grad_row [1:5]
            most_concentrated_serie = get_most_concentrated_two_components(info_serie)
            temp_df2 = pd.concat ([temp_df2, grad_row[0], grad_row[5]], axis = 1)
            for conc_index in most_concentrated_serie.index:
                letter = conc_index.split (' ')[0]
                temp_df2 = pd.concat ([temp_df2, get_eluent_composition_by_index(eluent_datas[letter],index)], axis = 1)

add_gradient_data(2, eluent_datas)