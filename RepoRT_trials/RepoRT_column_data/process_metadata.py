"""
The main use of this python script is to update the METADATA part of RepoRT data, the gradient info is not treated here. In concrete, these methods were implemented:
    1. For temperature, column length, column id, particle size and flowrate. If any of them is missing and if so the name of the column, the global mean is filled
    However, if the name is available, the mean of the subset of the dataset of the exact same name is used here; in case of the mean is not available,
    the global mean is used instead (This use to be the cases where the column is only used for one time).
    2. t0 value, is inferred using the length of the column (mm given, have to pass to cm), flowrate (mL/min) and the inner diameter (cm).
    t0 = V0 / 100F, where V0 is the void volume of the column (~66% total volume) and F is the flowrate, and here in RepoRT, the t0 is scaled by a factor of 100, maybe for
    machine learning training, but not sure.
"""

import pandas as pd
import numpy as np

filename = "C:/Users/leonz/Github_reps/TFG_git/RepoRT_trials/RepoRT_column_data/RepoRT_column_data.tsv"
df = pd.read_csv (filename, sep="\t", encoding= "utf-8")

def infer_t0_val (diameter, length, fr):
    """
    Used for inferring the t0. Here t0 is calculated as V0/T.
    In RepoRT, inner diameter is given in cm, length in mm and fr in mL/min.
    So we have to pass length (mm) to cm.
    """
    base_area = np.pi * (diameter / 2)**2
    return round(((0.66*base_area*length/10)/fr)/100, 5)

def process_column_data (df, save_filename):
    """
    Given the RepoRT METADATA dataframe.
    Returns an updated dataframe with the methods explained in the very beginning of this Script and save it to a .tsv file using the save_filename.
    """
    #Get a smaller df for faster iteration. The id column is not used.
    temp_df = df [df.columns[1:8]]
    # Create a dictionary with the column names as keys and the GLOBAL MEANS as the values.
    means_dict = {column : round(np.mean(temp_df[column]), 2) for column in temp_df.columns [2:]}

    #The updating process
    for index,row in temp_df.iterrows():
        for column in temp_df.columns [2:]:
            if pd.isnull(row[column]) and pd.isnull(row["column.name"]):
                # If the NAME AND THE COLUMN value BOTH MISSING.
                temp_df.loc[index,column] = means_dict[column] #Global mean used
            elif pd.isnull(row[column]) and pd.notnull(row["column.name"]):
                # If the name is not missing
                # Get the mean of subset of df where the name is the same
                column_name = row["column.name"]
                temp_mean = round(temp_df[temp_df["column.name"] == column_name] [column].mean(),2)
                # Check if the mean is null. If so, global mean is used instead.
                if pd.isnull(temp_mean):
                    temp_df.loc[index,column] = means_dict[column]
                else:
                    temp_df.loc[index,column] = temp_mean
    # Update the df
    df.update (temp_df)
    del temp_df, means_dict
    #Updating t0 value
    for index, row in df.iterrows():
        if row["column.t0"] == 0:
            temp_t0 = infer_t0_val(np.float64(row["column.id"]),
                                   np.float64(row["column.length"]),
                                   np.float64(row["column.flowrate"]))
            df.loc[index, "column.t0"] = temp_t0
        else:
            continue
    df.to_csv(save_filename, sep="\t", index=False)
    return df
save_filename = "C:/Users/leonz/Github_reps/TFG_git/RepoRT_trials/RepoRT_column_data/RepoRT_column_updated_metadata.tsv"
df = process_column_data(df, save_filename)