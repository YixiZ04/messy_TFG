"""
The main objective here is to convert all the units to %, e.g., mM and uM.
The input .tsv file used here will be the file with the metadata updated.
The only difference between this result file are the scaling factors for converting mM or uM to % (m/v), thus the values for concentration are different.
Here the scaling factor is:
    % = mM * Mw /10000 = uM * Mw / 10000000
"""

import numpy as np
import pandas as pd

filename = "/home/user/YixiTFG/TFG_Yixi/RepoRT_trials/RepoRT_column_data/RepoRT_column_updated_metadata.tsv"
df = pd.read_csv(filename, sep = '\t')

# The dictionary for molecular weights, approx. values are used.
mws = {
    "acetic": 60,
    "phosphor": 98,
    "nh4ac": 77,
    "nh4form": 63,
    "nh4carb": 96,
    "nh4bicarb": 79,
    "nh4f": 37,
    "nh4oh": 35,
    "trieth": 101,
    "triprop": 143,
    "tribut": 185,
    "nndimethylhex":129,
    "medronic":176,
}

#Functions needed for processing gradient infos.

def get_molecule_name (column_name):
    """
    Given a column name in this format: Eluent.A.mol_name
    A string only containing the mol_name will be returned.
    """
    molecule = column_name.split(".")[2]
    return str(molecule)

def process_grad_data (df):
    """
    Given a df as input, all the units of concentration will be converted to %(m/v).
    Moreover, all the columns containing the "unit" information will be dropped.
    """
    # Iteration over rows in order to process the concentration info.
    for index, row in df.iterrows():
        col_index = 0
        for column in df.columns:
            col_index += 1
            if row [column] == "mM":
                mol_column = df.columns[col_index - 2] #Get access to the molecule's column.
                mol_name = get_molecule_name (mol_column)
                scale_factor = mws[mol_name] / 10000 # Mw/10000
                new_value = row[df.columns[col_index -2]] * scale_factor #mM*Mw/10000
                df [mol_column] =  df [mol_column].astype(np.float64) #Necessary because the dtype in the original dset is np.int64
                df.loc[index, mol_column] = np.float64(new_value)
            elif row [column] == "µM":
                mol_column = df.columns[col_index - 2]
                mol_name = get_molecule_name(mol_column)
                scale_factor = mws[mol_name] / 10000000 #The only difference here.
                new_value = row[df.columns[col_index - 2]] * scale_factor #uM*Mw/10000000
                df [mol_column] =  df [mol_column].astype(np.float64)
                df.loc[index, mol_column] = new_value
            else:
                continue
    # As all concentration data is expressed in % (m/v), the unit's columns are no longer needed, so just drop them.
    drop_column_array = []
    for column in df.columns:
        if ".unit" in column or ".start" in column or ".end" in column:
            drop_column_array.append (column)
        else:
            continue
    df = df.drop (drop_column_array, axis =1)
    del drop_column_array
    return df  #This df contains the values of the concentration changed to % and no longer contains the "unit" columns.

new_df = process_grad_data (df)
save_filename = "/home/user/YixiTFG/TFG_Yixi/RepoRT_trials/RepoRT_column_data/RepoRT_column_units_changed.tsv"
new_df.to_csv(save_filename, sep = '\t', index = False)