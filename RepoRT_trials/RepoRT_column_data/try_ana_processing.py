import pandas as pd

filename = "C:/Users/leonz/Github_reps/TFG_git/RepoRT_trials/RepoRT_column_data/RepoRT_column_data.tsv"
df = pd.read_csv (filename, sep="\t", encoding= "utf-8")

# (processing_data function usage:)
# Filling column temperature, flow rate and column length with means of the similar columns (based on their names)
# It is followed by t0 updating using the new values inferred. Here t0 is calculated as V0/F, where F is the flowrate and V0 is the void volume (66% is used) of the column
for column in df.columns[2:8]:
    lines_null =df[df[column].isnull()] #Consigue las filas vacías
    for column_name in lines_null["column.name"]:
        if pd.notnull(column_name):
            same_lines = df[df['column.name'] == column_name]
            mean = same_lines[column].mean()
            if pd.isnull(mean):
                same_pattern = df[df['column.name'].fillna('').str.contains(column_name[0:15])]
                mean = same_pattern[column].mean()
                if pd.isnull(mean):
                    mean = df[column].mean()
            df.loc[(df[column].isnull()) & (df['column.name'] == column_name), column] = mean

new_filename = "/RepoRT_trials/RepoRT_column_data/RepoRT_column_data_Ana.tsv"
df.to_csv(new_filename, sep="\t", index=False)

print (mean)