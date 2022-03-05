import zipfile
import pandas as pd

df1 = pd.read_csv('files/UA.csv')
df2 = pd.read_csv('files/CGA.csv')
path = r'/Volumes/Seagate Basic'

with zipfile.ZipFile('mwf.zip', 'w') as zipobj:
    for index, row in df1.iterrows():
        mwf_path = path + row['path']
        zipobj.write(mwf_path)

    for index, row in df2.iterrows():
        mwf_path = path + row['mwf_path']
        zipobj.write(mwf_path)

