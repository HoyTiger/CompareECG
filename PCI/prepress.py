import pandas as pd

sheet1 = pd.read_excel('PCI患者数据库2021.12.07.xlsx', sheet_name='Sheet1')
sheet2 = pd.read_excel('PCI患者数据库2021.12.07.xlsx', sheet_name='Sheet2')

case = pd.read_excel('PCI患者数据库2021.12.07.xlsx', sheet_name='case')
control = pd.read_excel('PCI患者数据库2021.12.07.xlsx', sheet_name='control')

case_list = pd.DataFrame()
control_list = pd.DataFrame()

for index, row in case.iterrows():
    if row['ID+times'] in sheet1['ID+times'].to_list():
        row = sheet1[sheet1['ID+times']==row['ID+times']][case.columns]
    elif row['ID+times'] in sheet2['ID+times'].to_list():
        row = sheet2[sheet2['ID+times'] == row['ID+times']][case.columns]
    case_list = case_list.append(row, ignore_index=True)

for index, row in control.iterrows():
    if row['ID+times'] in  sheet1['ID+times'].to_list():
        row = sheet1[sheet1['ID+times']==row['ID+times']][control.columns]
    elif row['ID+times'] in sheet2['ID+times'].to_list():
        row = sheet2[sheet2['ID+times'] == row['ID+times']][control.columns]
    control_list = control_list.append(row, ignore_index=True)

pd.DataFrame(case_list).to_csv('case.csv')
pd.DataFrame(control_list).to_csv('control.csv')
print()
