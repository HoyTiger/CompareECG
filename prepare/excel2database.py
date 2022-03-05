import os

import pandas as pd

UA = r'/Volumes/Seagate Basic/天津三中心/临床信息/UA'
NSTEMI = r'/Volumes/Seagate Basic/天津三中心/临床信息/NSTEMI'
AMI = r'/Volumes/Seagate Basic/天津三中心/临床信息/AMI前'
AMI2 = r'/Volumes/Seagate Basic/天津三中心/临床信息/AMI下'
AF = r'/Volumes/Seagate Basic/天津三中心/临床信息/AF'

patient_ecg = pd.read_csv('files/report.csv')
patient_record = []
patient_record2 = []

dic = {
    '科室': 'department',
    '病区': 'unit',
    '姓名': 'patient_name',
    '性别': 'gender',
    '年龄': 'age',
    '病案号': 'medical_record_number',
    '管床医师': 'resident_name',
    '入科时间': 'admission_date',
    '出院时间': 'discharge_date',
    '主要诊断代码': 'icd_10',
    '主要诊断': 'main_diagnosis'
}

dic2 = {
    '住院号': 'medical_record_number',
    '住院次数': 'medical_times',
    '病人姓名': 'patient_name',
    '年龄': 'age',
    '性别': 'gender',
    '入院日期': 'admission_date',
    '入院科室': 'admission_department',
    '入院病区': 'admission_unit',
    '出院日期': 'discharge_date',
    '出院科室': 'discharge_department',
    '出院病区': 'discharge_unit',
    '住院天数': 'hospitalization_days',
    '入院诊断': 'admission_diagnosis',
    '入院途径': 'admission_route',
    '出院诊断': 'discharge_diagnosis',
    'ICD10': 'icd_10',
    '手术日期': 'pci_date',
    '费用合计': 'total_cost'
}


def read_excel(path):
    disease_type = os.path.basename(path)
    for file in os.listdir(path):
        excel = pd.read_excel(os.path.join(path, file), header=1)
        excel = excel.drop(columns=['床号', '病历状态'])
        excel = excel.rename(columns=dic)
        excel['disease_type'] = disease_type
        for index, row in excel.iterrows():
            if row['medical_record_number'] in patient_ecg['patient_source_id'].to_list():
                patient_record.append(dict(row))


read_excel(UA)
read_excel(NSTEMI)
read_excel(AMI)
read_excel(AMI)
read_excel(AF)

disease_type = 'CAG'
excel = pd.read_excel('files/冠脉造影.xlsx', header=0)
excel = excel.drop(
    columns=['状态', '状态', '入院病情', '科主任', '主副任医师', '主治医师', '住院医师', '手术名称', '手术编码', '术者', '一助', '二助', '手术级别', '手术级别'])
excel = excel.rename(columns=dic2)
excel['disease_type'] = disease_type
for index, row in excel.iterrows():
    if row['medical_record_number'] in patient_ecg['patient_source_id'].to_list():
        patient_record2.append(dict(row))

pd.DataFrame(patient_record2).to_csv('files/CAG.csv')
pd.DataFrame(patient_record).to_csv('files/bingli.csv')
