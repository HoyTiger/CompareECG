'''
将ECG文件夹中的XML文件导入database
'''

import os
import pandas as pd
import xml.etree.ElementTree as ET

from pdf2image import convert_from_path


def convert_pdf2img(path, save_path='/Volumes/Seagate Basic/天津三中心/image'):
    base = os.path.basename(path).replace('.pdf', '.jpg')
    try:
        if not os.path.exists(os.path.join(save_path, base)):
            pages = convert_from_path(path, 500)
            for page in pages:
                page.save(os.path.join(save_path, base), 'JPEG')
    except:
        pass


def walk(path):
    files = os.listdir(path)
    count = 1
    for file in files:
        if count == 1:
            temp_path = os.path.join(path, file)
            if os.path.isdir(temp_path):
                walk(temp_path)
            else:
                if temp_path.endswith('.xml'):
                    count += 1
                    tree = ET.parse(temp_path)
                    root = tree.getroot()
                    patient_source_id = root.find('Ecgview').find('Inspection').find('Patient_section').find(
                        'Patient_source_id')
                    Acquition_date = root.find('Ecgview').find('Inspection').find('Patient_section').find(
                        'Acquition_date').attrib
                    date = f"{Acquition_date['year']}-{Acquition_date['month']}-{Acquition_date['day']} {Acquition_date['hour']}:{Acquition_date['minute']}:{Acquition_date['second']}"
                    if patient_source_id.text is not None and len(patient_source_id.text)==6:
                        # f.write(patient_source_id.text + ',' + temp_path[22:].replace('.xml', '.mwf') + ',' + date +'\n')
                        print(patient_source_id.text + ',' + temp_path[22:].replace('.xml', '.mwf') + '\n')

def walk2(path):
    files = os.listdir(path)
    for file in files:
        temp_path = os.path.join(path, file)
        if os.path.isdir(temp_path):
            walk2(temp_path)
        else:
            if temp_path.endswith('.pdf'):
                print(temp_path)
                convert_pdf2img(temp_path)


if __name__ == '__main__':

    basic = u'/Volumes/Seagate Basic/心电数据/厦门/20220125222051/'
    files = os.listdir(basic)
    for file in files:
        if file.endswith('.pdf'):
            pdfpath = os.path.join(basic, file)
            print(file)
            convert_pdf2img(pdfpath, u'/Volumes/Seagate Basic/心电数据/厦门/image')

    # df = pd.read_csv('files/xml_copy.csv')
    # bingli = pd.read_csv('files/bingli.csv')
    # cag = pd.read_csv('files/CAG.csv')
    # for index, row in df.iterrows():
    #     if row['patient_source_id'] in cag['medical_record_number'].to_list() or row['patient_source_id'] in bingli[
    #         'medical_record_number'].to_list():
    #         pdfpath = basic + os.path.dirname(row['path']).replace('ECG', 'Report')
    #         if os.path.exists(pdfpath):
    #             for file in os.listdir(pdfpath):
    #                 convert_pdf2img(os.path.join(pdfpath, file))
    #                 print(file)


    # pd.DataFrame(record_list).to_csv('files/report.csv')
