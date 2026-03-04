import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

biomarkers

result = biomarkers[biomarkers['MRN'] == '35134116.0']
result

up_to_date=pd.read_csv("/gpfs/mskmind_ess/mad1/neuroblastoma/biomarkers/updated_dataset.csv")
biomarkers=pd.read_excel("/gpfs/mskmind_ess/mad1/neuroblastoma/biomarkers/new.xlsx", skiprows=[0])
df=up_to_date.merge(radiology_mrn[['rad_dt_scan','id_did_accession_radiology_xnat']], left_on='Xnat_ID', right_on='id_did_accession_radiology_xnat', how='inner')
df.drop_duplicates()
df_updated=biomarkers.merge(df, left_on='MRN', right_on='MRN', how='outer')
df_updated
#df.to_csv("/gpfs/mskmind_ess/mad1/neuroblastoma/biomarkers/subjects_wScans_outer.csv")

result = df_updated[df_updated['Xnat_ID'] == '456821']
result

biomarkers=pd.read_excel("/gpfs/mskmind_ess/mad1/neuroblastoma/biomarkers/new.xlsx", skiprows=[0])
biomarkers

radiology_mrn=pd.read_csv("/gpfs/mskmind_ess/mad1/neuroblastoma/biomarkers/mrn_mapping.csv")
radiology_mrn
#biomarkers=pd.read_csv("/gpfs/mskmind_ess/mad1/neuroblastoma/biomarkers/biomarkers.csv",sep='\t', engine='python', encoding='cp1252')
#radiology=pd.read_csv("/gpfs/mskmind_ess/mad1/neuroblastoma/biomarkers/updated_dataset.csv", sep='\t', engine='python', encoding='cp1252')
#biomarkers=pd.read_excel("/gpfs/mskmind_ess/mad1/neuroblastoma/biomarkers/new.xlsx", skiprows=[0])
#radiology=pd.read_csv("/gpfs/mskmind_ess/mad1/neuroblastoma/biomarkers/updated_dataset.csv")

# print(radiology['MRN'])
# print(biomarkers['MRN'])

# df=radiology.merge(biomarkers, left_on='MRN', right_on='MRN', how='outer')
# df.to_csv("/gpfs/mskmind_ess/mad1/neuroblastoma/biomarkers/subjects_wbiomarker_outer.csv")

df=biomarkers.merge(radiology, left_on='MRN', right_on='MRN', how='outer')
df.to_csv("/gpfs/mskmind_ess/mad1/neuroblastoma/biomarkers/subjects_wScans_outer.csv")
