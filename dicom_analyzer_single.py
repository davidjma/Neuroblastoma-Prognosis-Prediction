# importing libraries 
import os
import matplotlib.pyplot as plt
import numpy as np
import pydicom as pyd
import nibabel as nb
#import tensorflow as tf
import pandas as pd
from pydicom import dcmread
import itk

#working_dir=[d for d in os.listdir('/gpfs/mskmind_ess/mad1/whole_body_dicom/') if os.path.isdir('/gpfs/mskmind_ess/mad1/whole_body_dicom/'+d)]
dicom_dir='/gpfs/mskmind_ess/mad1/whole_body_dicom/456667/PET_All_NC/'
#upper=dcmread(dicom_dir+os.listdir(dicom_dir)[0])
#lower=dcmread(dicom_dir+os.listdir(dicom_dir)[1])

#print(working_dir)

# for all series ID
#for dir in working_dir:
#   dicom_dir=f'/gpfs/mskmind_ess/mad1/whole_body_dicom/{dir}/CT_All_NC/'
#   pet_dir=f'/gpfs/mskmind_ess/mad1/whole_body_dicom/{dir}/PET_All_NC/'

   # ct scans
 #  print('--------CT scans---------')
for file in os.listdir(dicom_dir):
   #reference=dcmread(dicom_dir+os.listdir(dicom_dir)[0])
   #reference_SeriesInstanceUID = reference.SeriesInstanceUID
   #print('dicom Name: ', os.listdir(dicom_dir)[0])   
   #print(f'Reference SeriesInstanceUID: {reference_SeriesInstanceUID}')
   #print(f'Reference SeriesNumber: {reference.SeriesNumber}')
   #print(f'Reference InstanceNumber: {reference.InstanceNumber}')
#   print(reference)
   #print(f'base: {reference_SeriesInstanceUID}'   
   sample=dcmread(dicom_dir+file)
   #sample.preamble
      #sample.SeriesInstanceUID=reference_SeriesInstanceUID
 #     sample[0x0020,0x000e].value=reference_SeriesInstanceUID
   print(f'{file}: {sample.SeriesInstanceUID}')
 #     sample.save_as(ct_dir+file)   
   print(f"{file} Series Number: {sample.SeriesNumber}")
   print(f"{file} Instance Number: {sample.InstanceNumber}")
   #print(reference)
namesGenerator = itk.GDCMSeriesFileNames.New()
namesGenerator.SetUseSeriesDetails(False)
namesGenerator.AddSeriesRestriction("0008|0021")
namesGenerator.SetGlobalWarningDisplay(True)
namesGenerator.SetDirectory(dicom_dir)

seriesUIDs = namesGenerator.GetSeriesUIDs()
num_dicoms = len(seriesUIDs)
print(f'SeriesUID for each directory {dir}: {seriesUIDs}')
#print(reference)
   # pet scans
  # print('---------PET scans---------')
   #for file in os.listdir(pet_dir):
   #   reference=dcmread(pet_dir+os.listdir(pet_dir)[0])
   #   reference_SeriesInstanceUID = reference.SeriesInstanceUID
     
   #   print(f'base: {reference_SeriesInstanceUID}')
 #     sample=dcmread(pet_dir+file)
  #    sample.preamble
      #sample.SeriesInstanceUID=reference_SeriesInstanceUID
   #   sample[0x0020,0x000e].value=reference_SeriesInstanceUID
    #  print(f'{file}: {sample.SeriesInstanceUID}')   
   #   sample.save_as(pet_dir+file)

#print(reference)
   #namesGenerator = itk.GDCMSeriesFileNames.New()
   #namesGenerator.SetUseSeriesDetails(True)
   #namesGenerator.AddSeriesRestriction("0008|0021")
   #namesGenerator.SetGlobalWarningDisplay(False)
   #namesGenerator.SetDirectory(pet_dir)

   #seriesUIDs = namesGenerator.GetSeriesUIDs()
   #num_dicoms = len(seriesUIDs)
  # print(f'SeriesUID for each directory {dir}: {seriesUIDs}')

#print(f'Patient Name: {str(upper.PatientName)}, Accession Name: {str(upper.AccessionNumber)}, \
#        SeriesInstanceUID:  {str(upper.SeriesInstanceUID)}, SeriesNumber: {str(upper.SeriesNumber)},\
#        Modality: {str(upper.Modality)}')

#print(f'Patient Name: {str(lower.PatientName)}, Accession Name: {str(lower.AccessionNumber)},\
#   SeriesInstanceUID:  {str(lower.SeriesInstanceUID)}, SeriesNumber: {str(lower.SeriesNumber)},\
#        Modality: {str(lower.Modality)}') 
#print(lower)

#print(dcmread(dicom_dir+upper))
#print(dcmread(dicom_dir+lower))
#print(dcmread(dicom_dir+upper)==dcmread(dicom_dir+lower))
