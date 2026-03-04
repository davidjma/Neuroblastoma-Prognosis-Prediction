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
import pdb #pdb.set_trace()

working_dir=[d for d in os.listdir('/gpfs/mskmind_ess/mad1/whole_body_dicom/') if os.path.isdir('/gpfs/mskmind_ess/mad1/whole_body_dicom/'+d)]

# for all series ID
for dir in working_dir:
   ct_dir=f'/gpfs/mskmind_ess/mad1/whole_body_dicom/{dir}/CT_All_NC/'
   pet_dir=f'/gpfs/mskmind_ess/mad1/whole_body_dicom/{dir}/PET_All_NC/'
   #print(print(ct_dir))
   #print(os.listdir(ct_dir))
   #print([s for s in os.listdir(ct_dir) if 'qk1.dcm' in s])
   
   print(f'\nLooking at this directory: {dir}')
   ref_ct=[s for s in os.listdir(ct_dir) if 'qk1.dcm' in s][0]
 #  print(ref_ct)
   ref_pet=[s for s in os.listdir(pet_dir) if 'qk1.dcm' in s][0]
 #  print(ref_pet)

   # ct scans
   print('--------CT scans---------')
   for file in os.listdir(ct_dir):
      reference=dcmread(ct_dir+ref_ct)
      reference_SeriesInstanceUID = reference.SeriesInstanceUID
      reference_SeriesNumber=reference.SeriesNumber   
#      print(f'base: {reference_SeriesInstanceUID}')
      
      # samples
      sample=dcmread(ct_dir+file)
      sample.preamble
      sample.SeriesInstanceUID=reference_SeriesInstanceUID
      sample.SeriesNumber=reference_SeriesNumber
      
      sampleFile=file.replace('.dcm', '')
      Instance_index=sampleFile.index('k')
      InstanceN=sampleFile[Instance_index+1:]
      sample.InstanceNumber=InstanceN
 
#     sample[0x0020,0x000e].value=reference_SeriesInstanceUID
 #     print(f'{file}: {sample.SeriesInstanceUID}')
      sample.save_as(ct_dir+file)   
   
   namesGenerator = itk.GDCMSeriesFileNames.New()
   namesGenerator.SetUseSeriesDetails(False)
   namesGenerator.AddSeriesRestriction("0008|0021")
   namesGenerator.SetGlobalWarningDisplay(False)
   namesGenerator.SetDirectory(ct_dir)

   #pdb.set_trace()
   seriesUIDs = namesGenerator.GetSeriesUIDs()
   num_dicoms = len(seriesUIDs)
   print(f'SeriesUID for each directory {dir}: {seriesUIDs}')

   # pet scans
   print('---------PET scans---------')
   for file in os.listdir(pet_dir):
      reference=dcmread(pet_dir+ref_pet)
      reference_SeriesInstanceUID = reference.SeriesInstanceUID
      reference_SeriesNumber=reference.SeriesNumber

      sample=dcmread(pet_dir+file)
      sample.preamble

      sampleFile=file.replace('.dcm', '')
      z=sampleFile.index('k')
      InstanceN=sampleFile[z+1:]
      sample.InstanceNumber=InstanceN
      sample.SeriesNumber=reference_SeriesNumber
      sample.SeriesInstanceUID=reference_SeriesInstanceUID

   #   sample[0x0020,0x000e].value=reference_SeriesInstanceUID
    #  print(f'{file}: {sample.SeriesInstanceUID}')   
      sample.save_as(pet_dir+file)

   print(ref_pet)
   namesGenerator = itk.GDCMSeriesFileNames.New()
   namesGenerator.SetUseSeriesDetails(True)
   namesGenerator.AddSeriesRestriction("0008|0021")
   namesGenerator.SetGlobalWarningDisplay(False)
   namesGenerator.SetDirectory(pet_dir)

   seriesUIDs = namesGenerator.GetSeriesUIDs()
   num_dicoms = len(seriesUIDs)
   print(f'SeriesUID for each directory {dir}: {seriesUIDs}')

