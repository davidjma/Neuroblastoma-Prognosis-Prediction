import os
import numpy as np
import pydicom as pyd
from pydicom import dcmread
import matplotlib.pyplot as plt
import nibabel as nb
import pandas as pd
import itk

dicom_dir='/gpfs/mskmind_ess/mad1/whole_body_dicom/456692/PET_All_NC/'

dcm_dir=os.listdir(dicom_dir)
ref_dcm=[s for s in dcm_dir if 'qk1.dcm' in s][0]

for file in os.listdir(dicom_dir):
   reference=dcmread(dicom_dir+ref_dcm)
   reference_SeriesInstanceUID = reference.SeriesInstanceUID
   reference_SeriesNumber=reference.SeriesNumber
   
   # samples
   sample=dcmread(dicom_dir+file)
   sample.preamble
   
   sampleFile=file.replace('.dcm', '')
   z=sampleFile.index('k')
   InstanceN=sampleFile[z+1:]
   sample.InstanceNumber=InstanceN
   sample.SeriesNumber=reference_SeriesNumber
   sample.SeriesInstanceUID=reference_SeriesInstanceUID
   
   print(f'Sample {file}, SampleInstanceNumber: {sample.InstanceNumber}, SampleSeriesNumber: {sample.SeriesNumber}')

