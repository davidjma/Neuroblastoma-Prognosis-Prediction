import pydicom as pyd
from pydicom import dcmread
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
import glob

dicom_dir='/gpfs/mskmind_ess/mad1/neuroblastoma/NB_16-1335/NB_CT_SKELETAL_MASK_WITH_PROGRESSION_1.00mm/RIA_16-1335_000026/CT_VOLUME/'
upper='1.3.12.2.1107.5.1.4.48483.5.0.1119732716821650.4551251220090911_volumetric_image.nii'
lower='1.3.12.2.1107.5.1.4.48483.5.0.1121831695011127.3551251220090911_volumetric_image.nii'

nii_list = glob.glob(dicom_dir+'*.nii')
print(nii_list)
nifti=[upper, lower]
all_func=[dicom_dir+folder for folder in nifti]
#print(all_func)

ni2_funcs = (nib.Nifti2Image.from_image(nib.load(func)) for func in nii_list)
ni2_concat = nib.concat_images(ni2_funcs, check_affines=False, axis=2)
print(ni2_concat.shape)
ni2_concat.to_filename(dicom_dir+lower+'.gz')
