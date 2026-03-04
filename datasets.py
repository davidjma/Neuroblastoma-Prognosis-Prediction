# importing libraries
import sys, os
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torchvision import transforms
import numpy as np
import numpy as np
from patient import Patient
import pickle
from pathlib import Path
from customtypes import ScanData, PathLike, Patients
from torchvision.transforms import Compose, RandomHorizontalFlip, RandomVerticalFlip, RandomRotation
import logging
import matplotlib.pyplot as plt
import pyarrow.parquet as pq
import math
from utils import elevate
from utils import cache_to_array, array_to_cache
from skimage import measure
import scipy 
from luna.radiology.mirp.imageReaders import read_itk_image, read_itk_segmentation
import pandas as pd
from radiomics import featureextractor  # This module is used for interaction with pyradiomics
import SimpleITK as sitk

# logger = logging.getLogger(__name__)
# logger.setLevel(0)

def get_configured_logger(name):
      logger = logging.getLogger(name)
      if (len(logger.handlers) == 0):                
                FORMAT = "%(asctime)s [%(processName)s] %(name)s - %(levelname)s: %(message)s"
                formatter = logging.Formatter(fmt=FORMAT)                                 
                handler = logging.StreamHandler()
                handler.setFormatter(formatter)
                logger.addHandler(handler)
                logger.setLevel(20)        
      return logger

logger = get_configured_logger("datasets")

def get_radiomics_frame(extractor, sitk_image, sitk_mask, prefix):
    result = extractor.execute(sitk_image, sitk_mask)

    df_result = pd.Series(result).to_frame().T
    df_result = df_result.loc[:, ~df_result.columns.str.contains("diagnostics")].add_prefix(prefix)

    return df_result

def crop_center(img, rel_crop_size):
    _, x, y  = img.shape

    pad_x = np.floor(rel_crop_size * x).astype(int)
    pad_y = np.floor(rel_crop_size * y).astype(int)

    return img[:, pad_x:x - pad_x, pad_y:y - pad_y]

def load_patient_data(data_file='/gpfs/mskmind_ess/mad1/neuroblastoma/NB-16-1335/tables/DISCOVERY__NB_CT_SKELETAL_MASK_WITH_PROGRESSION', patient_key="radiology_accession_number", quiet=False):
    """
    Reads patient data from data_file

    params: 
    - data_file (str): path to data file
    - quiet (bool): if quiet, no warnings will be created

    returns:
    - list of Patient objects
    """
    # if quiet:
    #     logger.setLevel(50)

    DATA = pq.read_table(data_file).to_pandas().reset_index().set_index("SEGMENT_ID")

    # create a list of Patient objects
    patients = []
    patient_id_list = np.unique(DATA[patient_key].sort_values().to_numpy())
    for patient_id in patient_id_list:
        ct_file = DATA[DATA['radiology_modality'].str.contains('CT') & DATA[patient_key]
                    .str.contains(patient_id)]['npy_volume']
        pt_file = DATA[DATA['radiology_modality'].str.contains('PT') & DATA[patient_key]
                    .str.contains(patient_id)]['npy_volume']

#        print (ct_file)
#        print (pt_file)
        if len(pt_file) != 1:
            logger.warning(f'Number of PET files for patient {patient_id} not equal to one. Skipping this patient')
            continue
        if len(ct_file) != 2:
            # raise Exception(f'Number of CT or PET file for patient {patient_id} not equal to one')
            logger.warning(f'Number of CT files for patient {patient_id} not equal to two. Skipping this patient')
            continue

        ct_file = ct_file[0]
        pt_file = pt_file[0]
        mask_file = DATA[DATA['radiology_modality'].str.contains('CT') & DATA[patient_key]
                    .str.contains(patient_id)]['npy_labels'][1]

        npy_edt_labels = DATA[DATA['radiology_modality'].str.contains('CT') & DATA[patient_key]
                    .str.contains(patient_id)]['npy_edt_labels'][1]

        itk_labels = DATA[DATA['radiology_modality'].str.contains('CT') & DATA[patient_key]
                    .str.contains(patient_id)]['itk_labels'][1]

        patient_outcome = DATA[DATA[patient_key].str.contains(patient_id)]['pt_prog_status']
        if len(patient_outcome) != 3:
            raise Exception(f'Number of progression status for patient {patient_id} not equal to three')
        try:
            patient_outcome = int(patient_outcome[0])
        except:
            logger.warning(f"Failed to convert patient {patient_id}'s outcome to integer, \
                which is {patient_outcome}. Skipping this patient.")
            continue

        p = Patient(patient_id=patient_id, patient_outcome=patient_outcome,
                    patient_ct_scan_file=ct_file, patient_pt_scan_file=pt_file, 
                    patient_mask_file=mask_file, npy_edt_labels=npy_edt_labels, itk_labels=itk_labels)

        patients.append(p)
    
    return np.array(patients)

class CTDataset(Dataset):
    # the __getitem__ method returns the numpy array of a PET scan
    # each item should be in the shape of [len, 1, w, h]
    def __init__(self, patients: Patients, radius=1.0):
        self.patients = patients
        self.radius = radius

        # self.data = {}
        # for idx, _ in enumerate (self.patients):
        #     logger.info(f"Loading {self.patients[idx].patient_ct_scan_file=}")
        #     self.data[idx] = np.load(self.patients[idx].patient_ct_scan_file)

    def __len__(self):
        return len(self.patients)
    
    def __getitem__(self, idx: int) -> torch.Tensor:
        ct_file = self.patients[idx].patient_ct_scan_file
        logger.info(f"Loading: {ct_file=}")
        ct_arr = np.load(ct_file)
        return ct_arr

class PETDataset(Dataset):
    # the __getitem__ method returns the numpy array of a PET scan
    # each item should be in the shape of [len, 1, w, h]
    def __init__(self, patients: Patients):
        self.patients = patients

        # self.data = {}
        # for idx, _ in enumerate (self.patients):
        #     logger.info(f"Loading {self.patients[idx].patient_pt_scan_file=}")
        #     self.data[idx] = np.load(self.patients[idx].patient_pt_scan_file)

    def __len__(self):
        return len(self.patients)
 
    def __getitem__(self, idx: int) -> torch.Tensor:
        pt_file = self.patients[idx].patient_pt_scan_file
        logger.info(f"Loading: {pt_file=}")
        pt_arr = np.load(pt_file)
        return pt_arr

class MaskDataset(Dataset):
    # the __getitem__ method returns the numpy array of a PET scan
    # each item should be in the shape of [len, 1, w, h]
    def __init__(self, patients: Patients):
        self.patients = patients

    def __len__(self):
        return len(self.patients)
 
    def __getitem__(self, idx: int) -> torch.Tensor:
        mk_file = self.patients[idx].npy_edt_labels
        logger.info(f"Loading: {mk_file=}")
        mk_arr = np.load(mk_file)
        return mk_arr

class OutcomesDataset(Dataset):
    # the __getitem__ method returns outcome of a patient
    # should just be a float value (not even a tensor)
    def __init__(self, patients: Patients):
        self.patients = patients

    def __len__(self):
        return len(self.patients)

    def __getitem__(self, idx) -> torch.Tensor:
        outcome = self.patients[idx].patient_outcome

        return outcome

def get_parquet_from_cache(layers):
    layers.append ("data.parquet")
    path = os.path.join(*layers)
    # logger.info(f"Checking cache {path=}")
    if os.path.exists(path):
        return pd.read_parquet(path)
    else:
        return None

def put_parquet_to_cache(layers, data):
    path = os.path.join(*layers)
    logger.info(f"Putting to cache {path=}")
    os.makedirs(path, exist_ok=True)
    data.to_parquet(path + "/data.parquet")

from skimage.filters import threshold_otsu

class CombinedDatasetPETComponetBag(Dataset):
    # outputs a dict of CT data and PET mean data with their outcome
    # the dictionary keys are ["CT", "Masked PET Mean", "Outcome"]
    def __init__(self, patients, hu_cutoff, layer0, layer1):
        self.patients = patients
        self.ct   = CTDataset       (patients)
        self.pt   = PETDataset      (patients)
        self.oc   = OutcomesDataset (patients)

        self.hu_cutoff = hu_cutoff

        self.layer0 = layer0
        self.layer1 = layer1
        self.layer2 = "CombinedDatasetPETComponetBagFilters"


    def __len__(self):
        return len(self.patients)

    def get_all_data(self):

        data_pkg = {}
        for idx in range (len(self)):
            data_pkg[str(self.patients[idx].patient_id)] = self[idx]

        return data_pkg

    def __getitem__(self, idx):
        # logger.info (f"Loading: {idx}")
        px_layer = self.patients[idx].patient_id
        outcome  = self.oc[idx]

        data = get_parquet_from_cache([self.layer0, self.layer1, self.layer2, px_layer])

        if data is None:
            data = self.generate_data(idx)
            put_parquet_to_cache([self.layer0, self.layer1, self.layer2, px_layer], data)

        outcome = torch.tensor(outcome).float()

        return data, outcome

    def generate_data(self, idx):
        px_layer = self.patients[idx].patient_id
        ct_arr  = self.ct[idx]
        pt_arr  = self.pt[idx]

        itk_label_volume = read_itk_image(str(self.patients[idx].itk_labels))

        # logger.info (f"Masking: {idx}")
        # Mask areas < x mm of skeletal mask, and with PET SUV > 0.05 (removes non-bodily background)
        mk_arr_base = (ct_arr >= self.hu_cutoff) & (pt_arr > 0.05)

        out = threshold_otsu ( pt_arr.flatten()[mk_arr_base.flatten().astype(bool)] )
        logger.info(f"Otsu threshold={out}")

        mk_arr_signal = (pt_arr > out)

        logger.info("Creating blob bags...")
        bli_arr = measure.label(mk_arr_signal, connectivity=2) # bone-label-instance array
        logger.info (f"Found {len(np.unique(bli_arr))} instances!")

        image_class_object_volume = read_itk_image(str(self.patients[idx].itk_labels))
        image_class_object_volume.set_voxel_grid(bli_arr.astype(np.uint8))
        image_path = image_class_object_volume.export(str(self.patients[idx].itk_labels.parent) + "/lesionlabels/")
        logger.info (image_path)

        instance_features = []

        sitk_ct_image  = sitk.GetImageFromArray(ct_arr.astype(np.float32))
        sitk_pt_image  = sitk.GetImageFromArray(pt_arr.astype(np.float32))

        ct_extractor = featureextractor.RadiomicsFeatureExtractor(binWidth=10)
        pt_extractor = featureextractor.RadiomicsFeatureExtractor(binWidth=0.1)

        for extractor in (ct_extractor, pt_extractor):
            extractor.enableImageTypeByName("LoG")
            extractor.enableImageTypeByName("Gradient")
            extractor.enableImageTypeByName("LBP3D")
            extractor.enableImageTypeByName("Wavelet")

        instance_list = []

        for label in np.unique(bli_arr):
            if label == 0: continue
            if not (bli_arr==label).sum() > 125: continue

            logger.info(f"Processing instance {label=}, n_voxels={(bli_arr==label).sum()}")

            results_list = []

            sitk_mask = sitk.GetImageFromArray((bli_arr==label).astype(np.uint8))

            df_ct_result = get_radiomics_frame(ct_extractor, sitk_ct_image, sitk_mask, "ct__")
            df_pt_result = get_radiomics_frame(pt_extractor, sitk_pt_image, sitk_mask, "pt__")

            df_instance_features = pd.concat((df_ct_result, df_pt_result), axis=1)
            df_instance_features['lesion_index'] = label

            instance_list.append(df_instance_features)

        df_bone_features = pd.concat(instance_list, axis=0).astype(np.float32)

        return df_bone_features

class CombinedDatasetBoneBags(Dataset):
    # outputs a dict of CT data and PET mean data with their outcome
    # the dictionary keys are ["CT", "Masked PET Mean", "Outcome"]
    def __init__(self, patients, hu_cutoff, layer0, layer1):
        self.patients = patients
        self.ct   = CTDataset       (patients)
        self.pt   = PETDataset      (patients)
        self.oc   = OutcomesDataset (patients)

        self.hu_cutoff = hu_cutoff

        self.layer0 = layer0
        self.layer1 = layer1
        self.layer2 = "CombinedDatasetBoneBag"


    def __len__(self):
        return len(self.patients)

    def __getitem__(self, idx):
        # logger.info (f"Loading: {idx}")
        px_layer = self.patients[idx].patient_id
        outcome  = self.oc[idx]

        data = get_parquet_from_cache([self.layer0, self.layer1, self.layer2, px_layer])

        if data is None:
            data = self.generate_data(idx)
            put_parquet_to_cache([self.layer0, self.layer1, self.layer2, px_layer], data)

        outcome = torch.tensor(outcome).float()

        return data, outcome

    def generate_data(self, idx):
        ct_arr  = self.ct[idx]
        pt_arr  = self.pt[idx]

        itk_label_volume = read_itk_image(str(self.patients[idx].itk_labels))

        # logger.info (f"Masking: {idx}")
        # Mask areas < x mm of skeletal mask, and with PET SUV > 0.05 (removes non-bodily background)
        mk_arr = (ct_arr >= self.hu_cutoff) & (pt_arr > 0.05)

        logger.info("Creating bone bags...")
        bli_arr = measure.label(mk_arr, connectivity=2) # bone-label-instance array
        logger.info (f"Found {len(np.unique(bli_arr))} instances!")

        instance_features = []

        sitk_ct_image  = sitk.GetImageFromArray(ct_arr.astype(np.float32))
        sitk_pt_image  = sitk.GetImageFromArray(pt_arr.astype(np.float32))

        ct_extractor = featureextractor.RadiomicsFeatureExtractor(binWidth=16)
        pt_extractor = featureextractor.RadiomicsFeatureExtractor(binWidth=0.2)

        instance_list = []

        for label in np.unique(bli_arr):
            if label == 0: continue
            if not np.where(bli_arr==label, 1, 0).sum() > 100: continue

            logger.info(f"Processing bone {label=}")

            results_list = []

            sitk_mask = sitk.GetImageFromArray((bli_arr==label).astype(np.uint8))

            df_result = get_radiomics_frame(ct_extractor, sitk_ct_image, sitk_mask, "ct_")
            results_list.append(df_result)
            
            for pt_level in [0.0, 0.5, 1.0]:
                bone_pt_mask = ((bli_arr==label) & (pt_arr >= pt_level)).astype(np.uint8)

                if not bone_pt_mask.sum() > 10: break

                sitk_mask = sitk.GetImageFromArray(bone_pt_mask)

                df_result = get_radiomics_frame(pt_extractor, sitk_pt_image, sitk_mask, f"pt_level{pt_level}_")
                results_list.append(df_result)

            df_instance_features = pd.concat(results_list, axis=1)

            instance_list.append(df_instance_features)

        df_bone_features = pd.concat(instance_list, axis=0).astype(np.float32)

        if not len(df_bone_features.columns) == 428: logger.info("No PET level 2.0 in all bones, be carefull!!!!!!")

        return df_bone_features


class CombinedDataset(Dataset):
    # outputs a dict of CT data and PET mean data with their outcome
    # the dictionary keys are ["CT", "Masked PET Mean", "Outcome"]
    def __init__(self, patients, downsample_ratio: int=1, train=True, dropout=0.8, radius=1.0):
        self.patients = patients
        self.ct   = CTDataset       (patients)
        self.pt   = PETDataset      (patients)
        self.mk   = MaskDataset     (patients)
        self.oc   = OutcomesDataset (patients)

        self.downsample_ratio = downsample_ratio
        self.dropout = dropout
        self.radius  = radius

        self.train = train 
        self.transform = Compose([
            RandomVerticalFlip(0.5),
            RandomHorizontalFlip(0.5),            
        ])

    @staticmethod
    def sigmoid(x):
        return 1 / (1 + math.exp(-x))

    def __len__(self):
        return len(self.patients)

    def __getitem__(self, idx):
        # logger.info (f"Loading: {idx}")
        ct_arr  = self.ct[idx]
        pt_arr  = self.pt[idx]
        mk_arr  = self.mk[idx]
        outcome = self.oc[idx]

        # logger.info (f"Cropping: {idx}")
        # Crop lightly
        pt_arr = crop_center(pt_arr, 0.1)
        ct_arr = crop_center(ct_arr, 0.1)
        mk_arr = crop_center(mk_arr, 0.1)

        # logger.info (f"Masking: {idx}")
        # Mask areas < x mm of skeletal mask, and with PET SUV > 0.05 (removes non-bodily background)
        mk_arr = (mk_arr <= self.radius) & (pt_arr > 0.05)

        # Apply mask
        ct_arr = np.where(mk_arr, ct_arr, np.min(ct_arr))
        pt_arr = np.where(mk_arr, pt_arr, 0) 

        # logger.info (f"Preprocessing: {idx}")
        # Calculate SUV over VOI
        pt_mean = pt_arr.sum(axis=(1,2)) / ( mk_arr.sum(axis=(1,2))  + 1e-4 )

        # Image impute preprocessing: Clip and rescale to sensible range in CT and PET
        ct_arr = np.clip  (ct_arr, -50, 800)  / 800
        pt_arr = np.log10 (pt_arr + 1)

        # Reshape our arrays
        ct_arr, pt_arr = np.expand_dims(ct_arr, 1), np.expand_dims(pt_arr, 1)

        # logger.info (f"Creating Tensors: {idx}")
        # Create tensors
        ct = torch.Tensor(np.concatenate((ct_arr, pt_arr), axis=1)).float()
        pt = torch.Tensor(pt_mean).float()
        outcome = torch.tensor(outcome).float()

        if not os.path.exists(f"figs/px_{self.patients[idx].patient_id}_ct.png"):
            plt.imshow(ct[:, 0, :, :].mean(dim=1))
            plt.savefig(f"figs/px_{self.patients[idx].patient_id}_ct.png")
        if not os.path.exists(f"figs/px_{self.patients[idx].patient_id}_pt.png"):
            plt.imshow(ct[:, 1, :, :].mean(dim=1))
            plt.savefig(f"figs/px_{self.patients[idx].patient_id}_pt.png")

        # logger.info (f"Augmenting: {idx}")
        if self.train:
            keep_slices, _ = torch.sort( torch.randperm(ct.shape[0])[:np.floor(self.dropout * ct.shape[0]).astype(int)] )

            ct = ct[keep_slices]
            pt = pt[keep_slices]

            for i in range(ct.shape[0]):
                ct[i] = self.transform(ct[i])

        out = {
            "CT": ct,
            "PT": pt,
            "Outcome": outcome
        }
        # logger.info (f"Returning: {idx}")
        return out

from sklearn.preprocessing import PowerTransformer
from sklearn.feature_selection import VarianceThreshold
from sklearn.impute import SimpleImputer

class Normalizer:
    def __init__(self, radiomics_subset):
        self.data = pd.DataFrame()
        self.power_transform = PowerTransformer()
        self.simple_imputer = SimpleImputer()
        self.radiomics_subset = radiomics_subset

        self.n_features = None

    def add_instance(self, x):
        self.data = pd.concat((self.data, x))

    def fit(self):
        self.column_schema = self.data.columns
        self.column_schema = self.column_schema[~self.column_schema.str.contains("lesion_index")]
        self.column_schema = self.column_schema[ self.column_schema.str.contains("|".join(self.radiomics_subset))]

        self.n_features = len(self.column_schema)

        X = self.data[self.column_schema].astype(float)
        X = self.power_transform.fit_transform(X)
        X = self.simple_imputer.fit_transform(X)

    def transform(self, input_data):
        X = input_data[self.column_schema].astype(float)
        X = self.power_transform.transform(X)
        X = self.simple_imputer.transform(X)
        return torch.Tensor(X)

    def print(self):
        print (self.data)
        print ("n_features=", self.n_features)

