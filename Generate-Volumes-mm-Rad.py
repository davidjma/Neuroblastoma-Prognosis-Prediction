import sys
#sys.path.append('/gpfs/mskmind_ess/mad1/neuroblastoma/luna/pyluna-common')
from luna.common.connectors import DremioDataframeConnector
import getpass

dremio = DremioDataframeConnector(
            scheme = 'grpc+tcp',
            hostname = "tlvidreamcord1", #"pllimsksparky3",  #"tllihpcmind6",
            flightport = 32010, #9047, #9007, # 32010,
            dremio_user = getpass.getpass(prompt='Username: '),
            dremio_password = getpass.getpass(prompt='Password: '),
            connection_args = {})

import pandas as pd
from luna.common.utils import LunaCliClient
from concurrent.futures import ThreadPoolExecutor
from sklearn.model_selection import train_test_split

df_rad = dremio.get_table("nb-16-1335", "RAD_CURATION_PETCT_SELECTIONS").set_index("radiology_patient_id")
df_rad

config = [
    {"dataset_id": "NB_CT_SKELETAL_MASK_WITH_PROGRESSION_1.00mm", "resample_pixel_spacing":1.00},
    {"dataset_id": "NB_CT_SKELETAL_MASK_WITH_PROGRESSION_1.25mm", "resample_pixel_spacing":1.25},
    {"dataset_id": "NB_CT_SKELETAL_MASK_WITH_PROGRESSION_1.50mm", "resample_pixel_spacing":1.50},
    {"dataset_id": "NB_CT_SKELETAL_MASK_WITH_PROGRESSION_1.75mm", "resample_pixel_spacing":1.75},
    {"dataset_id": "NB_CT_SKELETAL_MASK_WITH_PROGRESSION_2.00mm", "resample_pixel_spacing":2.00},
    {"dataset_id": "NB_CT_SKELETAL_MASK_WITH_PROGRESSION_2.25mm", "resample_pixel_spacing":2.25},
    {"dataset_id": "NB_CT_SKELETAL_MASK_WITH_PROGRESSION_2.50mm", "resample_pixel_spacing":2.50},
    {"dataset_id": "NB_CT_SKELETAL_MASK_WITH_PROGRESSION_2.75mm", "resample_pixel_spacing":2.75},
    {"dataset_id": "NB_CT_SKELETAL_MASK_WITH_PROGRESSION_3.00mm", "resample_pixel_spacing":3.00},
]

#env DATASET_URL=http://pllimskhpc1:6077/

def pipeline(index, dataset_id, resample_pixel_spacing):
    df_rad_pt = df_rad.loc[index, ["curation_tag", "dicom_folder"]].dropna()

    if not "CT-AC" in df_rad_pt["curation_tag"].values and not "PET-AC" in df_rad_pt["curation_tag"].values: return
    
    client = LunaCliClient(base_dir=f"/gpfs/mskmind_emc/data_lake/NB_16-1335/{dataset_id}", uuid=index)
    
    ct_dicom_folder = df_rad_pt.loc[df_rad_pt["curation_tag"]=='CT-AC',  "dicom_folder"].item()
    pt_dicom_folder = df_rad_pt.loc[df_rad_pt["curation_tag"]=='PET-AC', "dicom_folder"].item()
    
    client.configure(
            "python3.9 -m luna.radiology.cli.dicom_to_itk", 
            ct_dicom_folder, 
            itk_image_type='nii', 
            itk_c_type="float"
    ).run("CT_VOLUME")
    
    client.configure(
            "python3.9 -m luna.radiology.cli.dicom_to_itk", 
            pt_dicom_folder, 
            itk_image_type='nii', 
            itk_c_type="float", 
            convert_to_suv=True
    ).run("PT_VOLUME")
    
    client.configure(
            "python3.9 -m luna.radiology.cli.coregister_volumes", 
            "CT_VOLUME", 
            "CT_VOLUME",
            resample_pixel_spacing=resample_pixel_spacing, 
            order=1, 
            save_npy=True,
            dataset_id=f"RAW_{dataset_id}"
    ).run("CT_VOLUME_REGISTERED")
    
    client.configure(
            "python3.9 -m luna.radiology.cli.coregister_volumes", 
            "PT_VOLUME", 
            "CT_VOLUME",
            resample_pixel_spacing=resample_pixel_spacing, 
            order=1, 
            save_npy=True,
            dataset_id=f"RAW_{dataset_id}"
    ).run("PT_VOLUME_REGISTERED")
        
    client.configure(
            "python3.9 -m luna.radiology.cli.generate_threshold_mask", 
            "CT_VOLUME_REGISTERED", 
            threshold=225, 
            area_closing_radius=100, 
            expansion_radius=10,
            save_npy=True,
            dataset_id=f"RAW_{dataset_id}"
    ).run("CT_SKELETAL_MASK")

for job_config in config:
    dataset_id = job_config["dataset_id"]
    resample_pixel_spacing = job_config["resample_pixel_spacing"]
    
    with ThreadPoolExecutor(max_workers=25) as executor:
        for count, index in enumerate(df_rad.index.unique()): pass
#             future = executor.submit(pipeline, index, dataset_id, resample_pixel_spacing)
     
    # Get outcomes
    df_rad_outcomes = dremio.get_table("nb-16-1335", "RAD_CLINICAL_PROGRESSION").set_index("id_did_patient_radiology_xnat")
    df_rad_split    = dremio.get_table("nb-16-1335", "RAD_COHORT_SPLIT").set_index("id_did_patient_radiology_xnat")
   
    # Split based on outcome index
    px_indices = df_rad_outcomes.index.unique().sort_values()
    px_train, px_test = df_rad_split[df_rad_split["cohort"]=="DISCOVERY"].index, df_rad_split[df_rad_split["cohort"]=="VALIDATION"].index

    # Get rad features
    df_rad_features = pd.read_parquet(f"/gpfs/mskmind_emc/data_lake/NB-16-1335/tables/RAW_{dataset_id}").reset_index().set_index('radiology_patient_name')
    
    # Join
    df_rad_combined = df_rad_features.join(df_rad_outcomes['pt_prog_status']).reset_index().rename(columns={"index": "id_did_patient_radiology_xnat"} ).set_index("id_did_patient_radiology_xnat")

    # Save split datasets
    df_rad_combined.to_parquet(f"/gpfs/mskmind_emc/data_lake/NB-16-1335/tables/ALL__{dataset_id}")
    df_rad_combined.loc[df_rad_combined.index.intersection(px_train)].to_parquet(f"/gpfs/mskmind_emc/data_lake/NB-16-1335/tables/DISCOVERY__{dataset_id}")
    df_rad_combined.loc[df_rad_combined.index.intersection(px_test)] .to_parquet(f"/gpfs/mskmind_emc/data_lake/NB-16-1335/tables/VALIDATION__{dataset_id}")    

for job_config in config:
    dataset_id = job_config["dataset_id"]

    print (dataset_id, len(pd.read_parquet(f"/gpfs/mskmind_emc/data_lake/NB-16-1335/tables/VALIDATION__{dataset_id}").index.unique()))
    print (dataset_id, len(pd.read_parquet(f"/gpfs/mskmind_emc/data_lake/NB-16-1335/tables/DISCOVERY__{dataset_id}").index.unique()))
    
    for job_config in config:
        dataset_id = job_config["dataset_id"]

pd.read_parquet(f"/gpfs/mskmind_emc/data_lake/NB-16-1335/tables/VALIDATION__NB_CT_SKELETAL_MASK_WITH_PROGRESSION_3.00mm").reset_index()
