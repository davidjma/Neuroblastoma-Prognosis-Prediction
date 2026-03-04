#!bin/bash

base=/gpfs/mskmind_ess/mad1/whole_body_dicom

# creating one folder for all dicoms for each subject
for dir in $(ls -d $base/*/)
do

   # deleting existing new directory if exists
   rm -rf ${dir}CT_All_NC
   rm -rf ${dir}PET_All_NC
   mkdir -p ${dir}CT_All_NC
   mkdir -p ${dir}PET_All_NC
   
   ##### CT -AC #####   

   # defining variables
   len=0
   ct=${dir}/CT-AC

   # moving files in CT-AC file
   for file in $(ls $ct)
   do
#      echo $file
      num=`echo ${file} | cut -d- -f3`
      id=`echo ${file} | cut -d- -f1,2`
      cp ${dir}CT-AC/$(basename ${file}) ${dir}CT_All_NC/${id}-$((${num}+$len))-qk$((${num}+$len)).dcm
   done
   
   ####  PET-AC ####

   # defining variables
   pet=${dir}/PET-AC
   len=0

   # transfering files
   for file in $(ls $pet)
   do
 #     echo $file
      num=`echo ${file} | cut -d- -f3`
      id=`echo ${file} | cut -d- -f1,2`
      cp ${dir}PET-AC/$(basename ${file}) ${dir}PET_All_NC/${id}-$((${num}+${len}))-qk$((${num}+${len})).dcm
   done
   
   
   #### CT-AC-Lower ####
   ct_l=${dir}CT-AC-LOWER
   len=$(ls ${dir}CT-AC | wc -l)
 #  echo ${ct_l}
 #  echo ${len}

   # transferring files and renaming them 
   for file in $(ls ${ct_l})
   do
  #    echo ${file}
      num=`echo ${file} | cut -d- -f3`
      id=`echo ${file} | cut -d- -f1,2`
      cp ${dir}CT-AC-LOWER/$(basename ${file}) ${dir}CT_All_NC/${id}-$((${num}+$len))-qk$((${num}+$len)).dcm
   done   

   #### PET-AC-LOWER ####

   # defining variables
   pet_l=${dir}/PET-AC-LOWER
   len=$(ls ${dir}PET-AC | wc -l)

   # transfering files from one directory
   for file in $(ls $pet_l)
   do
      num=`echo ${file} | cut -d- -f3`
      id=`echo ${file} | cut -d- -f1,2`
      cp ${dir}PET-AC-LOWER/$(basename ${file}) ${dir}PET_All_NC/${id}-$((${num}+$len))-qk$((${num}+$len)).dcm
   done
done 
