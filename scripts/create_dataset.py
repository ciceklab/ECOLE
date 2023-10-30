'''
ECOLÉ samples preprocessing source code for training.
This script generates and processes the samples to perform CNV call training.
'''
from decimal import DecimalException
import numpy as np
from sklearn.metrics import confusion_matrix as cm
from tensorflow.keras.preprocessing import sequence
import torch
from performer_pytorch import Performer
from einops import rearrange, repeat
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.utils.data import TensorDataset, DataLoader,Dataset
import pandas as pd
import os
from itertools import groupby
from tqdm import tqdm
import argparse
import datetime




description = "ECOLÉ is a deep learning based WES CNV caller tool. \
            For academic research the software is completely free, although for \
            commercial usage the software is licensed. \n \
            please see ciceklab.cs.bilkent.edu.tr/ECOLÉlicenceblablabla."

parser = argparse.ArgumentParser(description=description)

required_args = parser.add_argument_group('Required Arguments')

required_args.add_argument("-i", "--input", help="Please provide the path of the input to data.\n This path should contain .npy files for each human sample.", required=True)

parser.add_argument("-V", "--version", help="show program version", action="store_true")
args = parser.parse_args()

data_path = args.input 

sample_files = os.listdir(data_path)
all_samples_names = [item.split("_")[0] for item in sample_files]

DELETION_FOLDER = "./processed_finetuning_dataset/deletion"
DUPLICATION_FOLDER = "./processed_finetuning_dataset/duplication"
NOCALL_FOLDER = "./processed_finetuning_dataset/nocall"

os.makedirs("./processed_finetuning_dataset/", exist_ok=True)
os.makedirs(DELETION_FOLDER, exist_ok=True)
os.makedirs(DUPLICATION_FOLDER, exist_ok=True)
os.makedirs(NOCALL_FOLDER, exist_ok=True)
print(f"creating the training dataset")
for item in tqdm(all_samples_names):

    gtcalls = np.load(data_path+item+"_labeled_data.npy", allow_pickle=True)
    print(f"processing sample: {item}")

    sampnames_data = []
    chrs_data = []
    readdepths_data = []
    start_inds_data = []
    end_inds_data = []
    wgscalls_data = []

    temp_sampnames = gtcalls[:,0]
    temp_chrs = gtcalls[:,1]
    temp_start_inds = gtcalls[:,2]
    temp_end_inds = gtcalls[:,3]
    temp_readdepths = gtcalls[:,4]
    temp_wgscalls = gtcalls[:,5]

    for i in range(len(temp_chrs)):
            
        crs = temp_chrs[i]
       
     
      
        if(len(crs) == 4):
            if(crs[3] == "Y"):
                crs = 23
            elif (crs[3] == "X"):
                crs = 24
            else:
                crs = int(crs[3])
        elif(len(crs) == 5):
            crs = int(crs[3:5])
       

        temp_chrs[i] = crs
        temp_readdepths[i] = list(temp_readdepths[i])
        arr = np.array(temp_readdepths[i],dtype=float)
        temp_readdepths[i].insert(len(temp_readdepths[i]), 0)
        temp_readdepths[i].insert(len(temp_readdepths[i]), 0)
        temp_readdepths[i].insert(len(temp_readdepths[i]), temp_end_inds[i])
        temp_readdepths[i].insert(len(temp_readdepths[i]), temp_start_inds[i]) 
        temp_readdepths[i].insert(len(temp_readdepths[i]), crs) 
        

    sampnames_data.extend(temp_sampnames)
    chrs_data.extend(temp_chrs)
    readdepths_data.extend(temp_readdepths)
    start_inds_data.extend(temp_start_inds)
    end_inds_data.extend(temp_end_inds)
    wgscalls_data.extend(temp_wgscalls)


    lens = [len(v) for v in readdepths_data]

    lengthfilter = [True if v < 1006  else False for v in lens]

    sampnames_data = np.asarray(sampnames_data)[lengthfilter]
    chrs_data = np.asarray(chrs_data)[lengthfilter]
    readdepths_data = np.asarray(readdepths_data)[lengthfilter]
    start_inds_data = np.asarray(start_inds_data)[lengthfilter]
    end_inds_data = np.asarray(end_inds_data)[lengthfilter]
    wgscalls_data = np.asarray(wgscalls_data)[lengthfilter]

    wgscalls_data[wgscalls_data == '<NO-CALL>'] = 0
    wgscalls_data[wgscalls_data == '<DUP>'] = 1
    wgscalls_data[wgscalls_data == '<DEL>'] = 2
    wgscalls_data= wgscalls_data.astype(int)

    readdepths_data = np.asarray([np.asarray(k) for k in readdepths_data])
    readdepths_data = sequence.pad_sequences(readdepths_data, maxlen=1005,dtype=np.float32,value=-1)
    readdepths_data = readdepths_data[ :, None, :]
    tot_nc = 0
    tot_del = 0
    tot_dp = 0
    print("creating exon samples:")
    for i in tqdm(range(len(wgscalls_data))):
        call_made =  wgscalls_data[i]
        exon_sample = readdepths_data[i]

        if call_made == 0:
            np.save(os.path.join(NOCALL_FOLDER,f"{item}_datapoint_{i}.npy"), exon_sample)
            tot_nc += 1
        elif call_made == 1:
            np.save(os.path.join(DUPLICATION_FOLDER,f"{item}_datapoint_{i}.npy"), exon_sample)
            tot_dp += 1
        elif call_made == 2:
            np.save(os.path.join(DELETION_FOLDER,f"{item}_datapoint_{i}.npy"), exon_sample)
            tot_del += 1
    print(tot_nc, tot_del, tot_dp)