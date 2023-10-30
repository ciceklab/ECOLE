'''
ECOLÉ samples preprocessing source code.
This script generates and processes the samples to perform CNV calling.
'''
import numpy as np
import os
from os import listdir
import pdb
import csv
import pandas as pd
import time
from tqdm import tqdm
from multiprocessing.pool import Pool
import argparse


description = "ECOLÉ is a deep learning based WES CNV caller tool. \
            For academic research the software is completely free, although for \
            commercial usage the software is licensed. \n \
            please see ciceklab.cs.bilkent.edu.tr/ECOLÉlicenceblablabla."

parser = argparse.ArgumentParser(description=description)

required_args = parser.add_argument_group('Required Arguments')

required_args.add_argument("-rd", "--readdepth", help="Please provide the exon-wise readdepth path.", required=True)

required_args.add_argument("-o", "--output", help="Please provide the output path for the preprocessing.", required=True)

required_args.add_argument("-t", "--target", help="Please provide the path of exon target file.", required=True)

parser.add_argument("-V", "--version", help="show program version", action="store_true")
args = parser.parse_args()

exon_wise_readdepths_path = args.readdepth
output_path = args.output
target_list_path = args.target

import os

exon_wise_readdepth_files = listdir(exon_wise_readdepths_path)
for file_name in exon_wise_readdepth_files:

    labeled_data = []
    sample_name = file_name.split(".")[0]
    target_data = pd.read_csv(target_list_path, sep="\t", header=None).values
    read_depth_data = pd.read_csv(os.path.join(exon_wise_readdepths_path, file_name), sep="\t").values

    wgs_calls_data = pd.read_csv(f"./finetune_example_data/ground_truth_labels/{sample_name}.csv" , sep=",",comment='#',header=None).values
    wgs_ends_list = np.asarray([int(x) for x in wgs_calls_data[:,2]])
    wgs_start_list = np.asarray([int(x) for x in wgs_calls_data[:,1]])
    wgs_chr = np.asarray([str(x) for x in wgs_calls_data[:,0]])

    chromosomes = np.unique(target_data[:,0])
    
    for chr_ in tqdm(chromosomes):
        chr_formatted = str(chr_[3:])
        cond_1 = read_depth_data[:,0]== chr_
        cond_2 = (wgs_chr == chr_formatted)
        if not np.any(cond_1):
            continue
        
        cond_3 = target_data[:,0] == chr_
        
        rd_d = read_depth_data[cond_1][:,2]
        rd_data_1 = read_depth_data[cond_1][:,1].astype(int) 
        min_val = min(rd_data_1)
        max_val = max(rd_data_1)
        
        maps = np.full(max_val,np.inf)
     
        maps[rd_data_1 - 1] = rd_d
        
        ct= 0
        target_data_cond = target_data[cond_3]
        for i in range(target_data_cond.shape[0]):
        
            target_chr = target_data_cond[i,0]
            target_st = target_data_cond[i,1]
            target_end = target_data_cond[i,2]
            
       
            rd_seq = maps[target_st:target_end] 
            indices = np.where(rd_seq != np.inf)[0]

            if indices.shape[0] > 0:
                target_end = target_st + np.max(indices) +1
                target_st = target_st + np.min(indices) 
            
            rd_seq = np.delete(rd_seq, np.where(rd_seq == np.inf))

            appropriate_inds_wgscalls =   (wgs_start_list[cond_2] <= target_end) * ( wgs_ends_list[cond_2] >= target_st)
                    
            if np.sum(appropriate_inds_wgscalls) == 0:
                groundtruthcall = "<NO-CALL>"
            else:
                groundtruthcall =  wgs_calls_data[cond_2][appropriate_inds_wgscalls,3][0] 
            if groundtruthcall == "<INV>":
                continue
            elif groundtruthcall == "<INS>":
                groundtruthcall = "<DUP>"
            
            data_point = []
            data_point.append(sample_name)
            data_point.append(chr_)
            data_point.append(target_st)
            data_point.append(target_end)
            data_point.append(rd_seq)
            data_point.append(groundtruthcall)
            
            labeled_data.append(data_point)
        
        
    labeled_data = np.asarray(labeled_data)
    np.save(os.path.join(output_path,sample_name+"_labeled_data.npy"), labeled_data)

    



        
