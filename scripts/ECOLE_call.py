'''
ECOLÉ source code.
ECOLÉ is a deep learning based WES CNV caller tool.
This script, ECOLÉ_call.py, is only used to load the weights of pre-trained models
and use them to perform CNV calls.
'''
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



'''
Helper function to print informative messages to the user.
'''
def message(message):
    print("[",datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),"]\t", "ECOLÉ:\t", message)


'''
Helper function to convert string chromosome to integer.
'''
def convert_chr_to_integer(chr_list):
    for i in range(len(chr_list)):
            crs = chr_list[i]
            if(len(crs) == 4):
                if(crs[3] == "Y"):
                    crs = 23
                elif (crs[3] == "X"):
                    crs = 24
                else:
                    crs = int(crs[3])
            elif(len(crs) == 5):
                crs = int(crs[3:5])
            chr_list[i] = crs

cur_dirname = os.path.dirname(__file__)
try:
    os.mkdir(os.path.join(cur_dirname,"../tmp/"))
    os.mkdir(os.path.join(cur_dirname,"../tmp2/"))
except OSError:
    print ("Creation of the directory failed")
else:
    print ("Successfully created the directory")



''' 
Perform I/O operations.
'''

description = "ECOLÉ is a deep learning based WES CNV caller tool. \
            For academic research the software is completely free, although for \
            commercial usage the software is licensed. \n \
            please see ciceklab.cs.bilkent.edu.tr/ECOLÉlicenceblablabla."

parser = argparse.ArgumentParser(description=description)

'''
Required arguments group:
(i) -m, pretrained models of the paper, one of the following: (1) ecole, (2) ecole-ft-expert, (3) ecole-ft-somatic. 
(ii) -i, input data path comprised of WES samples with read depth data.
(iii) -o, relative or direct output directory path to write ECOLÉ output file.
(v) -c, Depending on the level of resolution you desire, choose one of the options: (1) exonlevel, (2) merged
(vi) -t, Path of exon target file.
(vii) -n, The path for mean&std stats of read depth values.
'''

required_args = parser.add_argument_group('Required Arguments')
opt_args = parser.add_argument_group("Optional Arguments")

required_args.add_argument("-m", "--model", help="If you want to use pretrained ECOLÉ weights choose one of the options: \n \
                   (i) ecole \n (ii) ecole-ft-expert \n (iii) ecole-ft-somatic.", required=True)

required_args.add_argument("-bs", "--batch_size", help="Batch size to be used in the finetuning.", required=True)

required_args.add_argument("-i", "--input", help="Relative or direct path to input files for ECOLÉ CNV caller, these are the processed samples.", required=True)

required_args.add_argument("-o", "--output", help="Relative or direct output directory path to write ECOLÉ output file.", required=True)

required_args.add_argument("-c", "--cnv", help="Depending on the level of resolution you desire, choose one of the options: \n \
                                                (i) exonlevel, (ii) merged", required=True)

required_args.add_argument("-n", "--normalize", help="Please provide the path for mean&std stats of read depth values to normalize. \n \
                                                    These values are obtained precalculated from the training dataset before the pretraining.", required=True)


opt_args.add_argument("-g", "--gpu", help="Specify gpu", required=False)

'''
Optional arguments group:
-v or --version, version check
-h or --help, help
-g or --gpu, specify gpu
-
'''

parser.add_argument("-V", "--version", help="show program version", action="store_true")
args = parser.parse_args()

if args.version:
    print("ECOLÉ version 0.1")

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   
os.environ["CUDA_VISIBLE_DEVICES"] = ""

if args.gpu:
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"]= args.gpu
    message("Using GPU!")
else:
    message("Using CPU!")

os.makedirs(args.output, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class PositionalEmbedding(nn.Module):
    
        def __init__(self, channels):
            super(PositionalEmbedding, self).__init__()
            inv_freq = 1. / (1000000000 ** (torch.arange(0, channels, 2).float() / channels))
            self.register_buffer('inv_freq', inv_freq)
            
            
        def forward(self, tensor,chrms,strt,ends):
            siz = 1001
            bs = tensor.shape[0]
        
            
        
            
            pos = torch.linspace(strt[0,0].item(), ends[0,0].item(), siz, device=tensor.device).type(self.inv_freq.type()) 

            sin_inp = torch.einsum("i,j->ij", pos, self.inv_freq)
            emb = torch.cat((sin_inp.sin(), sin_inp.cos()), dim=-1)

            emb = emb[None,:,:]
            
            
            for i in range(1,bs):
                pos = torch.linspace(strt[i,0].item(), ends[i,0].item(), siz, device=tensor.device).type(self.inv_freq.type()) 
                sin_inp = torch.einsum("i,j->ij", pos, self.inv_freq)
                emb_i = torch.cat((sin_inp.sin(), sin_inp.cos()), dim=-1)
                emb_i = emb_i[None,:,:]
                
                
                emb = torch.cat((emb, emb_i), 0)
            return emb
            
class CNVcaller(nn.Module):
    def __init__(self, exon_size, patch_size, depth,embed_dim,num_class,channels = 1):
        super().__init__()
        assert exon_size % patch_size == 0, 'Image dimensions must be divisible by the patch size.'
        num_patches = (exon_size // patch_size) 
        patch_dim = channels * patch_size 
        self.patch_size = patch_size
        self.exon_size = exon_size
        self.pos_emb = PositionalEmbedding(embed_dim)
        self.patch_to_embedding = nn.Linear(patch_dim, embed_dim)

        self.chromosome_token = nn.Parameter(torch.randn(1, 24, embed_dim))
        self.to_cls_token = nn.Identity()
        self.attention = Performer(
    dim = embed_dim,
    depth = depth,
    heads = 8
)
              
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, embed_dim),
            nn.GELU()
        )
        self.mlp_head2 = nn.Linear(embed_dim, num_class)

    

    def forward(self, exon, mask):
        chrs = exon[:,:,-1]
        strt = exon[:,:,-2]
        ends = exon[:,:,-3]
        p = self.patch_size

        
        all_ind = list(range(self.exon_size))
        

        indices = torch.tensor(all_ind).to(exon.device)
        exon = torch.index_select(exon, 2, indices).to(exon.device)
        
        all_ind = list(range(self.exon_size + 1))
        
        
        
        indices = torch.tensor(all_ind).to(exon.device)
        mask = torch.index_select(mask, 1, indices).to(exon.device)

        x = rearrange(exon, 'b c (h p1) -> b h (p1 c)', p1 = p)
        x = self.patch_to_embedding(x)
        batch_size, n, _ = x.shape

        
        crs = self.chromosome_token[:, int(chrs[0,0].item()-1): int(chrs[0,0].item()), :]
    
        
        for i in range(1,batch_size):
            crs_ = self.chromosome_token[:, int(chrs[i,0].item()-1): int(chrs[i,0].item()), :]
            
            crs = torch.cat((crs, crs_), 0)
        
        x = torch.cat((crs, x), dim=1)
        
        
        x += self.pos_emb(x,chrs,strt,ends)
        
        x = self.attention(x,input_mask = mask)

    
        x = self.to_cls_token(x[:, 0])

        
    
        y = self.mlp_head(x)

        z = self.mlp_head2(y)
        return z

model = CNVcaller(1000, 1, 3, 192, 3)

if args.model == "ecole":
    model.load_state_dict(torch.load(os.path.join(cur_dirname, "../models/ecole_model.pt"), map_location=device))
elif args.model == "ecole-ft-expert":
    model.load_state_dict(torch.load(os.path.join(cur_dirname,"../models/ecole_ft_expert_model.pt"), map_location=device))
elif args.model == "ecole-ft-somatic":
    model.load_state_dict(torch.load(os.path.join(cur_dirname,"../models/ecole_ft_somatic_model.pt"), map_location=device))

model.eval()
model = model.to(device)

input_files = os.listdir(args.input)
all_samples_names = [file.split("_labeled_data.npy")[0] for file in input_files]

message("Calling for CNV regions...")

for sample_name in tqdm(all_samples_names):

    sampledata = np.load(os.path.join(args.input, sample_name+"_labeled_data.npy"), allow_pickle=True)
  
    message(f"calling sample: {sample_name}")

    sampnames_data = []
    chrs_data = []
    readdepths_data = []
    start_inds_data = []
    end_inds_data = []
    wgscalls_data = []

    temp_sampnames = sampledata[:,0]
    temp_chrs = sampledata[:,1]
    temp_start_inds = sampledata[:,2]
    temp_end_inds = sampledata[:,3]
    temp_readdepths = sampledata[:,4]

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

    lens = [len(v) for v in readdepths_data]

    lengthfilter = [True if v < 1006  else False for v in lens]

    sampnames_data = np.asarray(sampnames_data)[lengthfilter]
    chrs_data = np.asarray(chrs_data)[lengthfilter]
    readdepths_data = np.asarray(readdepths_data)[lengthfilter]
    start_inds_data = np.asarray(start_inds_data)[lengthfilter]
    end_inds_data = np.asarray(end_inds_data)[lengthfilter]

    readdepths_data = np.asarray([np.asarray(k) for k in readdepths_data])
    readdepths_data = sequence.pad_sequences(readdepths_data, maxlen=1005,dtype=np.float32,value=-1)
    readdepths_data = readdepths_data[ :, None, :]

    file1 = open(args.normalize, 'r')
    line = file1.readline()
    means_ = float(line.split(",")[0])
    stds_ = float(line.split(",")[1])

    test_x = torch.FloatTensor(readdepths_data)
    test_x = TensorDataset(test_x)
    x_test = DataLoader(test_x, batch_size=50)

    allpreds = []

    for exons in tqdm(x_test):
    
        exons = exons[0].to(device)

        mask = torch.logical_not(exons == -1)

        exons[:,:,:1000] -= means_
        exons[:,:,:1000] /=  stds_

        mask = torch.squeeze(mask)
        real_mask = torch.ones(exons.size(0),1006, dtype=torch.bool).to(device)
        real_mask[:,1:] = mask

        output1 = model(exons,real_mask)
        
        _, predicted = torch.max(output1.data, 1)

        
        preds = list(predicted.cpu().numpy().astype(np.int64))
        allpreds.extend(preds)
        
 
 

    chrs_data = chrs_data.astype(int)
    allpreds = np.array(allpreds)
    for j in tqdm(range(1,25)):
        indices = chrs_data == j

        predictions = allpreds[indices]
        start_inds = start_inds_data[indices]
        end_inds = end_inds_data[indices]

        sorted_ind = np.argsort(start_inds)
        predictions = predictions[sorted_ind]
     
        end_inds = end_inds[sorted_ind]
        start_inds = start_inds[sorted_ind]
        chr_ = "chr"
        if j < 23:
            chr_ += str(j)
        elif j == 23:
            chr_ += "Y"
        elif j == 24:
            chr_ += "X"
        for k_ in range(len(end_inds)):
           
            os.makedirs(os.path.dirname(os.path.join(cur_dirname,"../tmp/")  + sample_name + ".csv"), exist_ok=True)
            f = open(os.path.join(cur_dirname,"../tmp/") + sample_name + ".csv", "a")
            f.write(chr_ + "," + str(start_inds[k_]) + "," + str(end_inds[k_]) + ","+ str(predictions[k_]) + "\n")
            f.close()


message("Calling for regions without read depth information..")

for sample_name in tqdm(all_samples_names):
 
    message(f"Processing sample: {sample_name}")

    out_folder = args.output
    if args.cnv == "exonlevel":
        f = open(os.path.join(args.output, sample_name + ".csv"), "a")
        f.write("Sample Name" + "\t" +"Chromosome" + "\t" + "CNV Start Index" + "\t" + "CNV End Index" + "\t" + "ECOLÉ Prediction" + "\n")
        f.close()


    calls_ecole = pd.read_csv(os.path.join(cur_dirname,"../tmp/")+ sample_name + ".csv", sep=",", header=None).values
    target_data = pd.read_csv(os.path.join(cur_dirname,"../hglft_genome_64dc_dcbaa0.bed"), sep="\t", header=None).values

    convert_chr_to_integer(target_data[:,0])
    convert_chr_to_integer(calls_ecole[:,0])
 
    chrs_data = calls_ecole[:,0].astype(int)
    allpreds = calls_ecole[:,3].astype(int)
    start_inds_data = calls_ecole[:,1].astype(int)
    end_inds_data = calls_ecole[:,2].astype(int)

    chrs_data_target = target_data[:,0].astype(int)
    start_inds_data_target = target_data[:,1].astype(int)
    end_inds_data_target = target_data[:,2].astype(int)

    for l in range(1,25):
        indices = chrs_data == l
        indices_target = chrs_data_target == l

        if not any(indices):
            continue
      
        start_inds_target = start_inds_data_target[indices_target]
        end_inds_target = end_inds_data_target[indices_target]
        chrs_data_target_ = chrs_data_target[indices_target]
        sorted_ind_target = np.argsort(start_inds_target)
        start_inds_target = start_inds_target[sorted_ind_target]
        end_inds_target = end_inds_target[sorted_ind_target]
        chrs_data_target_ = chrs_data_target_[sorted_ind_target]
        
        predictions = allpreds[indices]
        start_inds = start_inds_data[indices]
        end_inds = end_inds_data[indices]
        sorted_ind = np.argsort(start_inds)
        predictions = predictions[sorted_ind]
        end_inds = end_inds[sorted_ind]
        start_inds = start_inds[sorted_ind]

        np_last_preds = np.zeros(len(sorted_ind_target))

        i = j = k = 0

        while i < len(start_inds_target) and j < len(start_inds):
        
            if start_inds[j] <= end_inds_target[i] and end_inds[j] >= start_inds_target[i]:
                np_last_preds[k] = predictions[j]
                i += 1
                j += 1
            else:
                
                np_last_preds[k] = 3
                i += 1
                
            k +=1
        
        while i < len(start_inds_target):
            np_last_preds[k] = 3
            i += 1
            k += 1        
        
        wind = 1
        a = np_last_preds
        np_last_preds_copy = np.zeros(len(np_last_preds))
        for idx in range(len(np_last_preds_copy)):
            if np_last_preds[idx] == 3:
                left_counter = 0
                right_counter = 0
                left_pointer = idx
                right_pointer = idx
                list_found = [0,0,0]
                while left_counter < wind  and left_pointer > 0:
                    left_pointer -= 1
                    if np_last_preds[left_pointer] == 3:
                        continue
                    else:
                        left_counter += 1
                        if np.abs(left_counter) == 0:
                            print(left_counter, idx)
                        dist = float(np.abs(left_counter))**-2
                        list_found[int(np_last_preds[left_pointer])] += dist
                
                while right_counter < wind + 1 and right_pointer < len(np_last_preds) - 1:
                    right_pointer += 1
                    if np_last_preds[right_pointer] == 3:
                        continue
                    else:
                        right_counter += 1
                        if np.abs(right_counter) == 0:
                            print(right_counter, idx)
                        dist = float(np.abs(right_counter))**-2
                        list_found[int(np_last_preds[right_pointer])] += dist

                
                np_last_preds_copy[idx] = np.argmax(np.array(list_found))
            else:
                np_last_preds_copy[idx] = np_last_preds[idx]
            
            chr_ = "chr"
            if chrs_data_target_[idx] < 23:
                chr_ += str(chrs_data_target_[idx])
            elif chrs_data_target_[idx] == 23:
                chr_ += "Y"
            elif chrs_data_target_[idx] == 24:
                chr_ += "X"

            
            if args.cnv == "exonlevel":
                call_made = np_last_preds_copy[idx]
                cnv_call_string = "NO-CALL"
                if call_made == 1:
                    cnv_call_string = "DUP"
                elif call_made == 2:
                    cnv_call_string = "DEL"

                f = open(os.path.join(args.output, sample_name + ".csv"), "a")
                f.write(sample_name + "\t" + chr_ + "\t" + str(start_inds_target[idx]) + "\t" + str(end_inds_target[idx]) + "\t"+ cnv_call_string+ "\n")
                f.close()                
            elif args.cnv == "merged":
                f = open(os.path.join(cur_dirname,"../tmp2/")  + sample_name + ".csv", "a")
                f.write(chr_ + "," + str(start_inds_target[idx]) + "," + str(end_inds_target[idx]) + ","+ str(np_last_preds_copy[idx]) + "\n")
                f.close()

def grouped_preds(preds):
    idx = 0
    result = []
    ele = -1
    for key, sub in groupby(preds):
        ele = len(list(sub))
        result.append((idx,idx + ele-1))
        idx += ele

    return result

if args.cnv == "merged":
    for sample_name in tqdm(all_samples_names):

        f = open(os.path.join(args.output, sample_name + ".csv"), "a")
        f.write("Sample Name" + "\t" +"Chromosome" + "\t" + "CNV Start Index" + "\t" + "CNV End Index" + "\t" + "ECOLÉ Prediction" + "\n")
        f.close()

        calls_ecole = pd.read_csv(os.path.join(cur_dirname,"../tmp2/")+ sample_name + ".csv", sep=",", header=None).values
        convert_chr_to_integer(calls_ecole[:,0])

        chrs_data = calls_ecole[:,0].astype(int)
        allpreds = calls_ecole[:,3].astype(int)
        start_inds_data = calls_ecole[:,1].astype(int)
        end_inds_data = calls_ecole[:,2].astype(int)

        for l in range(1,25):
            indices = chrs_data == l

            predictions = allpreds[indices]
            start_inds = start_inds_data[indices]
            end_inds = end_inds_data[indices]
            sorted_ind = np.argsort(start_inds)
            predictions = predictions[sorted_ind]
            end_inds = end_inds[sorted_ind]
            start_inds = start_inds[sorted_ind]

            S = grouped_preds(predictions)

            for i in range(0,len(S)):
        
                call_ecole = np.bincount(predictions[S[i][0]:S[i][1]+1],minlength=3).argmax()
                j = l
                chr_ = "chr"
                if j < 23:
                    chr_ += str(j)
                elif j == 23:
                    chr_ += "Y"
                elif j == 24:
                    chr_ += "X"
                
                call_made = np.bincount(predictions[S[i][0]:S[i][1]+1],minlength=3).argmax()
                cnv_call_string = "NO-CALL"
                if call_made == 1:
                    cnv_call_string = "DUP"
                elif call_made == 2:
                    cnv_call_string = "DEL"

                f = open(os.path.join(args.output, sample_name + ".csv"), "a")
                f.write(sample_name + "\t" + chr_ + "\t" + str(start_inds[S[i][0]]) + "\t" + str(end_inds[S[i][1]]) + "\t"+ cnv_call_string + "\n")
                f.close()



filelisttmp1 = os.listdir(os.path.join(cur_dirname,"../tmp/"))
filelisttmp2 = os.listdir(os.path.join(cur_dirname,"../tmp2/"))

for f in filelisttmp1:
    os.remove(os.path.join(os.path.join(cur_dirname,"../tmp/"), f))
for f in filelisttmp2:
    os.remove(os.path.join(os.path.join(cur_dirname,"../tmp2/"), f))

os.rmdir(os.path.join(cur_dirname,"../tmp/")) 
os.rmdir(os.path.join(cur_dirname,"../tmp2/")) 
