import os 

# Model 
patch_res = 16 
mlp_dim_D_S = 256 #512 
mlp_dim_D_C = 2048 #4096 
n_layers = 8 #24 
lite = False #True 

# Dataset 
batch_size = 2 
n_frames = 5 
img_size = 256 
path_base = r"D:\Dataset\AVSBench\s4_data\s4_data"
path_meta_csv = os.path.join(path_base, "s4_meta_data.csv") 
path_img = os.path.join(path_base, "visual_frames") 
path_audio_log_mel_img = os.path.join(path_base, "audio_log_mel_img") 

# Optimizer  
n_epoches = 15 
lr = 1e-4 
betas = (0.9, 0.999)
weight_decay = 1e-5 