###-----Configuration-----####
CUDA_DEVICE_ORDER="PCI_BUS_ID"
CUDA_VISIBLE_DEVICES="0"
PATCH_SIZE = 128 
BUFFER_SIZE = 128
BATCH_SIZE = 1
LABELS = [1,2,3,4,5,6,7,8,9,10,
                  11,13,14,16,17,18,20,22,
                   25,26,27,30] #labels used to train
### ----- Optimizer ------####
BETA_1 = 0.9 
BETA_2 = 0.999
EPSOLON = 1e-08 
DECAY = 0.0
AMSGRAD = False

NUM_PATCHES_PER_IMAGE = 32
NUM_PATCHES_PER_IMAGE_VAL = 32
VALIDATION_STEPS = 2
TRAIN_CSV_PATH = 'XCAT_50adult_train_1st.csv'
VAL_CSV_PATH = 'XCAT_50adult_val_1st.csv'
MODEL_OUTPUT_PATH = 'journal1_XCAT_DSC_CE_nw_fold1_model_nGPU'
MODEL_FIG_OUTPUT_PATH = 'journal1_XCAT_DSC_CE_nw_fold1_figs_nGPU'
NUM_PARALLEL_CALLS = 4 
NUM_PARALLEL_CALLS_VAL = 1

INIT_LR = 0.001 # at step 40, changed LR to 0.00001/2
V_STEPS = 1 # steps to save visualized output
