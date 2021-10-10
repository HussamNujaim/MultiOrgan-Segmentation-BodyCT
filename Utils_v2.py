from scipy.stats import truncnorm
import SimpleITK as sitk
import numpy as np
from Preprocess import resample_img
import Config as config

def get_truncated_normal(mean=0, sd=1, low=0, upp=10):
    return truncnorm(
        (low - mean) / sd, (upp - mean) / sd, loc=mean, scale=sd)

def _read_image_and_labels_reisze(input_path,patch_size,labels):
    """
    Summary:
        1. Read in images and lbls from file path
        2. Resize the images

    Args:
        input_path: input_path[0] is image path, input_path[1] is lbl path
        d_cfg: dataset related configurations
    Returns:
        np.float32: images np array
        np.uint8: lbls np array
        np.string: image names
    """
    img_path,lbl_path = input_path#.astype(str, copy=False)#not sure why return byte object
    #img_path,lbl_path = input_path
    img_name = img_path.split('/')[-1].split('.')[0]
    #img_path = tf.Print(img_path,[img_path], message='img_path_show::')

    #print(img_name)
    img_sitk = sitk.ReadImage(img_path)
    lbl_sitk = sitk.ReadImage(lbl_path)
    # generate random spacing
#     min_out_size = patch_size
#     max_space_xy=img_sitk.GetSize()[0]*img_sitk.GetSpacing()[0]/(min_out_size[0]+2)
#     max_space_z=img_sitk.GetSize()[2]*img_sitk.GetSpacing()[2]/(min_out_size[0]+2) # make 2 voxel more, to

#     # generate random spacing
#     # add a 0.5 chance of being 5mm

#     z_rand = np.random.uniform(low=0,high=1)
#     if z_rand <0.5:
#         z_space = 5
#     else:
#         z_space = np.minimum(get_truncated_normal(mean=2.5, sd=0.5, low=1.5, upp=3).rvs(), max_space_z)

#     z_space = np.minimum(z_space, max_space_z)
#     xy_space = np.minimum(get_truncated_normal(mean=0.85, sd=0.1, low=0.65, upp=1).rvs(),max_space_xy)

    #print(img_name,'z_space',z_space,'xy_space',xy_space)
    xy_space = 2.5
    z_space = 5
    img_sitk = resample_img(img_sitk,[xy_space,xy_space,z_space],is_label=False)
    lbl_sitk = resample_img(lbl_sitk,[xy_space,xy_space,z_space],is_label=True)

    images = sitk.GetArrayFromImage(img_sitk)

    images = np.clip(images, -1000., 800.).astype(np.float32)
    images = (images + 1000.) / 900. - 1. # distributed at -1 to 1


    # Normalize the images
    #images = images.astype(np.float32)
    #mean = np.mean(images)
    #std = np.std(images)
    #if std>0:
        #images = (images-mean)/std
    #else:
        #images = images*0

    # Change labels for training
    lbls_orig = sitk.GetArrayFromImage(lbl_sitk)

    # !!!! TEMPORARY:
    lbls_orig[lbls_orig==36]=0
    lbls_orig[lbls_orig==12]=11 # clavicle L -> clavicle R
    lbls_orig[lbls_orig==15]=14 # scalpula L -> scalpula R
    lbls_orig[lbls_orig==19]=18 # pelvis L -> pelvis R
    lbls_orig[lbls_orig==21]=20 # femur L -> femur R
    lbls_orig[lbls_orig==23]=22 # arm L -> arm R

    lbls = np.zeros_like(lbls_orig)
    for io in range(len(labels)):
        lbls[lbls_orig==labels[io]]=io+1

    # pad image boundaries
    pad_z = 8
    images = np.pad(images,((pad_z,pad_z),(0,0),(0,0)),'edge')
    lbls = np.pad(lbls,((pad_z,pad_z),(0,0),(0,0)),'edge')


    # pad images if they are smaller than input size
    padding = [np.maximum([config.PATCH_SIZE,config.PATCH_SIZE,config.PATCH_SIZE][dim]-images.shape[dim],0) for dim in range(3)]
    if (padding[0]>0) |(padding[1]>0)|(padding[2]>0):
        print(img_name)
        images = np.pad(images,((int(np.floor(padding[0]/2)),int(np.ceil(padding[0]/2))),
                               (int(np.floor(padding[1]/2)),int(np.ceil(padding[1]/2))),
                               (int(np.floor(padding[2]/2)),int(np.ceil(padding[2]/2))),),'constant',constant_values=-1)
        lbls = np.pad( lbls,((int(np.floor(padding[0]/2)),int(np.ceil(padding[0]/2))),
                               (int(np.floor(padding[1]/2)),int(np.ceil(padding[1]/2))),
                               (int(np.floor(padding[2]/2)),int(np.ceil(padding[2]/2))),),'constant',constant_values=0)
        print(images.shape)


    return images.astype(np.float32),lbls.astype(np.uint8),img_name#.decode("utf-8")

