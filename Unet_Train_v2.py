from Config import INIT_LR
import os
import Config as config
from Utils import _read_image_and_labels_reisze
import numpy as np
import pandas as pd
from tensorflow.python.keras import layers
from Unet_Model import Unet3D
from Unet_Loss import *
from tensorflow.python.keras import optimizers
import matplotlib.pyplot as plt
import tensorflow as tf

os.environ["CUDA_DEVICE_ORDER"]=config.CUDA_DEVICE_ORDER
os.environ["CUDA_VISIBLE_DEVICES"]=config.CUDA_VISIBLE_DEVICES

def Training():
    d_cfg = dict() #dataset configureation
    # parameters for tf.data.Dataset
    d_cfg["patch_size"]= [config.PATCH_SIZE,config.PATCH_SIZE,config.PATCH_SIZE]
    d_cfg["buffer_size"] = config.BUFFER_SIZE
    d_cfg["batch_size"] = config.BATCH_SIZE
    d_cfg["labels"] = config.LABELS
    d_cfg["num_patches_per_image"] =config.NUM_PATCHES_PER_IMAGE

    #d_cfg["num_patches_per_image_val"] =NUM_PATCHES_PER_IMAGE_VAL
    #d_cfg["dtype"] = (tf.float32, tf.uint8)


    train_file_path = pd.read_csv(config.TRAIN_CSV_PATH, dtype='object', keep_default_na=False,na_values=[]).as_matrix()
    val_file_path   = pd.read_csv(config.VAL_CSV_PATH, dtype='object', keep_default_na=False,na_values=[]).as_matrix()

    ##-----------Making the Storage Path------------##############
    if not os.path.exists(config.MODEL_OUTPUT_PATH):
        os.makedirs(config.MODEL_OUTPUT_PATH)
    else:
        print(config.MODEL_OUTPUT_PATH,'already exists')
    if not os.path.exists(config.MODEL_FIG_OUTPUT_PATH):
        os.makedirs(config.MODEL_FIG_OUTPUT_PATH)
        print(config.MODEL_FIG_OUTPUT_PATH,'already exists')


    img_shape = (d_cfg["patch_size"][0],d_cfg["patch_size"][1],d_cfg["patch_size"][2],1)
    n_labels = len(d_cfg["labels"]) +1 #background
    print(img_shape)

    inputs = layers.Input(shape=img_shape)
    model, outputs= Unet3D(inputs,n_labels )

    input_train = train_file_path
    input_val = val_file_path

    lbl_weights = np.ones(len(d_cfg["labels"])+1)
    lbl_weights[17:20]=3
    lbl_weights[21]=3

    pred_shape=[int(outputs.get_shape()[1].value),
                int(outputs.get_shape()[2].value),
                int(outputs.get_shape()[3].value)]

    filename_to_patches_fn = lambda input: tf.compat.v1.py_func(func=_image_name_to_3Dpatches,
                                                inp=[input,d_cfg["patch_size"],d_cfg["labels"],
                                                    lbl_weights,
                                                    pred_shape,
                                                    d_cfg["num_patches_per_image"]],
                                                Tout=[tf.float32, tf.float32])

    # seems not able to define inp prior, it can only be defined after :
    ds_train = (tf.data.Dataset.from_tensor_slices(input_train).shuffle(len(input_train))
        .map(filename_to_patches_fn ,num_parallel_calls=config.NUM_PARALLEL_CALLS)
            #.apply(tf.data.experimental.unbatch())
            .apply(tf.data.experimental.unbatch())
            .shuffle(buffer_size=d_cfg["buffer_size"]).repeat()
            .batch(d_cfg["batch_size"])
            .prefetch(32))


    #lbl_weights = np.ones(len(d_cfg["labels"])+1)
    filename_to_patches_fn_val_flag = lambda input: tf.compat.v1.py_func(func=_image_name_to_3Dpatches_val_flag,
                                                inp=[input,d_cfg["patch_size"],d_cfg["labels"],
                                                    lbl_weights,
                                                    pred_shape,
                                                    d_cfg["num_patches_per_image"]],
                                                Tout=[tf.float32, tf.float32])

    ds_val = (tf.data.Dataset.from_tensor_slices(input_val).shuffle(len(input_val))
        .map(filename_to_patches_fn_val_flag ,num_parallel_calls=config.NUM_PARALLEL_CALLS_VAL)
            #.apply(tf.data.experimental.unbatch())
            .apply(tf.data.experimental.unbatch())
            #.shuffle(buffer_size=32)
            #.repeat()
            .batch(d_cfg["batch_size"]))
            #.prefetch(8))


    save_model_path =os.path.join(config.MODEL_OUTPUT_PATH,'weights.{epoch:02d}-{val_loss:.2f}.hdf5')
    checkpoint = tf.keras.callbacks.ModelCheckpoint(filepath=save_model_path,
                                                    monitor='val_dice_loss_multi_weighted',
                                                    save_best_only=False,
                                                    verbose=0,
                                                    save_weights_only=False,
                                                    mode='auto',
                                                    period=1)

    tensor_board = tf.keras.callbacks.TensorBoard(log_dir=config.MODEL_OUTPUT_PATH ,
                                histogram_freq=1,
                                batch_size=1,
                                write_graph=True,
                                write_grads=False,
                                write_images=False)#,
                                #embeddings_freq=0,
                                #embeddings_layer_names=None,
                                #embeddings_metadata=None)#,
                                #embeddings_data=None)
    cp = [checkpoint,tensor_board]



    n_patches_per_image = d_cfg["num_patches_per_image"]
    n_patches_per_lbl = (lbl_weights/lbl_weights.sum()*n_patches_per_image).astype(int)
    print(n_patches_per_lbl)

    # 1. Read file paths

    #num_parallel_calls=num_parallel_calls
    num_parallel_calls=1
    num_parallel_calls_val=1
    filename_to_image_fn = lambda input: tf.compat.v1.py_func(func=_read_image_and_labels_reisze,
                                                inp=[input,d_cfg["patch_size"],d_cfg["labels"]],
                                                Tout=[tf.float32, tf.uint8,tf.string])
    lbl_weights = np.ones(len(d_cfg["labels"])+1)
    pred_shape=[int(outputs.get_shape()[1].value),
                int(outputs.get_shape()[2].value),
                int(outputs.get_shape()[3].value)]

    image_to_patches_fn = lambda images,lbls,images_name: tf.compat.v1.py_func(func=image_to_3Dpatches_weighted_train_demo,
                                                inp=[images,lbls,images_name,
                                                    lbl_weights,
                                                    d_cfg["patch_size"],
                                                    pred_shape,# from last dataset
                                                    2],
                                                    #d_cfg["num_patches_per_image"]],
                                                Tout=[tf.float32, tf.float32])#,tf.string])

    ds_pred_train = (tf.data.Dataset.from_tensor_slices(input_train)
        .map(filename_to_image_fn,num_parallel_calls=config.NUM_PARALLEL_CALLS_VAL)
            .map(image_to_patches_fn,num_parallel_calls=config.NUM_PARALLEL_CALLS_VAL)
            #.apply(tf.data.experimental.unbatch())
            .apply(tf.data.experimental.unbatch())
            #.shuffle(128)
            .batch(8))

    ds_pred_val = (tf.data.Dataset.from_tensor_slices(input_val)
        .map(filename_to_image_fn,num_parallel_calls=config.NUM_PARALLEL_CALLS_VAL)
            .map(image_to_patches_fn,num_parallel_calls=config.NUM_PARALLEL_CALLS_VAL)
            #.apply(tf.data.experimental.unbatch())
            .apply(tf.data.experimental.unbatch())
            #.shuffle(128)
            .batch(8))

    element = tf.compat.v1.data.make_one_shot_iterator(ds_pred_train).get_next()
    with tf.compat.v1.Session() as sess:
            batch_of_imgs_train, batch_of_label_train = sess.run(element)
    element = tf.compat.v1.data.make_one_shot_iterator(ds_pred_val).get_next()
    with tf.compat.v1.Session() as sess:
            batch_of_imgs_val, batch_of_label_val= sess.run(element)


    idx_train = []
    for i in range(len(batch_of_imgs_train)):
        mask = batch_of_label_train[i,:,:,:,0]==1
        bbx = np.where(mask)
        if any(bbx[0]):
            #print(batch_of_label_train[i,:,:,:,0].sum())
            #print(batch_of_label_train[i,:,:,:,0].sum()/(128**3))
            idx_train.append(int((np.min(bbx[0])+np.max(bbx[0]))/2))
        else:
            idx_train.append(int(mask.shape[0]/2))
    idx_val = []
    for i in range(len(batch_of_imgs_val)):
        bbx = np.where(batch_of_label_val[i,:,:,:,0]==1)
        if any(bbx[0]):
            idx_val.append(int((np.min(bbx[0])+np.max(bbx[0]))/2))
        else:
            idx_val.append(int(mask.shape[0]/2))
    print(idx_train)
    print(idx_val)

    steps_per_epochs_train = int(len(train_file_path)*d_cfg["num_patches_per_image"]/d_cfg["batch_size"])
    steps_per_epochs_val = int(len(val_file_path)*d_cfg["num_patches_per_image"]/d_cfg["batch_size"])
    print(steps_per_epochs_train )
    print(steps_per_epochs_val)
    print(config.MODEL_OUTPUT_PATH)

    weights = tf.constant([1,1,1,1,1,#0,1,2,3,4
                        1,1,1,1,#5,6,7,8
                        1, 1,1,#9,10,11+12
                        1,1,1,#13,14+15,16
                        1,1,1,#17,18+19,20+21
                        1,1,#22+23,25
                        1,1,1],#26,27,30
                        dtype=tf.float32)




    adamopt = optimizers.Adam(lr=INIT_LR, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0, amsgrad=False)
    model.compile(optimizer=adamopt, loss=ce_dice_loss_multi_weighted(n_labels,weights), metrics=[dice_loss_multi_weighted(n_labels,weights),
                                                                                        ce_loss_multi(n_labels,weights)])

    for ip in range(0,100,config.V_STEPS):
        #tart_time = time.time()
        #print('---prediction for visualization start--')

        plt.figure(figsize=(14, 30))
        for i in range(8):
            pred = model.predict(batch_of_imgs_train[i,:,:,:,:][np.newaxis,:,:,:,:])
            plt.subplot(8, 4, 4 * i + 1)
            plt.imshow(batch_of_imgs_train[i,idx_train[i],:,:,0],cmap='gray',vmin=-0.3,vmax=0.3)
            plt.title("Input image")
            plt.subplot(8, 4, 4 * i + 2)
            plt.imshow(batch_of_label_train[i,idx_train[i], :, :, 0],vmin=0,vmax=28)
            plt.title("Actual Mask")
            plt.subplot(8, 4, 4 * i + 3)
            #plt.imshow(pred[0,idx_train[i], :, :, 0],vmin=0,vmax=1)
            pred_plot = np.argmax(pred,axis=-1)
            plt.imshow(pred_plot[0,idx_train[i], :, :],vmin=0,vmax=28)
            plt.title("Predicted Mask")
            plt.subplot(8,4,4 * i +4)
            print('---train set ------')
            score = dice_loss_multi_np(batch_of_label_train[i,:, :, :, 0],pred)

            plt.text(0.1,0.8,'DSC=%4.3f' % score )
            plt.text(0.1,0.6,'min_pred_val=%4.3f' % np.min(pred))
            plt.text(0.1,0.4,'max_pred_val=%4.3f'% np.max(pred))
            plt.axis('off')
        #plt.show()
        plt.savefig(os.path.join(config.MODEL_FIG_OUTPUT_PATH,'training_trian_'+str(ip)+'.png'))

        plt.figure(figsize=(14, 30))
        for i in range(8):
            pred = model.predict(batch_of_imgs_val[i,:,:,:,:][np.newaxis,:,:,:,:])
            plt.subplot(8, 4, 4 * i + 1)
            plt.imshow(batch_of_imgs_val[i,idx_val[i],:,:,0],cmap='gray',vmin=-0.3,vmax=0.3)
            plt.title("Input image")
            plt.subplot(8, 4, 4 * i + 2)
            plt.imshow(batch_of_label_val[i,idx_val[i], :, :, 0],vmin=0,vmax=28)
            plt.title("Actual Mask")
            plt.subplot(8, 4, 4 * i + 3)
            #plt.imshow(pred[0,idx_val[i], :, :, 0],vmin=0,vmax=1)
            pred_plot = np.argmax(pred,axis=-1)
            plt.imshow(pred_plot[0,idx_val[i], :, :],vmin=0,vmax=28)
            plt.title("Predicted Mask")
            plt.subplot(8,4,4 * i +4)
            print('---val set ------')
            score = dice_loss_multi_np(batch_of_label_val[i,:, :, :, 0],pred)
            plt.text(0.1,0.8,'DSC=%4.3f' % score )
            plt.text(0.1,0.6,'min_pred_val=%4.3f' % np.min(pred))
            plt.text(0.1,0.4,'max_pred_val=%4.3f'% np.max(pred))
            plt.axis('off')
        #plt.show()
        plt.savefig(os.path.join(config.MODEL_FIG_OUTPUT_PATH,'training_val_'+str(ip)+'.png'))
        #print('---prediction for visualization finished--')
        if ip %50 ==0: # reduce learning rate

            init_lr = INIT_LR*0.1
            adamopt = optimizers.Adam(lr=init_lr, beta_1=config.BETA_1, beta_2=config.BETA_2, epsilon=config.EPSOLON, decay=config.DECAY, amsgrad=config.AMSGRAD)
            model.compile(optimizer=adamopt, loss=ce_dice_loss_multi_weighted(n_labels,weights), metrics=[dice_loss_multi_weighted(n_labels,weights),ce_loss_multi(n_labels,weights)])

    #        print('Train epoch',ip,'lr reduced to ',init_lr)
        history = model.fit(ds_train,
                        epochs=ip+config.V_STEPS,
                        steps_per_epoch =int(steps_per_epochs_train/4),
                        validation_data=ds_val,
                        validation_steps=config.VALIDATION_STEPS,#32,#int(steps_per_epochs_val),
                        callbacks=cp,
                    initial_epoch=ip)
        #print('--Train epoch',ip,'to',ip+5,'time used',time.time()-start_time,'s')


def image_to_3Dpatches_weighted(images,lbls,image_name,
                                lbl_weights,
                                patch_size,# = [128,128,128],
                                pred_shape,
                               n_patches_per_image=config.NUM_PATCHES_PER_IMAGE):
    """ Generate image patches similar to the code of DLTK

        Args:
            images (np.ndarray)
            lbls (np.ndarray)
            patch_size


        Returns:
            images (5D np.ndarray): shape is (batch_size=n_patches_per_image,patch_dim1,patch_dim2,patch_dim3,channel=1)
            lbls (5D np.ndarray): shape is (batch_size=n_patches_per_image,patch_dim1,patch_dim2,patch_dim3,channel=1)
    """
# 1. Determine # patches to extract from each class
    #print('Extracting patches from ',image_name)
    #print('Labels in the segmentation model',range(len(lbl_weights)))
    #print(image_name)#,' has label',np.unique(lbls))
    # validate lablels
    lbl_valid = np.intersect1d(range(len(lbl_weights)),np.unique(lbls))
    # make invalide labels to 0 weight
    lbl_invalid = np.setdiff1d(range(len(lbl_weights)),np.unique(lbls))
    _lbl_weights = np.copy(lbl_weights)
    _lbl_weights[lbl_invalid] =0
    n_patches_per_lbl = (_lbl_weights/_lbl_weights.sum()*n_patches_per_image).astype(int)
    # make sure n_patches_per_image is extracted
    for ip in range(n_patches_per_image-n_patches_per_lbl.sum()):
        n_patches_per_lbl[lbl_valid[ip]]=n_patches_per_lbl[lbl_valid[ip]]+1
    #print(n_patches_per_lbl)
# 2. Get patches
    image_patches = []
    lbl_patches = []
    images_name = []

    # loop across lbls
    for lbl in lbl_valid:
        idx_all = np.argwhere(lbls == lbl)

        #print(image_name,': extracting',n_patches_per_lbl[lbl],'patches with lbl=',lbl)
        idx_temp = np.random.choice(len(idx_all),n_patches_per_lbl[lbl])
        idx = idx_all[idx_temp]

        # get patch_start
        patch_half1  = np.round(np.array(patch_size/2)).astype(int)
        patches_start = idx - patch_half1
        patches_start = np.maximum(patches_start,0)
        patches_start = np.minimum(patches_start,np.array(images.shape)-patch_size)
        for ip in range(len(patches_start)):
            slicer = [slice(patches_start[ip][dim], patches_start[ip][dim] + patch_size[dim]) for dim in range(images.ndim)]
            im_patch = images[tuple(slicer)][:,:,:,np.newaxis]#(patch_size,1),channel last

            # crop batches to output shape
            crop_half = np.round(np.array((d_cfg["patch_size"]- np.array(pred_shape))/2)).astype(int)

            slicer = [slice(patches_start[ip][dim]+crop_half[dim],
                            patches_start[ip][dim]+crop_half[dim]
                            + pred_shape[dim]) for dim in range(images.ndim)]
            lbl_patch = lbls[tuple(slicer)][:,:,:,np.newaxis] #(patch_size,1)

            #print('img_patch_shape',im_patch.shape)
            #print('lbl_patch_shape',lbl_patch.shape)
            image_patches.append(im_patch)
            lbl_patches.append(lbl_patch)
            images_name.append(image_name)

    image_patches = np.asarray(image_patches)
    #print('image_patches shape',image_patches.shape)
    lbl_patches = np.asarray(lbl_patches)
    #print('lble_patches shape',lbl_patches.shape)
    return image_patches,lbl_patches.astype(np.float32)#,images_name


def _image_name_to_3Dpatches(input_path,patch_size,labels,
                            lbl_weights,
                            pred_shape,
                            n_patches_per_image=32):
    #start_time = time.time()
    images,lbls,img_name= _read_image_and_labels_reisze(input_path,patch_size,labels)
    image_patches,lbl_patches = image_to_3Dpatches_weighted(images,lbls,img_name,
                                lbl_weights,
                                patch_size,# = [128,128,128],
                                pred_shape,
                               n_patches_per_image)
    #print('extracted train patches', img_name,len(image_patches),'time', time.time()-start_time)
    return image_patches,lbl_patches

def _image_name_to_3Dpatches_val_flag(input_path,patch_size,labels,
                            lbl_weights,
                            pred_shape,
                            n_patches_per_image=32):
    #start_time = time.time()
    images,lbls,img_name= _read_image_and_labels_reisze(input_path,patch_size,labels)
    image_patches,lbl_patches = image_to_3Dpatches_weighted(images,lbls,img_name,
                                lbl_weights,
                                patch_size,# = [128,128,128],
                                pred_shape,
                               n_patches_per_image)
    #print('extracted val patches', img_name,len(image_patches),'time', time.time()-start_time)
    return image_patches,lbl_patches

def image_to_3Dpatches_weighted_train_demo(images,lbls,image_name,
                                lbl_weights,
                                patch_size,# = [128,128,128],
                                pred_shape,
                               n_patches_per_image=32):
    """ Generate image patches similar to the code of DLTK

        Args:
            images (np.ndarray)
            lbls (np.ndarray)
            patch_size


        Returns:
            images (5D np.ndarray): shape is (batch_size=n_patches_per_image,patch_dim1,patch_dim2,patch_dim3,channel=1)
            lbls (5D np.ndarray): shape is (batch_size=n_patches_per_image,patch_dim1,patch_dim2,patch_dim3,channel=1)
    """
#  Use random seed
    #1. Determine # patches to extract from each class
    #print('Extracting patches from ',image_name)
    #print('Labels in the segmentation model',range(len(lbl_weights)))
    #print(image_name)#,' has label',np.unique(lbls))

    # validate lablels
    lbl_valid = np.intersect1d(range(len(lbl_weights)),np.unique(lbls))
    # make invalide labels to 0 weight
    lbl_invalid = np.setdiff1d(range(len(lbl_weights)),np.unique(lbls))
    #lbl_weights[lbl_invalid] =0
    n_patches_per_lbl = (lbl_weights/lbl_weights.sum()*n_patches_per_image).astype(int)
    # make sure n_patches_per_image is extracted
    for ip in range(n_patches_per_image-n_patches_per_lbl.sum()):
        # Add additional patches to random labels
        idx_add = np.random.choice(lbl_valid[1:])
        n_patches_per_lbl[idx_add]=n_patches_per_lbl[idx_add]+1
    print('lbl valid',lbl_valid)
    print('n_patches_per_lbl',n_patches_per_lbl)


    #print('n_patches_per_lbl',n_patches_per_lbl)
    #print(n_patches_per_lbl)
# 2. Get patches
    image_patches = []
    lbl_patches = []
    images_name = []
    # loop across lbls
    for lbl in lbl_valid:
        idx_all = np.argwhere(lbls == lbl)

        #print(image_name,': extracting',n_patches_per_lbl[lbl],'patches with lbl=',lbl)
        #np.random.seed(412)
        idx_temp = np.random.choice(len(idx_all),n_patches_per_lbl[lbl])
        idx = idx_all[idx_temp]
        # get patch_start
        patch_half1  = np.round(np.array(patch_size/2)).astype(int)
        patches_start = idx - patch_half1
        patches_start = np.maximum(patches_start,0)
        patches_start = np.minimum(patches_start,np.array(images.shape)-patch_size)
        for ip in range(len(patches_start)):
            slicer = [slice(patches_start[ip][dim], patches_start[ip][dim] + patch_size[dim]) for dim in range(images.ndim)]
            im_patch = images[tuple(slicer)][:,:,:,np.newaxis]#(patch_size,1),channel last

            # crop batches to output shape
            crop_half = np.round(np.array((d_cfg["patch_size"]- np.array(pred_shape))/2)).astype(int)

            slicer = [slice(patches_start[ip][dim]+crop_half[dim],
                            patches_start[ip][dim]+crop_half[dim]
                            + pred_shape[dim]) for dim in range(images.ndim)]
            lbl_patch = lbls[tuple(slicer)][:,:,:,np.newaxis] #(patch_size,1)

            #print('img_patch_shape',im_patch.shape)
            #print('lbl_patch_shape',lbl_patch.shape)
            image_patches.append(im_patch)
            lbl_patches.append(lbl_patch)
            images_name.append(image_name)

    image_patches = np.asarray(image_patches)
    #print('image_patches shape',image_patches.shape)
    lbl_patches = np.asarray(lbl_patches)
    #print('lble_patches shape',lbl_patches.shape)
    return image_patches,lbl_patches.astype(np.float32)#,images_name


if __name__ == '__main__':
   Training()