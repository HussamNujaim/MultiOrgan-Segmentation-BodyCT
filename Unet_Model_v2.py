from tensorflow.python.keras import layers
from tensorflow.python.keras import models

# Encoder blocker
def conv_block_same_padding(input_tensor, num_filters):
  encoder = layers.Conv3D(num_filters, (3,3,3), padding='same')(input_tensor)
  encoder = layers.advanced_activations.LeakyReLU(alpha=0.1)(encoder)
  encoder = layers.Conv3D(num_filters*2, (3, 3,3), padding='same')(encoder)
  encoder = layers.advanced_activations.LeakyReLU(alpha=0.1)(encoder)
  return encoder

def conv_block_same_padding_same_ft(input_tensor, num_filters):
  encoder = layers.Conv3D(num_filters, (3,3,3), padding='same')(input_tensor)
  encoder = layers.advanced_activations.LeakyReLU(alpha=0.1)(encoder)
  encoder = layers.Conv3D(num_filters, (3, 3,3), padding='same')(encoder)
  encoder = layers.advanced_activations.LeakyReLU(alpha=0.1)(encoder)
  return encoder

def encoder_block_same_padding(input_tensor, num_filters):
  encoder = conv_block_same_padding(input_tensor, num_filters)
  encoder_pool = layers.MaxPooling3D((2, 2,2), strides=(2,2,2))(encoder)
  return encoder_pool, encoder

def decoder_block_same_padding(input_tensor, concat_tensor, num_filters):
    decoder = layers.Conv3DTranspose(num_filters*2, (2,2,2), strides=(2,2,2), padding='same')(input_tensor)
    decoder = layers.concatenate([concat_tensor, decoder], axis=-1)
    decoder = layers.advanced_activations.LeakyReLU(alpha=0.1)(decoder)
    decoder = layers.Conv3D(num_filters, (3,3,3), padding='same')(decoder)
    decoder = layers.advanced_activations.LeakyReLU(alpha=0.1)(decoder)
    decoder = layers.Conv3D(num_filters, (3,3,3), padding='same')(decoder)
    decoder = layers.advanced_activations.LeakyReLU(alpha=0.1)(decoder)
    return decoder

def Unet3D(inputs,
        n_labels):

    #img_shape  = (d_cfg["patch_size"][0],d_cfg["patch_size"][1],d_cfg["patch_size"][2],1)
    #inputs = layers.Input(shape=img_shape)
    encoder0_pool, encoder0 = encoder_block_same_padding(inputs, 32)
    encoder1_pool, encoder1 = encoder_block_same_padding(encoder0_pool, 64)
    encoder2_pool, encoder2 = encoder_block_same_padding(encoder1_pool, 128)
    center = conv_block_same_padding_same_ft(encoder2_pool, 256)
    decoder2 = decoder_block_same_padding(center , encoder2,256)
    decoder1 = decoder_block_same_padding(decoder2, encoder1, 128)
    decoder0 = decoder_block_same_padding(decoder1, encoder0, 64)
    #outputs = layers.Conv3D(n_labels, (1,1,1), activation='sigmoid')(decoder0)
    outputs = layers.Conv3D(n_labels, (1,1,1), activation='softmax')(decoder0)
    model = models.Model(inputs=[inputs], outputs=[outputs])
    model.summary(line_length=120)
    return model, outputs 
