#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 29 02:18:54 2022

@author: Xiao
pix2pix model
"""
import sys
# if local
#sys.path.insert(1,'/home/xiao/Inverse_MS_Design/Python/Common_lib/')
# if HPC
sys.path.insert(1,'/home/xshang93/projects/def-yuzou/xshang93/Inverse_MS_Design/Python/Common_lib/')
import tensorflow as tf

import os
import time
import datetime
import numpy as np

from image_vis_3D import plot_3D_res
from joblib import load
from data_preprocessing import train_dataset_gen_cGAN, DeNormalizeData

    
work_dir = './'
data_dirs = [
        "/home/xshang93/projects/def-yuzou/xshang93/ML/dataset/10_cGAN/preprocessed/",
        "/home/xshang93/projects/def-yuzou/xshang93/ML/dataset/15_cGAN/preprocessed/",
        "/home/xshang93/projects/def-yuzou/xshang93/ML/dataset/20_cGAN/preprocessed/",
        "/home/xshang93/projects/def-yuzou/xshang93/ML/dataset/25_cGAN/preprocessed/",
        "/home/xshang93/projects/def-yuzou/xshang93/ML/dataset/30_cGAN/preprocessed/",
        "/home/xshang93/projects/def-yuzou/xshang93/ML/dataset/35_cGAN/preprocessed/",
        "/home/xshang93/projects/def-yuzou/xshang93/ML/dataset/40_cGAN/preprocessed/",
        "/home/xshang93/projects/def-yuzou/xshang93/ML/dataset/45_cGAN/preprocessed/",
        "/home/xshang93/projects/def-yuzou/xshang93/ML/dataset/50_cGAN/preprocessed/",
             ]
OUTPUT_CHANNELS = 1 #stress
save_name = 'no_norm_20220819'
train_ratio = 0.8
pre_trained_model = load(work_dir+'HP_resultSGD_alldata_200eps.pkl').best_estimator_.model_


# Functions to build the generator

def downsample(filters, transfer_layer='', transfer=True, apply_batchnorm=True):
  initializer = tf.random_normal_initializer(0., 0.02)

  result = tf.keras.Sequential()

  result.add(
      tf.keras.layers.Conv3D(filters, kernel_size=(3, 3, 3), strides=1, padding='valid',
                             kernel_initializer=initializer, use_bias=False))
  # if transfer learning is activated
  if transfer:
      result.trainable = False
      result.set_weights =pre_trained_model.layers[transfer_layer].get_weights()
  if apply_batchnorm:
    result.add(tf.keras.layers.BatchNormalization())
    if transfer:
        result.trainable = False
        result.set_weights =pre_trained_model.layers[transfer_layer+1].get_weights()

  result.add(tf.keras.layers.LeakyReLU())

  return result

def upsample(filters, apply_dropout=False):
  initializer = tf.random_normal_initializer(0., 0.02)

  result = tf.keras.Sequential()
  result.add(
    tf.keras.layers.Conv3DTranspose(filters, kernel_size=(3, 3, 3), strides=1,
                                    padding='valid',
                                    kernel_initializer=initializer,
                                    use_bias=False))

  result.add(tf.keras.layers.BatchNormalization())

  if apply_dropout:
      result.add(tf.keras.layers.Dropout(0.5))

  result.add(tf.keras.layers.ReLU())

  return result

def Generator():
  inputs = tf.keras.layers.Input(shape=[32, 32, 32, 4])#instantiate a Keras Tensor

  down_stack = [
    downsample(32, transfer_layer = 1, transfer = True),  # (batch_size, 30, 30, 30, 32)
    downsample(32, transfer_layer = 3,transfer = True),  # (batch_size, 28, 28, 28, 32)
    downsample(64, transfer_layer = 5,transfer = True),  # (batch_size, 26, 26, 26, 64)
    downsample(128, transfer_layer = 7,transfer = True),  # (batch_size, 24, 24, 24, 128)
  ]#put everything in a list and then glue them together

  up_stack = [
#    upsample(512, 4, apply_dropout=True),  # (batch_size, 2, 2, 1024)
#    upsample(512, 4, apply_dropout=True),  # (batch_size, 4, 4, 1024)
#    upsample(512, 4, apply_dropout=True),  # (batch_size, 8, 8, 1024)
    upsample(128),  # (batch_size, 26, 26, 26, 128)
    upsample(64),  # (batch_size, 28, 28, 28, 64)
    upsample(32),  # (batch_size, 30, 30, 30, 32)
    upsample(32),  # (batch_size, 32, 32, 32, 32)
  ]

  initializer = tf.random_normal_initializer(0., 0.02)
  last = tf.keras.layers.Conv3DTranspose(OUTPUT_CHANNELS, kernel_size=(3, 3, 3),
                                         strides=1,
                                         padding='valid',
                                         kernel_initializer=initializer,
                                         activation='tanh')  # (batch_size, 32, 32, 32, 4)

  x = inputs

  # Downsampling through the model
  skips = []
  for down in down_stack:
    x = down(x)
    skips.append(x)

  skips = reversed(skips[:-1])

  # Upsampling and establishing the skip connections
  for up, skip in zip(up_stack, skips):
    x = up(x)
    x = tf.keras.layers.Concatenate()([x, skip])

  x = last(x)

  return tf.keras.Model(inputs=inputs, outputs=x)

# Define the generator loss

LAMBDA = 100

loss_object = tf.keras.losses.BinaryCrossentropy(from_logits=True)

def generator_loss(disc_generated_output, gen_output, target):
  gan_loss = loss_object(tf.ones_like(disc_generated_output), disc_generated_output)

  # Mean absolute error
  l1_loss = tf.reduce_mean(tf.abs(target - gen_output))

  total_gen_loss = gan_loss + (LAMBDA * l1_loss)

  return total_gen_loss, gan_loss, l1_loss

# Functions to build the discriminator
def Discriminator():
  initializer = tf.random_normal_initializer(0., 0.02)

  inp = tf.keras.layers.Input(shape=[32, 32, 32, 4], name='input_image')
  tar = tf.keras.layers.Input(shape=[32, 32, 32, 1], name='target_image')

  x = tf.keras.layers.concatenate([inp, tar])  # (batch_size, 256, 256, channels*2)

  down1 = downsample(32, transfer = False, apply_batchnorm = False)(x)  # (batch_size, 32, 32, 32, 32)
  down2 = downsample(32,transfer = False)(down1)  # (batch_size, 30, 30, 30, 32)
  down3 = downsample(64,transfer = False)(down2)  # (batch_size, 28, 28, 28, 64)

  zero_pad1 = tf.keras.layers.ZeroPadding3D()(down3)  # (batch_size, 26, 26, 26, 128)
  conv = tf.keras.layers.Conv3D(128, kernel_size = (3, 3, 3), strides=1,
                                kernel_initializer=initializer,
                                use_bias=False)(zero_pad1)  # (batch_size, 31, 31, 512)

  batchnorm1 = tf.keras.layers.BatchNormalization()(conv)

  leaky_relu = tf.keras.layers.LeakyReLU()(batchnorm1)

  zero_pad2 = tf.keras.layers.ZeroPadding3D()(leaky_relu)  # (batch_size, 33, 33, 512)

  last = tf.keras.layers.Conv3D(1, kernel_size = (3, 3, 3), strides=1,
                                kernel_initializer=initializer)(zero_pad2)  # (batch_size, 30, 30, 1)

  return tf.keras.Model(inputs=[inp, tar], outputs=last)

# Define the discriminator loss
def discriminator_loss(disc_real_output, disc_generated_output):
  real_loss = loss_object(tf.ones_like(disc_real_output), disc_real_output)

  generated_loss = loss_object(tf.zeros_like(disc_generated_output), disc_generated_output)

  total_disc_loss = real_loss + generated_loss

  return total_disc_loss

# Generate images for checking
def generate_images(model, test_input, tar):
  prediction = model(test_input, training=True)
  prediction = DeNormalizeData(prediction,data_min,data_max)
  plot_3D_res(prediction,0,save_dir = './pred')
  plot_3D_res(tar,0,save_dir = './true')
#  
#  
# Functions for training

log_dir="logs/"

summary_writer = tf.summary.create_file_writer(
  log_dir + "fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))

@tf.function
def train_step(input_image, target, step):
  with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
    gen_output = generator(input_image, training=True)

    disc_real_output = discriminator([input_image, target], training=True)
    disc_generated_output = discriminator([input_image, gen_output], training=True)

    gen_total_loss, gen_gan_loss, gen_l1_loss = generator_loss(disc_generated_output, gen_output, target)
    disc_loss = discriminator_loss(disc_real_output, disc_generated_output)

  generator_gradients = gen_tape.gradient(gen_total_loss,
                                          generator.trainable_variables)
  discriminator_gradients = disc_tape.gradient(disc_loss,
                                               discriminator.trainable_variables)

  generator_optimizer.apply_gradients(zip(generator_gradients,
                                          generator.trainable_variables))
  discriminator_optimizer.apply_gradients(zip(discriminator_gradients,
                                              discriminator.trainable_variables))

  with summary_writer.as_default():
    tf.summary.scalar('gen_total_loss', gen_total_loss, step=step//1000)
    tf.summary.scalar('gen_gan_loss', gen_gan_loss, step=step//1000)
    tf.summary.scalar('gen_l1_loss', gen_l1_loss, step=step//1000)
    tf.summary.scalar('disc_loss', disc_loss, step=step//1000)

def fit(train_ds, steps):
#  example_input, example_target = next(iter(test_ds.take(1)))
  start = time.time()

  for step, (input_image, target) in train_ds.repeat().take(steps).enumerate():
    if (step) % 1000 == 0:

      if step != 0:
        print(f'Time taken for 1000 steps: {time.time()-start:.2f} sec\n')

      start = time.time()

#      generate_images(generator, example_input, example_target)
      print(f"Step: {step//1000}k")

    train_step(input_image, target, step)

    # Training step
    if (step+1) % 50 == 0:
      print('.', end='', flush=True)
      
    # Save (checkpoint) the model every 20000 steps (u.e. about 4 epochs)
    if (step + 1) % 20000 == 0:
        checkpoint.save(file_prefix=checkpoint_prefix)
      
# ---------------------------- Body -------------------------------------- #
if __name__ == "__main__":
    # The batch size of 1 produced better results for the U-Net in the original pix2pix experiment
    BUFFER_SIZE = 100000;BATCH_SIZE = 1
    
    # Build input pipelines
    train_input,train_target,test_input,test_target,data_min,data_max = train_dataset_gen_cGAN(save_name,data_dirs,train_ratio,1,5)
    np.save('min_max_'+save_name,[data_min,data_max])
    
    train_dataset = tf.data.Dataset.from_tensor_slices((train_input,train_target))
    train_dataset = train_dataset.shuffle(BUFFER_SIZE)#buffer size should >= dataset size
    train_dataset = train_dataset.batch(BATCH_SIZE)
    
    test_dataset = tf.data.Dataset.from_tensor_slices((test_input,test_target))
    test_dataset = test_dataset.batch(BATCH_SIZE)
    
    # Build the generator
    generator = Generator()
#    tf.keras.utils.plot_model(generator, show_shapes=True, dpi=64)
    #
    ## Test the generator
    #gen_output = generator(tf.reshape(test_input[0],shape=[-1,32,32,32,4]), training=False)#training is false set it into inference mode, where dropout is disabled
    #plot_3D_res(gen_output,0)
    ##
    # Build the discriminator
    discriminator = Discriminator()
    
    ## Test the discriminator
    #disc_out = discriminator([tf.reshape(test_input[0],shape=[-1,32,32,32,4]), gen_output], training=False)#tf.newaxis used to add an aixs to match gen_output
    ##
    # Define the optimizers and checkpoint savers
    generator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
    discriminator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
    
    checkpoint_dir = './training_checkpoints'
    checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
    checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer,
                                     discriminator_optimizer=discriminator_optimizer,
                                     generator=generator,
                                     discriminator=discriminator)
    
    # Train the model
    fit(train_dataset, steps=100000)
    
    # save the model
    generator.save(save_name+'_Generator.h5')
    #tf.data.experimental.save(test_dataset, './')
