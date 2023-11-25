import os
import numpy as np 
import cv2
import requests
import sys

from PIL import Image
from io import BytesIO
from matplotlib import pyplot
import modelCore
import pdb
import tensorflow as tf
from keras.models import Model
from keras.layers import Input, ConvLSTM2D
from keras.layers import Conv2D
from keras import backend as K

manTraNet_root = '/srv/share4/apal72/LS/ManTraNet'
manTraNet_srcDir = os.path.join( manTraNet_root, 'src' )
sys.path.insert( 0, manTraNet_srcDir )
manTraNet_modelDir = os.path.join( manTraNet_root, 'pretrained_weights' )

manTraNet_dataDir = os.path.join( manTraNet_root, 'data' )
sample_file = os.path.join( manTraNet_dataDir, 'samplePairs.csv' )
assert os.path.isfile( sample_file ), "ERROR: can NOT find sample data, check `manTraNet_root`"
with open( sample_file ) as IN :
    sample_pairs = [line.strip().split(',') for line in IN.readlines() ]
L = len(sample_pairs)
print("INFO: in total, load", L, "samples")
    
def get_a_random_pair() :
    idx = np.random.randint(0,L)
    return ( os.path.join( manTraNet_dataDir, this ) for this in sample_pairs[idx] ) 


manTraNet = modelCore.load_pretrain_model_by_index( 4, manTraNet_modelDir )
# ManTraNet Architecture 
# print(manTraNet.summary(line_length=120))
# manTraNet.layers[1].summary()

# manTraNet.summary()
# input_shape = (None, None,None ,64) 
# input_layer = Input(shape=input_shape)

# cn = manTraNet.get_layer('cLSTM')
# conv_lstm_output = cn(input_layer)
# get_conv_lstm_output = Model(inputs=input_layer, outputs=conv_lstm_output) 
# input_data = np.zeros((1,4,64,64,64))
# conv_lstm_result = get_conv_lstm_output.predict(input_data)
# input_tensor = tf.convert_to_tensor(conv_lstm_result, dtype=tf.float32)
# conv2d_layer = Conv2D(1, (7, 7), padding='same', activation='sigmoid', name='pred')
# pred_out = conv2d_layer(input_tensor)

# # sigmoid_output = tf.sigmoid(input_tensor[0])
# pdb.set_trace()
# with tf.compat.v1.Session() as sess:
#     input_data = sess.run(input_tensor)
#     pred_result = sess.run(pred_out)
    # sigmoid_result = sess.run(sigmoid_output)


# print(pred_result.shape)

# devf_model = tf.keras.Model(inputs=manTraNet.input, outputs=output_devf)
# devf_output = devf_model.predict(input_tensor)
# Image Manipulation Classification Network

# input_layer = Input(shape=(None,None,3))
# # input_layer = Input(shape=img_in)
# IMCFeatex = manTraNet.get_layer('Featex')
# imc_output = IMCFeatex(input_layer)
# get_imc_output = Model(inputs=input_layer, outputs=imc_output) 
# input_data = (cv2.imread('/srv/share4/apal72/LS/ManTraNet/data/forged/I00_dgzb67i_0.jpg', 1)[...,::-1])
# input_data = input_data.reshape(1,input_data.shape[0], input_data.shape[1],input_data.shape[2])
# imc_result = get_imc_output.predict(input_data)

# input_layer_2 = Input(shape=(None,None,256))
# conv = manTraNet.get_layer('outlierTrans')
# conv_output = conv(input_layer_2)
# get_conv_output = Model(inputs=input_layer_2, outputs=conv_output) 

# conv_result =  get_conv_output.predict(imc_result)
# input_layer_3 = Input(shape=(None,None,64))
# bnorm = manTraNet.get_layer('bnorm')
# bnorm_output = bnorm(input_layer_3)
# get_bnorm_output = Model(inputs=input_layer_3, outputs=bnorm_output) 
# bnorm_result =  get_bnorm_output.predict(conv_result)

# input_layer_4 = Input(shape=(None,None,64))
# nest = manTraNet.get_layer('nestedAvgFeatex')
# nest_output = nest(input_layer_4)
# get_nest_output = Model(inputs=input_layer_4, outputs=nest_output) 
# nest_result =  get_nest_output.predict(bnorm_result)


# input_layer_5 = Input(shape=(None,None,64))
# glb = manTraNet.get_layer('glbStd')
# glb_output = glb(input_layer_5)
# get_glb_output = Model(inputs=input_layer_5, outputs=glb_output) 
# glb_result =  get_glb_output.predict(bnorm_result)


# input_layer_6 = Input(shape=(None,None,64))
# sigma5 = manTraNet.get_layer('expTime')
# sigma5_output = sigma5(input_layer_6)
# get_sigma5_output = Model(inputs=input_layer_6, outputs=sigma5_output) 
# sigma5_result =  get_sigma5_output.predict(glb_result)

# # pdb.set_trace()
# # input_layer_7 = Input(shape=(None,None,None,64))
# input_layer_7 = Input(shape=(None,None,None,64))
# input_layer_8 = Input(shape=(None,None,None,64))
# dev5d = manTraNet.get_layer('divStd')
# dev5d_output = dev5d([input_layer_7,input_layer_8])
# get_dev5d_output = Model(inputs=[input_layer_7,input_layer_8], outputs=dev5d_output) 
# # dev5d_result =  K.abs(nest_result/sigma5_result)
# dev5d_result =  get_dev5d_output.predict([nest_result,sigma5_result])


# input_shape = (None, None,None ,64) 
# input_layer_9 = Input(shape=input_shape)
# cn = manTraNet.get_layer('cLSTM')
# conv_lstm_output = cn(input_layer_9)
# get_conv_lstm_output = Model(inputs=input_layer_9, outputs=conv_lstm_output) 
# conv_lstm_result = get_conv_lstm_output.predict(dev5d_result)

# # pdb.set_trace()
# input_shape = (None,None ,8) 
# input_layer_10 = Input(shape=input_shape)
# pred = manTraNet.get_layer('pred')
# pred_output = pred(input_layer_10)
# get_pred_output = Model(inputs=input_layer_10, outputs=pred_output) 
# pred_result = get_pred_output.predict(conv_lstm_result)

# print(IMCFeatex.summary(line_length=120))

from datetime import datetime 
def read_rgb_image( image_file ) :
    rgb = cv2.imread( image_file, 1 )[...,::-1]
    return rgb
    
def decode_an_image_array( rgb, manTraNet, dn=1 ) :
    x = np.expand_dims( rgb.astype('float32')/255.*2-1, axis=0 )[:,::dn,::dn]
    t0 = datetime.now()
    y = manTraNet.predict(x)[0,...,0]
    t1 = datetime.now()
    return y, t1-t0

def decode_an_image_file( image_file, manTraNet, dn=1 ) :
    rgb = read_rgb_image( image_file )
    mask, ptime = decode_an_image_array( rgb, manTraNet, dn )
    return rgb[::dn,::dn], mask, ptime.total_seconds()

# pdb.set_trace()
for k in range(8) :
    # get a sample
    forged_file, original_file = get_a_random_pair()
    # load the original image just for reference
    ori = read_rgb_image( original_file )
    rgb = read_rgb_image(original_file)
    # input_layer = Input(shape=(None,None,3))
    # input_layer = Input(shape=img_in)
    # IMCFeatex = manTraNet.get_layer('Featex')
    # imc_output = IMCFeatex(input_layer)
    # get_imc_output = Model(inputs=input_layer, outputs=imc_output) 
    # # input_data = (cv2.imread('/srv/share4/apal72/LS/ManTraNet/data/forged/I00_dgzb67i_0.jpg', 1)[...,::-1])
    # input_data = rgb.reshape(1,rgb.shape[0], rgb.shape[1],rgb.shape[2])
    # imc_result = get_imc_output.predict(input_data)

    # input_layer_2 = Input(shape=(None,None,256))
    # conv = manTraNet.get_layer('outlierTrans')
    # conv_output = conv(input_layer_2)
    # get_conv_output = Model(inputs=input_layer_2, outputs=conv_output) 

    # conv_result =  get_conv_output.predict(imc_result)
    # input_layer_3 = Input(shape=(None,None,64))
    # bnorm = manTraNet.get_layer('bnorm')
    # bnorm_output = bnorm(input_layer_3)
    # get_bnorm_output = Model(inputs=input_layer_3, outputs=bnorm_output) 
    # bnorm_result =  get_bnorm_output.predict(conv_result)

    # input_layer_4 = Input(shape=(None,None,64))
    # nest = manTraNet.get_layer('nestedAvgFeatex')
    # nest_output = nest(input_layer_4)
    # get_nest_output = Model(inputs=input_layer_4, outputs=nest_output) 
    # nest_result =  get_nest_output.predict(bnorm_result)


    # input_layer_5 = Input(shape=(None,None,64))
    # glb = manTraNet.get_layer('glbStd')
    # glb_output = glb(input_layer_5)
    # get_glb_output = Model(inputs=input_layer_5, outputs=glb_output) 
    # glb_result =  get_glb_output.predict(bnorm_result)


    # input_layer_6 = Input(shape=(None,None,64))
    # sigma5 = manTraNet.get_layer('expTime')
    # sigma5_output = sigma5(input_layer_6)
    # get_sigma5_output = Model(inputs=input_layer_6, outputs=sigma5_output) 
    # sigma5_result =  get_sigma5_output.predict(glb_result)

    # # pdb.set_trace()
    # # input_layer_7 = Input(shape=(None,None,None,64))
    # input_layer_7 = Input(shape=(None,None,None,64))
    # input_layer_8 = Input(shape=(None,None,None,64))
    # dev5d = manTraNet.get_layer('divStd')
    # dev5d_output = dev5d([input_layer_7,input_layer_8])
    # get_dev5d_output = Model(inputs=[input_layer_7,input_layer_8], outputs=dev5d_output) 
    # # dev5d_result =  K.abs(nest_result/sigma5_result)
    # dev5d_result =  get_dev5d_output.predict([nest_result,sigma5_result])


    # input_shape = (None, None,None ,64) 
    # input_layer_9 = Input(shape=input_shape)
    # cn = manTraNet.get_layer('cLSTM')
    # conv_lstm_output = cn(input_layer_9)
    # get_conv_lstm_output = Model(inputs=input_layer_9, outputs=conv_lstm_output) 
    # conv_lstm_result = get_conv_lstm_output.predict(dev5d_result)

    # pdb.set_trace()
    # input_shape = (None,None ,8) 
    # input_layer_10 = Input(shape=input_shape)
    # pred = manTraNet.get_layer('pred')
    # pred_output = pred(input_layer_10)
    # get_pred_output = Model(inputs=input_layer_10, outputs=pred_output) 
    # pred_result = get_pred_output.predict(conv_lstm_result)
    # mask = pred_result[0]
    # manipulation detection using ManTraNet
    rgb, mask, ptime = decode_an_image_file( original_file, manTraNet )
    # cv2.imwrite('ori_rgb.png',ori)
    # cv2.imwrite('forged_rgb.png',rgb)
    # cv2.imwrite('mask.png',mask)
    # show results
    fig = pyplot.figure( figsize=(15,5) )
    pyplot.subplot(131)
    pyplot.imshow( ori )
    pyplot.title('Original Image')
    pyplot.subplot(132)
    pyplot.imshow( rgb )
    pyplot.title('Forged Image (ManTra-Net Input)')
    pyplot.subplot(133)
    pyplot.imshow( mask, cmap='gray' )
    pyplot.title('Predicted Mask (ManTra-Net Output)')
    # pyplot.suptitle('Decoded {} of size {} for {:.2f} seconds'.format( os.path.basename( forged_file ), rgb.shape ) )
    fig.savefig(str(k)+'.png', bbox_inches='tight')
    pyplot.close()