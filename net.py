# -*- coding:utf-8 -*- 
__author__ = 'wingniuqichao'
 
import caffe
from caffe import layers as L, params as P, to_proto
from caffe.proto import caffe_pb2
''' 
layer {
  name: "data"
  type: "Python"
  top: "data"
  top: "label"
  python_param{
    module: 'mydatalayer'
    layer: 'DataLayer'
    }
}
''' 
frozen_weight_param = dict(lr_mult=1)#权重
frozen_bias_param = dict(lr_mult=2)#偏执值
froozen_param = [frozen_weight_param, frozen_bias_param]



def write_layer():
    filters = [16, 32, 64, 128, 192, 256]
    nClasses = 2

    net = caffe.NetSpec()

    net.data, net.label = L.HDF5Data(batch_size=16, source='train.h5list', ntop=2)
    # 第一层编码
    net.enc1_conv1 = L.Convolution(net.data,
                                param=froozen_param,  # 这里通过定义一个list，来整合到param的字典,也就是：param=[]
                                num_output=filters[0],
                                pad=1,
                                kernel_size=3,
                                stride=1,
                                weight_filler=dict(type='xavier'), bias_filler=dict(type='constant')
                                )
    net.enc1_norm1 = L.BatchNorm(net.enc1_conv1, moving_average_fraction=0.9, in_place=True, use_global_stats=False)
    net.enc1_scale1 = L.Scale(net.enc1_norm1, bias_term=True, in_place=True)
    net.enc1_relu1 = L.ReLU(net.enc1_scale1, in_place=True)
    net.enc1_conv2 = L.Convolution(net.enc1_relu1,
                                param=froozen_param,  # 这里通过定义一个list，来整合到param的字典,也就是：param=[]
                                num_output=filters[0],
                                pad=1,
                                kernel_size=3,
                                stride=1,
                                weight_filler=dict(type='xavier'), bias_filler=dict(type='constant')
                                )
    net.enc1_norm2 = L.BatchNorm(net.enc1_conv2, moving_average_fraction=0.9, in_place=True, use_global_stats=False)
    net.enc1_scale2 = L.Scale(net.enc1_norm2, bias_term=True, in_place=True)
    net.enc1_relu2=L.ReLU(net.enc1_scale2, in_place=True)
    net.enc1_pool1 = L.Pooling(net.enc1_relu2,
                          pool=caffe.params.Pooling.MAX,
                          kernel_size=2,
                          stride=2
                          )

    # 第二层编码
    net.enc2_conv1 = L.Convolution(net.enc1_pool1,
                                param=froozen_param,
                                num_output=filters[1],
                                pad=1,
                                kernel_size=3,
                                stride=1,
                                weight_filler=dict(type='xavier'), bias_filler=dict(type='constant')
                                )
    net.enc2_norm1 = L.BatchNorm(net.enc2_conv1, moving_average_fraction=0.9, in_place=True, use_global_stats=False)
    net.enc2_scale1 = L.Scale(net.enc2_norm1, bias_term=True, in_place=True)
    net.enc2_relu1 = L.ReLU(net.enc2_scale1, in_place=True)
    net.enc2_conv2 = L.Convolution(net.enc2_relu1,
                                param=froozen_param,
                                num_output=filters[1],
                                pad=1,
                                kernel_size=3,
                                stride=1,
                                weight_filler=dict(type='xavier'), bias_filler=dict(type='constant')
                                )
    net.enc2_norm2 = L.BatchNorm(net.enc2_conv2, moving_average_fraction=0.9, in_place=True, use_global_stats=False)
    net.enc2_scale2 = L.Scale(net.enc2_norm2, bias_term=True, in_place=True)
    net.enc2_relu2=L.ReLU(net.enc2_scale2, in_place=True)
    net.enc2_pool1 = L.Pooling(net.enc2_relu2,
                          pool=caffe.params.Pooling.MAX,
                          kernel_size=2,
                          stride=2
                          )

    # 第三层卷积层
    net.enc3_conv1 = L.Convolution(net.enc2_pool1,
                                param=froozen_param,
                                num_output=filters[2],
                                pad=1,
                                kernel_size=3,
                                stride=1,
                                weight_filler=dict(type='xavier'), bias_filler=dict(type='constant')
                                )
    net.enc3_norm1 = L.BatchNorm(net.enc3_conv1, moving_average_fraction=0.9, in_place=True, use_global_stats=False)
    net.enc3_scale1 = L.Scale(net.enc3_norm1, bias_term=True, in_place=True)
    net.enc3_relu1 = L.ReLU(net.enc3_scale1, in_place=True)
    net.enc3_conv2 = L.Convolution(net.enc3_relu1,
                                param=froozen_param,
                                num_output=filters[2],
                                pad=1,
                                kernel_size=3,
                                stride=1,
                                weight_filler=dict(type='xavier'), bias_filler=dict(type='constant')
                                )
    net.enc3_norm2 = L.BatchNorm(net.enc3_conv2, moving_average_fraction=0.9, in_place=True, use_global_stats=False)
    net.enc3_scale2 = L.Scale(net.enc3_norm2, bias_term=True, in_place=True)
    net.enc3_relu2=L.ReLU(net.enc3_scale2, in_place=True)
    net.enc3_pool1 = L.Pooling(net.enc3_relu2,
                          pool=caffe.params.Pooling.MAX,
                          kernel_size=2,
                          stride=2
                          )

    # 第四层卷积层
    net.enc4_conv1 = L.Convolution(net.enc3_pool1,
                                param=froozen_param,
                                num_output=filters[3],
                                pad=1,
                                kernel_size=3,
                                stride=1,
                                weight_filler=dict(type='xavier'), bias_filler=dict(type='constant')
                                )
    net.enc4_norm1 = L.BatchNorm(net.enc4_conv1, moving_average_fraction=0.9, in_place=True, use_global_stats=False)
    net.enc4_scale1 = L.Scale(net.enc4_norm1, bias_term=True, in_place=True)
    net.enc4_relu1 = L.ReLU(net.enc4_scale1, in_place=True)
    net.enc4_conv2 = L.Convolution(net.enc4_relu1,
                                param=froozen_param,
                                num_output=filters[3],
                                pad=1,
                                kernel_size=3,
                                stride=1,
                                weight_filler=dict(type='xavier'), bias_filler=dict(type='constant')
                                )
    net.enc4_norm2 = L.BatchNorm(net.enc4_conv2, moving_average_fraction=0.9, in_place=True, use_global_stats=False)
    net.enc4_scale2 = L.Scale(net.enc4_norm2, bias_term=True, in_place=True)
    net.enc4_relu2=L.ReLU(net.enc4_scale2, in_place=True)
    net.enc4_pool1 = L.Pooling(net.enc4_relu2,
                          pool=caffe.params.Pooling.MAX,
                          kernel_size=2,
                          stride=2
                          )

    # 第五层卷积层
    net.enc5_conv1 = L.Convolution(net.enc4_pool1,
                                param=froozen_param,
                                num_output=filters[4],
                                pad=1,
                                kernel_size=3,
                                stride=1,
                                weight_filler=dict(type='xavier'), bias_filler=dict(type='constant')
                                )
    net.enc5_norm1 = L.BatchNorm(net.enc5_conv1, moving_average_fraction=0.9, in_place=True, use_global_stats=False)
    net.enc5_scale1 = L.Scale(net.enc5_norm1, bias_term=True, in_place=True)
    net.enc5_relu1 = L.ReLU(net.enc5_scale1, in_place=True)
    net.enc5_conv2 = L.Convolution(net.enc5_relu1,
                                param=froozen_param,
                                num_output=filters[4],
                                pad=1,
                                kernel_size=3,
                                stride=1,
                                weight_filler=dict(type='xavier'), bias_filler=dict(type='constant')
                                )
    net.enc5_norm2 = L.BatchNorm(net.enc5_conv2, moving_average_fraction=0.9, in_place=True, use_global_stats=False)
    net.enc5_scale2 = L.Scale(net.enc5_norm2, bias_term=True, in_place=True)
    net.enc5_relu2=L.ReLU(net.enc5_scale2, in_place=True)
    net.enc5_pool1 = L.Pooling(net.enc5_relu2,
                          pool=caffe.params.Pooling.MAX,
                          kernel_size=2,
                          stride=2
                          )

    # 第六层卷积层
    net.enc6_conv1 = L.Convolution(net.enc5_pool1,
                                param=froozen_param,
                                num_output=filters[5],
                                pad=1,
                                kernel_size=3,
                                stride=1,
                                weight_filler=dict(type='xavier'), bias_filler=dict(type='constant')
                                )
    net.enc6_norm1 = L.BatchNorm(net.enc6_conv1, moving_average_fraction=0.9, in_place=True, use_global_stats=False)
    net.enc6_scale1 = L.Scale(net.enc6_norm1, bias_term=True, in_place=True)
    net.enc6_relu1 = L.ReLU(net.enc6_scale1, in_place=True)
    net.enc6_conv2 = L.Convolution(net.enc6_relu1,
                                param=froozen_param,
                                num_output=filters[5],
                                pad=1,
                                kernel_size=3,
                                stride=1,
                                weight_filler=dict(type='xavier'), bias_filler=dict(type='constant')
                                )
    net.enc6_norm2 = L.BatchNorm(net.enc6_conv2, moving_average_fraction=0.9, in_place=True, use_global_stats=False)
    net.enc6_scale2 = L.Scale(net.enc6_norm2, bias_term=True, in_place=True)
    net.enc6_relu2=L.ReLU(net.enc6_scale2, in_place=True)
    net.enc6_pool1 = L.Pooling(net.enc6_relu2,
                          pool=caffe.params.Pooling.MAX,
                          kernel_size=2,
                          stride=2
                          )

    # 中间层
    net.mid_conv1 = L.Convolution(net.enc6_pool1,
                                param=froozen_param,
                                num_output=filters[5],
                                pad=1,
                                kernel_size=3,
                                stride=1,
                                weight_filler=dict(type='xavier'), bias_filler=dict(type='constant')
                                )
    net.mid_norm1 = L.BatchNorm(net.mid_conv1, moving_average_fraction=0.9, in_place=True, use_global_stats=False)
    net.mid_scale1 = L.Scale(net.mid_norm1, bias_term=True, in_place=True)
    net.mid_relu1 = L.ReLU(net.mid_scale1, in_place=True)
    net.mid_conv2 = L.Convolution(net.mid_relu1,
                                param=froozen_param,
                                num_output=filters[5],
                                pad=1,
                                kernel_size=3,
                                stride=1,
                                weight_filler=dict(type='xavier'), bias_filler=dict(type='constant')
                                )
    net.mid_norm2 = L.BatchNorm(net.mid_conv2, moving_average_fraction=0.9, in_place=True, use_global_stats=False)
    net.mid_scale2 = L.Scale(net.mid_norm2, bias_term=True, in_place=True)
    net.mid_relu2=L.ReLU(net.mid_scale2, in_place=True)

    # 第一层解码
    net.dec1_deconv1 = L.Deconvolution(net.mid_relu2,
                                param=froozen_param,
                                convolution_param=dict(
                                    num_output=filters[5],
                                    pad=0,
                                    kernel_size=2,
                                    stride=2,
                                    weight_filler=dict(type='xavier'), bias_filler=dict(type='constant')
                                )
                            )
    net.dec1_norm0 = L.BatchNorm(net.dec1_deconv1, moving_average_fraction=0.9, in_place=True, use_global_stats=False)
    net.dec1_scale0 = L.Scale(net.dec1_norm0, bias_term=True, in_place=True)
    net.dec1_concat1 = caffe.layers.Concat(net.dec1_scale0, net.enc6_relu2, concat_param=dict(concat_dim=1))
    net.dec1_conv1 = L.Convolution(net.dec1_concat1,
                                param=froozen_param,
                                num_output=filters[5],
                                pad=1,
                                kernel_size=3,
                                stride=1,
                                weight_filler=dict(type='xavier'), bias_filler=dict(type='constant')
                                )
    net.dec1_norm1 = L.BatchNorm(net.dec1_conv1, moving_average_fraction=0.9, in_place=True, use_global_stats=False)
    net.dec1_scale1 = L.Scale(net.dec1_norm1, bias_term=True, in_place=True)
    net.dec1_relu1 = L.ReLU(net.dec1_scale1, in_place=True)
    net.dec1_conv2 = L.Convolution(net.dec1_relu1,
                                param=froozen_param,
                                num_output=filters[5],
                                pad=1,
                                kernel_size=3,
                                stride=1,
                                weight_filler=dict(type='xavier'), bias_filler=dict(type='constant')
                                )
    net.dec1_norm2 = L.BatchNorm(net.dec1_conv2, moving_average_fraction=0.9, in_place=True, use_global_stats=False)
    net.dec1_scale2 = L.Scale(net.dec1_norm2, bias_term=True, in_place=True)
    net.dec1_relu2=L.ReLU(net.dec1_scale2, in_place=True)

    # 第二层解码
    net.dec2_deconv1 = L.Deconvolution(net.dec1_relu2,
                                param=froozen_param,
                                convolution_param=dict(
                                    num_output=filters[4],
                                    pad=0,
                                    kernel_size=2,
                                    stride=2,
                                    weight_filler=dict(type='xavier'), bias_filler=dict(type='constant')
                                )
                            )
    net.dec2_norm0 = L.BatchNorm(net.dec2_deconv1, moving_average_fraction=0.9, in_place=True, use_global_stats=False)
    net.dec2_scale0 = L.Scale(net.dec2_norm0, bias_term=True, in_place=True)
    net.dec2_concat1 = caffe.layers.Concat(net.dec2_scale0, net.enc5_relu2, concat_param=dict(concat_dim=1))
    net.dec2_conv1 = L.Convolution(net.dec2_concat1,
                                param=froozen_param,
                                num_output=filters[4],
                                pad=1,
                                kernel_size=3,
                                stride=1,
                                weight_filler=dict(type='xavier'), bias_filler=dict(type='constant')
                                )
    net.dec2_norm1 = L.BatchNorm(net.dec2_conv1, moving_average_fraction=0.9, in_place=True, use_global_stats=False)
    net.dec2_scale1 = L.Scale(net.dec2_norm1, bias_term=True, in_place=True)
    net.dec2_relu1 = L.ReLU(net.dec2_scale1, in_place=True)
    net.dec2_conv2 = L.Convolution(net.dec2_relu1,
                                param=froozen_param,
                                num_output=filters[4],
                                pad=1,
                                kernel_size=3,
                                stride=1,
                                weight_filler=dict(type='xavier'), bias_filler=dict(type='constant')
                                )
    net.dec2_norm2 = L.BatchNorm(net.dec2_conv2, moving_average_fraction=0.9, in_place=True, use_global_stats=False)
    net.dec2_scale2 = L.Scale(net.dec2_norm2, bias_term=True, in_place=True)
    net.dec2_relu2=L.ReLU(net.dec2_scale2, in_place=True)

    # 第三层解码
    net.dec3_deconv1 = L.Deconvolution(net.dec2_relu2,
                                param=froozen_param,
                                convolution_param=dict(
                                    num_output=filters[3],
                                    pad=0,
                                    kernel_size=2,
                                    stride=2,
                                    weight_filler=dict(type='xavier'), bias_filler=dict(type='constant')
                                )
                            )
    net.dec3_norm0 = L.BatchNorm(net.dec3_deconv1, moving_average_fraction=0.9, in_place=True, use_global_stats=False)
    net.dec3_scale0 = L.Scale(net.dec3_norm0, bias_term=True, in_place=True)
    net.dec3_concat1 = caffe.layers.Concat(net.dec3_scale0, net.enc4_relu2, concat_param=dict(concat_dim=1))
    net.dec3_conv1 = L.Convolution(net.dec3_concat1,
                                param=froozen_param,
                                num_output=filters[3],
                                pad=1,
                                kernel_size=3,
                                stride=1,
                                weight_filler=dict(type='xavier'), bias_filler=dict(type='constant')
                                )
    net.dec3_norm1 = L.BatchNorm(net.dec3_conv1, moving_average_fraction=0.9, in_place=True, use_global_stats=False)
    net.dec3_scale1 = L.Scale(net.dec3_norm1, bias_term=True, in_place=True)
    net.dec3_relu1 = L.ReLU(net.dec3_scale1, in_place=True)
    net.dec3_conv2 = L.Convolution(net.dec3_relu1,
                                param=froozen_param,
                                num_output=filters[3],
                                pad=1,
                                kernel_size=3,
                                stride=1,
                                weight_filler=dict(type='xavier'), bias_filler=dict(type='constant')
                                )
    net.dec3_norm2 = L.BatchNorm(net.dec3_conv2, moving_average_fraction=0.9, in_place=True, use_global_stats=False)
    net.dec3_scale2 = L.Scale(net.dec3_norm2, bias_term=True, in_place=True)
    net.dec3_relu2=L.ReLU(net.dec3_scale2, in_place=True)

    # 第四层解码
    net.dec4_deconv1 = L.Deconvolution(net.dec3_relu2,
                                param=froozen_param,
                                convolution_param=dict(
                                    num_output=filters[2],
                                    pad=0,
                                    kernel_size=2,
                                    stride=2,
                                    weight_filler=dict(type='xavier'), bias_filler=dict(type='constant')
                                )
                            )
    net.dec4_norm0 = L.BatchNorm(net.dec4_deconv1, moving_average_fraction=0.9, in_place=True, use_global_stats=False)
    net.dec4_scale0 = L.Scale(net.dec4_norm0, bias_term=True, in_place=True)
    net.dec4_concat1 = caffe.layers.Concat(net.dec4_scale0, net.enc3_relu2, concat_param=dict(concat_dim=1))
    net.dec4_conv1 = L.Convolution(net.dec4_concat1,
                                param=froozen_param,
                                num_output=filters[2],
                                pad=1,
                                kernel_size=3,
                                stride=1,
                                weight_filler=dict(type='xavier'), bias_filler=dict(type='constant')
                                )
    net.dec4_norm1 = L.BatchNorm(net.dec4_conv1, moving_average_fraction=0.9, in_place=True, use_global_stats=False)
    net.dec4_scale1 = L.Scale(net.dec4_norm1, bias_term=True, in_place=True)
    net.dec4_relu1 = L.ReLU(net.dec4_scale1, in_place=True)
    net.dec4_conv2 = L.Convolution(net.dec4_relu1,
                                param=froozen_param,
                                num_output=filters[2],
                                pad=1,
                                kernel_size=3,
                                stride=1,
                                weight_filler=dict(type='xavier'), bias_filler=dict(type='constant')
                                )
    net.dec4_norm2 = L.BatchNorm(net.dec4_conv2, moving_average_fraction=0.9, in_place=True, use_global_stats=False)
    net.dec4_scale2 = L.Scale(net.dec4_norm2, bias_term=True, in_place=True)
    net.dec4_relu2=L.ReLU(net.dec4_scale2, in_place=True)

    # 第五层解码
    net.dec5_deconv1 = L.Deconvolution(net.dec4_relu2,
                                param=froozen_param,
                                convolution_param=dict(
                                    num_output=filters[1],
                                    pad=0,
                                    kernel_size=2,
                                    stride=2,
                                    weight_filler=dict(type='xavier'), bias_filler=dict(type='constant')
                                )
                            )
    net.dec5_norm0 = L.BatchNorm(net.dec5_deconv1, moving_average_fraction=0.9, in_place=True, use_global_stats=False)
    net.dec5_scale0 = L.Scale(net.dec5_norm0, bias_term=True, in_place=True)
    net.dec5_concat1 = caffe.layers.Concat(net.dec5_scale0, net.enc2_relu2, concat_param=dict(concat_dim=1))
    net.dec5_conv1 = L.Convolution(net.dec5_concat1,
                                param=froozen_param,
                                num_output=filters[1],
                                pad=1,
                                kernel_size=3,
                                stride=1,
                                weight_filler=dict(type='xavier'), bias_filler=dict(type='constant')
                                )
    net.dec5_norm1 = L.BatchNorm(net.dec5_conv1, moving_average_fraction=0.9, in_place=True, use_global_stats=False)
    net.dec5_scale1 = L.Scale(net.dec5_norm1, bias_term=True, in_place=True)
    net.dec5_relu1 = L.ReLU(net.dec5_scale1, in_place=True)
    net.dec5_conv2 = L.Convolution(net.dec5_relu1,
                                param=froozen_param,
                                num_output=filters[1],
                                pad=1,
                                kernel_size=3,
                                stride=1,
                                weight_filler=dict(type='xavier'), bias_filler=dict(type='constant')
                                )
    net.dec5_norm2 = L.BatchNorm(net.dec5_conv2, moving_average_fraction=0.9, in_place=True, 
                                batch_norm_param = dict(
                                    use_global_stats=False)
                                )
    net.dec5_scale2 = L.Scale(net.dec5_norm2, bias_term=True, in_place=True)
    net.dec5_relu2=L.ReLU(net.dec5_scale2, in_place=True)

    # 第六层解码
    net.dec6_deconv1 = L.Deconvolution(net.dec5_relu2,
                                param=froozen_param,
                                convolution_param=dict(
                                    num_output=filters[0],
                                    pad=0,
                                    kernel_size=2,
                                    stride=2,
                                    weight_filler=dict(type='xavier'), bias_filler=dict(type='constant')
                                )
                            )
    net.dec6_norm0 = L.BatchNorm(net.dec6_deconv1, moving_average_fraction=0.9, in_place=True, use_global_stats=False)
    net.dec6_scale0 = L.Scale(net.dec6_norm0, bias_term=True, in_place=True)
    net.dec6_concat1 = caffe.layers.Concat(net.dec6_scale0, net.enc1_relu2, concat_param=dict(concat_dim=1))
    net.dec6_conv1 = L.Convolution(net.dec6_concat1,
                                param=froozen_param,
                                num_output=filters[0],
                                pad=1,
                                kernel_size=3,
                                stride=1,
                                weight_filler=dict(type='xavier'), bias_filler=dict(type='constant')
                                )
    net.dec6_norm1 = L.BatchNorm(net.dec6_conv1, moving_average_fraction=0.9, in_place=True, use_global_stats=False)
    net.dec6_scale1 = L.Scale(net.dec6_norm1, bias_term=True, in_place=True)
    net.dec6_relu1 = L.ReLU(net.dec6_scale1, in_place=True)
    net.dec6_conv2 = L.Convolution(net.dec6_relu1,
                                param=froozen_param,
                                num_output=filters[0],
                                pad=1,
                                kernel_size=3,
                                stride=1,
                                weight_filler=dict(type='xavier'), bias_filler=dict(type='constant')
                                )
    net.dec6_norm2 = L.BatchNorm(net.dec6_conv2, moving_average_fraction=0.9, in_place=True, use_global_stats=False)
    net.dec6_scale2 = L.Scale(net.dec6_norm2, bias_term=True, in_place=True)
    net.dec6_relu2=L.ReLU(net.dec6_scale2, in_place=True)


    
    net.conv_out = L.Convolution(
        net.dec6_relu2,
        param=froozen_param,  # 这里通过定义一个list，来整合到param的字典,也就是：param=[]
        num_output=nClasses,
        pad=1,
        kernel_size=3,
        stride=1,
        weight_filler=dict(type='xavier'), bias_filler=dict(type='constant')
    )


    net.loss=caffe.layers.SoftmaxWithLoss(net.conv_out, net.label)
    net.accuracy=caffe.layers.Accuracy(net.conv_out, net.label)
 
    return net.to_proto()

 
with open('train.prototxt', 'w') as f:
    f.write(str(write_layer()))
