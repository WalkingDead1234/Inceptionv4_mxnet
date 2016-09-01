#coding=utf-8
__author__ = 'zhangshuai'
modified_date = '16/7/11'

import find_mxnet
import mxnet as mx
import mxnet

def Conv(data, num_filter, kernel=(1, 1), stride=(1, 1), pad=(0, 0), act_type='relu', name=None, suffix='',bn_momentum=0.9,with_relu=False):
    conv = mx.symbol.Convolution(data=data, num_filter=num_filter, kernel=kernel, stride=stride, pad=pad, no_bias=True, name='%s%s_conv2d' %(name, suffix))
    bn = mx.symbol.BatchNorm(data=conv, name='%s%s_batchnorm' %(name, suffix),fix_gamma=True,momentum=bn_momentum, eps=2e-5)
                             # fix_gamma=False, momentum=bn_momentum, eps=2e-5)
    act = mx.symbol.Activation(data=bn, act_type=act_type, name='%s%s_%s' %(name, suffix, act_type))

    return (act if with_relu else bn)
    # return act

def Inception_stem(input, name=None):
    c = Conv(input, 32, kernel=(3, 3), stride=(2, 2), name='%s_conv1_3*3' % name)
    c = Conv(c, 32, kernel=(3, 3), name='%s_conv2_3*3' % name)
    c = Conv(c, 64, kernel=(3, 3), pad=(1, 1), name='%s_conv3_3*3' % name)

    p1 = mx.symbol.Pooling(c, kernel=(3, 3), stride=(2, 2), pool_type='max', name='%s_maxpool_1' % name)
    c2 = Conv(c, 96, kernel=(3, 3), stride=(2, 2), name='%s_conv4_3*3' % name)
    concat = mx.symbol.Concat(*[p1, c2], name='%s_concat_1' % name)

    c1 = Conv(concat, 64, kernel=(1, 1), pad=(0, 0), name='%s_conv5_1*1' % name)
    c1 = Conv(c1, 96, kernel=(3, 3), name='%s_conv6_3*3' % name)

    c2 = Conv(concat, 64, kernel=(1, 1), pad=(0, 0), name='%s_conv7_1*1' % name)
    c2 = Conv(c2, 64, kernel=(7, 1), pad=(3, 0), name='%s_conv8_7*1' % name)
    c2 = Conv(c2, 64, kernel=(1, 7), pad=(0, 3), name='%s_conv9_1*7' % name)
    c2 = Conv(c2, 96, kernel=(3, 3), pad=(0, 0), name='%s_conv10_3*3' % name)

    concat = mx.symbol.Concat(*[c1, c2], name='%s_concat_2' % name)

    c3 = Conv(concat, 192, kernel=(3, 3), stride=(2, 2), name='%s_conv11_3*3' % name)   #fix by paper
    p3 = mx.symbol.Pooling(concat, kernel=(3, 3), stride=(2, 2), pool_type='max', name='%s_maxpool_2' % name)

    concat = mx.symbol.Concat(*[c3, p3], name='%s_concat_3' % name)

    return concat


def Inception_resnet_A(input, name=None, scale_residual=True):
    identity_input = input
    c1 = Conv(input, 32, kernel=(1, 1), pad=(0, 0), name='%s_conv1_1*1' %name)

    c2 = Conv(input, 32, kernel=(1, 1), pad=(0, 0), name='%s_conv2_1*1' %name)
    c2 = Conv(c2, 32, kernel=(3, 3), pad=(1, 1), name='%s_conv3_3*3' %name)

    c3 = Conv(input, 32, kernel=(1, 1), pad=(0, 0), name='%s_conv4_1*1' %name)
    c3 = Conv(c3, 48, kernel=(3, 3), pad=(1, 1), name='%s_conv5_3*3' %name)
    c3 = Conv(c3, 64, kernel=(3, 3), pad=(1, 1), name='%s_conv6_3*3' %name)

    concat = mx.symbol.Concat(*[c1, c2, c3], name='%s_concat_1' %name)

    if scale_residual:
        scal_res = Conv(concat, 384, kernel=(1, 1), pad=(0, 0), act_type='relu', name='%s_conv5_1*1' %name)  #256
    #     concat = 0.2*scal_res
    # new_data = identity_input + concat
    new_data = identity_input + scal_res
    act = mx.symbol.Activation(new_data, act_type='relu', name='%s_act_1' %name)

    return new_data


def Reduction_A(input, name=None):
    p1 = mx.symbol.Pooling(input, kernel=(3, 3), stride=(2, 2), pool_type='max', name='%s_maxpool_1' % name)

    c2 = Conv(input, 384, kernel=(3, 3), stride=(2, 2), name='%s_conv1_3*3' % name)

    c3 = Conv(input, 256, kernel=(1, 1), pad=(0, 0), name='%s_conv2_1*1' % name)
    c3 = Conv(c3, 256, kernel=(3, 3), pad=(1, 1), name='%s_conv3_3*3' % name)
    c3 = Conv(c3, 384, kernel=(3, 3), stride=(2, 2), pad=(0, 0), name='%s_conv4_3*3' % name)

    concat = mx.symbol.Concat(*[p1, c2, c3], name='%s_concat_1' % name)

    return concat


def Inception_resnet_B(input, name=None, scale_residual=True):
    identity_input = input

    c1 = Conv(input, 192, kernel=(1, 1), pad=(0, 0), name='%s_conv1_1*1' %name)

    c2 = Conv(input, 128, kernel=(1, 1), pad=(0, 0), name='%s_conv2_1*1' %name)
    c2 = Conv(c2, 160, kernel=(1, 7), pad=(0, 3), name='%s_conv3_1*7' %name)
    c2 = Conv(c2, 192, kernel=(7, 1), pad=(3, 0), name='%s_conv4_7*1' %name)

    concat = mx.symbol.Concat(*[c1, c2], name='%s_concat_1' %name)

    if scale_residual:
        scal_res = Conv(concat, 1152, kernel=(1, 1), pad=(0, 0), act_type='relu', name='%s_conv5_1*1' %name)   #1154
    #     concat = 0.2*scal_res  #why
    #
    # new_data = identity_input + scal_res
    new_data = identity_input + scal_res
    act = mx.symbol.Activation(new_data, act_type='relu', name='%s_act_1' % name)

    return new_data


def Reduction_B(input, name=None):
    p1 = mx.symbol.Pooling(input, kernel=(3, 3), stride=(2, 2), pool_type='max', name='%s_maxpool_1' %name)

    c2 = Conv(input, 256, kernel=(1, 1), pad=(0, 0), name='%s_conv1_1*1' %name)
    c2 = Conv(c2, 384, kernel=(3, 3), stride=(2, 2), name='%s_conv2_3*3' %name)

    c3 = Conv(input, 256, kernel=(1, 1), pad=(0, 0), name='%s_conv3_1*1' %name)
    c3 = Conv(c3, 288, kernel=(3, 3), stride=(2, 2), name='%s_conv4_3*3' %name)

    c4 = Conv(input, 256, kernel=(1, 1), pad=(0, 0), name='%s_conv5_1*1' %name)
    c4 = Conv(c4, 288, kernel=(3, 3), pad=(1, 1), name='%s_conv6_3*3' %name)
    c4 = Conv(c4, 320, kernel=(3, 3), stride=(2, 2), name='%s_conv7_3*3' %name)

    concat = mx.symbol.Concat(*[p1, c2, c3, c4], name='%s_concat_1' %name)

    return concat


def Inception_resnet_C(input, name=None, scale_residual=True):
    identity_input = input

    c1 = Conv(input, 192, kernel=(1, 1), pad=(0, 0), name='%s_conv1_1*1' %name)

    c2 = Conv(input, 192, kernel=(1, 1), pad=(0, 0), name='%s_conv2_1*1' %name)
    c2 = Conv(c2, 224, kernel=(1, 3), pad=(0, 1), name='%s_conv3_1*3' %name)
    c2 = Conv(c2, 256, kernel=(3, 1), pad=(1, 0), name='%s_conv4_3*1' %name)

    concat = mx.symbol.Concat(*[c1, c2], name='%s_concat_1' %name)

    if scale_residual:
        scal_res = Conv(concat, 2144, kernel=(1, 1), pad=(0, 0), act_type='relu', name='%s_conv5_1*1' %name) # 2048
        concat = 0.2*scal_res
    # new_data = identity_input + concat
    new_data = identity_input + scal_res
    # new_data = scal_res + identity_input
    act = mx.symbol.Activation(new_data, act_type='relu', name='%s_act_1' %name)


    return new_data

def get_symbol(num_classes=1000):
    data = mx.symbol.Variable("data")
    x = Inception_stem(data, name='in_stem')

    #5 * Inception-resnet-A
    # x = Inception_resnet_A(x, name='in_res_1a', scale_residual=True)
    # x = Inception_resnet_A(x, name='in_res_2a', scale_residual=True)
    # x = Inception_resnet_A(x, name='in_res_3a', scale_residual=True)
    # x = Inception_resnet_A(x, name='in_res_4a', scale_residual=True)
    # x = Inception_resnet_A(x, name='in_res_5a', scale_residual=True)

    for i in range(5):
        x = Inception_resnet_A(x, name='in_res_%da' %(i+1), scale_residual=True)

    #Reduction-A
    x = Reduction_A(x, name='re_1a')

    #10 * Inception-resnet-B
    # x = Inception_resnet_B(x, name='in_res_1b', scale_residual=True)
    # x = Inception_resnet_B(x, name='in_res_2b', scale_residual=True)
    # x = Inception_resnet_B(x, name='in_res_3b', scale_residual=True)
    # x = Inception_resnet_B(x, name='in_res_4b', scale_residual=True)
    # x = Inception_resnet_B(x, name='in_res_5b', scale_residual=True)


    for i in range(10):
        x = Inception_resnet_B(x, name='in_res_%db' %(i+1), scale_residual=True)

    #Reduction-B
    x = Reduction_B(x, name='re_1b')

    #5 * Inception-resnet-C

    for i in range(5):
        x = Inception_resnet_C(x, name='in_res_%dc' %(i+1), scale_residual=True)

    # Average Pooling
    x = mx.symbol.Pooling(x, kernel=(8, 8), pad=(1, 1), pool_type='avg', name='global_avgpool')

    # Dropout
    x = mx.symbol.Dropout(x, p=0.2)

    flatten = mx.symbol.Flatten(x, name='flatten')
    fc1 = mx.symbol.FullyConnected(flatten, num_hidden=num_classes, name='fc1')
    softmax = mx.symbol.SoftmaxOutput(fc1, name='softmax')

    return softmax



