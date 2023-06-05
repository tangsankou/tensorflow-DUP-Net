""" PointNet++ Layers

Author: Charles R. Qi
Date: November 2017
"""

import os
import sys
from tf_ops.sampling.tf_sampling import farthest_point_sample, gather_point
from tf_ops.grouping.tf_grouping import query_ball_point, group_point, knn_point
from tf_ops.interpolation.tf_interpolate import three_nn, three_interpolate
import tensorflow as tf
import numpy as np
from utils import tf_util

def sample_and_group(npoint, radius, nsample, xyz, points, tnet_spec=None, knn=False, use_xyz=True):
    '''
    Input:
        npoint: int32
        radius: float32
        nsample: int32
        xyz: (batch_size, ndataset, 3) TF tensor
        points: (batch_size, ndataset, channel) TF tensor, if None will just use xyz as points
        tnet_spec: dict (keys: mlp, mlp2, is_training, bn_decay), if None do not apply tnet
        knn: bool, if True use kNN instead of radius search
        use_xyz: bool, if True concat XYZ with local point features, otherwise just use point features
    Output:
        new_xyz: (batch_size, npoint, 3) TF tensor
        new_points: (batch_size, npoint, nsample, 3+channel) TF tensor
        idx: (batch_size, npoint, nsample) TF tensor, indices of local points as in ndataset points
        grouped_xyz: (batch_size, npoint, nsample, 3) TF tensor, normalized point XYZs
            (subtracted by seed point XYZ) in local regions
    '''

    '''
    根据fps算法从输入xyz中选取npoint个点，返回他们的index
    然后在由gather_point采样出对应的点，返回子集点云new_xyz  (batch_size, npoint, 3)。
    '''
    new_xyz = gather_point(xyz, farthest_point_sample(npoint, xyz)) # (batch_size, npoint, 3)
    
    '''
    然后为new_xyz中的每个点在xyz中找到他的local region neighbor。
    根据grouping算法不同,对于new_xyz中的每个点,knn会返回metric space上最近的nsample个点。
    query_ball_point则会根据nsamples和radius两方面的限制来提取pts_cnt个点(不一定能达到nsample)。
    idx是一个(batch_size, npoint, nsample)的变量,nsample是int32的array, indices to input points in xyz。
    '''

    if knn:
        _,idx = knn_point(nsample, xyz, new_xyz)
    else:
        if np.isscalar(radius):
            idx, pts_cnt = query_ball_point(radius, nsample, xyz, new_xyz)
        else:
            idx_list = []
            for radius_one, xyz_one, new_xyz_one in zip(tf.unstack(radius,axis=0), tf.unstack(xyz, axis=0),tf.unstack(new_xyz, axis=0)):
                idx_one, _ = query_ball_point(radius_one, nsample, tf.expand_dims(xyz_one, axis=0), tf.expand_dims(new_xyz_one, axis=0))
                idx_list.append(idx_one)
            idx = tf.stack(idx_list, axis=0)
            idx = tf.squeeze(idx, axis=1)

    '''
    group_point是根据idx返回的序号将点云local_region的点云组织成有效数据结构(batch_size, npoint, nsample, 3)。
    '''

    grouped_xyz = group_point(xyz, idx) # (batch_size, npoint, nsample, 3)
    grouped_xyz -= tf.tile(tf.expand_dims(new_xyz, 2), [1,1,nsample,1]) # translation normalization
    
    '''
    points: (batch_size, ndataset, channel) 是考虑范围内所有点的feature channel是feature的维度
    grouped_points: (batch_size, npoint, nsample, channel) 是根据idx把特征提取出来
    use_xyz==True的时候,将坐标和特征拼接输出(batch_size, npoint, nsample, 3+channel)
    否则，只输出特征，不输出坐标。
    '''

    if tnet_spec is not None:
        grouped_xyz = tnet(grouped_xyz, tnet_spec)
    if points is not None:
        grouped_points = group_point(points, idx) # (batch_size, npoint, nsample, channel)
        if use_xyz:
            # new_points = tf.concat([grouped_xyz, tf.tile(tf.expand_dims(new_xyz, 2), [1,1,nsample,1]),grouped_points], axis=-1) # (batch_size, npoint, nample, 3+channel)
            new_points = tf.concat([grouped_xyz, grouped_points],axis=-1)  # (batch_size, npoint, nample, 3+channel)
        else:
            new_points = grouped_points
    else:
        # new_points =  tf.concat([grouped_xyz, tf.tile(tf.expand_dims(new_xyz, 2), [1,1,nsample,1])], axis=-1)
        new_points = grouped_xyz

    '''
    new_xyz是新生成的子集点云的中心点集合
    new_points可能是所有参与运算的点的三维坐标也有可能是所有参与运算点的特征,也有可能是坐标+特征
    idx是所有参与运算的点在xyz中的序号
    grouped_xyz:(batch_size, npoint, nsample, 3)是所有参与运算的点的三维坐标
    '''

    return new_xyz, new_points, idx, grouped_xyz


#Equivalent to sample_and_group with npoint=1, radius=inf, use (0,0,0) as the centroid
def sample_and_group_all(xyz, points, use_xyz=True):
    '''
    Inputs:
        xyz: (batch_size, ndataset, 3) TF tensor
        points: (batch_size, ndataset, channel) TF tensor, if None will just use xyz as points
        use_xyz: bool, if True concat XYZ with local point features, otherwise just use point features
    Outputs:
        new_xyz: (batch_size, 1, 3) as (0,0,0)
        new_points: (batch_size, 1, ndataset, 3+channel) TF tensor
    Note:
        Equivalent to sample_and_group with npoint=1, radius=inf, use (0,0,0) as the centroid
    '''
    batch_size = xyz.get_shape()[0].value
    nsample = xyz.get_shape()[1].value
    new_xyz = tf.constant(np.tile(np.array([0,0,0]).reshape((1,1,3)), (batch_size,1,1)),dtype=tf.float32) # (batch_size, 1, 3)
    idx = tf.constant(np.tile(np.array(range(nsample)).reshape((1,1,nsample)), (batch_size,1,1)))
    grouped_xyz = tf.reshape(xyz, (batch_size, 1, nsample, 3)) # (batch_size, npoint=1, nsample, 3)
    if points is not None:
        if use_xyz:
            new_points = tf.concat([xyz, points], axis=2) # (batch_size, 16, 259)
        else:
            new_points = points
        new_points = tf.expand_dims(new_points, 1) # (batch_size, 1, 16, 259)
    else:
        new_points = grouped_xyz
    return new_xyz, new_points, idx, grouped_xyz


def pointnet_sa_module(xyz, points, npoint, radius, nsample, mlp, mlp2, group_all, is_training,
                       bn_decay, scope, bn=True, ibn=False, pooling='max', tnet_spec=None, knn=False, use_xyz=True):
    ''' PointNet Set Abstraction (SA) Module
        Input:
            xyz: (batch_size, ndataset, 3) TF tensor
            points: (batch_size, ndataset, channel) TF tensor
            npoint: int32 -- #points sampled in farthest point sampling
            radius: float32 -- search radius in local region
            batch_radius: the size of each object
            nsample: int32 -- how many points in each local region
            mlp: list of int32 -- output size for MLP on each point
            mlp2: list of int32 -- output size for MLP on each region
            group_all: bool -- group all points into one PC if set true, OVERRIDE
                npoint, radius and nsample settings
            use_xyz: bool, if True concat XYZ with local point features, otherwise just use point features
        Return:
            new_xyz: (batch_size, npoint, 3) TF tensor
            new_points: (batch_size, npoint, mlp[-1] or mlp2[-1]) TF tensor
            idx: (batch_size, npoint, nsample) int32 -- indices for local regions
    '''
    with tf.compat.v1.variable_scope(scope) as sc:
        if group_all:
            nsample = xyz.get_shape()[1].value
            new_xyz, new_points, idx, grouped_xyz = sample_and_group_all(xyz, points, use_xyz)
        else:
            new_xyz, new_points, idx, grouped_xyz = sample_and_group(npoint, radius, nsample, xyz, points, tnet_spec, knn, use_xyz)
        #Point Feature Embedding
        if mlp2 is None: mlp2 = []
        for i, num_out_channel in enumerate(mlp):
            new_points = tf_util.conv2d_ibn(new_points, num_out_channel, [1,1],
                                            padding='VALID', stride=[1,1],
                                            bn=bn, ibn=ibn, is_training=is_training,
                                            scope='conv%d'%(i), bn_decay=bn_decay) 
        #Pooling in Local Regions池化方法
        if pooling=='avg':
            new_points = tf.compat.v1.layers.average_pooling2d(new_points, [1,nsample], [1,1], padding='VALID', name='avgpool1')
        elif pooling=='weighted_avg':
            with tf.compat.v1.variable_scope('weighted_avg1'):
                dists = tf.norm(tensor=grouped_xyz,axis=-1,ord=2,keepdims=True)
                exp_dists = tf.exp(-dists * 5)
                weights = exp_dists/tf.reduce_sum(input_tensor=exp_dists,axis=2,keepdims=True) # (batch_size, npoint, nsample, 1)
                new_points *= weights # (batch_size, npoint, nsample, mlp[-1])
                new_points = tf.reduce_sum(input_tensor=new_points, axis=2, keepdims=True)
        elif pooling=='max':
            new_points = tf.reduce_max(input_tensor=new_points, axis=[2], keepdims=True)
        elif pooling=='min':
            new_points = tf.compat.v1.layers.max_pooling2d(-1 * new_points, [1, nsample], [1, 1], padding='VALID',name='minpool1')
        elif pooling=='max_and_avg':
            avg_points = tf.compat.v1.layers.max_pooling2d(new_points, [1,nsample], [1,1], padding='VALID', name='maxpool1')
            max_points = tf.compat.v1.layers.average_pooling2d(new_points, [1,nsample],[1,1], padding='VALID', name='avgpool1')
            new_points = tf.concat([avg_points, max_points], axis=-1)

        #[Optional]Feature Processing对点云进行卷积和特征抽取   
        if mlp2 is None: mlp2 = []
        for i, num_out_channel in enumerate(mlp2):
            new_points = tf_util.conv2d_ibn(new_points, num_out_channel, [1,1],
                                            padding='VALID', stride=[1,1],
                                            bn=bn, ibn=ibn,is_training=is_training,
                                            scope='conv_post_%d'%(i), bn_decay=bn_decay) 
        new_points = tf.squeeze(new_points, [2]) # (batch_size, npoints, mlp2[-1])
        #新的子集中心点坐标，对应特征，包含指向的local region的所有参与运算点的index（batch_size,npoint,nsample）
        return new_xyz, new_points, idx

def pointnet_sa_module_msg(xyz, points, npoint, radius_list, nsample_list, mlp_list, is_training, bn_decay, scope, bn=True, ibn = False, use_xyz=True):
    ''' PointNet Set Abstraction (SA) module with Multi-Scale Grouping (MSG)
        Input:
            xyz: (batch_size, ndataset, 3) TF tensor
            points: (batch_size, ndataset, channel) TF tensor
            npoint: int32 -- #points sampled in farthest point sampling
            radius: list of float32 -- search radius in local region
            nsample: list of int32 -- how many points in each local region
            mlp: list of list of int32 -- output size for MLP on each point
            use_xyz: bool, if True concat XYZ with local point features, otherwise just use point features
        Return:
            new_xyz: (batch_size, npoint, 3) TF tensor
            new_points: (batch_size, npoint, \sum_k{mlp[k][-1]}) TF tensor
    '''
    with tf.compat.v1.variable_scope(scope) as sc:
        new_xyz = gather_point(xyz, farthest_point_sample(npoint, xyz))
        new_points_list = []
        for i in range(len(radius_list)):
            radius = radius_list[i]
            nsample = nsample_list[i]
            idx, pts_cnt = query_ball_point(radius, nsample, xyz, new_xyz)
            grouped_xyz = group_point(xyz, idx)
            grouped_xyz -= tf.expand_dims(new_xyz, 2)
            if points is not None:
                grouped_points = group_point(points, idx)
                if use_xyz:
                    grouped_points = tf.concat([grouped_points, grouped_xyz], axis=-1)
            else:
                grouped_points = grouped_xyz
            for j,num_out_channel in enumerate(mlp_list[i]):
                grouped_points = tf_util.conv2d_ibn(grouped_points, num_out_channel, [1,1],
                                                    padding='VALID', stride=[1,1], bn=bn, ibn=ibn,is_training=is_training,
                                                    scope='conv%d_%d'%(i,j), bn_decay=bn_decay)
            new_points = tf.reduce_max(input_tensor=grouped_points, axis=[2])
            new_points_list.append(new_points)
        new_points_concat = tf.concat(new_points_list, axis=-1)
        return new_xyz, new_points_concat


    """先使用卷积处理xyz1, xyz2坐标
    随后再利用稀疏的point2特征进行插值
    将特征进行上采样插值"""
def pointnet_fp_module(xyz1, xyz2, points1, points2, mlp, is_training, bn_decay, scope, bn=True,ibn=False):
    ''' PointNet Feature Propogation (FP) Module
        Input:                                                                                                      
            xyz1: (batch_size, ndataset1, 3) TF tensor                                                              
            xyz2: (batch_size, ndataset2, 3) TF tensor, sparser than xyz1                                           
            points1: (batch_size, ndataset1, nchannel1) TF tensor                                                   
            points2: (batch_size, ndataset2, nchannel2) TF tensor
            mlp: list of int32 -- output size for MLP on each point                                                 
        Return:
            new_points: (batch_size, ndataset1, mlp[-1]) TF tensor
    '''
    with tf.compat.v1.variable_scope(scope) as sc:
        dist, idx = three_nn(xyz1, xyz2)
        dist = tf.maximum(dist, 1e-10)
        norm = tf.reduce_sum(input_tensor=(1.0/dist),axis=2,keepdims=True)#归一化
        norm = tf.tile(norm,[1,1,3])
        weight = (1.0/dist) / norm
        #特征插值函数from  tf_ops/interpolation/tf_interpolate.py, 利用cpp实现的tf_ops
        interpolated_points = three_interpolate(points2, idx, weight)

        if points1 is not None:
            new_points1 = tf.concat(axis=2, values=[interpolated_points, points1]) # B,ndataset1,nchannel1+nchannel2
        else:
            new_points1 = interpolated_points
        new_points1 = tf.expand_dims(new_points1, 2)
        for i, num_out_channel in enumerate(mlp):
            new_points1 = tf_util.conv2d_ibn(new_points1, num_out_channel, [1,1],
                                             padding='VALID', stride=[1,1],
                                             bn=bn, ibn=ibn,is_training=is_training,
                                             scope='conv_%d'%(i), bn_decay=bn_decay)#用1*1卷积进行维度缩减
        new_points1 = tf.squeeze(new_points1, [2]) # B,ndataset1,mlp[-1]
        return new_points1


if __name__ == "__main__":
    
    pointnet_sa_module(l0_xyz, l0_points, npoint=num_point, radius=bradius*0.05,bn=use_bn,ibn = use_ibn,
                                                           nsample=32, mlp=[32, 32, 64], mlp2=None, group_all=False,
                                                           is_training=is_training, bn_decay=bn_decay, scope='layer1')