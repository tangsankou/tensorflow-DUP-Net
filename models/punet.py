import tensorflow as tf
import os
import sys
""" BASE_DIR = os.path.dirname(os.path.abspath(__file__))
print(BASE_DIR)
sys.path.append(BASE_DIR) """
sys.path.append('/home/user_tp/workspace/code/dupnet/tensorflow2-DUP-Net')
from utils import tf_util
from utils.pointnet_util import pointnet_sa_module,pointnet_fp_module

#输入位置的占位符，包括了输入点云、基准、点云归一化、多尺度处理半径等
def placeholder_inputs(batch_size, num_point,up_ratio = 4):
    pointclouds_pl = tf.compat.v1.placeholder(tf.float32, shape=(batch_size, num_point, 6))
    pointclouds_gt = tf.compat.v1.placeholder(tf.float32, shape=(batch_size, num_point*up_ratio, 3))
    pointclouds_normal = tf.compat.v1.placeholder(tf.float32, shape=(batch_size, num_point * up_ratio, 3))
    pointclouds_radius = tf.compat.v1.placeholder(tf.float32, shape=(batch_size))
    return pointclouds_pl, pointclouds_gt, pointclouds_normal, pointclouds_radius

#定义模型，整体定义在sc这个scope下面，统一符号的命名空间范围
def get_gen_model(point_cloud, is_training, scope, bradius = 1.0, reuse=None, use_rv=False, use_bn = False,use_ibn = False,
                  use_normal=False, bn_decay=None, up_ratio = 4, idx=None):

    with tf.compat.v1.variable_scope(scope,reuse=reuse) as sc:#变量共享
        # print("type(point_cloud.get_shape()[0]):",type(point_cloud.get_shape()[0]))
        batch_size = point_cloud.get_shape()[0]#B批大小
        # print("batch_size:",batch_size)
        num_point = point_cloud.get_shape()[1]#N点的数目
        l0_xyz = point_cloud[:,:,0:3]#0级的点云
        if use_normal:
            l0_points = point_cloud[:,:,3:]#0级normal值
        else:
            l0_points = None
        # Layer 1 在四个级不同的尺度上进行下采样，如下图中红色部分
        # 其中采样数量npoint在不断减小(变稀疏了）
        # 采样半径radius从0.05,0.1,0.2,0.3扩大.
        # 点特征输出维度mlp也在不断增加，因为扩大后包含更多语义信息了

        """
        从原始点云中选出num_point个点来,每个点在其周围选择至多nsample=32个点作为local region。
        l1_xyz : (batch_size, ?num_point, 3)
        l1_points: (batch_size, ?, ?128)
        l1_indices:(batch_size, ?, ?32)
        """
        l1_xyz, l1_points, l1_indices = pointnet_sa_module(l0_xyz, l0_points, npoint=num_point, radius=bradius*0.05,bn=use_bn,ibn = use_ibn,
                                                           nsample=32, mlp=[32, 32, 64], mlp2=None, group_all=False,
                                                           is_training=is_training, bn_decay=bn_decay, scope='layer1')

        l2_xyz, l2_points, l2_indices = pointnet_sa_module(l1_xyz, l1_points, npoint=num_point/2, radius=bradius*0.1,bn=use_bn,ibn = use_ibn,
                                                           nsample=32, mlp=[64, 64, 128], mlp2=None, group_all=False,
                                                           is_training=is_training, bn_decay=bn_decay, scope='layer2')

        l3_xyz, l3_points, l3_indices = pointnet_sa_module(l2_xyz, l2_points, npoint=num_point/4, radius=bradius*0.2,bn=use_bn,ibn = use_ibn,
                                                           nsample=32, mlp=[128, 128, 256], mlp2=None, group_all=False,
                                                           is_training=is_training, bn_decay=bn_decay, scope='layer3')

        l4_xyz, l4_points, l4_indices = pointnet_sa_module(l3_xyz, l3_points, npoint=num_point/8, radius=bradius*0.3,bn=use_bn,ibn = use_ibn,
                                                           nsample=32, mlp=[256, 256, 512], mlp2=None, group_all=False,
                                                           is_training=is_training, bn_decay=bn_decay, scope='layer4')

        # 将特征上采样到C.相同的维度64,以便随后进行特征拓展
        # Feature Propagation layers
        up_l4_points = pointnet_fp_module(l0_xyz, l4_xyz, None, l4_points, [64], is_training, bn_decay,
                                       scope='fa_layer1',bn=use_bn,ibn = use_ibn)

        up_l3_points = pointnet_fp_module(l0_xyz, l3_xyz, None, l3_points, [64], is_training, bn_decay,
                                       scope='fa_layer2',bn=use_bn,ibn = use_ibn)

        up_l2_points = pointnet_fp_module(l0_xyz, l2_xyz, None, l2_points, [64], is_training, bn_decay,
                                       scope='fa_layer3',bn=use_bn,ibn = use_ibn)

        #衔接特征随后进行特征拓展
        ###concat feature
        with tf.compat.v1.variable_scope('up_layer',reuse=reuse):
            new_points_list = []
            #上采样率,上采样几倍就来几次拓展,将特征先衔接再拓展维度上进行两次卷积C1C2，最后在新维度上concate得到net输入
            for i in range(up_ratio):
                #64*4+1=259维度
                concat_feat = tf.concat([up_l4_points, up_l3_points, up_l2_points, l1_points, l0_xyz], axis=-1)
                concat_feat = tf.expand_dims(concat_feat, axis=2)
                concat_feat = tf_util.conv2d_ibn(concat_feat, 256, [1, 1],
                                                 padding='VALID', stride=[1, 1],
                                                 bn=False, is_training=is_training,
                                                 scope='fc_layer0_%d'%(i), bn_decay=bn_decay)

                new_points = tf_util.conv2d_ibn(concat_feat, 128, [1, 1],
                                                padding='VALID', stride=[1, 1],
                                                bn=use_bn, is_training=is_training,
                                                scope='conv_%d' % (i),
                                                bn_decay=bn_decay)
                new_points_list.append(new_points)
            net = tf.concat(new_points_list,axis=1)

        #全连接层，利用1*1卷积来代替得到最终的三维点云输入r*N*3
        #get the xyz
        coord = tf_util.conv2d_ibn(net, 64, [1, 1],
                                   padding='VALID', stride=[1, 1],
                                   bn=False, is_training=is_training,
                                   scope='fc_layer1', bn_decay=bn_decay)

        coord = tf_util.conv2d_ibn(coord, 3, [1, 1],
                                   padding='VALID', stride=[1, 1],
                                   bn=False, is_training=is_training,
                                   scope='fc_layer2', bn_decay=bn_decay,
                                   activation_fn=None, weight_decay=0.0)  # B*(2N)*1*3
        coord = tf.squeeze(coord, [2])  # B*(2N)*3 去掉维度为1的维度，将张量变为三维输入B*(rN)*3

    return coord,None

if __name__ == "__main__":
    tf.compat.v1.disable_eager_execution()
    pointclouds_ipt = tf.compat.v1.placeholder(tf.float32, shape=(1, 1024, 3))
    print("pointclouds_ipt.type:",type(pointclouds_ipt))
    #载入计算图
    pred, _ = get_gen_model(pointclouds_ipt, is_training=False, scope='generator', bradius=1.0,
                                      reuse=None, use_normal=False, use_bn=False, use_ibn=False, bn_decay=0.95, up_ratio=2)
    
    print("pred.shape()",pred.shape)