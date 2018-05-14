import tensorflow as tf
import numpy as np

def neighbor_shift(x):
    x = tf.expand_dims(x, 2)
    x_neigbor = tf.concat([
        tf.pad(x[1:, :, :], ((1, 0), (0, 0), (0, 0))),
        tf.pad(x[:-1, :, :], ((0, 1), (0, 0), (0, 0))),
        tf.pad(x[:, 1:, :], ((0, 0), (1, 0), (0, 0))),
        tf.pad(x[:, :-1, :], ((0, 0), (0, 1), (0, 0))),

        tf.pad(x[1:, 1:, :], ((1, 0), (1, 0), (0, 0))),
        tf.pad(x[1:, :-1, :], ((1, 0), (0, 1), (0, 0))),
        tf.pad(x[:-1, 1:, :], ((0, 1), (1, 0), (0, 0))),
        tf.pad(x[:-1, :-1, :], ((0, 1), (0, 1), (0, 0)))
    ],2)
    return x_neigbor

def get_Tv_Kn():
    # 10x10x8 for now, should be able to reduce locally
    Tv = tf.zeros((10,10,8))
    Kn = tf.zeros((10,10,8))
    return Tv,Kn

def get_distance(Positions_orig):
    dx_y = tf.concat([
    tf.expand_dims(tf.pad(Positions_orig[1:] - Positions_orig[:-1], ((1, 0), (0, 0), (0, 0))), 0), #above
    tf.expand_dims(tf.pad(Positions_orig[:-1] - Positions_orig[1:], ((0, 1), (0, 0), (0, 0))), 0), #below
    tf.expand_dims(tf.pad(Positions_orig[:,1:] - Positions_orig[:,:-1], ((0, 0), (1, 0), (0, 0))), 0), #left
    tf.expand_dims(tf.pad(Positions_orig[:,:-1] - Positions_orig[:,1:], ((0, 0), (0, 1), (0, 0))), 0), #right

    tf.expand_dims(tf.pad(Positions_orig[1:,1:] - Positions_orig[:-1,:-1], ((1, 0), (1, 0), (0, 0))), 0), #upper left
    tf.expand_dims(tf.pad(Positions_orig[1:,:-1] - Positions_orig[:-1,1:], ((1, 0), (0, 1), (0, 0))), 0), #upper right
    tf.expand_dims(tf.pad(Positions_orig[:-1,1:] - Positions_orig[1:,:-1], ((0, 1), (1, 0), (0, 0))), 0), #lower left
    tf.expand_dims(tf.pad(Positions_orig[:-1,:-1] - Positions_orig[1:,1:], ((0, 1), (0, 1), (0, 0))), 0), #lower right
    ],0)  # 8x10x10x2, only four of them are unique
    dx, dy = tf.transpose(dx_y[:,:,:,0],(1,2,0)), tf.transpose(dx_y[:,:,:,1],(1,2,0))
    dxy = tf.concat([tf.expand_dims(dx,3), tf.expand_dims(dy,3)], 3)
    distance = tf.sqrt(dx**2 + dx**2)#10x10x8
    return dxy, distance

def f2x(distance_new, distance_orig, strech, bondsign, Tv):

    sig1 = tf.sigmoid(strech - distance_new)  # 10x10x8, sig=0 means crack
    distance_inc = distance_new - distance_orig
    dL = distance_inc * sig1 * bondsign  # 10x10x8
    dL += (1 - bondsign) * (tf.sigmoid(-distance_inc)) * distance_inc

    dL_total = tf.concat([tf.expand_dims(tf.reduce_sum(dL[:, :, 4:], 2),2), tf.expand_dims(tf.reduce_sum(dL[:, :, :4], 2),2)],2)  # 10x10x2
    TdL = dL * Tv
    TdL_total = tf.concat([tf.expand_dims(tf.reduce_sum(TdL[:, :, 4:], 2),2), tf.expand_dims(tf.reduce_sum(TdL[:, :, :4], 2),2)],2)  # 10x10x2

    return dL, dL_total, TdL_total

def x2f(Kn, dL, dL_total, Tv, TdL_total, distance, dxy, bondsign):
    part1 = tf.tile(tf.expand_dims(2. * Kn * dL, 3), (1,1,1,2))  # 10x10x8x2
    part2 = 0.5 * tf.tile(tf.expand_dims(Tv, 3), (1,1,1,2))  \
            * tf.concat([tf.expand_dims(neighbor_shift(dL_total[:,:,0]),3),
                         tf.expand_dims(neighbor_shift(dL_total[:,:,1]),3)], 3)   # 10x10x8x2
    part3 = 0.5 * tf.concat([tf.expand_dims(neighbor_shift(TdL_total[:,:,0]),3),
                             tf.expand_dims(neighbor_shift(TdL_total[:,:,1]),3)], 3)   # 10x10x8x2
    # dxy: 10x10x8x2, distance: 10x10x8
    f = (part1 + part2 + part3) * dxy / tf.expand_dims(distance, 3)
    f = f * tf.expand_dims((bondsign - tf.sigmoid(dL)), 3)  # 10x10x8x2
    netF = tf.reduce_sum(f, 2)

    return netF

def apply_bc( Lvel, Lacc):
    # clip&padding to enforce boundary condition
    Lvel_left_x = -0.
    Lvel_left_y = 0.
    Lvel_left = tf.concat([tf.ones((10, 1, 1)) * Lvel_left_x, tf.ones((10, 1, 1)) * Lvel_left_y], 2)
    Lvel_right_x = 0.
    Lvel_right_y = 0.
    Lvel_right = tf.concat([tf.ones((10, 1, 1)) * Lvel_right_x, tf.ones((10, 1, 1)) * Lvel_right_y], 2)
    Lvel = tf.concat([Lvel_left, Lvel[:, 1:-1, :], Lvel_right], 1)

    Lacc_left_x = -0.5
    Lacc_left_y = 0.
    Lacc_left = tf.concat([tf.ones((10, 1, 1)) * Lacc_left_x, tf.ones((10, 1, 1)) * Lacc_left_y], 2)
    Lacc_right_x = 0.5
    Lacc_right_y = 0.
    Lacc_right = tf.concat([tf.ones((10, 1, 1)) * Lacc_right_x, tf.ones((10, 1, 1)) * Lacc_right_y], 2)
    Lacc = tf.concat([Lacc_left, Lacc[:, 1:-1, :], Lacc_right], 1)

    return Lvel, Lacc

if __name__ == '__main__':
    num_particle_x = num_particle_y = 10
    ndim = 2  # physical spatial dimension of the problem
    t_step = 0.01  # time steps of the simulation
    steps = 10  # number of steps for simulation
    Mass = 1.

    Positions = tf.placeholder(tf.float32, (num_particle_x, num_particle_y, ndim)) # position x
    Lvel = tf.placeholder(tf.float32, (10, 10, 2)) # velocity \dot x
    Lacc = tf.placeholder(tf.float32, (10, 10, 2)) # accelration \dot \dot x
    strech = tf.placeholder(tf.float32, (10, 10, 8)) # threshold on displacement before crack
    bondsign = tf.placeholder(tf.float32, (num_particle_x, num_particle_y, 8)) # inital crack

    Tv, Kn = get_Tv_Kn()
    dxy_orig, distance_orig = get_distance(Positions)
    distance_old = distance_orig
    bondsign_old = bondsign
    Positions_old = Positions
    # pos_his = []
    # netF = []

    for step_i in range(steps):

        Positions_new = Positions_old + Lvel * t_step + Lacc * t_step ** 2 / 2.  # 10x10x2
        dxy_new, distance_new = get_distance(Positions_new)

        dL, dL_total, TdL_total = f2x(distance_new, distance_orig, strech, bondsign, Tv)

        netF = x2f(Kn, dL, dL_total, Tv, TdL_total, distance_new, dxy_new, bondsign)

        Lacc = netF / Mass #10x10x2
        Lvel = Lvel + Lacc * t_step #10x10x2
        Lvel, Lacc = apply_bc(Lvel, Lacc)

        # pos_his += [Positions_new]
        print('done')

    tf.gradients(netF, Positions)