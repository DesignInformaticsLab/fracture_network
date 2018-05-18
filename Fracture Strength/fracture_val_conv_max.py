# This is the tensorflow version of the phase expansion code by Yang Jiao
# This is based on Houpu Yao and Ruijin Cang's previous codes
import tensorflow as tf
import numpy as np

def neighbor_shift(x): #this is specific for 3D
    x = tf.expand_dims(x, 3)
    x_neigbor = tf.concat([
        tf.pad(x[1:, :, :, :], ((0, 1), (0, 0), (0, 0), (0, 0))), #below
        tf.pad(x[:-1, :, :, :], ((1, 0), (0, 0), (0, 0), (0, 0))), # above
        tf.pad(x[:, 1:, :, :], ((0, 0), (0, 1), (0, 0), (0, 0))), # right
        tf.pad(x[:, :-1, :, :], ((0, 0), (1, 0), (0, 0), (0, 0))), # left
        tf.pad(x[:, :, 1:, :], ((0, 0), (0, 0), (0, 1), (0, 0))), # back ***
        tf.pad(x[:, :, :-1, :], ((0, 0), (0, 0), (1, 0), (0, 0))), # front

        tf.pad(x[1:, 1:, :, :], ((0, 1), (0, 1), (0, 0), (0, 0))), # bottom right
        tf.pad(x[1:, :-1, :, :], ((0, 1), (1, 0), (0, 0), (0, 0))), # bottom left
        tf.pad(x[:-1, 1:, :, :], ((1, 0), (0, 1), (0, 0), (0, 0))), # upper right
        tf.pad(x[:-1, :-1, :, :], ((1, 0), (1, 0), (0, 0), (0, 0))), # upper left
        tf.pad(x[:, 1:, 1:, :], ((0, 0), (0, 1), (0, 1), (0, 0))), # right back ***
        tf.pad(x[:, 1:, :-1, :], ((0, 0), (0, 1), (1, 0), (0, 0))), # right front
        tf.pad(x[:, :-1, 1:, :], ((0, 0), (1, 0), (0, 1), (0, 0))), # left back ***
        tf.pad(x[:, :-1, :-1, :], ((0, 0), (1, 0), (1, 0), (0, 0))), # left front
        tf.pad(x[1:, :, 1:, :], ((0, 1), (0, 0), (0, 1), (0, 0))), # bottom back ***
        tf.pad(x[1:, :, :-1, :], ((0, 1), (0, 0), (1, 0), (0, 0))), # bottom front
        tf.pad(x[:-1, :, 1:, :], ((1, 0), (0, 0), (0, 1), (0, 0))), # upper back ***
        tf.pad(x[:-1, :, :-1, :], ((1, 0), (0, 0), (1, 0), (0, 0))) # upper front
        #
        # tf.pad(x[1:, 1:, 1:, :], ((0, 1), (0, 1), (0, 1), (0, 0))), # bottom right back ***
        # tf.pad(x[1:, 1:, :-1, :], ((0, 1), (0, 1), (1, 0), (0, 0))), # bottom right front
        # tf.pad(x[1:, :-1, 1:, :], ((0, 1), (1, 0), (0, 1), (0, 0))), # bottom left back ***
        # tf.pad(x[1:, :-1, :-1, :], ((0, 1), (1, 0), (1, 0), (0, 0))), # bottom left front
        # tf.pad(x[:-1, 1:, 1:, :], ((1, 0), (0, 1), (0, 1), (0, 0))), # upper right back ***
        # tf.pad(x[:-1, 1:, :-1, :], ((1, 0), (0, 1), (1, 0), (0, 0))), # upper right front
        # tf.pad(x[:-1, :-1, 1:, :], ((1, 0), (1, 0), (0, 1), (0, 0))), # upper left back ***
        # tf.pad(x[:-1, :-1, :-1, :], ((1, 0), (1, 0), (1, 0), (0, 0))) # upper left front
    ],3)
    return x_neigbor

def get_const_tensors(scale, dim):
    # 10x10x10x18 for now, should be able to reduce locally
    dis= 0.01/scale
    rho = 4.43e2
    sigY = 2e5
    E0 = 1e7
    mu0 = 0.3
    E1 = 1e7
    mu1 = 0.3
    E2 = 1e8
    mu2 = 0.3
    Kn01 = 2.0*E0/(1.0 + mu0)
    Kn02 = 2.0*E0/(1.0 + mu0)
    Tv0 = E0*(4.0*mu0 - 1.0)/((9+4*np.sqrt(2))*(1.0 + mu0)*(1.0 - 2.0*mu0))
    Kn11 = 2.0*E1/(1.0 + mu1)
    Kn12 = 2.0*E1/(1.0 + mu1)
    Tv1 = E1*(4.0*mu1 - 1.0)/((9+4*np.sqrt(2))*(1.0 + mu1)*(1.0 - 2.0*mu1))
    Kn21 = 2.0*E2/(1.0 + mu2)
    Kn22 = 2.0*E2/(1.0 + mu2)
    Tv2 = E2*(4.0*mu2 - 1.0)/((9+4*np.sqrt(2))*(1.0 + mu2)*(1.0 - 2.0*mu2))

    half_scale = int(scale/2)
    Tv_top = np.ones((scale,scale,half_scale,18))*Tv0
    Tv_bottom = np.ones((scale,scale,half_scale,18))*Tv0

    Kn_top = np.concatenate((np.ones((scale,scale,half_scale,6))*Kn01,np.ones((scale,scale,half_scale,12))*Kn02),3)
    Kn_bottom = np.concatenate((np.ones((scale,scale,half_scale,6))*Kn01,np.ones((scale,scale,half_scale,12))*Kn02),3)

    stretch_top = np.concatenate((np.ones((scale,scale,half_scale,6))*(120*sigY/(2.0*Kn01) + 1.0)*dis,
                             np.ones((scale,scale,half_scale,12))*(120*sigY/(2.0*Kn01) + 1.0*np.sqrt(2.0))*dis),3)

    stretch_bottom = np.concatenate((np.ones((scale,scale,half_scale,6))*(120*sigY/(2.0*Kn01) + 1.0)*dis,
                             np.ones((scale,scale,half_scale,12))*(120*sigY/(2.0*Kn01) + 1.0*np.sqrt(2.0))*dis),3)

    # model the interface
    connection_index_top = (4,10,12,14,16)
    connection_index_bottom = (5,11,13,15,17)
    Tv_top[:,:,-1,connection_index_top] = 2*(Tv0*Tv1/(Tv0+Tv1)) # -1 is the last square in height (the interface from the top
    # piece), 4 is the link to the back (bottom) (see neighbor_shift)
    Tv_bottom[:,:,0,connection_index_bottom] = 2*(Tv0*Tv1/(Tv0+Tv1)) # 0 is the first square in height (the interface from the bottom
    # piece), 5 is the link to the front (top) (see neighbor_shift)
    Kn_top[:,:,-1,4] = 2.0*(Kn01*Kn11/(Kn01+Kn11))
    Kn_top[:,:,-1,(10,12,14,16)] = 2.0*(Kn02*Kn12/(Kn02+Kn12))
    Kn_bottom[:,:,-1,5] = 2.0*(Kn01*Kn11/(Kn01+Kn11))
    Kn_bottom[:,:,-1,(11,13,15,17)] = 2.0*(Kn02*Kn12/(Kn02+Kn12))
    stretch_top[:,:,-1,4] = (60*sigY/(4.0*(Kn01*Kn11/(Kn01+Kn11))) + 1.0)*dis
    stretch_top[:,:,-1,(10,12,14,16)] = (60*sigY/(4.0*(Kn01*Kn11/(Kn01+Kn11))) + np.sqrt(2.0))*dis
    stretch_bottom[:,:,-1,5] = (60*sigY/(4.0*(Kn01*Kn11/(Kn01+Kn11))) + 1.0)*dis
    stretch_bottom[:,:,-1,(11,13,15,17)] = (60*sigY/(4.0*(Kn01*Kn11/(Kn01+Kn11))) + np.sqrt(2.0))*dis

    # assemble
    Tv = tf.convert_to_tensor(np.concatenate((Tv_top, Tv_bottom), 2), dtype=tf.float32)
    Kn = tf.convert_to_tensor(np.concatenate((Kn_top, Kn_bottom), 2), dtype=tf.float32)
    stretch = tf.convert_to_tensor(np.concatenate((stretch_top, stretch_bottom), 2), dtype=tf.float32)

    return Tv, Kn, stretch

def get_distance(Positions_orig):
    dx_y_z = tf.abs(tf.concat([
    tf.expand_dims(tf.pad(Positions_orig[1:, :, :] - Positions_orig[:-1, :, :], ((1, 0), (0, 0), (0, 0), (0, 0))), 0), #upper
    tf.expand_dims(tf.pad(Positions_orig[:-1, :, :] - Positions_orig[1:, :, :], ((0, 1), (0, 0), (0, 0), (0, 0))), 0), #lower
    tf.expand_dims(tf.pad(Positions_orig[:,1:,:] - Positions_orig[:,:-1,:], ((0, 0), (1, 0), (0, 0), (0, 0))), 0), #left
    tf.expand_dims(tf.pad(Positions_orig[:,:-1,:] - Positions_orig[:,1:,:], ((0, 0), (0, 1), (0, 0), (0, 0))), 0), #right
    tf.expand_dims(tf.pad(Positions_orig[:,:,1:] - Positions_orig[:,:,:-1], ((0, 0), (0, 0), (1, 0), (0, 0))), 0), #left
    tf.expand_dims(tf.pad(Positions_orig[:,:,:-1] - Positions_orig[:,:,1:], ((0, 0), (0, 0), (0, 1), (0, 0))), 0), #right

    tf.expand_dims(tf.pad(Positions_orig[1:,1:,:] - Positions_orig[:-1,:-1,:], ((1, 0), (1, 0), (0, 0), (0, 0))), 0), #upper left
    tf.expand_dims(tf.pad(Positions_orig[1:,:-1,:] - Positions_orig[:-1,1:,:], ((1, 0), (0, 1), (0, 0), (0, 0))), 0), #upper right
    tf.expand_dims(tf.pad(Positions_orig[:-1,1:,:] - Positions_orig[1:,:-1,:], ((0, 1), (1, 0), (0, 0), (0, 0))), 0), #lower left
    tf.expand_dims(tf.pad(Positions_orig[:-1,:-1,:] - Positions_orig[1:,1:,:], ((0, 1), (0, 1), (0, 0), (0, 0))), 0), #lower right

    tf.expand_dims(tf.pad(Positions_orig[:,1:,1:] - Positions_orig[:,:-1,:-1], ((0, 0), (0, 1), (0, 1), (0, 0))), 0), #left front
    tf.expand_dims(tf.pad(Positions_orig[:,1:,:-1] - Positions_orig[:,:-1,1:], ((0, 0), (1, 0), (0, 1), (0, 0))), 0), #left back
    tf.expand_dims(tf.pad(Positions_orig[:,:-1,1:] - Positions_orig[:,1:,:-1], ((0, 0), (0, 1), (1, 0), (0, 0))), 0), #right front
    tf.expand_dims(tf.pad(Positions_orig[:,:-1,:-1] - Positions_orig[:,1:,1:], ((0, 0), (0, 1), (0, 1), (0, 0))), 0), #right back

    tf.expand_dims(tf.pad(Positions_orig[1:,:,1:] - Positions_orig[:-1,:,:-1], ((1, 0), (0, 0), (1, 0), (0, 0))), 0), #upper front
    tf.expand_dims(tf.pad(Positions_orig[1:,:,:-1] - Positions_orig[:-1,:,1:], ((1, 0), (0, 0), (0, 1), (0, 0))), 0), #upper back
    tf.expand_dims(tf.pad(Positions_orig[:-1,:,1:] - Positions_orig[1:,:,:-1], ((0, 1), (0, 0), (1, 0), (0, 0))), 0), #lower front
    tf.expand_dims(tf.pad(Positions_orig[:-1,:,:-1] - Positions_orig[1:,:,1:], ((0, 1), (0, 0), (0, 1), (0, 0))), 0) #lower back
    #
    # tf.expand_dims(tf.pad(Positions_orig[1:,1:,1:] - Positions_orig[:-1,:-1,:-1], ((1, 0), (1, 0), (1, 0), (0, 0))), 0), #upper left front
    # tf.expand_dims(tf.pad(Positions_orig[1:,1:,:-1] - Positions_orig[:-1,:-1,1:], ((1, 0), (1, 0), (0, 1), (0, 0))), 0), #upper left back
    # tf.expand_dims(tf.pad(Positions_orig[1:,:-1,1:] - Positions_orig[:-1,1:,:-1], ((1, 0), (0, 1), (1, 0), (0, 0))), 0), #upper right front
    # tf.expand_dims(tf.pad(Positions_orig[1:,:-1,:-1] - Positions_orig[:-1,1:,1:], ((1, 0), (0, 1), (0, 1), (0, 0))), 0), #upper right back
    # tf.expand_dims(tf.pad(Positions_orig[:-1,1:,1:] - Positions_orig[1:,:-1,:-1], ((0, 1), (1, 0), (1, 0), (0, 0))), 0), #lower left front
    # tf.expand_dims(tf.pad(Positions_orig[:-1,1:,:-1] - Positions_orig[1:,:-1,1:], ((0, 1), (1, 0), (0, 1), (0, 0))), 0), #lower left back
    # tf.expand_dims(tf.pad(Positions_orig[:-1,:-1,1:] - Positions_orig[1:,1:,:-1], ((0, 1), (0, 1), (1, 0), (0, 0))), 0), #lower right front
    # tf.expand_dims(tf.pad(Positions_orig[:-1,:-1,:-1] - Positions_orig[1:,1:,1:], ((0, 1), (0, 1), (0, 1), (0, 0))), 0), #lower right back
    ],0))  # 18x10x10x10x2, only four of them are unique
    dx, dy, dz = tf.transpose(dx_y_z[:,:,:,:,0],(1,2,3,0)), tf.transpose(dx_y_z[:,:,:,:,1],(1,2,3,0)), \
                 tf.transpose(dx_y_z[:,:,:,:,2],(1,2,3,0))
    dxyz = tf.concat([tf.expand_dims(dx,4), tf.expand_dims(dy,4), tf.expand_dims(dz,4)], 4)
    distance = tf.sqrt(dx**2 + dy**2 + dz**2)#10x10x10x18
    return dxyz, distance

def f2x(distance_new, distance_orig, stretch, bondsign, Tv):

    sig1 = tf.sigmoid(stretch - distance_new)  # 10x10x10x18, sig=0 means crack
    distance_inc = distance_new - distance_orig
    dL = distance_inc * sig1 * bondsign  # 10x10x10x18
    dL += (1 - bondsign) * (tf.sigmoid(-distance_inc)) * distance_inc

    # dL_total = tf.concat([tf.expand_dims(tf.reduce_sum(dL[:, :, :, 6:], 3),3), tf.expand_dims(tf.reduce_sum(dL[:, :, :, :6], 3),3)],3)  # 10x10x10x2
    dL_total = tf.reduce_sum(dL, 3)  # 10x10x10x1
    TdL = dL * Tv
    # TdL_total = tf.concat([tf.expand_dims(tf.reduce_sum(TdL[:, :, :, 6:], 3),3), tf.expand_dims(tf.reduce_sum(TdL[:, :, :, :6], 3),3)],3)  # 10x10x10x2
    TdL_total = tf.reduce_sum(TdL, 3)  # 10x10x10x1

    return dL, dL_total, TdL_total

def x2f(Kn, dL, dL_total, Tv, TdL_total, distance, dxyz, bondsign):
    part1 = tf.tile(tf.expand_dims(2. * Kn * dL, 4), (1,1,1,1,3))  # 10x10x10x18x3
    part2 = 0.5 * tf.tile(tf.expand_dims(Tv * (tf.expand_dims(dL_total,3) + neighbor_shift(dL_total)), 4), (1,1,1,1,3))   # 10x10x10x18x3
    part3 = 0.5 * tf.tile(tf.expand_dims(tf.expand_dims(TdL_total,3) + neighbor_shift(TdL_total), 4), (1,1,1,1,3)) # 10x10x10x18x3
    # dxyz: 10x10x10x18x3, distance: 10x10x10x18
    f = (part1 + part2 + part3) * dxyz / tf.expand_dims(distance, 4)
    f = f * tf.expand_dims((bondsign - tf.sigmoid(dL)), 4)  # 10x10x10x18x3

    return tf.reduce_sum(f, 3), tf.reduce_sum(tf.reduce_sum(f[:,:,0,:,2], 2)) #all forces and force from top

def apply_bc(Lvel, Lacc, scale):
    # clip&padding to enforce boundary condition
    Lvel_bottom_x = 0.
    Lvel_bottom_y = 0.
    Lvel_bottom_z = 0.
    Lvel_bottom = tf.concat([tf.ones((scale, scale, 1, 1)) * Lvel_bottom_x, tf.ones((scale, scale, 1, 1)) * Lvel_bottom_y,
                             tf.ones((scale, scale, 1, 1)) * Lvel_bottom_z], 3)
    Lvel_top_x = 0.
    Lvel_top_y = 0.
    Lvel_top_z = -0.5
    Lvel_top = tf.concat([tf.ones((scale, scale, 1, 1)) * Lvel_top_x, tf.ones((scale, scale, 1, 1)) * Lvel_top_y,
                          tf.ones((scale, scale, 1, 1)) * Lvel_top_z], 3)
    Lvel = tf.concat([Lvel_top, Lvel[:, :, 1:-1, :], Lvel_bottom], 2)

    Lacc_bottom_x = 0.
    Lacc_bottom_y = 0.
    Lacc_bottom_z = 0.
    Lacc_bottom = tf.concat([tf.ones((scale, scale, 1, 1)) * Lacc_bottom_x, tf.ones((scale, scale, 1, 1)) * Lacc_bottom_y,
                             tf.ones((scale, scale, 1, 1)) * Lacc_bottom_z], 3)
    Lacc_top_x = 0.
    Lacc_top_y = 0.
    Lacc_top_z = -0.5
    Lacc_top = tf.concat([tf.ones((scale, scale, 1, 1)) * Lacc_top_x, tf.ones((scale, scale, 1, 1)) * Lacc_top_y,
                          tf.ones((scale, scale, 1, 1)) * Lacc_top_z], 3)
    Lacc = tf.concat([Lacc_top, Lacc[:, :, 1:-1, :], Lacc_bottom], 2)

    return Lvel, Lacc

if __name__ == '__main__':
    num_particle_x = num_particle_y = num_particle_z = scale = 10
    ndim = 3  # physical spatial dimension of the problem
    t_step = 0.01  # time steps of the simulation
    steps = 10  # number of steps for simulation
    Mass = 1.
    len = 0.01 # physical size of the problem

    # Positions = tf.placeholder(tf.float32, (scale, scale, scale, ndim)) # position x
    grid = np.linspace(0, len, scale)
    Positions = tf.convert_to_tensor(
        np.asarray([[grid[i], grid[j], grid[k]] for i in range(scale) for j in range(scale) for k in range(scale)]
                   ).reshape(scale, scale, scale, ndim), dtype=tf.float32)

    Lvel = tf.placeholder(tf.float32, (scale, scale, scale, ndim)) # velocity \dot x
    Lacc = tf.placeholder(tf.float32, (scale, scale, scale, ndim)) # accelration \dot \dot x
    # stretch = tf.placeholder(tf.float32, (scale, scale, scale, 3**ndim-9)) # threshold on displacement before crack
    bondsign = tf.placeholder(tf.float32, (scale, scale, scale, 3**ndim-9)) # inital crack

    Tv, Kn, stretch = get_const_tensors(scale=scale, dim=ndim)
    dxy_orig, distance_orig = get_distance(Positions)
    distance_old = distance_orig
    bondsign_old = bondsign
    Positions_old = Positions
    # pos_his = []
    # netF = []

    for step_i in range(steps):

        Positions_new = Positions_old + Lvel * t_step + Lacc * t_step ** 2 / 2.  # 10x10x10x3
        dxy_new, distance_new = get_distance(Positions_new)

        dL, dL_total, TdL_total = f2x(distance_new, distance_orig, stretch, bondsign, Tv)

        netF, netF_top = x2f(Kn, dL, dL_total, Tv, TdL_total, distance_new, dxy_new, bondsign)

        Lacc = netF / Mass #10x10x10x3
        Lvel = Lvel + Lacc * t_step #10x10x10x3
        Lvel, Lacc = apply_bc(Lvel, Lacc, scale)

        # pos_his += [Positions_new]
        print('done')

    # tf.gradients(netF, Positions)


