# This is the tensorflow version of the phase expansion code by Yang Jiao
# This is based on Houpu Yao and Ruijin Cang's previous codes
import tensorflow as tf
import numpy as np

def neighbor_shift(x): #this is specific for 3D
    x = tf.expand_dims(x, 3)
    # 1st dimension - down(forward direction)/up(backward direction),
    # 2nd dimension - right/left,
    # 3nd dimension - bottom/top
    x_neigbor = tf.concat([
        tf.pad(x[1:, :, :, :], ((0, 1), (0, 0), (0, 0), (0, 0))), # down
        tf.pad(x[:-1, :, :, :], ((1, 0), (0, 0), (0, 0), (0, 0))), # up
        tf.pad(x[:, 1:, :, :], ((0, 0), (0, 1), (0, 0), (0, 0))), # right
        tf.pad(x[:, :-1, :, :], ((0, 0), (1, 0), (0, 0), (0, 0))), # left
        tf.pad(x[:, :, 1:, :], ((0, 0), (0, 0), (0, 1), (0, 0))), # bottom ***
        tf.pad(x[:, :, :-1, :], ((0, 0), (0, 0), (1, 0), (0, 0))), # top

        tf.pad(x[1:, 1:, :, :], ((0, 1), (0, 1), (0, 0), (0, 0))), # down right
        tf.pad(x[1:, :-1, :, :], ((0, 1), (1, 0), (0, 0), (0, 0))), # down left
        tf.pad(x[:-1, 1:, :, :], ((1, 0), (0, 1), (0, 0), (0, 0))), # up right
        tf.pad(x[:-1, :-1, :, :], ((1, 0), (1, 0), (0, 0), (0, 0))), # up left
        tf.pad(x[:, 1:, 1:, :], ((0, 0), (0, 1), (0, 1), (0, 0))), # right bottom ***
        tf.pad(x[:, 1:, :-1, :], ((0, 0), (0, 1), (1, 0), (0, 0))), # right top
        tf.pad(x[:, :-1, 1:, :], ((0, 0), (1, 0), (0, 1), (0, 0))), # left bottom ***
        tf.pad(x[:, :-1, :-1, :], ((0, 0), (1, 0), (1, 0), (0, 0))), # left top
        tf.pad(x[1:, :, 1:, :], ((0, 1), (0, 0), (0, 1), (0, 0))), # down back ***
        tf.pad(x[1:, :, :-1, :], ((0, 1), (0, 0), (1, 0), (0, 0))), # down top
        tf.pad(x[:-1, :, 1:, :], ((1, 0), (0, 0), (0, 1), (0, 0))), # up bottom ***
        tf.pad(x[:-1, :, :-1, :], ((1, 0), (0, 0), (1, 0), (0, 0))) # up top
        #
        # tf.pad(x[1:, 1:, 1:, :], ((0, 1), (0, 1), (0, 1), (0, 0))), # down right bottom ***
        # tf.pad(x[1:, 1:, :-1, :], ((0, 1), (0, 1), (1, 0), (0, 0))), # down right top
        # tf.pad(x[1:, :-1, 1:, :], ((0, 1), (1, 0), (0, 1), (0, 0))), # down left bottom ***
        # tf.pad(x[1:, :-1, :-1, :], ((0, 1), (1, 0), (1, 0), (0, 0))), # down left top
        # tf.pad(x[:-1, 1:, 1:, :], ((1, 0), (0, 1), (0, 1), (0, 0))), # up right bottom ***
        # tf.pad(x[:-1, 1:, :-1, :], ((1, 0), (0, 1), (1, 0), (0, 0))), # up right top
        # tf.pad(x[:-1, :-1, 1:, :], ((1, 0), (1, 0), (0, 1), (0, 0))), # up left bottom ***
        # tf.pad(x[:-1, :-1, :-1, :], ((1, 0), (1, 0), (1, 0), (0, 0))) # up left top
    ],3)
    return x_neigbor

def neighbor_shift2d_top(x): #this is specific for 2D
    # x = tf.expand_dims(x, 3)
    x_neigbor = tf.concat([
        x,
        tf.pad(x[1:, :], ((0, 1), (0, 0), (0, 0), (0, 0))), # down
        tf.pad(x[:-1, :], ((1, 0), (0, 0), (0, 0), (0, 0))), # up
        tf.pad(x[:, 1:], ((0, 0), (0, 1), (0, 0), (0, 0))), # right
        tf.pad(x[:, :-1], ((0, 0), (1, 0), (0, 0), (0, 0))), # left
     ], 3)
    return x_neigbor

def neighbor_shift2d_around(x): #this is specific for 2D
    # x = tf.expand_dims(x, 3)
    x_neigbor = tf.concat([
        tf.pad(x[1:, :], ((0, 1), (0, 0), (0, 0), (0, 0))),  # down
        tf.pad(x[:-1, :], ((1, 0), (0, 0), (0, 0), (0, 0))),  # up
        tf.pad(x[:, 1:], ((0, 0), (0, 1), (0, 0), (0, 0))),  # right
        tf.pad(x[:, :-1], ((0, 0), (1, 0), (0, 0), (0, 0))),  # left
        tf.pad(x[1:, 1:], ((0, 1), (0, 1), (0, 0), (0, 0))),  # down right
        tf.pad(x[1:, :-1], ((0, 1), (1, 0), (0, 0), (0, 0))),  # down left
        tf.pad(x[:-1, 1:], ((1, 0), (0, 1), (0, 0), (0, 0))),  # up right
        tf.pad(x[:-1, :-1], ((1, 0), (1, 0), (0, 0), (0, 0)))  # up left
     ], 3)
    return x_neigbor

def insert(tensor_target, tensor_slice, location, dim_idx=3):
    # tensor_slcie: n,10,10,10
    if dim_idx == 3:
        for i, loc_i in enumerate(location):
            tensor_target = tf.concat([tensor_target[:, :, :, :loc_i],
                                       tf.expand_dims(tensor_slice[:, :, :, i], 3),
                                       tensor_target[:, :, :, loc_i:]], 3)
    return tensor_target

def get_const_tensors(scale, dim, defect):
    defect = tf.expand_dims(defect, 3)
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

    Kn01 = 2.0*E0/(1.0 + mu0)   # Kn for close neighbours, same material
    Kn02 = 2.0*E0/(1.0 + mu0)  # Kn for far neighbours, same material
    Tv0 = E0*(4.0*mu0 - 1.0)/((9+4*np.sqrt(2))*(1.0 + mu0)*(1.0 - 2.0*mu0))  # Tv for all neighbours, same material
    S01 = (120*sigY/(2.0*Kn01) + 1.0)*dis  # stretch threshold for close neighbours, same material
    S02 = (120*sigY/(2.0*Kn01) + 1.0*np.sqrt(2.0))*dis  # stretch threshold for far neighbours, same material
    #TODO: should stretch thresholds be different for close or far neighbours?

    Kn11 = 1e6  # Kn for close neighbours, interface
    Kn12 = 1e6  # Kn for far neighbours, interface
    Tv1 = 1e6  # Tv on the interface
    S11 = 0.000105  # stretch threshold for close neighbours on the interface
    S12 = 0.000149  # stretch threshold for far neighbours on the interface

    Kn21 = 0  # connected to defect
    Kn22 = 0  # connected to defect
    Tv2 = 0  # connected to defect
    S21 = 0  # connected to defect
    S22 = 0  # connected to defect

    # fixed parts include upper piece without the two layers close to the interface and the lower piece without the one
    # layer at the interface

    half_scale = int(scale/2)
    Tv_top = tf.ones((scale,scale,half_scale-2,18),tf_dtype)*Tv0  # the upper piece without the last two layers
    Tv_bottom = tf.ones((scale,scale,half_scale-1,18),tf_dtype)*Tv0  # the bottom piece with out the first layer

    # the upper piece without the last two layers
    Kn_top = tf.concat([tf.ones((scale,scale,half_scale-2,6),tf_dtype)*Kn01,  # 6 close neighbours
                        tf.ones((scale,scale,half_scale-2,12),tf_dtype)*Kn02],3)  # 12 far neighbours

    # the bottom piece with out the first layer
    Kn_bottom = tf.concat([tf.ones((scale,scale,half_scale-1,6),tf_dtype)*Kn01,  # 6 close neighbours
                           tf.ones((scale,scale,half_scale-1,12),tf_dtype)*Kn02],3)  # 12 far neighbours

    # the upper piece without the last two layers
    stretch_top = tf.concat([tf.ones((scale,scale,half_scale-2,6),tf_dtype)*S01,
                             tf.ones((scale,scale,half_scale-2,12),tf_dtype)*S02],3)

    # the bottom piece with out the first layer
    stretch_bottom = tf.concat([tf.ones((scale,scale,half_scale-1,6),tf_dtype)*S01,  # 6 close neighbours
                               tf.ones((scale,scale,half_scale-1,12),tf_dtype)*S02],3)  # 12 far neighbours

    bondsign_top = tf.ones((scale,scale,half_scale-2,18),tf_dtype)
    bondsign_bottom = tf.ones((scale,scale,half_scale-1,18),tf_dtype)

    # for the second last layer of the upper piece, we will check whether particles are linked to defects on the last
    # layer of the upper piece, thus we need all indices for the bottom links (total 5)
    connection_index_bottom = (4,10,12,14,16)  # 4-bottom, 10-right bottom, 12-left bottom, 14-down back, 16-up bottom

    # similarly, for the first layer of the bottom piece, we check whether particles are linked to defects on the last
    # layer of the upper piece (which are on top), thus we need all indices for the top links (total 5)
    connection_index_top = (5,11,13,15,17) # 5-top, 11-right top, 13-left top, 15-down top, 17-up top

    # upper piece second layer to the surface
    shifted_defect_top_2 = neighbor_shift2d_top(defect)  # check whether defects exists at the bottom (5 locations)
    Tv_top_2 = insert(tensor_target=tf.ones((scale,scale,1,13),tf_dtype)*Tv0,  # all 13 locations other than the 5 at the bottom
                      tensor_slice=tf.ones((scale,scale,1,5),tf_dtype)*Tv0*shifted_defect_top_2 +  # no defect
                      tf.ones((scale,scale,1,5),tf_dtype)*Tv2*(1-shifted_defect_top_2),  # defect, average the two
                      location=connection_index_bottom)  # locations where the bottom 5 should be inserted

    Kn_top_2 = insert(tensor_target=tf.concat([tf.ones((scale,scale,1,5),tf_dtype)*Kn01,tf.ones((scale,scale,1,8),tf_dtype)*Kn02],3),
                      tensor_slice=tf.concat([tf.ones((scale,scale,1,1),tf_dtype)*Kn01,
                                              tf.ones((scale,scale,1,4),tf_dtype)*Kn02],3)*shifted_defect_top_2 +
                      tf.concat([tf.ones((scale,scale,1,1),tf_dtype)*Kn21,
                                 tf.ones((scale,scale,1,4),tf_dtype)*Kn22],3)*(1-shifted_defect_top_2),
                      location=connection_index_bottom)

    stretch_top_2 = insert(tensor_target=tf.concat([tf.ones((scale,scale,1,5),tf_dtype)*S01,tf.ones((scale,scale,1,8),tf_dtype)*S02],3),
                           tensor_slice=tf.concat([tf.ones((scale,scale,1,1),tf_dtype)*S01,
                                          tf.ones((scale,scale,1,4),tf_dtype)*S02],3)*shifted_defect_top_2 +
                               tf.concat([tf.ones((scale,scale,1,1),tf_dtype)*S21,
                                          tf.ones((scale,scale,1,4),tf_dtype)*S22],3)*(1-shifted_defect_top_2),
                           location=connection_index_bottom)

    bondsign_top_2 = insert(tensor_target=tf.ones((scale,scale,1,13),tf_dtype),  # all 13 locations other than the 5 at the bottom
                            tensor_slice=tf.ones((scale,scale,1,5),tf_dtype)*shifted_defect_top_2,  # no defect
                            location=connection_index_bottom)  # locations where the bottom 5 should be inserted

    # upper piece surface layer
    # for the last layer of the upper piece, we need to look around since the defects are right in this layer
    connection_index_around = (0,1,2,3)  # all 8 neighbours around

    shifted_defect_top_1 = neighbor_shift2d_around(defect) # check whether defects exists around (8 locations)

    Tv_top_1 = insert(tensor_target=tf.concat([(tf.ones((scale,scale,1,8),tf_dtype)*Tv0*shifted_defect_top_1 +  # 8 neighbours around
                                                tf.ones((scale,scale,1,8),tf_dtype)*Tv2*(1-shifted_defect_top_1))*defect,
                                                tf.ones((scale,scale,1,5),tf_dtype)*Tv0*defect],3),  # 5 neighbours on top
                      tensor_slice=tf.ones((scale,scale,1,5),tf_dtype)*Tv1*defect,  # 5 neighbours at bottom, all interface
                      location=connection_index_bottom)
    Kn_top_1 = insert(tensor_target=tf.concat([(tf.ones((scale,scale,1,8),tf_dtype)*Kn01*shifted_defect_top_1 +  # 8 neighbours around
                                                tf.ones((scale,scale,1,8),tf_dtype)*Kn21*(1-shifted_defect_top_1))*defect,
                                                tf.ones((scale,scale,1,1),tf_dtype)*Kn01*defect, # 1 close neighbour on top
                                                tf.ones((scale,scale,1,4),tf_dtype)*Kn02*defect],3), # 4 far neighbours on top
                      tensor_slice=tf.concat([tf.ones((scale,scale,1,1),tf_dtype)*Kn11*defect, # 1 close neighbour at bottom
                                              tf.ones((scale,scale,1,4),tf_dtype)*Kn12*defect],3), # 4 far neighbours at bottom
                      location=connection_index_bottom)
    stretch_top_1 = insert(tensor_target=tf.concat([(tf.ones((scale,scale,1,8),tf_dtype)*S01*shifted_defect_top_1+
                                                     tf.ones((scale,scale,1,8),tf_dtype)*S21*(1-shifted_defect_top_1))*defect,
                                                     tf.ones((scale,scale,1,1),tf_dtype)*S01*defect,
                                                     tf.ones((scale,scale,1,4),tf_dtype)*S02*defect],3),
                           tensor_slice=tf.concat([tf.ones((scale,scale,1,1),tf_dtype)*S11*defect,
                                                   tf.ones((scale,scale,1,4),tf_dtype)*S12*defect],3),
                           location=connection_index_bottom)

    bondsign_top_1 = insert(tensor_target=tf.concat([tf.ones((scale,scale,1,8),tf_dtype)*shifted_defect_top_1*defect,   # 8 neighbours around
                                               tf.ones((scale,scale,1,5),tf_dtype)*defect],3),  # 5 neighbours on top
                      tensor_slice=tf.ones((scale,scale,1,5),tf_dtype)*defect,  # 5 neighbours at bottom, all interface
                      location=connection_index_bottom)

    # lower piece surface layer
    Tv_bottom_1 = insert(tensor_target=tf.ones((scale,scale,1,13),tf_dtype)*Tv0,  # 13 neighbours except the ones on top
                         tensor_slice=tf.ones((scale,scale,1,5),tf_dtype)*Tv1*shifted_defect_top_2 +  # 5 neighbours on top
                                      tf.ones((scale,scale,1,5),tf_dtype)*Tv2*(1-shifted_defect_top_2),
                         location=connection_index_top)
    Kn_bottom_1 = insert(tensor_target=tf.concat([tf.ones((scale,scale,1,5),tf_dtype)*Kn01,  # 5 close neighbours except on top
                                                  tf.ones((scale,scale,1,8),tf_dtype)*Kn02],3),  # 8 far neighbours except on top
                         tensor_slice=tf.concat([tf.ones((scale,scale,1,1),tf_dtype)*Kn11,  # 1 close neighbour on top, interface
                                                 tf.ones((scale,scale,1,4),tf_dtype)*Kn12],3)*shifted_defect_top_2 +  # 4 far neighbours on top, interface
                             tf.concat([tf.ones((scale,scale,1,1),tf_dtype)*Kn21,  # 1 close neighbour on top, defect
                                        tf.ones((scale,scale,1,4),tf_dtype)*Kn22],3)*(1-shifted_defect_top_2),  # 4 far neighbours on top, defect
                         location=connection_index_top)
    stretch_bottom_1 = insert(tensor_target=tf.concat([tf.ones((scale,scale,1,5),tf_dtype)*S01,
                                                       tf.ones((scale,scale,1,8),tf_dtype)*S02],3),
                              tensor_slice=tf.concat([tf.ones((scale,scale,1,1),tf_dtype)*S11,
                                                      tf.ones((scale,scale,1,4),tf_dtype)*S12],3)*shifted_defect_top_2 +
                                           tf.concat([tf.ones((scale,scale,1,1),tf_dtype)*S21,
                                                      tf.ones((scale,scale,1,4),tf_dtype)*S22],3)*(1-shifted_defect_top_2),
                              location=connection_index_top)
    bondsign_bottom_1 = insert(tensor_target=tf.ones((scale,scale,1,13),tf_dtype),  # 13 neighbours except the ones on top
                               tensor_slice=tf.ones((scale,scale,1,5),tf_dtype)*shifted_defect_top_2,  # 5 neighbours on top
                               location=connection_index_top)


    # # model the interface
    # Tv_top[:,:,-1,connection_index_top] = 2*(Tv0*Tv1/(Tv0+Tv1)) # -1 is the last square in height (the interface from the top
    # # piece), 4 is the link to the back (bottom) (see neighbor_shift)
    # Tv_bottom[:,:,0,connection_index_bottom] = 2*(Tv0*Tv1/(Tv0+Tv1)) # 0 is the first square in height (the interface from the bottom
    # # piece), 5 is the link to the front (top) (see neighbor_shift)
    # Kn_top[:,:,-1,4] = 2.0*(Kn01*Kn11/(Kn01+Kn11))
    # Kn_top[:,:,-1,(10,12,14,16)] = 2.0*(Kn02*Kn12/(Kn02+Kn12))
    # Kn_bottom[:,:,-1,5] = 2.0*(Kn01*Kn11/(Kn01+Kn11))
    # Kn_bottom[:,:,-1,(11,13,15,17)] = 2.0*(Kn02*Kn12/(Kn02+Kn12))
    # stretch_top[:,:,-1,4] = (60*sigY/(4.0*(Kn01*Kn11/(Kn01+Kn11))) + 1.0)*dis
    # stretch_top[:,:,-1,(10,12,14,16)] = (60*sigY/(4.0*(Kn01*Kn11/(Kn01+Kn11))) + np.sqrt(2.0))*dis
    # stretch_bottom[:,:,-1,5] = (60*sigY/(4.0*(Kn01*Kn11/(Kn01+Kn11))) + 1.0)*dis
    # stretch_bottom[:,:,-1,(11,13,15,17)] = (60*sigY/(4.0*(Kn01*Kn11/(Kn01+Kn11))) + np.sqrt(2.0))*dis

    # assemble
    # Tv = tf.convert_to_tensor(np.concatenate((Tv_top, Tv_bottom), 2), dtype=tf.float32)
    # Kn = tf.convert_to_tensor(np.concatenate((Kn_top, Kn_bottom), 2), dtype=tf.float32)
    # stretch = tf.convert_to_tensor(np.concatenate((stretch_top, stretch_bottom), 2), dtype=tf.float32)
    Tv = tf.concat([Tv_top, Tv_top_2, Tv_top_1, Tv_bottom_1, Tv_bottom], 2)
    Kn = tf.concat([Kn_top, Kn_top_2, Kn_top_1, Kn_bottom_1, Kn_bottom], 2)
    stretch = tf.concat([stretch_top, stretch_top_2, stretch_top_1, stretch_bottom_1, stretch_bottom], 2)
    bondsign = tf.concat([bondsign_top, bondsign_top_2, bondsign_top_1, bondsign_bottom_1, bondsign_bottom], 2)
    return Tv, Kn, stretch, bondsign

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

def boundary_correction(f):
    # set fs on the boundary to zero
    # zero distance: [0,:,:,0], [-1,:,:,1], [:,0,:,2], [:,-1,:,3], [:,:,0,4], [:,:,-1,5]
    # [0,:,:,6],[:,0,:,6],   [0,:,:,7],[:,-1,:,7],   [-1,:,:,8],[:,0,:,8],    [-1,:,:,9],[:,-1,:,9]
    # [:,-1,:,10],[:,:,-1,10],   [:,0,:,11],[:,:,-1,11],   [:,-1,:,12],[:,:,0,12],    [:,-1,:,13],[:,:,-1,13]
    # [:,:,0,14],[0,:,:,14],   [0,:,:,15],[:,:,-1,15],   [:,:,0,16],[-1,:,:,16],    [:,:,-1,17],[-1,:,:,17]
    f0 = tf.pad(f[1:, :, :, 0:1, :], ((1, 0), (0, 0), (0, 0), (0, 0), (0, 0)))
    f1 = tf.pad(f[:-1, :, :, 1:2, :], ((0, 1), (0, 0), (0, 0), (0, 0), (0, 0)))
    f2 = tf.pad(f[:, 1:, :, 2:3, :], ((0, 0), (1, 0), (0, 0), (0, 0), (0, 0)))
    f3 = tf.pad(f[:, :-1, :, 3:4, :], ((0, 0), (0, 1), (0, 0), (0, 0), (0, 0)))
    f4 = tf.pad(f[:, :,  1:,4:5, :], ((0, 0), (0, 0), (1, 0), (0, 0), (0, 0)))
    f5 = tf.pad(f[:, :, :-1, 5:6, :], ((0, 0), (0, 0), (0, 1), (0, 0), (0, 0)))

    f6 = tf.pad(f[1:, :, :, 6:7, :], ((1, 0), (0, 0), (0, 0), (0, 0), (0, 0)))
    f6 = tf.pad(f6[:, 1:, :, :, :], ((0, 0), (1, 0), (0, 0), (0, 0), (0, 0)))
    f7 = tf.pad(f[1:, :, :, 7:8, :], ((1, 0), (0, 0), (0, 0), (0, 0), (0, 0)))
    f7 = tf.pad(f7[:, :-1, :, :, :], ((0, 0), (0, 1), (0, 0), (0, 0), (0, 0)))
    f8 = tf.pad(f[:-1, :, :,8:9, :], ((0, 1), (0, 0), (0, 0), (0, 0), (0, 0)))
    f8 = tf.pad(f8[:, 1:, :, :, :], ((0, 0), (1, 0), (0, 0), (0, 0), (0, 0)))
    f9 = tf.pad(f[:-1, :, :,9:10, :], ((0, 1), (0, 0), (0, 0), (0, 0), (0, 0)))
    f9 = tf.pad(f9[:, :-1, :, :, :], ((0, 0), (0, 1), (0, 0), (0, 0), (0, 0)))

    f10 = tf.pad(f[:, :-1, :, 10:11, :], ((0, 0), (0, 1), (0, 0), (0, 0), (0, 0)))
    f10 = tf.pad(f10[:, :, :-1, :, :], ((0, 0), (0, 0), (0, 1), (0, 0), (0, 0)))
    f11 = tf.pad(f[:, 1:, :, 11:12, :], ((0, 0), (1, 0), (0, 0), (0, 0), (0, 0)))
    f11 = tf.pad(f11[:, :, :-1, :, :], ((0, 0), (0, 0), (0, 1), (0, 0), (0, 0)))
    f12 = tf.pad(f[:, :-1, :, 12:13, :], ((0, 0), (0, 1), (0, 0), (0, 0), (0, 0)))
    f12 = tf.pad(f12[:, :, 1:, :, :], ((0, 0), (0, 0), (1, 0), (0, 0), (0, 0)))
    f13 = tf.pad(f[:, :-1, :,13:14, :], ((0, 0), (0, 1), (0, 0), (0, 0), (0, 0)))
    f13 = tf.pad(f13[:, :, :-1, :, :], ((0, 0), (0, 0), (0, 1), (0, 0), (0, 0)))

    f14 = tf.pad(f[:, :, 1:, 14:15, :], ((0, 0), (0, 0), (1, 0), (0, 0), (0, 0)))
    f14 = tf.pad(f14[1:, :, :, :, :], ((1, 0), (0, 0), (0, 0), (0, 0), (0, 0)))
    f15 = tf.pad(f[1:, :, :, 15:16, :], ((1, 0), (0, 0), (0, 0), (0, 0), (0, 0)))
    f15 = tf.pad(f15[:, :, :-1, :, :], ((0, 0), (0, 0), (0, 1), (0, 0), (0, 0)))
    f16 = tf.pad(f[:, :, 1:, 16:17, :], ((0, 0), (0, 0), (1, 0), (0, 0), (0, 0)))
    f16 = tf.pad(f16[:-1, :, :, :, :], ((0, 1), (0, 0), (0, 0), (0, 0), (0, 0)))
    f17 = tf.pad(f[:, :, :-1,17:18, :], ((0, 0), (0, 0), (0, 1), (0, 0), (0, 0)))
    f17 = tf.pad(f17[:-1, :, :, :, :], ((0, 1), (0, 0), (0, 0), (0, 0), (0, 0)))

    f = tf.concat([f0,f1,f2,f3,f4,f5,f6,f7,f8,f9,f10,f11,f12,f13,f14,f15,f16,f17],3)
    return f


def f2x(distance_new, distance_orig, stretch, bondsign, Tv):

    sig1 = tf.sigmoid(beta* (stretch - distance_new))  # 10x10x10x18, sig=0 means crack
    distance_inc = distance_new - distance_orig
    flag_phase = tf.ones((10,10,10,18),tf_dtype)
    dL = bondsign * flag_phase * sig1 * distance_inc # 10x10x10x18
    bondsign_new = bondsign  * flag_phase * (1-sig1) * bondsign ######### Double check on the trasition
    dL += bondsign * (1-flag_phase) * distance_inc # 10x10x10x18
    sig2 = tf.sigmoid(-beta*(distance_inc+1e-10)) ##########Double check how to handle distance_inc==0
    dL += (1 - bondsign) * (sig2) * distance_inc
    dL = boundary_correction(tf.expand_dims(dL,4))[:,:,:,:,0]
    # dL_total = tf.concat([tf.expand_dims(tf.reduce_sum(dL[:, :, :, 6:], 3),3), tf.expand_dims(tf.reduce_sum(dL[:, :, :, :6], 3),3)],3)  # 10x10x10x2
    dL_total = tf.reduce_sum(dL, 3)  # 10x10x10x1
    TdL = dL * Tv
    # TdL_total = tf.concat([tf.expand_dims(tf.reduce_sum(TdL[:, :, :, 6:], 3),3), tf.expand_dims(tf.reduce_sum(TdL[:, :, :, :6], 3),3)],3)  # 10x10x10x2
    TdL_total = tf.reduce_sum(TdL, 3)  # 10x10x10x1

    dbg_val = {}
    dbg_val['sig1'] = sig1
    dbg_val['sig2'] = sig2
    dbg_val['distance_inc'] = distance_inc

    return dL, dL_total, TdL_total, bondsign_new, dbg_val

def x2f(Kn, dL, dL_total, Tv, TdL_total, distance, dxyz, bondsign):
    part1 = tf.tile(tf.expand_dims(2. * Kn * dL, 4), (1,1,1,1,3))  # 10x10x10x18x3
    part2 = 0.5 * tf.tile(tf.expand_dims(Tv * (tf.expand_dims(dL_total,3) + neighbor_shift(dL_total)), 4), (1,1,1,1,3))   # 10x10x10x18x3
    part3 = 0.5 * tf.tile(tf.expand_dims(tf.expand_dims(TdL_total,3) + neighbor_shift(TdL_total), 4), (1,1,1,1,3)) # 10x10x10x18x3
    # dxyz: 10x10x10x18x3, distance: 10x10x10x18
    f = (part1 + part2 + part3) * dxyz / tf.expand_dims(distance, 4)
    f = boundary_correction(f)
    sig1 = tf.sigmoid(beta * dL)
    f = f * tf.expand_dims((bondsign - sig1), 4)  # 10x10x10x18x3

    dbg_val = {}
    dbg_val['sig1'] = sig1
    return tf.reduce_sum(f, 3), tf.reduce_sum(tf.reduce_sum(f[:,:,0,:,2], 2)), dbg_val #all forces and force from top

def apply_bc(Lvel, Lacc, scale):
    # clip&padding to enforce boundary condition
    Lvel_bottom_x = 0.
    Lvel_bottom_y = 0.
    Lvel_bottom_z = 0.
    Lvel_bottom = tf.concat([tf.ones((scale, scale, 1, 1),tf_dtype) * Lvel_bottom_x, tf.ones((scale, scale, 1, 1),tf_dtype) * Lvel_bottom_y,
                             tf.ones((scale, scale, 1, 1),tf_dtype) * Lvel_bottom_z], 3)
    Lvel_top_x = 0.
    Lvel_top_y = 0.
    Lvel_top_z = -0.5
    Lvel_top = tf.concat([tf.ones((scale, scale, 1, 1),tf_dtype) * Lvel_top_x, tf.ones((scale, scale, 1, 1),tf_dtype) * Lvel_top_y,
                          tf.ones((scale, scale, 1, 1),tf_dtype) * Lvel_top_z], 3)
    Lvel = tf.concat([Lvel_top, Lvel[:, :, 1:-1, :], Lvel_bottom], 2)

    Lacc_bottom_x = 0.
    Lacc_bottom_y = 0.
    Lacc_bottom_z = 0.
    Lacc_bottom = tf.concat([tf.ones((scale, scale, 1, 1),tf_dtype) * Lacc_bottom_x, tf.ones((scale, scale, 1, 1),tf_dtype) * Lacc_bottom_y,
                             tf.ones((scale, scale, 1, 1),tf_dtype) * Lacc_bottom_z], 3)
    Lacc_top_x = 0.
    Lacc_top_y = 0.
    Lacc_top_z = -0.5
    Lacc_top = tf.concat([tf.ones((scale, scale, 1, 1),tf_dtype) * Lacc_top_x, tf.ones((scale, scale, 1, 1),tf_dtype) * Lacc_top_y,
                          tf.ones((scale, scale, 1, 1),tf_dtype) * Lacc_top_z], 3)
    Lacc = tf.concat([Lacc_top, Lacc[:, :, 1:-1, :], Lacc_bottom], 2)

    return Lvel, Lacc






if __name__ == '__main__':
    tf_dtype = tf.float64
    num_particle_x = num_particle_y = num_particle_z = scale = 10
    ndim = 3  # physical spatial dimension of the problem
    t_step = 1e-9  # time steps of the simulation
    steps = 500  # number of steps for simulation
    Mass = 0.01**2 * 4.43E2 / 1000
    len = 0.01 # physical size of the problem
    beta = 1e11 # slop of sigmoid

    # Positions = tf.placeholder(tf.float32, (scale, scale, scale, ndim)) # position x
    grid = np.linspace(0.0005, 0.0095, scale)#np.linspace(0, len, scale)
    Positions = tf.convert_to_tensor(
        np.asarray([[grid[i], grid[j], grid[k]] for i in range(scale) for j in range(scale) for k in range(scale)]
                   ).reshape(scale, scale, scale, ndim), dtype=tf_dtype)

    # Defect pattern: 1-no defect, 0-defect
    Defects = tf.convert_to_tensor(
        np.asarray([1. for i in range(scale) for j in range(scale)]
                   ).reshape(scale, scale, 1), dtype=tf_dtype)  # Default defect pattern on the bottom surface of the upper piece

    # Lvel, Lacc = apply_bc(np.zeros((scale, scale, scale, ndim)), np.zeros((scale, scale, scale, ndim)), scale)
    Lvel = tf.Variable(np.zeros((scale, scale, scale, ndim)), tf_dtype)
    Lacc = tf.Variable(np.zeros((scale, scale, scale, ndim)), tf_dtype)


    # Lvel = tf.placeholder(tf.float32, (scale, scale, scale, ndim))  # velocity \dot x
    # Lacc = tf.placeholder(tf.float32, (scale, scale, scale, ndim))  # accelration \dot \dot x
    # stretch = tf.placeholder(tf.float32, (scale, scale, scale, 3**ndim-9))  # threshold on displacement before crack
    # bondsign = tf.placeholder(tf.float32, (scale, scale, scale, 3**ndim-9))  # inital crack

    Tv, Kn, stretch, bondsign = get_const_tensors(scale=scale, dim=ndim, defect=Defects)
    dxy_orig, distance_orig = get_distance(Positions)
    distance_old = distance_orig
    bondsign_old = bondsign
    Positions_old = Positions
    # pos_his = []
    # netF = []

    ## training starts ###
    FLAGS = tf.app.flags.FLAGS
    tfconfig = tf.ConfigProto(
        allow_soft_placement=True,
        log_device_placement=True,
    )
    tfconfig.gpu_options.allow_growth = True
    sess = tf.Session(config=tfconfig)
    init = tf.global_variables_initializer()
    sess.run(init)

    np.savetxt('Position_pred_{}.txt'.format(0),np.squeeze(sess.run(Positions_old)).transpose(2, 1, 0, 3).reshape(-1, 3).tolist())
    np.savetxt('Lvel_pred_{}.txt'.format(0),np.squeeze(sess.run(Lvel)).transpose(2, 1, 0, 3).reshape(-1, 3).tolist())
    np.savetxt('Lacc_pred_{}.txt'.format(0),np.squeeze(sess.run(Lacc)).transpose(2, 1, 0, 3).reshape(-1, 3).tolist())
    # np.savetxt('dL_pred_{}.txt'.format(0), np.squeeze(sess.run()).transpose(2, 1, 0, 3).reshape(-1, 3).tolist())
    # np.savetxt('netF_pred_{}.txt'.format(0), np.squeeze(sess.run(netF)).transpose(2, 1, 0, 3).reshape(-1, 3).tolist())
    for step_i in range(steps):
        bondsign_interface = tf.stack([bondsign_old[:,:,4,i] for i in [5,11,13,15,17]])
        bondsign_interface_val = sess.run(bondsign_interface)

        Positions_new = Positions_old + Lvel * t_step + Lacc * t_step ** 2 / 2.  # 10x10x10x3
        Lvel = Lvel + Lacc * t_step /2.0 #10x10x10x3

        np.savetxt('Position_pred_{}.txt'.format(step_i+1), np.squeeze(sess.run(Positions_new)).transpose(2, 1, 0, 3).reshape(-1, 3).tolist())
        np.savetxt('Lvel_pred_{}.txt'.format(step_i+1), np.squeeze(sess.run(Lvel)).transpose(2, 1, 0, 3).reshape(-1, 3).tolist())
        np.savetxt('Lacc_pred_{}.txt'.format(step_i+1), np.squeeze(sess.run(Lacc)).transpose(2, 1, 0, 3).reshape(-1, 3).tolist())
        dxy_new, distance_new = get_distance(Positions_new)

        dL, dL_total, TdL_total, bondsign_new, dbg_val_f2x = f2x(distance_new, distance_orig, stretch, bondsign_old, Tv)
        np.savetxt('dL_total_pred_{}.txt'.format(step_i+1), np.squeeze(sess.run(dL_total)).transpose(2, 1, 0).reshape(-1).tolist())
        np.savetxt('TdL_total_pred_{}.txt'.format(step_i+1), np.squeeze(sess.run(dL_total)).transpose(2, 1, 0).reshape(-1).tolist())

        netF, netF_top, dbg_val_x2f = x2f(Kn, dL, dL_total, Tv, TdL_total, distance_new, dxy_new, bondsign)
        np.savetxt('netF_pred_{}.txt'.format(step_i+1), np.squeeze(sess.run(netF)).transpose(2, 1, 0,3).reshape(-1).tolist())

        Lacc = netF / Mass #10x10x10x3
        Lvel = Lvel + Lacc * t_step /2.0 #10x10x10x3
        Lvel, Lacc = apply_bc(Lvel, Lacc, scale)

        # pos_his += [Positions_new]
        print('done')
        bondsign_old = bondsign_new
    # tf.gradients(netF, Positions)



