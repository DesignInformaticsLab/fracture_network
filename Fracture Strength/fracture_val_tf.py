import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
import tensorflow as tf

def Initialization(index_fil):
    delta = (Box[0, 1] - Box[0, 0]) / particles_per_row
    Radius = delta / 2.0
    rows = np.int32(np.round((Box[1, 1] - Box[1, 0]) / delta))
    nparticle = np.int32(particles_per_row * rows)
    # clculate the position of each particle
    Positions = np.zeros([nparticle, ndim])

    k = 0
    for j in range(rows):
        y = Box[1, 0] + delta * (j - 0) + Radius
        for i in range(particles_per_row):
            x = Box[0, 0] + delta * (i - 0) + Radius
            if k <= nparticle - 1:
                Positions[k, 0] = x
                Positions[k, 1] = y
            k = k + 1
    j = 0
    k = 0
    m = 0
    n = 0

    for i in range(nparticle):
        if (Positions[i, 1] >= Box[1, 1] - 2. * Radius):
            j = j + 1

        if (Positions[i, 1] <= Box[1, 0] + 2. * Radius):
            k = k + 1

        if (Positions[i, 0] <= Box[0, 0] + 2. * Radius):
            m = m + 1

        if (Positions[i, 0] >= Box[0, 1] - 2. * Radius):
            n = n + 1

    Top = np.zeros([j])
    Bottom = np.zeros([k])
    Left = np.zeros([m])
    Right = np.zeros([n])

    j = 0
    k = 0
    m = 0
    n = 0

    for i in range(nparticle):
        if (Positions[i, 1] >= Box[1, 1] - 2. * Radius):
            Top[j] = i
            j = j + 1

        if (Positions[i, 1] <= Box[1, 0] + 2. * Radius):
            Bottom[k] = i
            k = k + 1

        if (Positions[i, 0] <= Box[0, 0] + 2. * Radius):
            Left[m] = i
            m = m + 1

        if (Positions[i, 0] >= Box[0, 1] - 2. * Radius):
            Right[n] = i
            n = n + 1

            # read in the phase information
    phase01 = np.zeros([nparticle])
    #     WB=sio.loadmat('alloy_mat/WB_sm.mat')['WB_sm'][index_fil-1]
    WB = sio.loadmat('alloy_mat/filter_1000')['filter'].T[index_fil - 1]
    WB = (WB.reshape(10, 10).T).reshape(-1)
    Phase01 = WB

    initialP = np.zeros([nparticle, ndim])
    distance = np.zeros([nparticle, nneighbors])
    origindistance = np.zeros([nparticle, nneighbors])
    dL = np.zeros([nparticle, nneighbors])
    dL_total = np.zeros([nparticle, 2])
    TdL_total = np.zeros([nparticle, 2])
    neighbors = np.zeros([nparticle, nneighbors])
    nsign = np.zeros([nparticle, nneighbors])
    NB = np.zeros([nparticle])
    bondsign = np.ones([nparticle, nneighbors])
    Stretch = np.zeros([nparticle, nneighbors])
    Kn = np.zeros([nparticle, nneighbors])
    Tv = np.zeros([nparticle, nneighbors])
    netF = np.zeros([nparticle, ndim])
    Lvel = np.zeros([nparticle, ndim])
    Lacc = np.zeros([nparticle, ndim])

    return Phase01, initialP, distance, origindistance, dL, dL_total, \
           TdL_total, neighbors, nsign, NB, bondsign, Stretch, Kn, Tv, netF, Lvel, \
           Lacc, Positions, Top, Bottom, Left, Right, delta, Radius, rows, nparticle


def Searchneighbor(Phase01, initialP, distance, origindistance, dL, dL_total,
                   TdL_total, neighbors, nsign, NB, bondsign, Stretch, Kn, Tv, netF, Lvel, Lacc, Positions):
    for i in range(np.int32(nparticle)):
        #         print('particle number=',i+1)
        index = -1
        for j in range(np.int32(nparticle)):
            dis = np.sqrt((Positions[j, 0] - Positions[i, 0]) ** 2 + (Positions[j, 1] - Positions[i, 1]) ** 2)
            if (j != i and dis < 2.01 * Radius):
                index = index + 1
                neighbors[i, index] = j + 1
                nsign[i, index] = 1
                origindistance[i, index] = dis

                if (Phase01[i] == 0 and Phase01[j] == 0):
                    Kn[i, index] = Kn01
                    Tv[i, index] = Tv0
                    Stretch[i, index] = (1.2 * sigY / (2. * Kn01) + 1.) * dis
                elif (Phase01[i] == 1 and Phase01[j] == 1):
                    Kn[i, index] = Kn11
                    Tv[i, index] = Tv1
                    Stretch[i, index] = (1.2 * sigY / (2. * Kn11) + 1.) * dis
                elif (Phase01[i] == 2 and Phase01[j] == 2):
                    Kn[i, index] = Kn21
                    Tv[i, index] = Tv2
                    Stretch[i, index] = (1.2 * sigY / (2. * Kn21) + 1.) * dis
                elif ((Phase01[i] == 0 and Phase01[j] == 1) or (Phase01[i] == 1 and Phase01[j] == 0)):
                    Kn[i, index] = 2. * (Kn01 * Kn11 / (Kn01 + Kn11))
                    if (Tv0 == 0 and Tv1 == 0):
                        Tv[i, index] == 0
                    else:
                        Tv[i, index] = 2. * (Tv0 * Tv1 / (Tv0 + Tv1))
                    Stretch[i, index] = (0.6 * sigY / (2. * 2. * (Kn01 * Kn11 / (Kn01 + Kn11))) + 1.) * dis

                elif ((Phase01[i] == 0 and Phase01[j] == 2) or (Phase01[i] == 2 and Phase01[j] == 0)):
                    Kn[i, index] = 2. * (Kn01 * Kn21 / (Kn01 + Kn21))
                    if (Tv0 == 0 and Tv2 == 0):
                        Tv[i, index] = 0
                    else:
                        Tv[i, index] = 2. * (Tv0 * Tv2 / (Tv0 + Tv2))
                    Stretch[i, index] = (0.6 * sigY / (2. * 2. * (Kn01 * Kn21 / (Kn01 + Kn21))) + 1.) * dis

                elif ((Phase01[i] == 1 and Phase01[j] == 2) or (Phase01[i] == 2 and Phase01[j] == 1)):
                    Kn[i, index] = 2. * (Kn11 * Kn21 / (Kn11 + Kn21))
                    if (Tv1 == 0 and Tv2 == 0):
                        Tv[i, index] = 0
                    else:
                        Tv[i, index] = 2. * (Tv1 * Tv2 / (Tv1 + Tv2))

                    Stretch[i, index] = (0.6 * sigY / (2. * 2. * (Kn11 * Kn21 / (Kn11 + Kn21))) + 1.) * dis
            elif (dis < 2.01 * np.sqrt(2.) * Radius and dis > 2.01 * Radius):
                index = index + 1
                neighbors[i, index] = j + 1
                nsign[i, index] = 2
                origindistance[i, index] = dis

                if (Phase01[i] == 0 and Phase01[j] == 0):
                    Kn[i, index] = Kn02
                    Tv[i, index] = Tv0
                    Stretch[i, index] = (1.2 / np.sqrt(2) * sigY / (2. * Kn01) + 1.) * dis
                elif (Phase01[i] == 1 and Phase01[j] == 1):
                    Kn[i, index] = Kn12
                    Tv[i, index] = Tv1
                    Stretch[i, index] = (1.2 / np.sqrt(2) * sigY / (2. * Kn11) + 1.) * dis
                elif (Phase01[i] == 2 and Phase01[j] == 2):
                    Kn[i, index] = Kn22
                    Tv[i, index] = Tv2
                    Stretch[i, index] = (1.2 / np.sqrt(2) * sigY / (2. * Kn21) + 1.) * dis
                elif ((Phase01[i] == 0 and Phase01[j] == 1) or (Phase01[i] == 1 and Phase01[j] == 0)):
                    Kn[i, index] = 2. * (Kn02 * Kn12 / (Kn02 + Kn12))
                    if (Tv0 == 0 and Tv1 == 0):
                        Tv[i, index] == 0
                    else:
                        Tv[i, index] = 2. * (Tv0 * Tv1 / (Tv0 + Tv1))
                    Stretch[i, index] = (0.6 / np.sqrt(2) * sigY / (2. * 2. * (Kn02 * Kn12 / (Kn02 + Kn12))) + 1.) * dis

                elif ((Phase01[i] == 0 and Phase01[j] == 2) or (Phase01[i] == 2 and Phase01[j] == 0)):
                    Kn[i, index] = 2. * (Kn02 * Kn22 / (Kn02 + Kn22))
                    if (Tv0 == 0 and Tv2 == 0):
                        Tv[i, index] = 0
                    else:
                        Tv[i, index] = 2. * (Tv0 * Tv2 / (Tv0 + Tv2))
                    Stretch[i, index] = (0.6 / np.sqrt(2) * sigY / (2. * 2. * (Kn02 * Kn22 / (Kn02 + Kn22))) + 1.) * dis

                elif ((Phase01[i] == 2 and Phase01[j] == 1) or (Phase01[i] == 1 and Phase01[j] == 2)):
                    Kn[i, index] = 2. * (Kn22 * Kn12 / (Kn22 + Kn12))
                    if (Tv2 == 0 and Tv1 == 0):
                        Tv[i, index] = 0
                    else:
                        Tv[i, index] = 2. * (Tv2 * Tv1 / (Tv2 + Tv1))

                    Stretch[i, index] = (0.6 / SQRT(2.) * sigY / (2. * 2. * (Kn22 * Kn12 / (Kn22 + Kn12))) + 1.) * dis
        NB[i] = index + 1

    return neighbors, nsign, origindistance, Kn, Tv, Stretch, NB


def Update(NB, Positions, neighbors, Stretch, origindistance, nsign, bondsign):
    #     dL = 0.
    #     dL_total = 0.
    #     TdL_total = 0.
    #     distance = 0.

    distance = np.zeros([nparticle, nneighbors])
    dL = np.zeros([nparticle, nneighbors])
    dL_total = np.zeros([nparticle, 2])
    TdL_total = np.zeros([nparticle, 2])

    for i in range(nparticle):
        for j in range(np.int32(NB[i])):
            distance[i, j] = np.sqrt((Positions[i, 0] - Positions[np.int32(neighbors[i, j] - 1), 0]) ** 2 +
                                     (Positions[i, 1] - Positions[np.int32(neighbors[i, j] - 1), 1]) ** 2)
            if (nsign[i, j] == 1):
                if (bondsign[i, j] == 1):
                    if (distance[i, j] <= Stretch[i, j]):
                        dL[i, j] = distance[i, j] - origindistance[i, j]
                    else:
                        bondsign[i, j] = 0
                elif (bondsign[i, j] == 0 and distance[i, j] < origindistance[i, j]):
                    dL[i, j] = distance[i, j] - origindistance[i, j]
                dL_total[i, 0] = dL_total[i, 0] + dL[i, j]
                TdL_total[i, 0] = TdL_total[i, 0] + dL[i, j] * Tv[i, j]
            elif (nsign[i, j] == 2):
                if (bondsign[i, j] == 1):
                    if (distance[i, j] <= Stretch[i, j]):
                        dL[i, j] = distance[i, j] - origindistance[i, j]
                    else:
                        bondsign[i, j] = 0
                elif (bondsign[i, j] == 0 and distance[i, j] < origindistance[i, j]):
                    dL[i, j] = distance[i, j] - origindistance[i, j]

                dL_total[i, 1] = dL_total[i, 1] + dL[i, j]
                TdL_total[i, 1] = TdL_total[i, 1] + dL[i, j] * Tv[i, j]

    return bondsign, dL, dL_total, TdL_total, distance


def Netinteraction(Positions, neighbors, nsign, dL, dL_total, TdL_total, Kn, Tv):
    #     netF=0
    netF = np.zeros([nparticle, ndim])
    for i in range(nparticle):
        for j in range(np.int32(NB[i])):
            dx = Positions[np.int32(neighbors[i, j] - 1), 0] - Positions[i, 0]
            dy = Positions[np.int32(neighbors[i, j] - 1), 1] - Positions[i, 1]
            if (nsign[i, j] == 1):
                if (bondsign[i, j] == 0 and dL[i, j] >= 0):
                    f = 0.
                else:
                    f = 2. * Kn[i, j] * dL[i, j] + 1. / 2. * Tv[i, j] * (
                    dL_total[i, 0] + dL_total[np.int32(neighbors[i, j] - 1), 0]) + \
                        1. / 2. * (TdL_total[i, 0] + TdL_total[np.int32(neighbors[i, j] - 1), 0])
            elif (nsign[i, j] == 2):
                if (bondsign[i, j] == 0 and dL[i, j] >= 0.):
                    f = 0.
                else:
                    f = 2. * Kn[i, j] * dL[i, j] + 1. / 2. * Tv[i, j] * (
                    dL_total[i, 1] + dL_total[np.int32(neighbors[i, j] - 1), 1]) + \
                        1. / 2. * (TdL_total[i, 1] + TdL_total[np.int32(neighbors[i, j] - 1), 1])

            netF[i, 0] = netF[i, 0] + dx * f / distance[i, j]
            netF[i, 1] = netF[i, 1] + dy * f / distance[i, j]

    return netF



#### MAIN ####
rho = np.float32(4.43e3)
PI = np.float32(3.1415926)
sigY=np.float32(9.5e8)
E0=np.float32(1.04e11)
mu0=np.float32(0.32)
E1=np.float32(1.15e11)
mu1 = np.float32(0.33)
E2=np.float32(1.15e10)
mu2=np.float32(0.33)

ndim = np.int32(2)
Box = np.array([[0.0,0.01],[0.0,0.01]],'float32')
particles_per_row = np.int32(10)
nneighbors = np.int(8)

for index_fil in range(1, 2):
    # plane strain
    Kn01 = E0 / (2. * (1. + mu0))
    Kn02 = E0 / (4. * (1. + mu0))
    Tv0 = E0 * (4. * mu0 - 1.) / (24. * (1. + mu0) * (1. - 2. * mu0))
    Kn11 = E1 / (2. * (1. + mu1))
    Kn12 = E1 / (4. * (1. + mu1))
    Tv1 = E1 * (4. * mu1 - 1.) / (24. * (1. + mu1) * (1. - 2. * mu1))

    Kn21 = E2 / (2. * (1. + mu2))
    Kn22 = E2 / (4. * (1. + mu2))
    Tv2 = E2 * (4. * mu2 - 1.) / (24. * (1. + mu2) * (1. - 2. * mu2))

    Phase01, initialP, distance, origindistance, dL, dL_total, \
    TdL_total, neighbors, nsign, NB, bondsign, Stretch, Kn, Tv, netF, Lvel, \
    Lacc, Positions, Top, Bottom, Left, Right, delta, Radius, rows, nparticle = Initialization(index_fil)

    initialP = Positions

    Mass = (Box[0, 1] - Box[0, 0]) * (Box[1, 1] - Box[1, 0]) * rho / nparticle

    neighbors, nsign, origindistance, Kn, Tv, Stretch, NB = Searchneighbor(Phase01, initialP, distance, origindistance,
                                                                           dL, dL_total,
                                                                           TdL_total, neighbors, nsign, NB, bondsign,
                                                                           Stretch, Kn, Tv, netF, Lvel, Lacc, Positions)

    t_end = 1.0
    t_start = 0.
    t_step = 1e-9
    steps = np.round((t_end - t_start) / t_step)

    # main integration part
    MaxF1 = 0
    MaxF2 = 0

    # System evolution
    Lvel[np.int32(Bottom.tolist()), 1] = 0
    Lvel[np.int32(Top.tolist()), 1] = 0
    Lvel[np.int32(Left.tolist()), 0] = -0.5
    Lvel[np.int32(Right.tolist()), 0] = 0.5

    # changing status
    index_status = 1

    Bforce = []

    for t in range(np.int32(steps)):  # np.int32(steps)
        if t % 100 == 0:
            print('processing step number=', t)
        # print('total bond=',np.sum(bondsign))
        Positions = Positions + Lvel * t_step + Lacc * t_step ** 2 / 2.

        bondsign, dL, dL_total, TdL_total, distance = Update(NB, Positions, neighbors, Stretch, origindistance, nsign,
                                                             bondsign)

        if t == 0:
            bondsign_last_Total_value = np.sum(bondsign)

        netF = Netinteraction(Positions, neighbors, nsign, dL, dL_total, TdL_total, Kn, Tv)

        Lacc = netF / Mass

        Lvel = Lvel + Lacc * t_step

        Lvel[np.int32(Top.tolist()), 1] = 0.
        Lacc[np.int32(Top.tolist()), 1] = 0.

        Lvel[np.int32(Bottom.tolist()), 1] = 0.
        Lacc[np.int32(Bottom.tolist()), 1] = 0.

        Lvel[np.int32(Left.tolist()), 0] = -0.5
        Lacc[np.int32(Left.tolist()), 0] = 0.

        Lvel[np.int32(Right.tolist()), 0] = 0.5
        Lacc[np.int32(Right.tolist()), 1] = 0.

        Bforce.append(np.sum(netF[np.int32(Left.tolist()), 0].reshape(-1)))

        if np.sum(bondsign) != bondsign_last_Total_value:
            print('bond status change, former=', bondsign_last_Total_value)
            print('bond status change, now=', np.sum(bondsign))
            sio.savemat('./status/WB_{}/Lvel_{}.mat'.format(index_fil, index_status), {'Lvel': Lvel})
            sio.savemat('./status/WB_{}/Bforce_{}.mat'.format(index_fil, index_status), {'Bforce': Bforce[t]})
            index_status = index_status + 1
            bondsign_last_Total_value = np.sum(bondsign)

        if (abs(np.sum(netF[np.int32(Left.tolist()), 0])) > MaxF1):
            MaxF1 = abs(np.sum(netF[np.int32(Left.tolist()), 0]))

        if (abs(np.sum(netF[np.int32(Top.tolist()), 1])) > MaxF2):
            MaxF2 = abs(np.sum(netF[np.int32(Top.tolist()), 1]))

        if (abs(np.sum(netF[np.int32(Left.tolist()), 0])) < MaxF1 / 10. or abs(
                np.sum(netF[np.int32(Top.tolist()), 1])) < MaxF2 / 10.):
            break

    crack = []
    for i in range(nparticle):
        if np.count_nonzero(bondsign[i, :] == 0) != 0:
            damage = 4
        else:
            damage = Phase01[i]

        crack.append(damage)
    sio.savemat('./status/Wb_{}/crack_{}.mat'.format(index_fil, index_fil), {'crack': crack})

    # save Bforce
    sio.savemat('./status/WB_{}/Bforce_total_{}.mat'.format(index_fil, index_fil), {'Bforce': Bforce})

    print('complished filter number =', index_fil)


