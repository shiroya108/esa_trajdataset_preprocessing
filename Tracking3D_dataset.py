import numpy as np
import pandas as pd
import quaternion
import scipy.interpolate
import h5py
import os
from os import path as osp
import json

# from keras.utils import Sequence
# import tr_env


def interpolate_3dvector_linear(input, input_timestamp, output_timestamp):
    assert input.shape[0] == input_timestamp.shape[0]
    func = scipy.interpolate.interp1d(input_timestamp, input, axis=0)
    interpolated = func(output_timestamp)
    return interpolated


def load_euroc_mav_dataset(imu_data_filename, gt_data_filename):
    gt_data = pd.read_csv(gt_data_filename).values    
    imu_data = pd.read_csv(imu_data_filename).values

    gyro_data = interpolate_3dvector_linear(imu_data[:, 1:4], imu_data[:, 0], gt_data[:, 0])
    acc_data = interpolate_3dvector_linear(imu_data[:, 4:7], imu_data[:, 0], gt_data[:, 0])
    pos_data = gt_data[:, 1:4]
    ori_data = gt_data[:, 4:8]

    return gyro_data, acc_data, pos_data, ori_data


def load_oxiod_dataset(imu_data_filename, gt_data_filename):
    imu_data = pd.read_csv(imu_data_filename).values
    gt_data = pd.read_csv(gt_data_filename).values

    imu_data = imu_data[1200:-300]
    gt_data = gt_data[1200:-300]

    gyro_data = imu_data[:, 4:7]
    acc_data = imu_data[:, 10:13]
    mag_data = imu_data[:,13:16]
    
    pos_data = gt_data[:, 2:5]
    ori_data = np.concatenate([gt_data[:, 8:9], gt_data[:, 5:8]], axis=1)

    return gyro_data, acc_data, mag_data, pos_data, ori_data

def load_ronin_dataset(imu_data_filename, info_json_filename):

    with open(info_json_filename, 'r') as f:
       info = json.load(f)

    with h5py.File(imu_data_filename,'r') as f:
        gyro_uncalib = f['synced']['gyro_uncalib']
        acce_uncalib = f['synced']['acce']
        gyro_data = gyro_uncalib - np.array(info['imu_init_gyro_bias'])
        acc_data = np.array(info['imu_acce_scale']) * (acce_uncalib - np.array(info['imu_acce_bias']))
        pos_data = np.array(f['pose']['tango_pos'])
        # init_tango_ori = quaternion.quaternion(*f['pose/tango_ori'][0])
        ori_data =  np.array(f['pose']['tango_ori'])
        mag_data =  np.array(f['synced']['magnet'])


        # pos_data = np.array(f['pose']['tango_pos'])
        # ori_data =  np.array(f['pose']['tango_ori'])
        # acc_data =  np.array(f['synced']['acce'])
        # gyro_data =  np.array(f['synced']['gyro'])
        # mag_data =  np.array(f['synced']['magnet'])

    length = min(gyro_data.shape[0], acc_data.shape[0], mag_data.shape[0], pos_data.shape[0], ori_data.shape[0])

    # length = 30000 if length>30000 else length

    gyro_data = gyro_data[0:length, :]
    acc_data =  acc_data[0:length, :]
    mag_data =  mag_data[0:length, :]
    pos_data =  pos_data[0:length, :]
    ori_data =  ori_data[0:length, :] 

    return gyro_data, acc_data, mag_data, pos_data, ori_data
    
def load_our_datasets(imu_data_filename, gt_data_filename):
    imu_data = pd.read_csv(imu_data_filename).values
    gt_data = pd.read_csv(gt_data_filename).values
    
    # imu_data = imu_data[200:-200]
    # gt_data = gt_data[200:-200]

    acc_data = imu_data[:, 0:3]
    gyro_data = imu_data[:, 3:6]
    mag_data = imu_data[:,6:9]
    #--0~2:center position 3~6:fixed quaternion 7~9:euler 10~13:common quaternion
    pos_data = gt_data[:, 0:3]
    ori_data = gt_data[:, 10:14]
    # ori_data = gt_data[:, 3:7]

    return gyro_data, acc_data, mag_data, pos_data, ori_data

def load_hsh_datasets(imu_data_filename, gt_data_filename):
    imu_data = pd.read_csv(imu_data_filename).values
    gt_data = pd.read_csv(gt_data_filename).values
    
    # imu_data = imu_data[200:-200]
    # gt_data = gt_data[200:-200]

    acc_data = imu_data[:, 0:3]
    gyro_data = imu_data[:, 3:6]
    mag_data = imu_data[:,6:9]
    #--0~2:center position 3~6:fixed quaternion 7~9:euler 10~13:common quaternion
    pos_data = gt_data[:, 0:3]
    ori_data = gt_data[:, 10:14]
    # ori_data = gt_data[:, 3:7]

    return gyro_data, acc_data, mag_data, pos_data, ori_data



def load_ridi_dataset(imu_data_filename, gt_data_filename):
    imu_data = pd.read_csv(imu_data_filename).values

    imu_data = imu_data[1200:-300]

    gyro_data = imu_data[:,2:5]
    acc_data = imu_data[:, 5:8]
    mag_data = imu_data[:,14:17]
    
    pos_data = imu_data[:, 17:20]
    ori_data = imu_data[:, 20:24]

    return gyro_data, acc_data, mag_data, pos_data, ori_data


def force_quaternion_uniqueness(q):

    q_data = quaternion.as_float_array(q)

    if np.absolute(q_data[0]) > 1e-05:
        if q_data[0] < 0:
            return -q
        else:
            return q
    elif np.absolute(q_data[1]) > 1e-05:
        if q_data[1] < 0:
            return -q
        else:
            return q
    elif np.absolute(q_data[2]) > 1e-05:
        if q_data[2] < 0:
            return -q
        else:
            return q
    else:
        if q_data[3] < 0:
            return -q
        else:
            return q


def cartesian_to_spherical_coordinates(point_cartesian):
    delta_l = np.linalg.norm(point_cartesian)

    if np.absolute(delta_l) > 1e-05:
        theta = np.arccos(point_cartesian[2] / delta_l)
        psi = np.arctan2(point_cartesian[1], point_cartesian[0])
        return delta_l, theta, psi
    else:
        return 0, 0, 0


def load_dataset_6d_rvec(imu_data_filename, gt_data_filename, window_size=200, stride=10):

    #imu_data = np.genfromtxt(imu_data_filename, delimiter=',')
    #gt_data = np.genfromtxt(gt_data_filename, delimiter=',')
    
    imu_data = pd.read_csv(imu_data_filename).values
    gt_data = pd.read_csv(gt_data_filename).values

    #imu_data = imu_data[1200:-300]
    #gt_data = gt_data[1200:-300]
    
    gyro_acc_data = np.concatenate([imu_data[:, 4:7], imu_data[:, 10:13]], axis=1)
    
    pos_data = gt_data[:, 2:5]
    ori_data = np.concatenate([gt_data[:, 8:9], gt_data[:, 5:8]], axis=1)

    init_q = quaternion.from_float_array(ori_data[window_size//2 - stride//2, :])
    
    init_rvec = np.empty((3, 1))
    cv2.Rodrigues(quaternion.as_rotation_matrix(init_q), init_rvec)

    init_tvec = pos_data[window_size//2 - stride//2, :]

    x = []
    y_delta_rvec = []
    y_delta_tvec = []

    for idx in range(0, gyro_acc_data.shape[0] - window_size - 1, stride):
        x.append(gyro_acc_data[idx + 1 : idx + 1 + window_size, :])

        tvec_a = pos_data[idx + window_size//2 - stride//2, :]
        tvec_b = pos_data[idx + window_size//2 + stride//2, :]

        q_a = quaternion.from_float_array(ori_data[idx + window_size//2 - stride//2, :])
        q_b = quaternion.from_float_array(ori_data[idx + window_size//2 + stride//2, :])

        rmat_a = quaternion.as_rotation_matrix(q_a)
        rmat_b = quaternion.as_rotation_matrix(q_b)

        delta_rmat = np.matmul(rmat_b, rmat_a.T)

        delta_rvec = np.empty((3, 1))
        cv2.Rodrigues(delta_rmat, delta_rvec)

        delta_tvec = tvec_b - np.matmul(delta_rmat, tvec_a.T).T

        y_delta_rvec.append(delta_rvec)
        y_delta_tvec.append(delta_tvec)


    x = np.reshape(x, (len(x), x[0].shape[0], x[0].shape[1]))
    y_delta_rvec = np.reshape(y_delta_rvec, (len(y_delta_rvec), y_delta_rvec[0].shape[0]))
    y_delta_tvec = np.reshape(y_delta_tvec, (len(y_delta_tvec), y_delta_tvec[0].shape[0]))

    return x, [y_delta_rvec, y_delta_tvec], init_rvec, init_tvec


def load_dataset_6d_quat(gyro_data, acc_data, pos_data, ori_data, window_size=200, stride=10):
    #gyro_acc_data = np.concatenate([gyro_data, acc_data], axis=1)

    init_p = pos_data[window_size//2 - stride//2, :]
    init_q = ori_data[window_size//2 - stride//2, :]

    #x = []
    x_gyro = []
    x_acc = []
    y_delta_p = []
    y_delta_q = []

    for idx in range(0, gyro_data.shape[0] - window_size - 1, stride):
        #x.append(gyro_acc_data[idx + 1 : idx + 1 + window_size, :])
        x_gyro.append(gyro_data[idx + 1 : idx + 1 + window_size, :])
        x_acc.append(acc_data[idx + 1 : idx + 1 + window_size, :])

        p_a = pos_data[idx + window_size//2 - stride//2, :]
        p_b = pos_data[idx + window_size//2 + stride//2, :]

        q_a = quaternion.from_float_array(ori_data[idx + window_size//2 - stride//2, :])
        q_b = quaternion.from_float_array(ori_data[idx + window_size//2 + stride//2, :])

        delta_p = np.matmul(quaternion.as_rotation_matrix(q_a).T, (p_b.T - p_a.T)).T

        delta_q = force_quaternion_uniqueness(q_a.conjugate() * q_b)

        y_delta_p.append(delta_p)
        y_delta_q.append(quaternion.as_float_array(delta_q))


    #x = np.reshape(x, (len(x), x[0].shape[0], x[0].shape[1]))
    x_gyro = np.reshape(x_gyro, (len(x_gyro), x_gyro[0].shape[0], x_gyro[0].shape[1]))
    x_acc = np.reshape(x_acc, (len(x_acc), x_acc[0].shape[0], x_acc[0].shape[1]))
    y_delta_p = np.reshape(y_delta_p, (len(y_delta_p), y_delta_p[0].shape[0]))
    y_delta_q = np.reshape(y_delta_q, (len(y_delta_q), y_delta_q[0].shape[0]))

    #return x, [y_delta_p, y_delta_q], init_p, init_q
    return [x_gyro, x_acc], [y_delta_p, y_delta_q], init_p, init_q


def load_dataset_6d_euler(gyro_data, acc_data, ori_data, eular_data, window_size=200, stride=10):

    init_p = pos_data[window_size//2 - stride//2, :]
    init_e = euler_data[window_size//2 - stride//2, :]

    x_gyro = []
    x_acc = []
    y_delta_p = []
    y_delta_q = []
    y_delta_e = []

    for idx in range(0, gyro_data.shape[0] - window_size - 1, stride):
        x_gyro.append(gyro_data[idx + 1 : idx + 1 + window_size, :])
        x_acc.append(acc_data[idx + 1 : idx + 1 + window_size, :])

        p_a = pos_data[idx + window_size//2 - stride//2, :]
        p_b = pos_data[idx + window_size//2 + stride//2, :]

        q_a = quaternion.from_float_array(ori_data[idx + window_size//2 - stride//2, :])
        q_b = quaternion.from_float_array(ori_data[idx + window_size//2 + stride//2, :])
        
        e_a = eular_data[idx + window_size//2 - stride//2, :]
        e_b = eular_data[idx + window_size//2 + stride//2, :]

        delta_p = np.matmul(quaternion.as_rotation_matrix(q_a).T, (p_b.T - p_a.T)).T
        delta_q = force_quaternion_uniqueness(q_a.conjugate() * q_b)
        delta_e = e_b - e_a
        
        y_delta_p.append(delta_p)
        y_delta_e.append(delta_e)
        #y_delta_q.append(quaternion.as_float_array(delta_q))
        
        
    x_gyro = np.reshape(x_gyro, (len(x_gyro), x_gyro[0].shape[0], x_gyro[0].shape[1]))
    x_acc = np.reshape(x_acc, (len(x_acc), x_acc[0].shape[0], x_acc[0].shape[1]))
    y_delta_p = np.reshape(y_delta_p, (len(y_delta_p), y_delta_p[0].shape[0]))
    y_delta_e = np.reshape(y_delta_e, (len(y_delta_e), y_delta_e[0].shape[0]))
    #y_delta_q = np.reshape(y_delta_q, (len(y_delta_q), y_delta_q[0].shape[0]))

    #return x, [y_delta_p, y_delta_q], init_p, init_q
    return [x_gyro, x_acc], [y_delta_p, y_delta_e], init_p, init_e


def load_dataset_9d_quat(gyro_data, acc_data, mag_data, pos_data, ori_data, window_size=200, stride=10):
    init_p = pos_data[window_size//2 - stride//2, :]
    init_q = ori_data[window_size//2 - stride//2, :]

    #x = []
    x_gyro = []
    x_acc = []
    x_mag = []
    y_delta_p = []
    y_delta_q = []
    
    for idx in range(0, gyro_data.shape[0] - window_size - 1, stride):
        #x.append(gyro_acc_data[idx + 1 : idx + 1 + window_size, :])
        x_gyro.append(gyro_data[idx + 1 : idx + 1 + window_size, :])
        x_acc.append(acc_data[idx + 1 : idx + 1 + window_size, :])
        x_mag.append(mag_data[idx + 1 : idx + 1 + window_size, :])

        p_a = pos_data[idx + window_size//2 - stride//2, :]
        p_b = pos_data[idx + window_size//2 + stride//2, :]

        q_a = quaternion.from_float_array(ori_data[idx + window_size//2 - stride//2, :])
        q_b = quaternion.from_float_array(ori_data[idx + window_size//2 + stride//2, :])

        delta_p = np.matmul(quaternion.as_rotation_matrix(q_a).T, (p_b.T - p_a.T)).T

        delta_q = force_quaternion_uniqueness(q_a.conjugate() * q_b)

        y_delta_p.append(delta_p)
        y_delta_q.append(quaternion.as_float_array(delta_q))


    #x = np.reshape(x, (len(x), x[0].shape[0], x[0].shape[1]))
    x_gyro = np.reshape(x_gyro, (len(x_gyro), x_gyro[0].shape[0], x_gyro[0].shape[1]))
    x_acc = np.reshape(x_acc, (len(x_acc), x_acc[0].shape[0], x_acc[0].shape[1]))
    x_mag = np.reshape(x_mag, (len(x_mag), x_mag[0].shape[0], x_mag[0].shape[1]))
    y_delta_p = np.reshape(y_delta_p, (len(y_delta_p), y_delta_p[0].shape[0]))
    y_delta_q = np.reshape(y_delta_q, (len(y_delta_q), y_delta_q[0].shape[0]))

    #return x, [y_delta_p, y_delta_q], init_p, init_q
    return [x_gyro, x_acc, x_mag], [y_delta_p, y_delta_q], init_p, init_q

def load_dataset_9d_quat_hsh(gyro_data, acc_data, mag_data, pos_data, ori_data, window_size=200, stride=10):
    init_p = pos_data[window_size//2 - stride//2, :]
    init_q = ori_data[window_size//2 - stride//2, :]

    #x = []
    x_gyro = []
    x_acc = []
    x_mag = []
    y_delta_p = []
    y_delta_q = []
    
    for idx in range(0, gyro_data.shape[0] - window_size - 1, stride):
        #x.append(gyro_acc_data[idx + 1 : idx + 1 + window_size, :])
        x_gyro.append(gyro_data[idx + 1 : idx + 1 + window_size, :])
        x_acc.append(acc_data[idx + 1 : idx + 1 + window_size, :])
        x_mag.append(mag_data[idx + 1 : idx + 1 + window_size, :])

        p_a = pos_data[idx + window_size//2 - stride//2, :]
        p_b = pos_data[idx + window_size//2 + stride//2  + stride%2, :]

        q_a = quaternion.from_float_array(ori_data[idx + window_size//2 - stride//2, :])
        q_b = quaternion.from_float_array(ori_data[idx + window_size//2 + stride//2 + stride%2, :])

        # delta_p = np.matmul(quaternion.as_rotation_matrix(q_a).T, (p_b.T - p_a.T)).T
        delta_p = p_b - p_a

        delta_q = force_quaternion_uniqueness(q_a.conjugate() * q_b)

        y_delta_p.append(delta_p)
        y_delta_q.append(quaternion.as_float_array(delta_q))

    x_gyro = np.reshape(x_gyro, (len(x_gyro), x_gyro[0].shape[0], x_gyro[0].shape[1]))
    x_acc = np.reshape(x_acc, (len(x_acc), x_acc[0].shape[0], x_acc[0].shape[1]))
    x_mag = np.reshape(x_mag, (len(x_mag), x_mag[0].shape[0], x_mag[0].shape[1]))
    y_delta_p = np.reshape(y_delta_p, (len(y_delta_p), y_delta_p[0].shape[0]))
    y_delta_q = np.reshape(y_delta_q, (len(y_delta_q), y_delta_q[0].shape[0]))

    return [x_gyro, x_acc, x_mag], [y_delta_p, y_delta_q], init_p, init_q

def load_dataset_9d_tango(gyro_data, acc_data, mag_data, pos_data, ori_data, window_size=200, stride=10, samples=0):
    init_p = pos_data[window_size//2 - stride//2, 0:2]
    init_q = ori_data[window_size//2 - stride//2, :]

    #x = []
    x_gyro = []
    x_acc = []
    x_mag = []
    y_delta_p = []
    y_delta_q = []
    

    for idx in range(0, gyro_data.shape[0] - window_size - 1, stride):
        #x.append(gyro_acc_data[idx + 1 : idx + 1 + window_size, :])
        x_gyro.append(gyro_data[idx + 1 : idx + 1 + window_size, :])
        x_acc.append(acc_data[idx + 1 : idx + 1 + window_size, :])
        x_mag.append(mag_data[idx + 1 : idx + 1 + window_size, :])

        p_a = pos_data[idx + window_size//2 - stride//2, 0:2]
        p_b = pos_data[idx + window_size//2 + stride//2, 0:2]

        q_a = quaternion.from_float_array(ori_data[idx + window_size//2 - stride//2, :])
        q_b = quaternion.from_float_array(ori_data[idx + window_size//2 + stride//2, :])

        # delta_p = np.matmul(quaternion.as_rotation_matrix(q_a).T, (p_b.T - p_a.T)).T
        delta_p = p_b - p_a

        delta_q = force_quaternion_uniqueness(q_a.conjugate() * q_b)

        y_delta_p.append(delta_p)
        y_delta_q.append(quaternion.as_float_array(delta_q))


    

    #x = np.reshape(x, (len(x), x[0].shape[0], x[0].shape[1]))
    x_gyro = np.reshape(x_gyro, (len(x_gyro), x_gyro[0].shape[0], x_gyro[0].shape[1]))
    x_acc = np.reshape(x_acc, (len(x_acc), x_acc[0].shape[0], x_acc[0].shape[1]))
    x_mag = np.reshape(x_mag, (len(x_mag), x_mag[0].shape[0], x_mag[0].shape[1]))
    y_delta_p = np.reshape(y_delta_p, (len(y_delta_p), y_delta_p[0].shape[0]))
    y_delta_q = np.reshape(y_delta_q, (len(y_delta_q), y_delta_q[0].shape[0]))


    if samples > 0 and samples < len(y_delta_p):
        x_gyro =  x_gyro[0:samples,:,:]
        x_acc =   x_acc[0:samples,:,:]
        x_mag = x_mag[0:samples,:,:]
        y_delta_p =  y_delta_p[0:samples,:]
        y_delta_q =  y_delta_q[0:samples,:]

    # print(x_gyro.shape)
    # print(y_delta_p.shape)
    

    #return x, [y_delta_p, y_delta_q], init_p, init_q
    return [x_gyro, x_acc, x_mag], [y_delta_p, y_delta_q], init_p, init_q


def load_ronin_9d_quat(gyro_data, acc_data, mag_data, pos_data, ori_data, window_size=200, stride=10):
    init_p = pos_data[window_size//2 - stride//2, :]
    init_q = ori_data[window_size//2 - stride//2, :]

    #x = []
    x_gyro = []
    x_acc = []
    x_mag = []
    y_delta_p = []
    y_delta_q = []
    
    for idx in range(0, gyro_data.shape[0] - window_size - 1, stride):
        #x.append(gyro_acc_data[idx + 1 : idx + 1 + window_size, :])
        x_gyro.append(gyro_data[idx + 1 : idx + 1 + window_size, :])
        x_acc.append(acc_data[idx + 1 : idx + 1 + window_size, :])
        x_mag.append(mag_data[idx + 1 : idx + 1 + window_size, :])

        p_a = pos_data[idx + window_size//2 - stride//2, :]
        p_b = pos_data[idx + window_size//2 + stride//2, :]

        q_a = quaternion.from_float_array(ori_data[idx + window_size//2 - stride//2, :])
        q_b = quaternion.from_float_array(ori_data[idx + window_size//2 + stride//2, :])

        delta_p = np.matmul(quaternion.as_rotation_matrix(q_a).T, (p_b.T - p_a.T)).T

        delta_q = force_quaternion_uniqueness(q_a.conjugate() * q_b)

        y_delta_p.append(delta_p)
        y_delta_q.append(quaternion.as_float_array(delta_q))


    #x = np.reshape(x, (len(x), x[0].shape[0], x[0].shape[1]))
    x_gyro = np.reshape(x_gyro, (len(x_gyro), x_gyro[0].shape[0], x_gyro[0].shape[1]))
    x_acc = np.reshape(x_acc, (len(x_acc), x_acc[0].shape[0], x_acc[0].shape[1]))
    x_mag = np.reshape(x_mag, (len(x_mag), x_mag[0].shape[0], x_mag[0].shape[1]))
    y_delta_p = np.reshape(y_delta_p, (len(y_delta_p), y_delta_p[0].shape[0]))
    y_delta_q = np.reshape(y_delta_q, (len(y_delta_q), y_delta_q[0].shape[0]))

    #return x, [y_delta_p, y_delta_q], init_p, init_q
    return [x_gyro, x_acc, x_mag], [y_delta_p, y_delta_q], init_p, init_q

def load_oxford_filename():
    imu_data = []
    gt_data = []
    head = tr_env.OXIOD_DATASET_DIR
    
    imu_data.append(head+'/handheld/data5/syn/imu3.csv')
    imu_data.append(head+'/handheld/data2/syn/imu1.csv')
    imu_data.append(head+'/handheld/data2/syn/imu2.csv')
    imu_data.append(head+'/handheld/data5/syn/imu2.csv')
    imu_data.append(head+'/handheld/data3/syn/imu4.csv')
    imu_data.append(head+'/handheld/data4/syn/imu4.csv')
    imu_data.append(head+'/handheld/data4/syn/imu2.csv')
    imu_data.append(head+'/handheld/data1/syn/imu7.csv')
    imu_data.append(head+'/handheld/data5/syn/imu4.csv')
    imu_data.append(head+'/handheld/data4/syn/imu5.csv')
    imu_data.append(head+'/handheld/data1/syn/imu3.csv')
    imu_data.append(head+'/handheld/data3/syn/imu2.csv')
    imu_data.append(head+'/handheld/data2/syn/imu3.csv')
    imu_data.append(head+'/handheld/data1/syn/imu1.csv')
    imu_data.append(head+'/handheld/data3/syn/imu3.csv')
    imu_data.append(head+'/handheld/data3/syn/imu5.csv')
    imu_data.append(head+'/handheld/data1/syn/imu4.csv')
    
    imu_data.append(head+'/slow walking/data1/syn/imu1.csv')
    imu_data.append(head+'/running/data1/syn/imu1.csv')
    imu_data.append(head+'/handbag/data1/syn/imu1.csv')
    imu_data.append(head+'/slow walking/data1/syn/imu2.csv')
    imu_data.append(head+'/running/data1/syn/imu2.csv')
    imu_data.append(head+'/handbag/data1/syn/imu2.csv')
    imu_data.append(head+'/slow walking/data1/syn/imu3.csv')
    imu_data.append(head+'/running/data1/syn/imu3.csv')
    imu_data.append(head+'/handbag/data1/syn/imu3.csv')
    imu_data.append(head+'/slow walking/data1/syn/imu4.csv')
    imu_data.append(head+'/running/data1/syn/imu4.csv')
    imu_data.append(head+'/handbag/data2/syn/imu1.csv')
    imu_data.append(head+'/slow walking/data1/syn/imu5.csv')
    imu_data.append(head+'/running/data1/syn/imu5.csv')
    imu_data.append(head+'/handbag/data2/syn/imu2.csv')
    imu_data.append(head+'/slow walking/data1/syn/imu6.csv')
    imu_data.append(head+'/running/data1/syn/imu6.csv')
    imu_data.append(head+'/handbag/data2/syn/imu3.csv')
    
    #imu_data.append('Oxford Inertial Odometry Dataset/Stop/imu1.csv')
    #imu_data.append('Oxford Inertial Odometry Dataset/Stop/imu2.csv')

    gt_data.append(head+'/handheld/data5/syn/vi3.csv')
    gt_data.append(head+'/handheld/data2/syn/vi1.csv')
    gt_data.append(head+'/handheld/data2/syn/vi2.csv')
    gt_data.append(head+'/handheld/data5/syn/vi2.csv')
    gt_data.append(head+'/handheld/data3/syn/vi4.csv')
    gt_data.append(head+'/handheld/data4/syn/vi4.csv')
    gt_data.append(head+'/handheld/data4/syn/vi2.csv')
    gt_data.append(head+'/handheld/data1/syn/vi7.csv')
    gt_data.append(head+'/handheld/data5/syn/vi4.csv')
    gt_data.append(head+'/handheld/data4/syn/vi5.csv')
    gt_data.append(head+'/handheld/data1/syn/vi3.csv')
    gt_data.append(head+'/handheld/data3/syn/vi2.csv')
    gt_data.append(head+'/handheld/data2/syn/vi3.csv')
    gt_data.append(head+'/handheld/data1/syn/vi1.csv')
    gt_data.append(head+'/handheld/data3/syn/vi3.csv')
    gt_data.append(head+'/handheld/data3/syn/vi5.csv')
    gt_data.append(head+'/handheld/data1/syn/vi4.csv')
    
    gt_data.append(head+'/slow walking/data1/syn/vi1.csv')
    gt_data.append(head+'/running/data1/syn/vi1.csv')
    gt_data.append(head+'/handbag/data1/syn/vi1.csv')
    gt_data.append(head+'/slow walking/data1/syn/vi2.csv')
    gt_data.append(head+'/running/data1/syn/vi2.csv')
    gt_data.append(head+'/handbag/data1/syn/vi2.csv')
    gt_data.append(head+'/slow walking/data1/syn/vi3.csv')
    gt_data.append(head+'/running/data1/syn/vi3.csv')
    gt_data.append(head+'/handbag/data1/syn/vi3.csv')
    gt_data.append(head+'/slow walking/data1/syn/vi4.csv')
    gt_data.append(head+'/running/data1/syn/vi4.csv')
    gt_data.append(head+'/handbag/data2/syn/vi1.csv')
    gt_data.append(head+'/slow walking/data1/syn/vi5.csv')
    gt_data.append(head+'/running/data1/syn/vi5.csv')
    gt_data.append(head+'/handbag/data2/syn/vi2.csv')
    gt_data.append(head+'/slow walking/data1/syn/vi6.csv')
    gt_data.append(head+'/running/data1/syn/vi6.csv')
    gt_data.append(head+'/handbag/data2/syn/vi3.csv')
    
    #gt_data.append('Oxford Inertial Odometry Dataset/Stop/vi1.csv')
    #gt_data.append('Oxford Inertial Odometry Dataset/Stop/vi2.csv')
    return imu_data, gt_data
    
def load_our_datasets_filename():
    head = tr_env.OUR_DATASET_DIR
    # head = '.\\..\\Vicon dataset\\Syn data new format/'
    act_class = ['Forward_back/', 'Hand wave/', 'Draw eight/', 'Move cup/', 'Walk/']
    direction = ['Horizontal/', 'Vertical/']
    wave_class = ['Left_to_Right/', 'Up_to_down/']
    speed = ['fast_', 'mid_', 'slow_']
    walk = ['Circle/', 'Random/', 'S/']
    walk_name = ['circle_', 'random_', 's_']
    imu_data = []
    gt_data = []
    num = 3
    
    # IMU dictionary
    for a in range(0, len(act_class)):
      if (a==0):
        for d in range(0, len(direction)):
          for s in range(0, len(speed)):
            for n in range(0, num):
              temp_v = head + act_class[a] + direction[d] + speed[s] + 'vicon_syn_' + str(n+1) + '.csv'
              temp_i = head + act_class[a] + direction[d] + speed[s] + 'imu_syn_' + str(n+1) + '.csv'
              gt_data.append(temp_v)
              imu_data.append(temp_i)
      elif (a==1):
        for w in range(0, len(wave_class)):
          for d in range(0, len(direction)):
            for s in range(0, len(speed)):
              for n in range(0, num):
                temp_v = head + act_class[a] + wave_class[w] + direction[d] + speed[s] + 'vicon_syn_' + str(n+1) + '.csv'
                temp_i = head + act_class[a] + wave_class[w] + direction[d] + speed[s] + 'imu_syn_' + str(n+1) + '.csv'
                gt_data.append(temp_v)
                imu_data.append(temp_i)
      elif (a==2):
        for s in range(0, len(speed)):
          for n in range(0, num):
            temp_v = head + act_class[a] + speed[s] + 'vicon_syn_' + str(n+1) + '.csv'
            temp_i = head + act_class[a] + speed[s] + 'imu_syn_' + str(n+1) + '.csv'
            gt_data.append(temp_v)
            imu_data.append(temp_i)
      elif (a==3):
        for n in range(0, num*5):
          temp_v = head + act_class[a] + 'MCR_vicon_syn_' + str(n+1) + '.csv'
          temp_i = head + act_class[a] + 'MCR_imu_syn_' + str(n+1) + '.csv'
          gt_data.append(temp_v)
          imu_data.append(temp_i)

      elif (a==4):
        for wk in range(0, len(walk)):
          for n in range(0, num*6-1):
            temp_v = head + act_class[a] + walk[wk] + walk_name[wk] + 'vicon_syn_' + str(n+1) + '.csv'
            temp_i = head + act_class[a] + walk[wk] + walk_name[wk] + 'imu_syn_' + str(n+1) + '.csv'
            gt_data.append(temp_v)
            imu_data.append(temp_i)
    
    
    imu_data.append(head+'Stop/stop_imu_syn.csv')
    gt_data.append( head+'Stop/stop_vicon_syn.csv')
    
    imu_data.append(head+'Forward_back/Horizontal/fast_imu_syn_4.csv')
    imu_data.append(head+'Forward_back/Horizontal/mid_imu_syn_4.csv')
    imu_data.append(head+'Forward_back/Horizontal/slow_imu_syn_4.csv')
    gt_data.append( head+'Forward_back/Horizontal/fast_vicon_syn_4.csv')
    gt_data.append( head+'Forward_back/Horizontal/mid_vicon_syn_4.csv')
    gt_data.append( head+'Forward_back/Horizontal/slow_vicon_syn_4.csv')

    return imu_data, gt_data

def load_our_datasets_filename_speeds():
    head = tr_env.OUR_DATASET_DIR
    act_class = ['Forward_back/', 'Hand wave/', 'Draw eight/', 'Move cup/', 'Walk/']
    direction = ['Horizontal/', 'Vertical/']
    wave_class = ['Left_to_Right/', 'Up_to_down/']
    speed = ['fast_', 'mid_', 'slow_']
    walk = ['Circle/', 'Random/', 'S/']
    walk_name = ['circle_', 'random_', 's_']
    imu_data_fast = []
    gt_data_fast = []
    imu_data_mid = []
    gt_data_mid = []
    imu_data_slow = []
    gt_data_slow = []
    num = 3
    
    # IMU dictionary
    for a in range(0, len(act_class)):
      if (a==0):
        for d in range(0, len(direction)):
            for n in range(0, num):
              temp_v = head + act_class[a] + direction[d] + 'fast_' + 'vicon_syn_' + str(n+1) + '.csv'
              temp_i = head + act_class[a] + direction[d] + 'fast_' + 'imu_syn_' + str(n+1) + '.csv'
              gt_data_fast.append(temp_v)
              imu_data_fast.append(temp_i)
              temp_v = head + act_class[a] + direction[d] + 'mid_' + 'vicon_syn_' + str(n+1) + '.csv'
              temp_i = head + act_class[a] + direction[d] + 'mid_' + 'imu_syn_' + str(n+1) + '.csv'
              gt_data_mid.append(temp_v)
              imu_data_mid.append(temp_i)
              temp_v = head + act_class[a] + direction[d] + 'slow_' + 'vicon_syn_' + str(n+1) + '.csv'
              temp_i = head + act_class[a] + direction[d] + 'slow_' + 'imu_syn_' + str(n+1) + '.csv'
              gt_data_slow.append(temp_v)
              imu_data_slow.append(temp_i)
      elif (a==1):
        for w in range(0, len(wave_class)):
          for d in range(0, len(direction)):
              for n in range(0, num):
                temp_v = head + act_class[a] + wave_class[w] + direction[d] + 'fast_' + 'vicon_syn_' + str(n+1) + '.csv'
                temp_i = head + act_class[a] + wave_class[w] + direction[d] + 'fast_' + 'imu_syn_' + str(n+1) + '.csv'
                gt_data_fast.append(temp_v)
                imu_data_fast.append(temp_i)
                temp_v = head + act_class[a] + wave_class[w] + direction[d] + 'mid_' + 'vicon_syn_' + str(n+1) + '.csv'
                temp_i = head + act_class[a] + wave_class[w] + direction[d] + 'mid_' + 'imu_syn_' + str(n+1) + '.csv'
                gt_data_mid.append(temp_v)
                imu_data_mid.append(temp_i)
                temp_v = head + act_class[a] + wave_class[w] + direction[d] + 'slow_' + 'vicon_syn_' + str(n+1) + '.csv'
                temp_i = head + act_class[a] + wave_class[w] + direction[d] + 'slow_' + 'imu_syn_' + str(n+1) + '.csv'
                gt_data_slow.append(temp_v)
                imu_data_slow.append(temp_i)
      elif (a==2):
          for n in range(0, num):
            temp_v = head + act_class[a] + 'fast_' + 'vicon_syn_' + str(n+1) + '.csv'
            temp_i = head + act_class[a] + 'fast_' + 'imu_syn_' + str(n+1) + '.csv'
            gt_data_fast.append(temp_v)
            imu_data_fast.append(temp_i)
            temp_v = head + act_class[a] + 'mid_' + 'vicon_syn_' + str(n+1) + '.csv'
            temp_i = head + act_class[a] + 'mid_' + 'imu_syn_' + str(n+1) + '.csv'
            gt_data_mid.append(temp_v)
            imu_data_mid.append(temp_i)
            temp_v = head + act_class[a] + 'slow_' + 'vicon_syn_' + str(n+1) + '.csv'
            temp_i = head + act_class[a] + 'slow_' + 'imu_syn_' + str(n+1) + '.csv'
            gt_data_slow.append(temp_v)
            imu_data_slow.append(temp_i)
      elif (a==3):
        for n in range(0, num*5):
          temp_v = head + act_class[a] + 'MCR_vicon_syn_' + str(n+1) + '.csv'
          temp_i = head + act_class[a] + 'MCR_imu_syn_' + str(n+1) + '.csv'
          gt_data_fast.append(temp_v)
          imu_data_fast.append(temp_i)

      elif (a==4):
        for wk in range(0, len(walk)):
          for n in range(0, num*6-1):
            temp_v = head + act_class[a] + walk[wk] + walk_name[wk] + 'vicon_syn_' + str(n+1) + '.csv'
            temp_i = head + act_class[a] + walk[wk] + walk_name[wk] + 'imu_syn_' + str(n+1) + '.csv'
            gt_data_slow.append(temp_v)
            imu_data_slow.append(temp_i)
    
    
    imu_data_slow.append(head+'Stop/stop_imu_syn.csv')
    gt_data_slow.append( head+'Stop/stop_vicon_syn.csv')
    
    imu_data_fast.append(head+'Forward_back/Horizontal/fast_imu_syn_4.csv')
    imu_data_mid.append(head+'Forward_back/Horizontal/mid_imu_syn_4.csv')
    imu_data_slow.append(head+'Forward_back/Horizontal/slow_imu_syn_4.csv')
    gt_data_fast.append( head+'Forward_back/Horizontal/fast_vicon_syn_4.csv')
    gt_data_mid.append( head+'Forward_back/Horizontal/mid_vicon_syn_4.csv')
    gt_data_slow.append( head+'Forward_back/Horizontal/slow_vicon_syn_4.csv')

    return imu_data_fast, gt_data_fast, imu_data_mid, gt_data_mid, imu_data_slow, gt_data_slow

def load_our_datasets_filename_actions():
    head = tr_env.OUR_DATASET_DIR
    act_class = ['Forward_back/', 'Hand wave/', 'Draw eight/', 'Move cup/', 'Walk/']
    direction = ['Horizontal/', 'Vertical/']
    wave_class = ['Left_to_Right/', 'Up_to_down/']
    speed = ['fast_', 'mid_', 'slow_']
    walk = ['Circle/', 'Random/', 'S/']
    walk_name = ['circle_', 'random_', 's_']
    imu_data_hand = []
    gt_data_hand = []
    imu_data_walk = []
    gt_data_walk = []
    num = 3
    
    # IMU dictionary
    for a in range(0, len(act_class)):
      if (a==0):
        for d in range(0, len(direction)):
          for s in range(0, len(speed)):
            for n in range(0, num):
              temp_v = head + act_class[a] + direction[d] + speed[s] + 'vicon_syn_' + str(n+1) + '.csv'
              temp_i = head + act_class[a] + direction[d] + speed[s] + 'imu_syn_' + str(n+1) + '.csv'
              gt_data_hand.append(temp_v)
              imu_data_hand.append(temp_i)
      elif (a==1):
        for w in range(0, len(wave_class)):
          for d in range(0, len(direction)):
            for s in range(0, len(speed)):
              for n in range(0, num):
                temp_v = head + act_class[a] + wave_class[w] + direction[d] + speed[s] + 'vicon_syn_' + str(n+1) + '.csv'
                temp_i = head + act_class[a] + wave_class[w] + direction[d] + speed[s] + 'imu_syn_' + str(n+1) + '.csv'
                gt_data_hand.append(temp_v)
                imu_data_hand.append(temp_i)
      elif (a==2):
        for s in range(0, len(speed)):
          for n in range(0, num):
            temp_v = head + act_class[a] + speed[s] + 'vicon_syn_' + str(n+1) + '.csv'
            temp_i = head + act_class[a] + speed[s] + 'imu_syn_' + str(n+1) + '.csv'
            gt_data_hand.append(temp_v)
            imu_data_hand.append(temp_i)
      elif (a==3):
        for n in range(0, num*5):
          temp_v = head + act_class[a] + 'MCR_vicon_syn_' + str(n+1) + '.csv'
          temp_i = head + act_class[a] + 'MCR_imu_syn_' + str(n+1) + '.csv'
          gt_data_hand.append(temp_v)
          imu_data_hand.append(temp_i)

      elif (a==4):
        for wk in range(0, len(walk)):
          for n in range(0, num*6-1):
            temp_v = head + act_class[a] + walk[wk] + walk_name[wk] + 'vicon_syn_' + str(n+1) + '.csv'
            temp_i = head + act_class[a] + walk[wk] + walk_name[wk] + 'imu_syn_' + str(n+1) + '.csv'
            gt_data_walk.append(temp_v)
            imu_data_walk.append(temp_i)
    
    
    imu_data_walk.append(head+'Stop/stop_imu_syn.csv')
    gt_data_walk.append( head+'Stop/stop_vicon_syn.csv')
    
    imu_data_hand.append(head+'Forward_back/Horizontal/fast_imu_syn_4.csv')
    imu_data_hand.append(head+'Forward_back/Horizontal/mid_imu_syn_4.csv')
    imu_data_hand.append(head+'Forward_back/Horizontal/slow_imu_syn_4.csv')
    gt_data_hand.append( head+'Forward_back/Horizontal/fast_vicon_syn_4.csv')
    gt_data_hand.append( head+'Forward_back/Horizontal/mid_vicon_syn_4.csv')
    gt_data_hand.append( head+'Forward_back/Horizontal/slow_vicon_syn_4.csv')

    return imu_data_hand, gt_data_hand, imu_data_walk, gt_data_walk


def load_ronin_dataset_filename():
    imu_data = []
    info_data = []
    head = './ronin/'

    trainlist = open(head+'list_train.txt', 'r').read()
    folders = trainlist.split("\n")    

    for folder in folders:
        path = head+folder+'/data.hdf5'
        info_path = head+folder+'/info.json'
        if os.path.exists(path):
            imu_data.append(path)
            info_data.append(info_path)
    return imu_data, info_data


def load_ronin_seen_filename():
    imu_data = []
    info_data = []
    head = tr_env.RONIN_DATASET_DIR
    
    trainlist = open(head+'list_test_seen.txt', 'r').read()
    folders = trainlist.split("\n")   

    for folder in folders:
        path = head+folder+'/data.hdf5'
        info_path = head+folder+'/info.json'
        if os.path.exists(path):
            imu_data.append(path)
            info_data.append(info_path)
    return imu_data, info_data

def load_ronin_unseen_filename():
    imu_data = []
    info_data = []
    head = tr_env.RONIN_DATASET_DIR
    
    trainlist = open(head+'list_test_unseen.txt', 'r').read()
    folders = trainlist.split("\n")   

    for folder in folders:
        path = head+folder+'/data.hdf5'
        info_path = head+folder+'/info.json'
        if os.path.exists(path):
            imu_data.append(path)
            info_data.append(info_path)
    return imu_data, info_data


def load_ridi_train_filename():
    imu_data = []
    gt_data = []
    head = tr_env.RONIN_DATASET_DIR
    datalist = ["hang_handheld_normal1",
    "hang_handheld_speed1",
    "hang_handheld_side3",
    "hang_handheld_side4",
    "hang_leg_new1",
    "hang_leg_new2",
    "hang_bag_normal1",
    "hang_bag_speed1",
    "hang_body_slow1",
    "hang_body_fast1",
    "hang_body_side1",
    "hang_body_normal1",
    "hang_body_backward1",
    "hang_body_backward3",
    "hang_body_backward4",
    "hang_body_stop1",
    "dan_handheld1",
    "dan_leg1",
    "dan_body1",
    "dan_body2",
    "dan_bag1",
    "huayi_handheld1",
    "huayi_leg_front1",
    "huayi_leg_front2",
    "huayi_bag1",
    "huayi_lopata1",
    "ma_body1",
    "ma_body2",
    "ma_handheld1",
    "ma_handheld3",
    "ma_bag_low2",
    "ma_bag_low3",
    "zhicheng_handheld1",
    "zhicheng_leg1",
    "zhicheng_bag1",
    "zhicheng_body1",
    "tang_handheld1",
    "tang_bag1",
    "tang_body1",
    "xiaojing_handheld1",
    "xiaojing_leg1",
    "xiaojing_body1",
    "hao_handheld1",
    "hao_leg1",
    "hao_bag1",
    "hao_body1",
    "yajie_handheld1",
    "yajie_bag1",
    "yajie_body1"]

    for entry in datalist:
        path = head+entry+"/processed/data.csv"
        if os.path.exists(path):
            imu_data.append(path)
            gt_data.append(path)
    return imu_data, gt_data


def load_ridi_test_filename():
    imu_data = []
    gt_data = []
    head = tr_env.RIDI_DATSET_DIR
    datalist = ["hang_handheld_test1",
    "hang_handheld_side_test2",
    "hang_bag_speed2",
    "hang_body_test1",
    "hang_body_backward2",
    "hang_leg_new3",
    "huayi_bag_test1",
    "huayi_handheld_test1",
    "huayi_leg_front3",
    "huayi_body_test1",
    "dan_body3",
    "dan_leg2",
    "dan_bag2",
    "zhicheng_handheld2",
    "zhicheng_leg2",
    "zhicheng_bag2",
    "zhicheng_body2",
    "ma_handheld2",
    "tang_handheld2",
    "tang_bag2",
    "tang_body2",
    "xiaojing_handheld2",
    "xiaojing_body2",
    "yajie_body2",
    "yajie_handheld2"]

    for entry in datalist:
        path = head+entry+'/processed/data.csv'
        if os.path.exists(path):
            imu_data.append(path)
            gt_data.append(path)
    return imu_data, gt_data
