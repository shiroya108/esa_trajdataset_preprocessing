import numpy as np
import quaternion



def generate_trajectory_6d_quat(init_p, init_q, y_delta_p, y_delta_q):
    cur_p = np.array(init_p)
    cur_q = quaternion.from_float_array(init_q)
    pred_p = []
    pred_p.append(np.array(cur_p))

    for [delta_p, delta_q] in zip(y_delta_p, y_delta_q):
        cur_p = cur_p + np.matmul(quaternion.as_rotation_matrix(cur_q), delta_p.T).T
        cur_q = cur_q * quaternion.from_float_array(delta_q).normalized()
        pred_p.append(np.array(cur_p))
    # print(delta_q)
    # print(type(delta_q))
    return np.reshape(pred_p, (len(pred_p), 3))

def generate_trajectory_6d_tango(init_p, init_q, y_delta_p, y_delta_q):
    cur_p = np.array(init_p)
    cur_q = quaternion.from_float_array(init_q)
    pred_p = []
    pred_p.append(np.array(cur_p))

    for [delta_p, delta_q] in zip(y_delta_p, y_delta_q):
        # cur_p = cur_p + np.matmul(quaternion.as_rotation_matrix(cur_q), delta_p.T).T
        cur_p = cur_p + delta_p
        cur_q = cur_q * quaternion.from_float_array(delta_q).normalized()
        pred_p.append(np.array(cur_p))
    # print(delta_q)
    # print(type(delta_q))
    return np.reshape(pred_p, (len(pred_p), 2))
    
def generate_position(in_p, in_q):
    
    p = np.zeros((len(in_p), 3))
    for j in range(0, in_p.shape[0]):
        cur_p = np.array(in_p[j,:])
        cur_q = quaternion.from_float_array(in_q[j,:])
        cur_p = np.matmul(quaternion.as_rotation_matrix(cur_q), cur_p)
        p[j,:] = cur_p

    return p

    
def generate_trajectory_only_p(init_p, y_delta_p):
    cur_p = np.array(init_p)
    pred_p = []
    pred_p.append(np.array(cur_p))

    for [delta_p] in zip(y_delta_p):
        cur_p = cur_p + delta_p
        pred_p.append(np.array(cur_p))

    return np.reshape(pred_p, (len(pred_p), 3))


def generate_trajectory_3d(init_l, init_theta, init_psi, y_delta_l, y_delta_theta, y_delta_psi):
    cur_l = np.array(init_l)
    cur_theta = np.array(init_theta)
    cur_psi = np.array(init_psi)
    pred_l = []
    pred_l.append(np.array(cur_l))

    for [delta_l, delta_theta, delta_psi] in zip(y_delta_l, y_delta_theta, y_delta_psi):
        cur_theta = cur_theta + delta_theta
        cur_psi = cur_psi + delta_psi
        cur_l[0] = cur_l[0] + delta_l * np.sin(cur_theta) * np.cos(cur_psi)
        cur_l[1] = cur_l[1] + delta_l * np.sin(cur_theta) * np.sin(cur_psi)
        cur_l[2] = cur_l[2] + delta_l * np.cos(cur_theta)
        pred_l.append(np.array(cur_l))

    return np.reshape(pred_l, (len(pred_l), 3))
    
def writing_csv(rmse, model_name, filename="RMSE_result.csv"):
    name=[]
    new_line=[]
    new_line.append('')
    name.append(model_name)
    
    with open(filename,'a') as f:
            np.savetxt(f, name, fmt="%s", newline=',')
            np.savetxt(f, rmse,fmt="%.6f", newline=',')
            np.savetxt(f, new_line,fmt="%s", delimiter=',')
    # groups = ["a", "b", "c", "d", "e", "f","g","h","i","j","k","l"]
    # dict = {"groups": groups, model_name: rmse}
    # rmse_df = pd.DataFrame(dict)
    # #print(rmse_df)
    # rmse_df.to_csv('Result_RMSE.csv', mode='a', header=False)
    
def tracking_output(pos, quat, filename, result_dir):
    
    input = np.zeros((1,6), dtype=float)
    with open(result_dir+'\\'+str(filename)+'.csv', 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            for j in range(0,len(pos[:,1])):
                input[0,0:3] = pos[j,:]
                input[0,3:6] = quat[j,:]
                writer.writerow(input)

def compute_relative_trajectory_error(est, gt, dataset, max_delta=-1):
    """
    The Relative Trajectory Error (RTE) defined in:
    A Benchmark for the evaluation of RGB-D SLAM Systems
    http://ais.informatik.uni-freiburg.de/publications/papers/sturm12iros.pdf

    Args:
        est: the estimated trajectory
        gt: the ground truth trajectory.
        delta: fixed window size. If set to -1, the average of all RTE up to max_delta will be computed.
        max_delta: maximum delta. If -1 is provided, it will be set to the length of trajectories.

    Returns:
        Relative trajectory error. This is the mean value under different delta.
    """

    if dataset == "oxiod":
        delta = 100 * 60
    else:
        delta = 50 * 60

    if max_delta == -1:
        max_delta = est.shape[0]
    deltas = np.array([delta]) if delta > 0 else np.arange(1, min(est.shape[0], max_delta))
    rtes = np.zeros(deltas.shape[0])
    for i in range(deltas.shape[0]):
        # For each delta, the RTE is computed as the RMSE of endpoint drifts from fixed windows
        # slided through the trajectory.
        err = est[deltas[i]: , :] + gt[:-deltas[i]  , :] - est[:-deltas[i] , :] - gt[deltas[i]: , :]
        rtes[i] = np.sqrt(np.mean(err ** 2))

    # The average of RTE of all window sized is returned.
    return np.mean(rtes)



    