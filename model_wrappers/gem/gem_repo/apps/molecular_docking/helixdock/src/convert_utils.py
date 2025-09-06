'''
    tools for converting mol from 6k to xyz
'''
import os
import numpy as np
import paddle
from paddle import sin, cos



def rotation_matrix(alpha, beta, gamma):
    '''
        tools for generate the rotation matrix
    '''
    alpha = alpha
    beta = beta
    gamma = gamma

    Rx_tensor = paddle.concat(((alpha + 1) / (alpha + 1), alpha - alpha, alpha - alpha,
                           alpha - alpha, paddle.cos(alpha), - paddle.sin(alpha),
                           alpha - alpha, paddle.sin(alpha), paddle.cos(alpha)), 
                           axis=0).reshape((9, -1)).T.reshape((-1, 3, 3))

    Ry_tensor = paddle.concat((paddle.cos(beta), beta - beta, - paddle.sin(beta),
                           beta - beta, (beta + 1) / (beta + 1), beta - beta,
                           paddle.sin(beta), beta - beta, paddle.cos(beta)), 
                           axis=0).reshape((9, -1)).T.reshape((-1, 3, 3))

    Rz_tensor = paddle.concat((paddle.cos(gamma), -paddle.sin(gamma), gamma - gamma,
                           paddle.sin(gamma), paddle.cos(gamma), gamma - gamma,
                           gamma - gamma, gamma - gamma, (gamma + 1) / (gamma + 1)), 
                           axis=0).reshape((9, -1)).T.reshape((
        -1, 3, 3))

    R = paddle.matmul(paddle.matmul(Rx_tensor, Ry_tensor), Rz_tensor)

    return R


def rodrigues(vector, theta):
    '''
        tools for generate the rotation matrix from Torsion vector
    '''
    a = vector[:, 0]
    b = vector[:, 1]
    c = vector[:, 2]

    R_list = [
        cos(theta) + paddle.pow(a, 2) * (1 - cos(theta)).reshape((1, -1)),
        a * b * (1 - cos(theta)) - c * sin(theta).reshape((1, -1)),
        a * c * (1 - cos(theta)) + b * sin(theta).reshape((1, -1)),
        a * b * (1 - cos(theta)) + c * sin(theta).reshape((1, -1)),
        cos(theta) + paddle.pow(b, 2) * (1 - cos(theta)).reshape((1, -1)),
        b * c * (1 - cos(theta)) - a * sin(theta).reshape((1, -1)),
        a * c * (1 - cos(theta)) - b * sin(theta).reshape((1, -1)),
        b * c * (1 - cos(theta)) + a * sin(theta).reshape((1, -1)),
        cos(theta) + paddle.pow(c, 2) * (1 - cos(theta)).reshape((1, -1))
    ]

    R_matrix = paddle.concat(R_list, axis=0).T.reshape((-1, 3, 3))

    return R_matrix


def vector_length(vector):
    '''
        tools for calculate the length of the vector
    '''
    nlist = vector.shape[1]
    vec_length = paddle.sqrt(paddle.sum(paddle.square(vector), axis=-1))  # shape: [-1, 1]
    vec_length = vec_length.reshape((-1, nlist, 1, 1))

    return vec_length


def relative_vector_rotation(vector, R):
    '''
        tools for relative_vector_rotation
    '''

    nlist = vector.shape[1]
    R_tensor = R

    vector = vector.reshape((-1, nlist, 1, 3))
    vec_length = vector_length(vector)  # shape [-1, 1, 1]
    vector = vector.reshape((-1, nlist, 3, 1))  # shape [-1, 1, 3]

    new_vector = paddle.matmul(R_tensor.unsqueeze(1), vector).reshape((-1, nlist, 1, 3))  # shape [-1, 1, 3]
    new_vec_length = vector_length(new_vector)  # shape [-1, 1, 1]

    new_vector = new_vector * vec_length / new_vec_length  # shape [-1, 1, 3]
    # new_vector = new_vector[:, 0, :]  # shape [-1, 3]
    new_vector = new_vector.squeeze(2)
    return new_vector


def relative_vector_center_rotation(vector, center, R):
    '''
        tools for relative_vector_rotation in center
    '''
    nlist = vector.shape[1]
    num_of_vec = len(vector)
    R_tensor = R
    center = center.reshape((-1, 1, 1, 3))

    vector = vector.reshape((num_of_vec, -1, 1, 3))  # shape [-1, 1, 3]
    vec_length = vector_length(vector)  # shape [-1, 1, 1]

    point_1 = vector
    point_0 = paddle.zeros(shape=(num_of_vec, nlist, 1, 3))  # shape [-1, 1, 3]

    new_point_1 = paddle.matmul(R_tensor.unsqueeze(1), (point_1 - center).reshape((-1, nlist, 3, 1))) \
                  + center.reshape((-1, 1, 3, 1))  # shape [-1, 3, 1]
    new_point_0 = paddle.matmul(R_tensor.unsqueeze(1), (point_0 - center).reshape((-1, nlist, 3, 1))) \
                  + center.reshape((-1, 1, 3, 1))  # shape [-1, 3, 1]

    new_vector = (new_point_1 - new_point_0).reshape((-1, nlist, 1, 3))  # shape [-1, 1, 3]

    new_vec_length = vector_length(new_vector)  # shape [-1, 1, 1]
    new_vector = new_vector * (vec_length / new_vec_length)  # shape [-1, 1, 3]
    # new_vector = new_vector[:, 0, :]  # shape [-1, 3]
    new_vector = new_vector.squeeze(2)

    return new_vector


def output_ligand_traj(out_dpath, ligand):
    '''
        use for generate the pdb file from xyz
    '''
    origin_heavy_atoms_lines = ligand.origin_heavy_atoms_lines
    origin_coords = ligand.init_heavy_atoms_coords
    new_coords = ligand.pose_heavy_atoms_coords
    poses_file_names = ligand.poses_file_names

    for f_name, coord in zip(poses_file_names, new_coords):
        output_fpath = out_dpath + "/" + f_name + "/optimized_traj.pdb"
        os.makedirs(out_dpath + "/" + f_name, exist_ok=True)

        lines = []
        for num, line in enumerate(origin_heavy_atoms_lines):
            x = coord[num][0]
            y = coord[num][1]
            z = coord[num][2]

            atom_type = line.split()[2]
            pre_element = line.split()[2]
            if pre_element[:2] == "CL":
                element = "Cl"
            elif pre_element[:2] == "BR":
                element = "Br"
            else:
                element = pre_element[0]

            line = "ATOM%7s%5s%4s%2s%4s%12s%8s%8s%6s%6s%12s" % (
                str(num + 1), atom_type, "LIG", "A", "1", "%.3f" % x, "%.3f" % y, "%.3f" % z, "1.00", "0.00", element)
            lines.append(line)

        MODEL_numner = 0
        if not os.path.exists(output_fpath):
            pass
        else:
            with open(output_fpath) as f:
                MODEL_numner = len([x for x in f.readlines() if x[:5] == "MODEL"])

        with open(output_fpath, 'a+') as f:

            f.writelines('MODEL%9s\n' % str(MODEL_numner + 1))

            for line in lines:
                f.writelines(line + '\n')
            f.writelines("TER\n")
            f.writelines("ENDMDL\n")

        # current_pose.pbd 是最后一步优化的pose，不断地被覆盖
        with open(out_dpath + "/" + f_name + "/current_pose.pdb", 'w') as f:
            for line in lines:
                f.writelines(line + '\n')

    # origin pdbqt to pdb
    for f_name, coord in zip(poses_file_names, origin_coords):
        lines = []
        for num, line in enumerate(origin_heavy_atoms_lines):
            x = coord[num][0]
            y = coord[num][1]
            z = coord[num][2]

            atom_type = line.split()[2]
            pre_element = line.split()[2]
            if pre_element[:2] == "CL":
                element = "Cl"
            elif pre_element[:2] == "BR":
                element = "Br"
            else:
                element = pre_element[0]

            line = "ATOM%7s%5s%4s%2s%4s%12s%8s%8s%6s%6s%12s" % (
                str(num + 1), atom_type, "LIG", "A", "1", "%.3f" % x, "%.3f" % y, "%.3f" % z, "1.00", "0.00", element)
            lines.append(line)

        with open(out_dpath + "/" + f_name + "/init_pose.pdb", 'w') as f:
            for line in lines:
                f.writelines(line + '\n')