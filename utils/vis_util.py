import torch
import numpy as np


def get_cross_prod_mat(pVec_Arr):
    """ Convert pVec_Arr of shape (3) to its cross product matrix
    """
    qCross_prod_mat = np.array([
        [0, -pVec_Arr[2], pVec_Arr[1]],
        [pVec_Arr[2], 0, -pVec_Arr[0]],
        [-pVec_Arr[1], pVec_Arr[0], 0],
    ])
    return qCross_prod_mat

def params_to_8points(box, degrees=False):
    """ Given bounding box as 7 parameters: w, l, h, cx, cy, cz, z, compute the 8 corners of the box
    """
    w, l, h, cx, cy, cz, z = box
    points = []
    for i in [-1, 1]:
        for j in [-1, 1]:
            for k in [-1, 1]:
                points.append([w.item()/2 * i, l.item()/2 * j, h.item()/2 * k])
    points = np.asarray(points)
    points = (get_rotation(z.item(), degree=degrees) @ points.T).T
    points += np.expand_dims(np.array([cx.item(), cy.item(), cz.item()]), 0)
    return points


def params_to_8points_no_rot(box):
    """ Given bounding box as 6 parameters (without rotation): w, l, h, cx, cy, cz, compute the 8 corners of the box.
        Works when the box is axis aligned
    """
    # w, l, h, cx, cy, cz = box
    cx, cy, cz, w, l, h = box
    points = []
    for i in [-1, 1]:
        for j in [-1, 1]:
            for k in [-1, 1]:
                points.append([w.item()/2 * i, l.item()/2 * j, h.item()/2 * k])
    points = np.asarray(points)
    points += np.expand_dims(np.array([cx.item(), cy.item(), cz.item()]), 0)
    return points

def fit_shapes_to_box(box, shape, withangle=True):
    """ Given normalized shape, transform it to fit the input bounding box.
        Expects denormalized bounding box with optional angle channel in degrees
        :param box: tensor
        :param shape: tensor
        :param withangle: boolean
        :return: transformed shape
    """
    box = box.detach().cpu().numpy()
    shape = shape.detach().cpu().numpy()
    if withangle:
        w, l, h, cx, cy, cz, z = box
    else:
        w, l, h, cx, cy, cz = box
    # scale
    shape_size = np.max(shape, axis=0) - np.min(shape, axis=0)
    shape = shape / shape_size
    shape *= box[:3]
    if withangle:
        # rotate
        shape = (get_rotation(z, degree=True).astype("float32") @ shape.T).T
    # translate
    shape += [cx, cy, cz]

    return shape


def refineBoxes(boxes, objs, triples, relationships, vocab):
    for idx in range(len(boxes)):
      child_box = boxes[idx]
      w, l, h, cx, cy, cz = child_box
      for t in triples:
         if idx == t[0] and relationships[t[1]] in ["supported by", "lying on", "standing on"]:
            parent_idx = t[2]
            cat = vocab['object_idx_to_name'][objs[parent_idx]].replace('\n', '')
            if cat != 'floor':
                continue
            parent_box = boxes[parent_idx]
            base = parent_box[5] + 0.0125

            new_bottom = base
            # new_h = cz + h / 2 - new_bottom
            new_cz = new_bottom + h / 2
            shift = new_cz - cz
            boxes[idx][:] = [w, l, h, cx, cy, new_cz]

            # fix adjusmets
            for t_ in triples:
                if t_[2] == t[0] and relationships[t_[1]] in ["supported by", "lying on", "standing on"]:
                    cat = vocab['object_idx_to_name'][t_[2]].replace('\n', '')
                    if cat != 'floor':
                        continue

                    w_, l_, h_, cx_, cy_, cz_ = boxes[t_[0]]
                    boxes[t_[0]][:] = [w_, l_, h_, cx_, cy_, cz_ + shift]
    return boxes


def get_rotation(z, degree=True):
    """ Get rotation matrix given rotation angle along the z axis.
    :param z: angle of z axos rotation
    :param degree: boolean, if true angle is given in degrees, else in radians
    :return: rotation matrix as np array of shape[3,3]
    """
    if degree:
        z = np.deg2rad(z)
    rot = np.array([[np.cos(z), -np.sin(z),  0],
                    [np.sin(z),  np.cos(z),  0],
                    [        0,          0,  1]])
    return rot


def normalize_box_params(box_params, scale=3):
    """ Normalize the box parameters for more stable learning utilizing the accumulated dataset statistics

    :param box_params: float array of shape [7] containing the box parameters
    :param scale: float scalar that scales the parameter distribution
    :return: normalized box parameters array of shape [7]
    """
    mean = np.array([1.3827214, 1.309359, 0.9488993, -0.12464812, 0.6188591, -0.54847, 0.73127955])
    std = np.array([1.7797655, 1.657638, 0.8501885, 1.9160025, 2.0038228, 0.70099753, 0.50347435])

    return scale * ((box_params - mean) / std)


def denormalize_box_params(box_params, scale=3, params=7):
    """ Denormalize the box parameters utilizing the accumulated dataset statistics

    :param box_params: float array of shape [params] containing the box parameters
    :param scale: float scalar that scales the parameter distribution
    :param params: number of bounding box parameters. Expects values of either 6 or 7. 6 omits the angle
    :return: denormalized box parameters array of shape [params]
    """
    if params == 6:
        mean = np.array([1.3827214, 1.309359, 0.9488993, -0.12464812, 0.6188591, -0.54847])
        std = np.array([1.7797655, 1.657638, 0.8501885, 1.9160025, 2.0038228, 0.70099753])
    elif params == 7:
        mean = np.array([1.3827214, 1.309359, 0.9488993, -0.12464812, 0.6188591, -0.54847, 0.73127955])
        std = np.array([1.7797655, 1.657638, 0.8501885, 1.9160025, 2.0038228, 0.70099753, 0.50347435])
    else:
        raise NotImplementedError
    return (box_params * std) / scale + mean


# BRIEF denormalize the predicted boundingbox.
def batch_torch_denormalize_box_params(box_params, scale=3):
    """ Denormalize the box parameters utilizing the accumulated dateaset statistics

    :param box_params: float tensor of shape [N, 6] containing the 6 box parameters, where N is the number of boxes
    :param scale: float scalar that scales the parameter distribution
    :return: float tensor of shape [N, 6], the denormalized box parameters
    """

    mean = torch.tensor([1.3827214, 1.309359, 0.9488993, -0.12464812, 0.6188591, -0.54847]).reshape(1,-1).float().cuda()
    std = torch.tensor([1.7797655, 1.657638, 0.8501885, 1.9160025, 2.0038228, 0.70099753]).reshape(1,-1).float().cuda()

    return (box_params * std) / scale + mean


def bool_flag(s):
    """Helper function to make argparse work with the input True and False.
    Otherwise it reads it as a string and is always set to True.

    :param s: string input
    :return: the boolean equivalent of the string s
    """
    if s == '1' or s == 'True':
      return True
    elif s == '0' or s == 'False':
      return False
    msg = 'Invalid value "%s" for bool flag (should be 0, False or 1, True)'
    raise ValueError(msg % s)
