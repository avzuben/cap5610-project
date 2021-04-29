import numpy as np

from dataset_utils import load_dataset

DIR_2D = './pred-2dn/'
DIR_2DPP = './pred-2dnpp/'
DIR_3D = './pred-3dn/'

SHAPE_GT = (10, 352, 352)

WEIGHT_MSE_2D_2DPP_3D = {
    1: [[0.33339201], [0.33334009], [0.33326789]],
    2: [[0.3333324], [0.3333339], [0.33333369]],
    3: [[0.33359929], [0.33353806], [0.33286265]],
    4: [[0.33349437], [0.33342641], [0.33307922]],
    5: [[0.33346418], [0.33347459], [0.33306123]],
}

def model_folder(pat):
    folder = pat % 5
    if folder == 0:
        folder = 5
    return folder


def resize_padding(arr, target_shape, pad_value=0):
    data = np.zeros(target_shape)
    data += pad_value
    s_offest = (target_shape[0] - arr.shape[0]) // 2
    h_offest = (target_shape[1] - arr.shape[1]) // 2
    w_offest = (target_shape[2] - arr.shape[2]) // 2

    t_s_s = max(s_offest, 0)
    t_s_e = t_s_s + min(arr.shape[0] + s_offest, arr.shape[0]) - max(0, -s_offest)
    t_h_s = max(h_offest, 0)
    t_h_e = t_h_s + min(arr.shape[1] + h_offest, arr.shape[1]) - max(0, -h_offest)
    t_w_s = max(w_offest, 0)
    t_w_e = t_w_s + min(arr.shape[2] + w_offest, arr.shape[2]) - max(0, -w_offest)

    s_s_s = max(0, -s_offest)
    s_s_e = s_s_s + t_s_e - t_s_s
    s_h_s = max(0, -h_offest)
    s_h_e = s_h_s + t_h_e - t_h_s
    s_w_s = max(0, -w_offest)
    s_w_e = s_w_s + t_w_e - t_w_s

    data[t_s_s:t_s_e, t_h_s:t_h_e, t_w_s:t_w_e] = arr[s_s_s:s_s_e, s_h_s:s_h_e, s_w_s:s_w_e]
    return data



def convert_seg_image_to_one_hot_encoding(image, n_classes=4):
    '''
    image must be either (x, y, z) or (x, y)
    Takes as input an nd array of a label map (any dimension). Outputs a one hot encoding of the label map.
    Example (3D): if input is of shape (x, y, z), the output will ne of shape (x, y, z, n_classes)
    '''
    classes = np.arange(n_classes)
    out_image = np.zeros(list(image.shape) + [len(classes)], dtype=image.dtype)
    for i, c in enumerate(classes):
        x = np.zeros((len(classes)))
        x[i] = 1
        out_image[image == c] = x
    return out_image


counters = {}
models_mse = {}
for m in range(1, 6, 1):
    models_mse[m] = {
        '2d': [],
        '2dpp': [],
        'w': [],
        '3d': [],
    }
    counters[m] = 0

patients = np.arange(1, 101)

ds = load_dataset(root_dir='./ds/')

mse_2d = np.zeros((len(patients)))
mse_2dpp = np.zeros((len(patients)))
mse_3d = np.zeros((len(patients)))

for i, pat in enumerate(patients):
    print('Evaluating patient ' + str(pat))

    model_n = model_folder(pat)
    folder = str(model_n)

    pred_multi_input_heart_locator_2d = np.load(
        DIR_2D + folder + '/' + str(pat).zfill(3) + '-multi-input-heart-locator.npy', allow_pickle=True)
    pred_multi_input_heart_locator_2dpp = np.load(
        DIR_2DPP + folder + '/' + str(pat).zfill(3) + '-multi-input-heart-locator.npy', allow_pickle=True)
    pred_multi_input_heart_locator_3d = np.load(
        DIR_3D + folder + '/' + str(pat).zfill(3) + '-multi-input-heart-locator.npy', allow_pickle=True)
    pred_multi_input_heart_locator_3d[np.round(np.sum(pred_multi_input_heart_locator_3d, -1)) == 2] = pred_multi_input_heart_locator_3d[np.round(np.sum(pred_multi_input_heart_locator_3d, -1)) == 2] / 2

    ed_idx = np.arange(0, len(pred_multi_input_heart_locator_2d), 2)
    es_idx = np.arange(1, len(pred_multi_input_heart_locator_2d), 2)

    images_idx = np.concatenate([ed_idx, es_idx])

    # Padding ground-truth
    gt = np.concatenate([resize_padding(ds[pat]['ed_gt'], ed_idx.shape + SHAPE_GT[1:]),
                         resize_padding(ds[pat]['es_gt'], es_idx.shape + SHAPE_GT[1:])],
                        axis=0)
    gt_one_hot = convert_seg_image_to_one_hot_encoding(gt)

    models_mse[model_n]['2d'].append(np.mean((gt_one_hot - pred_multi_input_heart_locator_2d)**2))
    models_mse[model_n]['2dpp'].append(np.mean((gt_one_hot - pred_multi_input_heart_locator_2dpp)**2))
    models_mse[model_n]['3d'].append(np.mean((gt_one_hot - pred_multi_input_heart_locator_3d)**2))
    models_mse[model_n]['w'].append(np.mean((gt_one_hot - (WEIGHT_MSE_2D_2DPP_3D[model_n][0] * pred_multi_input_heart_locator_2d +
                                                           WEIGHT_MSE_2D_2DPP_3D[model_n][1] * pred_multi_input_heart_locator_2dpp +
                                                           WEIGHT_MSE_2D_2DPP_3D[model_n][2] * pred_multi_input_heart_locator_3d))**2))


for m in range(1, 6, 1):
    print(m)
    mse_2d = np.mean(models_mse[m]['2d'])
    mse_2dpp = np.mean(models_mse[m]['2dpp'])
    mse_3d = np.mean(models_mse[m]['3d'])

    Cinv = np.linalg.inv(np.diag([mse_2d, mse_2dpp]))
    e = np.ones((2, 1))
    lamb = 2/(e.T@Cinv@e)
    w_mse_2d_2dpp = (lamb * (Cinv@e)/2)
    print(w_mse_2d_2dpp)

    Cinv = np.linalg.inv(np.diag([mse_2d, mse_2dpp, mse_3d]))
    e = np.ones((3, 1))
    lamb = 2 / (e.T @ Cinv @ e)
    w_mse_2d_2dpp_3d = (lamb * (Cinv @ e) / 2)
    print(w_mse_2d_2dpp_3d)

Cinv = np.linalg.inv(np.diag(
    [np.mean(models_mse[1]['w']),
     np.mean(models_mse[2]['w']),
     np.mean(models_mse[3]['w']),
     np.mean(models_mse[4]['w']),
     np.mean(models_mse[5]['w'])]))
e = np.ones((5, 1))
lamb = 2/(e.T@Cinv@e)
w_mse_seg = (lamb * (Cinv@e)/2)
print(w_mse_seg)
