import numpy as np

from dataset_utils import load_dataset

DIR_SEG = './output-hl/'

SHAPE_SEG = (10, 352, 352, 1)
SHAPE_GT = (10, 352, 352)


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


counters = {}
models_mse = {}
for m in range(1, 6, 1):
    models_mse[m] = {
        'seg': [],
    }
    counters[m] = 0

patients = np.arange(1, 101)

ds = load_dataset(root_dir='./ds/')

for i, pat in enumerate(patients):
    print('Evaluating patient ' + str(pat))

    model_n = model_folder(pat)
    folder = str(model_n)

    pred_seg = np.load(DIR_SEG + folder + '/' + str(pat).zfill(3) + '.npy', allow_pickle=True)

    ed_idx = np.arange(0, len(pred_seg), 2)
    es_idx = np.arange(1, len(pred_seg), 2)

    images_idx = np.concatenate([ed_idx, es_idx])

    # Resort predictions
    pred_seg = np.concatenate(
        [resize_padding(pred_seg[ed_idx], ed_idx.shape + SHAPE_SEG[1:]), resize_padding(pred_seg[es_idx], es_idx.shape + SHAPE_SEG[1:])], axis=0)

    # Padding ground-truth
    gt = np.concatenate([resize_padding(ds[pat]['ed_gt'], ed_idx.shape + SHAPE_GT[1:]),
                         resize_padding(ds[pat]['es_gt'], es_idx.shape + SHAPE_GT[1:])],
                        axis=0)
    # Create heart locator ground-truth
    gt_seg = np.zeros(gt.shape)
    gt_seg[gt > 0] = 1

    pred_seg = pred_seg.reshape(pred_seg.shape[:3])

    models_mse[model_n]['seg'].append(np.mean((gt_seg - pred_seg)**2))

Cinv = np.linalg.inv(np.diag(
    [np.mean(models_mse[1]['seg']),
     np.mean(models_mse[2]['seg']),
     np.mean(models_mse[3]['seg']),
     np.mean(models_mse[4]['seg']),
     np.mean(models_mse[5]['seg'])]))
e = np.ones((5, 1))
lamb = 2/(e.T@Cinv@e)
w_mse_seg = (lamb * (Cinv@e)/2)
print(w_mse_seg)
