import numpy as np

from medpy import metric
from skimage.morphology import label
from PIL import Image
from pathlib import Path

from dataset_utils import load_dataset, get_patients_spacing

DIR_2D = './pred-2dn/'
DIR_2DPP = './pred-2dnpp/'
DIR_3D = './pred-3dn/'
DIR_SEG = './output-hl/'

SHAPE_SEG = (10, 352, 352, 1)
SHAPE_2D = (10, 352, 352, 4)
SHAPE_3D = (20, 352, 352, 4)
SHAPE_GT = (10, 352, 352)

WEIGHT_MSE_2D_2DPP = {
    1: [[0.50003894], [0.49996106]],
    2: [[0.49999888], [0.50000112]],
    3: [[0.50004589], [0.49995411]],
    4: [[0.50005095], [0.49994905]],
    5: [[0.4999922], [0.5000078]],
}
WEIGHT_MSE_2D_2DPP_3D = {
    1: [[0.33339201], [0.33334009], [0.33326789]],
    2: [[0.3333324], [0.3333339], [0.33333369]],
    3: [[0.33359929], [0.33353806], [0.33286265]],
    4: [[0.33349437], [0.33342641], [0.33307922]],
    5: [[0.33346418], [0.33347459], [0.33306123]],
}


def normalize_input(image):
    image = image - image.mean()
    pixels = image.flatten()
    delta_index = int(round(((1 - 0.98) / 2) * len(pixels)))
    pixels = np.sort(pixels)
    min = pixels[delta_index]
    max = pixels[-(delta_index + 1)]
    image = 2 * ((image - min) / (max - min)) - 1
    image[image < -1] = -1
    image[image > 1] = 1
    return image


def input_to_image(input_image):
    input_image = (input_image + 1) / 2
    input_image = np.expand_dims(input_image, -1)
    return np.concatenate([input_image, input_image, input_image], -1)


def mask_to_image(mask):
    img = np.zeros(mask.shape[0:2] + (3,))
    # First class
    idx = mask == 1
    img[idx] = [1, 0, 0]
    # Second class
    idx = mask == 2
    img[idx] = [0, 1, 0]
    # Third class
    idx = mask == 3
    img[idx] = [0, 0, 1]
    return img


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


def postprocess_stack(seg):
    # basically look for connected components and choose the largest one, delete everything else
    if np.sum(seg) > 0:
        mask = seg != 0
        lbls = label(mask, 8)
        lbls_sizes = [np.sum(lbls == i) for i in np.unique(lbls)]
        largest_region = np.argmax(lbls_sizes[1:]) + 1
        seg[lbls != largest_region] = 0
    return seg


def postprocess_slice(seg, i):
    rv = get_largest_region(seg[i:i+1], 1)
    myo = get_largest_region(seg[i:i+1], 2)
    lv = get_largest_region(seg[i:i+1], 3)
    seg[i] = rv + myo + lv
    return seg


def postprocess_prediction(seg):
    if np.sum(seg) > 0:

        seg = postprocess_stack(seg)
        s = len(seg) // 2
        seg = postprocess_slice(seg, s)
        sb = s
        sa = s
        while True:
            seg = postprocess_stack(seg)
            sb = sb - 1
            sa = sa + 1
            # print(sb)
            # print(sa)
            if sb >= 0:
                seg = postprocess_slice(seg, sb)
            if sa <= len(seg) - 1:
                seg = postprocess_slice(seg, sa)
            if sb <= 0 and sa >= len(seg) - 1:
                break
        # seg = postprocess_stack(seg)
    return seg


def get_largest_region(seg, class_label):
    # basically look for connected components and choose the largest one, delete everything else
    new_seg = np.zeros(seg.shape)
    for i in range(seg.shape[0]):
        mask = seg[i] == class_label
        if np.sum(seg[i, mask]) > 0:
            lbls = label(mask, 8)
            lbls_sizes = [np.sum(lbls == i) for i in np.unique(lbls)]
            largest_region = np.argmax(lbls_sizes[1:]) + 1
            new_seg[i, lbls == largest_region] = seg[i, lbls == largest_region]
    return new_seg


def compute_metrics(seg_gt, seg_pred, labels, spacing):
    assert seg_gt.shape == seg_pred.shape
    mask_pred = np.zeros(seg_pred.shape, dtype=bool)
    mask_gt = np.zeros(seg_pred.shape, dtype=bool)

    for l in labels:
        mask_gt[seg_gt == l] = True
        mask_pred[seg_pred == l] = True

    vol_gt = np.sum(mask_gt)
    vol_pred = np.sum(mask_pred)

    try:
        dice = metric.dc(mask_pred, mask_gt)
        if np.sum(mask_gt) == 0:
            dice = np.nan
    except Exception as e:
        print(e)
        dice = np.nan

    try:
        hd = metric.hd(mask_gt, mask_pred, voxelspacing=(spacing[0], 1.25, 1.25))
    except Exception as e:
        print(e)
        hd = np.nan

    try:
        assd = metric.assd(mask_gt, mask_pred)
    except Exception as e:
        print(e)
        assd = np.nan

    return dice, hd, assd, vol_gt, vol_pred


metrics = {}
model_types = [
    'heart-locator',
    '2d-multi-input-heart-locator',
    '2dpp-multi-input-heart-locator',
    '2d-multi-input-avg', 
    '3d-multi-input-heart-locator', 
    'multi-input-avg'
]
model_numbers = ['1', '2', '3', '4', '5', 'all']

for mt in model_types:
    metrics[mt] = {}
    for mn in model_numbers:
        metrics[mt][mn] = {
            'lv': {
                'ed': {
                    'dc': list(),
                    'hd': list()
                },
                'es': {
                    'dc': list(),
                    'hd': list()
                },
                'total': {
                    'dc': list(),
                    'hd': list()
                }
            },
            'myo': {
                'ed': {
                    'dc': list(),
                    'hd': list()
                },
                'es': {
                    'dc': list(),
                    'hd': list()
                },
                'total': {
                    'dc': list(),
                    'hd': list()
                }
            },
            'rv': {
                'ed': {
                    'dc': list(),
                    'hd': list()
                },
                'es': {
                    'dc': list(),
                    'hd': list()
                },
                'total': {
                    'dc': list(),
                    'hd': list()
                }
            },
            'total': {
                'ed': {
                    'dc': list(),
                    'hd': list()
                },
                'es': {
                    'dc': list(),
                    'hd': list()
                },
                'total': {
                    'dc': list(),
                    'hd': list()
                }
            }
        }

patients = np.arange(1, 101)

ds = load_dataset(root_dir='./ds/')
pat_spacing = get_patients_spacing()

for pat in patients:
    print('Evaluating patient ' + str(pat))

    model_n = model_folder(pat)
    folder = str(model_n)

    pred_multi_input_heart_locator_2d = np.load(DIR_2D + folder + '/' + str(pat).zfill(3) + '-multi-input-heart-locator.npy', allow_pickle=True)
    pred_multi_input_heart_locator_2dpp = np.load(DIR_2DPP + folder + '/' + str(pat).zfill(3) + '-multi-input-heart-locator.npy', allow_pickle=True)
    pred_multi_input_heart_locator_3d = np.load(DIR_3D + folder + '/' + str(pat).zfill(3) + '-multi-input-heart-locator.npy', allow_pickle=True)

    pred_seg = np.load(DIR_SEG + folder + '/' + str(pat).zfill(3) + '.npy', allow_pickle=True)

    ed_idx = np.arange(0, len(pred_multi_input_heart_locator_2d), 2)
    es_idx = np.arange(1, len(pred_multi_input_heart_locator_2d), 2)

    images_idx = np.concatenate([ed_idx, es_idx])

    pred_seg = pred_seg.reshape(pred_seg.shape[:-1])
    pred_seg = np.concatenate(
        [postprocess_prediction(np.round(pred_seg[ed_idx], True)), postprocess_prediction(np.round(pred_seg[es_idx], True))],
        axis=0)

    gt = np.concatenate([resize_padding(ds[pat]['ed_gt'], ed_idx.shape + SHAPE_GT[1:]), resize_padding(ds[pat]['es_gt'], es_idx.shape + SHAPE_GT[1:])],
                        axis=0)

    data = np.zeros((len(ds[pat]['ed_data']) + len(ds[pat]['es_data']),) + SHAPE_GT[1:])
    for d in range(len(ds[pat]['ed_data'])):
        data[d,] = resize_padding(np.array([normalize_input(ds[pat]['ed_data'][d])]), (1,) + SHAPE_GT[1:], -1)
    for d in range(len(ds[pat]['es_data'])):
        data[d+len(ds[pat]['ed_data']),] = resize_padding(np.array([normalize_input(ds[pat]['es_data'][d])]), (1,) + SHAPE_GT[1:], -1)

    pred_multi_input_avg_2d = WEIGHT_MSE_2D_2DPP[model_n][0] * pred_multi_input_heart_locator_2d + WEIGHT_MSE_2D_2DPP[model_n][1] * pred_multi_input_heart_locator_2dpp

    pred_multi_input_avg = WEIGHT_MSE_2D_2DPP_3D[model_n][0] * pred_multi_input_heart_locator_2d + WEIGHT_MSE_2D_2DPP_3D[model_n][1] * pred_multi_input_heart_locator_2dpp + WEIGHT_MSE_2D_2DPP_3D[model_n][2] * pred_multi_input_heart_locator_3d

    pred_multi_input_avg_2d = np.concatenate(
        [postprocess_prediction(np.argmax(pred_multi_input_avg_2d[:len(pred_multi_input_avg_2d) // 2], axis=-1)),
         postprocess_prediction(np.argmax(pred_multi_input_avg_2d[len(pred_multi_input_avg_2d) // 2:], axis=-1))], axis=0)
    pred_multi_input_avg = np.concatenate(
        [postprocess_prediction(np.argmax(pred_multi_input_avg[:len(pred_multi_input_avg) // 2], axis=-1)),
         postprocess_prediction(np.argmax(pred_multi_input_avg[len(pred_multi_input_avg) // 2:], axis=-1))], axis=0)
    pred_multi_input_heart_locator_2d = np.concatenate(
        [postprocess_prediction(np.argmax(pred_multi_input_heart_locator_2d[:len(pred_multi_input_heart_locator_2d) // 2], axis=-1)),
         postprocess_prediction(np.argmax(pred_multi_input_heart_locator_2d[len(pred_multi_input_heart_locator_2d) // 2:], axis=-1))], axis=0)
    pred_multi_input_heart_locator_2dpp = np.concatenate(
        [postprocess_prediction(np.argmax(pred_multi_input_heart_locator_2dpp[:len(pred_multi_input_heart_locator_2dpp) // 2], axis=-1)),
         postprocess_prediction(np.argmax(pred_multi_input_heart_locator_2dpp[len(pred_multi_input_heart_locator_2dpp) // 2:], axis=-1))], axis=0)
    pred_multi_input_heart_locator_3d = np.concatenate(
        [postprocess_prediction(np.argmax(pred_multi_input_heart_locator_3d[:len(pred_multi_input_heart_locator_3d) // 2], axis=-1)),
         postprocess_prediction(np.argmax(pred_multi_input_heart_locator_3d[len(pred_multi_input_heart_locator_3d) // 2:], axis=-1))], axis=0)

    ##### Heart loactor

    ## ED
    # total
    dc, hd, _, _, _ = compute_metrics(gt[:len(pred_seg)//2], pred_seg[:len(pred_seg)//2], [1, 2, 3], pat_spacing[pat])
    if not np.isnan(dc) and not np.isnan(hd):
        metrics['heart-locator'][folder]['total']['ed']['dc'].append(dc)
        metrics['heart-locator'][folder]['total']['ed']['hd'].append(hd)
        metrics['heart-locator'][folder]['total']['total']['dc'].append(dc)
        metrics['heart-locator'][folder]['total']['total']['hd'].append(hd)
        metrics['heart-locator']['all']['total']['ed']['dc'].append(dc)
        metrics['heart-locator']['all']['total']['ed']['hd'].append(hd)
        metrics['heart-locator']['all']['total']['total']['dc'].append(dc)
        metrics['heart-locator']['all']['total']['total']['hd'].append(hd)

    ## ES
    # total
    dc, hd, _, _, _ = compute_metrics(gt[len(pred_seg)//2:], pred_seg[len(pred_seg)//2:], [1, 2, 3], pat_spacing[pat])
    if not np.isnan(dc) and not np.isnan(hd):
        metrics['heart-locator'][folder]['total']['es']['dc'].append(dc)
        metrics['heart-locator'][folder]['total']['es']['hd'].append(hd)
        metrics['heart-locator'][folder]['total']['total']['dc'].append(dc)
        metrics['heart-locator'][folder]['total']['total']['hd'].append(hd)
        metrics['heart-locator']['all']['total']['es']['dc'].append(dc)
        metrics['heart-locator']['all']['total']['es']['hd'].append(hd)
        metrics['heart-locator']['all']['total']['total']['dc'].append(dc)
        metrics['heart-locator']['all']['total']['total']['hd'].append(hd)

    # ##### 2D Multi Input Heart Locator

    # ## ED
    # RV
    dc, hd, _, _, _ = compute_metrics(gt[:len(pred_multi_input_heart_locator_2d)//2], pred_multi_input_heart_locator_2d[:len(pred_multi_input_heart_locator_2d)//2], [1], pat_spacing[pat])
    if not np.isnan(dc) and not np.isnan(hd):
        metrics['2d-multi-input-heart-locator'][folder]['rv']['ed']['dc'].append(dc)
        metrics['2d-multi-input-heart-locator'][folder]['rv']['ed']['hd'].append(hd)
        metrics['2d-multi-input-heart-locator'][folder]['rv']['total']['dc'].append(dc)
        metrics['2d-multi-input-heart-locator'][folder]['rv']['total']['hd'].append(hd)
        metrics['2d-multi-input-heart-locator']['all']['rv']['ed']['dc'].append(dc)
        metrics['2d-multi-input-heart-locator']['all']['rv']['ed']['hd'].append(hd)
        metrics['2d-multi-input-heart-locator']['all']['rv']['total']['dc'].append(dc)
        metrics['2d-multi-input-heart-locator']['all']['rv']['total']['hd'].append(hd)

    # MYO
    dc, hd, _, _, _ = compute_metrics(gt[:len(pred_multi_input_heart_locator_2d)//2], pred_multi_input_heart_locator_2d[:len(pred_multi_input_heart_locator_2d)//2], [2], pat_spacing[pat])
    if not np.isnan(dc) and not np.isnan(hd):
        metrics['2d-multi-input-heart-locator'][folder]['myo']['ed']['dc'].append(dc)
        metrics['2d-multi-input-heart-locator'][folder]['myo']['ed']['hd'].append(hd)
        metrics['2d-multi-input-heart-locator'][folder]['myo']['total']['dc'].append(dc)
        metrics['2d-multi-input-heart-locator'][folder]['myo']['total']['hd'].append(hd)
        metrics['2d-multi-input-heart-locator']['all']['myo']['ed']['dc'].append(dc)
        metrics['2d-multi-input-heart-locator']['all']['myo']['ed']['hd'].append(hd)
        metrics['2d-multi-input-heart-locator']['all']['myo']['total']['dc'].append(dc)
        metrics['2d-multi-input-heart-locator']['all']['myo']['total']['hd'].append(hd)

    # LV
    dc, hd, _, _, _ = compute_metrics(gt[:len(pred_multi_input_heart_locator_2d)//2], pred_multi_input_heart_locator_2d[:len(pred_multi_input_heart_locator_2d)//2], [3], pat_spacing[pat])
    if not np.isnan(dc) and not np.isnan(hd):
        metrics['2d-multi-input-heart-locator'][folder]['lv']['ed']['dc'].append(dc)
        metrics['2d-multi-input-heart-locator'][folder]['lv']['ed']['hd'].append(hd)
        metrics['2d-multi-input-heart-locator'][folder]['lv']['total']['dc'].append(dc)
        metrics['2d-multi-input-heart-locator'][folder]['lv']['total']['hd'].append(hd)
        metrics['2d-multi-input-heart-locator']['all']['lv']['ed']['dc'].append(dc)
        metrics['2d-multi-input-heart-locator']['all']['lv']['ed']['hd'].append(hd)
        metrics['2d-multi-input-heart-locator']['all']['lv']['total']['dc'].append(dc)
        metrics['2d-multi-input-heart-locator']['all']['lv']['total']['hd'].append(hd)

    # total
    dc, hd, _, _, _ = compute_metrics(gt[:len(pred_multi_input_heart_locator_2d)//2], pred_multi_input_heart_locator_2d[:len(pred_multi_input_heart_locator_2d)//2], [1, 2, 3], pat_spacing[pat])
    if not np.isnan(dc) and not np.isnan(hd):
        metrics['2d-multi-input-heart-locator'][folder]['total']['ed']['dc'].append(dc)
        metrics['2d-multi-input-heart-locator'][folder]['total']['ed']['hd'].append(hd)
        metrics['2d-multi-input-heart-locator'][folder]['total']['total']['dc'].append(dc)
        metrics['2d-multi-input-heart-locator'][folder]['total']['total']['hd'].append(hd)
        metrics['2d-multi-input-heart-locator']['all']['total']['ed']['dc'].append(dc)
        metrics['2d-multi-input-heart-locator']['all']['total']['ed']['hd'].append(hd)
        metrics['2d-multi-input-heart-locator']['all']['total']['total']['dc'].append(dc)
        metrics['2d-multi-input-heart-locator']['all']['total']['total']['hd'].append(hd)

    # ## ES
    # RV
    dc, hd, _, _, _ = compute_metrics(gt[len(pred_multi_input_heart_locator_2d)//2:], pred_multi_input_heart_locator_2d[len(pred_multi_input_heart_locator_2d)//2:], [1], pat_spacing[pat])
    if not np.isnan(dc) and not np.isnan(hd):
        metrics['2d-multi-input-heart-locator'][folder]['rv']['es']['dc'].append(dc)
        metrics['2d-multi-input-heart-locator'][folder]['rv']['es']['hd'].append(hd)
        metrics['2d-multi-input-heart-locator'][folder]['rv']['total']['dc'].append(dc)
        metrics['2d-multi-input-heart-locator'][folder]['rv']['total']['hd'].append(hd)
        metrics['2d-multi-input-heart-locator']['all']['rv']['es']['dc'].append(dc)
        metrics['2d-multi-input-heart-locator']['all']['rv']['es']['hd'].append(hd)
        metrics['2d-multi-input-heart-locator']['all']['rv']['total']['dc'].append(dc)
        metrics['2d-multi-input-heart-locator']['all']['rv']['total']['hd'].append(hd)

    # MYO
    dc, hd, _, _, _ = compute_metrics(gt[len(pred_multi_input_heart_locator_2d)//2:], pred_multi_input_heart_locator_2d[len(pred_multi_input_heart_locator_2d)//2:], [2], pat_spacing[pat])
    if not np.isnan(dc) and not np.isnan(hd):
        metrics['2d-multi-input-heart-locator'][folder]['myo']['es']['dc'].append(dc)
        metrics['2d-multi-input-heart-locator'][folder]['myo']['es']['hd'].append(hd)
        metrics['2d-multi-input-heart-locator'][folder]['myo']['total']['dc'].append(dc)
        metrics['2d-multi-input-heart-locator'][folder]['myo']['total']['hd'].append(hd)
        metrics['2d-multi-input-heart-locator']['all']['myo']['es']['dc'].append(dc)
        metrics['2d-multi-input-heart-locator']['all']['myo']['es']['hd'].append(hd)
        metrics['2d-multi-input-heart-locator']['all']['myo']['total']['dc'].append(dc)
        metrics['2d-multi-input-heart-locator']['all']['myo']['total']['hd'].append(hd)

    # LV
    dc, hd, _, _, _ = compute_metrics(gt[len(pred_multi_input_heart_locator_2d)//2:], pred_multi_input_heart_locator_2d[len(pred_multi_input_heart_locator_2d)//2:], [3], pat_spacing[pat])
    if not np.isnan(dc) and not np.isnan(hd):
        metrics['2d-multi-input-heart-locator'][folder]['lv']['es']['dc'].append(dc)
        metrics['2d-multi-input-heart-locator'][folder]['lv']['es']['hd'].append(hd)
        metrics['2d-multi-input-heart-locator'][folder]['lv']['total']['dc'].append(dc)
        metrics['2d-multi-input-heart-locator'][folder]['lv']['total']['hd'].append(hd)
        metrics['2d-multi-input-heart-locator']['all']['lv']['es']['dc'].append(dc)
        metrics['2d-multi-input-heart-locator']['all']['lv']['es']['hd'].append(hd)
        metrics['2d-multi-input-heart-locator']['all']['lv']['total']['dc'].append(dc)
        metrics['2d-multi-input-heart-locator']['all']['lv']['total']['hd'].append(hd)

    # total
    dc, hd, _, _, _ = compute_metrics(gt[len(pred_multi_input_heart_locator_2d)//2:], pred_multi_input_heart_locator_2d[len(pred_multi_input_heart_locator_2d)//2:], [1, 2, 3], pat_spacing[pat])
    if not np.isnan(dc) and not np.isnan(hd):
        metrics['2d-multi-input-heart-locator'][folder]['total']['es']['dc'].append(dc)
        metrics['2d-multi-input-heart-locator'][folder]['total']['es']['hd'].append(hd)
        metrics['2d-multi-input-heart-locator'][folder]['total']['total']['dc'].append(dc)
        metrics['2d-multi-input-heart-locator'][folder]['total']['total']['hd'].append(hd)
        metrics['2d-multi-input-heart-locator']['all']['total']['es']['dc'].append(dc)
        metrics['2d-multi-input-heart-locator']['all']['total']['es']['hd'].append(hd)
        metrics['2d-multi-input-heart-locator']['all']['total']['total']['dc'].append(dc)
        metrics['2d-multi-input-heart-locator']['all']['total']['total']['hd'].append(hd)

    # ##### 2D++ Multi Input Heart Locator

    # ## ED
    # RV
    dc, hd, _, _, _ = compute_metrics(gt[:len(pred_multi_input_heart_locator_2dpp)//2], pred_multi_input_heart_locator_2dpp[:len(pred_multi_input_heart_locator_2dpp)//2], [1], pat_spacing[pat])
    if not np.isnan(dc) and not np.isnan(hd):
        metrics['2dpp-multi-input-heart-locator'][folder]['rv']['ed']['dc'].append(dc)
        metrics['2dpp-multi-input-heart-locator'][folder]['rv']['ed']['hd'].append(hd)
        metrics['2dpp-multi-input-heart-locator'][folder]['rv']['total']['dc'].append(dc)
        metrics['2dpp-multi-input-heart-locator'][folder]['rv']['total']['hd'].append(hd)
        metrics['2dpp-multi-input-heart-locator']['all']['rv']['ed']['dc'].append(dc)
        metrics['2dpp-multi-input-heart-locator']['all']['rv']['ed']['hd'].append(hd)
        metrics['2dpp-multi-input-heart-locator']['all']['rv']['total']['dc'].append(dc)
        metrics['2dpp-multi-input-heart-locator']['all']['rv']['total']['hd'].append(hd)

    # MYO
    dc, hd, _, _, _ = compute_metrics(gt[:len(pred_multi_input_heart_locator_2dpp)//2], pred_multi_input_heart_locator_2dpp[:len(pred_multi_input_heart_locator_2dpp)//2], [2], pat_spacing[pat])
    if not np.isnan(dc) and not np.isnan(hd):
        metrics['2dpp-multi-input-heart-locator'][folder]['myo']['ed']['dc'].append(dc)
        metrics['2dpp-multi-input-heart-locator'][folder]['myo']['ed']['hd'].append(hd)
        metrics['2dpp-multi-input-heart-locator'][folder]['myo']['total']['dc'].append(dc)
        metrics['2dpp-multi-input-heart-locator'][folder]['myo']['total']['hd'].append(hd)
        metrics['2dpp-multi-input-heart-locator']['all']['myo']['ed']['dc'].append(dc)
        metrics['2dpp-multi-input-heart-locator']['all']['myo']['ed']['hd'].append(hd)
        metrics['2dpp-multi-input-heart-locator']['all']['myo']['total']['dc'].append(dc)
        metrics['2dpp-multi-input-heart-locator']['all']['myo']['total']['hd'].append(hd)

    # LV
    dc, hd, _, _, _ = compute_metrics(gt[:len(pred_multi_input_heart_locator_2dpp)//2], pred_multi_input_heart_locator_2dpp[:len(pred_multi_input_heart_locator_2dpp)//2:], [3], pat_spacing[pat])
    if not np.isnan(dc) and not np.isnan(hd):
        metrics['2dpp-multi-input-heart-locator'][folder]['lv']['ed']['dc'].append(dc)
        metrics['2dpp-multi-input-heart-locator'][folder]['lv']['ed']['hd'].append(hd)
        metrics['2dpp-multi-input-heart-locator'][folder]['lv']['total']['dc'].append(dc)
        metrics['2dpp-multi-input-heart-locator'][folder]['lv']['total']['hd'].append(hd)
        metrics['2dpp-multi-input-heart-locator']['all']['lv']['ed']['dc'].append(dc)
        metrics['2dpp-multi-input-heart-locator']['all']['lv']['ed']['hd'].append(hd)
        metrics['2dpp-multi-input-heart-locator']['all']['lv']['total']['dc'].append(dc)
        metrics['2dpp-multi-input-heart-locator']['all']['lv']['total']['hd'].append(hd)

    # total
    dc, hd, _, _, _ = compute_metrics(gt[:len(pred_multi_input_heart_locator_2dpp)//2], pred_multi_input_heart_locator_2dpp[:len(pred_multi_input_heart_locator_2dpp)//2], [1, 2, 3], pat_spacing[pat])
    if not np.isnan(dc) and not np.isnan(hd):
        metrics['2dpp-multi-input-heart-locator'][folder]['total']['ed']['dc'].append(dc)
        metrics['2dpp-multi-input-heart-locator'][folder]['total']['ed']['hd'].append(hd)
        metrics['2dpp-multi-input-heart-locator'][folder]['total']['total']['dc'].append(dc)
        metrics['2dpp-multi-input-heart-locator'][folder]['total']['total']['hd'].append(hd)
        metrics['2dpp-multi-input-heart-locator']['all']['total']['ed']['dc'].append(dc)
        metrics['2dpp-multi-input-heart-locator']['all']['total']['ed']['hd'].append(hd)
        metrics['2dpp-multi-input-heart-locator']['all']['total']['total']['dc'].append(dc)
        metrics['2dpp-multi-input-heart-locator']['all']['total']['total']['hd'].append(hd)

    # ## ES
    # RV
    dc, hd, _, _, _ = compute_metrics(gt[len(pred_multi_input_heart_locator_2dpp)//2:], pred_multi_input_heart_locator_2dpp[len(pred_multi_input_heart_locator_2dpp)//2:], [1], pat_spacing[pat])
    if not np.isnan(dc) and not np.isnan(hd):
        metrics['2dpp-multi-input-heart-locator'][folder]['rv']['es']['dc'].append(dc)
        metrics['2dpp-multi-input-heart-locator'][folder]['rv']['es']['hd'].append(hd)
        metrics['2dpp-multi-input-heart-locator'][folder]['rv']['total']['dc'].append(dc)
        metrics['2dpp-multi-input-heart-locator'][folder]['rv']['total']['hd'].append(hd)
        metrics['2dpp-multi-input-heart-locator']['all']['rv']['es']['dc'].append(dc)
        metrics['2dpp-multi-input-heart-locator']['all']['rv']['es']['hd'].append(hd)
        metrics['2dpp-multi-input-heart-locator']['all']['rv']['total']['dc'].append(dc)
        metrics['2dpp-multi-input-heart-locator']['all']['rv']['total']['hd'].append(hd)

    # MYO
    dc, hd, _, _, _ = compute_metrics(gt[len(pred_multi_input_heart_locator_2dpp)//2:], pred_multi_input_heart_locator_2dpp[len(pred_multi_input_heart_locator_2dpp)//2:], [2], pat_spacing[pat])
    if not np.isnan(dc) and not np.isnan(hd):
        metrics['2dpp-multi-input-heart-locator'][folder]['myo']['es']['dc'].append(dc)
        metrics['2dpp-multi-input-heart-locator'][folder]['myo']['es']['hd'].append(hd)
        metrics['2dpp-multi-input-heart-locator'][folder]['myo']['total']['dc'].append(dc)
        metrics['2dpp-multi-input-heart-locator'][folder]['myo']['total']['hd'].append(hd)
        metrics['2dpp-multi-input-heart-locator']['all']['myo']['es']['dc'].append(dc)
        metrics['2dpp-multi-input-heart-locator']['all']['myo']['es']['hd'].append(hd)
        metrics['2dpp-multi-input-heart-locator']['all']['myo']['total']['dc'].append(dc)
        metrics['2dpp-multi-input-heart-locator']['all']['myo']['total']['hd'].append(hd)

    # LV
    dc, hd, _, _, _ = compute_metrics(gt[len(pred_multi_input_heart_locator_2dpp)//2:], pred_multi_input_heart_locator_2dpp[len(pred_multi_input_heart_locator_2dpp)//2:], [3], pat_spacing[pat])
    if not np.isnan(dc) and not np.isnan(hd):
        metrics['2dpp-multi-input-heart-locator'][folder]['lv']['es']['dc'].append(dc)
        metrics['2dpp-multi-input-heart-locator'][folder]['lv']['es']['hd'].append(hd)
        metrics['2dpp-multi-input-heart-locator'][folder]['lv']['total']['dc'].append(dc)
        metrics['2dpp-multi-input-heart-locator'][folder]['lv']['total']['hd'].append(hd)
        metrics['2dpp-multi-input-heart-locator']['all']['lv']['es']['dc'].append(dc)
        metrics['2dpp-multi-input-heart-locator']['all']['lv']['es']['hd'].append(hd)
        metrics['2dpp-multi-input-heart-locator']['all']['lv']['total']['dc'].append(dc)
        metrics['2dpp-multi-input-heart-locator']['all']['lv']['total']['hd'].append(hd)

    # total
    dc, hd, _, _, _ = compute_metrics(gt[len(pred_multi_input_heart_locator_2dpp)//2:], pred_multi_input_heart_locator_2dpp[len(pred_multi_input_heart_locator_2dpp)//2:], [1, 2, 3], pat_spacing[pat])
    if not np.isnan(dc) and not np.isnan(hd):
        metrics['2dpp-multi-input-heart-locator'][folder]['total']['es']['dc'].append(dc)
        metrics['2dpp-multi-input-heart-locator'][folder]['total']['es']['hd'].append(hd)
        metrics['2dpp-multi-input-heart-locator'][folder]['total']['total']['dc'].append(dc)
        metrics['2dpp-multi-input-heart-locator'][folder]['total']['total']['hd'].append(hd)
        metrics['2dpp-multi-input-heart-locator']['all']['total']['es']['dc'].append(dc)
        metrics['2dpp-multi-input-heart-locator']['all']['total']['es']['hd'].append(hd)
        metrics['2dpp-multi-input-heart-locator']['all']['total']['total']['dc'].append(dc)
        metrics['2dpp-multi-input-heart-locator']['all']['total']['total']['hd'].append(hd)

    # ##### 3D Multi Input Heart Locator

    # ## ED
    # RV
    dc, hd, _, _, _ = compute_metrics(gt[:len(pred_multi_input_heart_locator_3d)//2], pred_multi_input_heart_locator_3d[:len(pred_multi_input_heart_locator_3d)//2], [1], pat_spacing[pat])
    if not np.isnan(dc) and not np.isnan(hd):
        metrics['3d-multi-input-heart-locator'][folder]['rv']['ed']['dc'].append(dc)
        metrics['3d-multi-input-heart-locator'][folder]['rv']['ed']['hd'].append(hd)
        metrics['3d-multi-input-heart-locator'][folder]['rv']['total']['dc'].append(dc)
        metrics['3d-multi-input-heart-locator'][folder]['rv']['total']['hd'].append(hd)
        metrics['3d-multi-input-heart-locator']['all']['rv']['ed']['dc'].append(dc)
        metrics['3d-multi-input-heart-locator']['all']['rv']['ed']['hd'].append(hd)
        metrics['3d-multi-input-heart-locator']['all']['rv']['total']['dc'].append(dc)
        metrics['3d-multi-input-heart-locator']['all']['rv']['total']['hd'].append(hd)

    # MYO
    dc, hd, _, _, _ = compute_metrics(gt[:len(pred_multi_input_heart_locator_3d)//2], pred_multi_input_heart_locator_3d[:len(pred_multi_input_heart_locator_3d)//2], [2], pat_spacing[pat])
    if not np.isnan(dc) and not np.isnan(hd):
        metrics['3d-multi-input-heart-locator'][folder]['myo']['ed']['dc'].append(dc)
        metrics['3d-multi-input-heart-locator'][folder]['myo']['ed']['hd'].append(hd)
        metrics['3d-multi-input-heart-locator'][folder]['myo']['total']['dc'].append(dc)
        metrics['3d-multi-input-heart-locator'][folder]['myo']['total']['hd'].append(hd)
        metrics['3d-multi-input-heart-locator']['all']['myo']['ed']['dc'].append(dc)
        metrics['3d-multi-input-heart-locator']['all']['myo']['ed']['hd'].append(hd)
        metrics['3d-multi-input-heart-locator']['all']['myo']['total']['dc'].append(dc)
        metrics['3d-multi-input-heart-locator']['all']['myo']['total']['hd'].append(hd)

    # LV
    dc, hd, _, _, _ = compute_metrics(gt[:len(pred_multi_input_heart_locator_3d)//2], pred_multi_input_heart_locator_3d[:len(pred_multi_input_heart_locator_3d)//2], [3], pat_spacing[pat])
    if not np.isnan(dc) and not np.isnan(hd):
        metrics['3d-multi-input-heart-locator'][folder]['lv']['ed']['dc'].append(dc)
        metrics['3d-multi-input-heart-locator'][folder]['lv']['ed']['hd'].append(hd)
        metrics['3d-multi-input-heart-locator'][folder]['lv']['total']['dc'].append(dc)
        metrics['3d-multi-input-heart-locator'][folder]['lv']['total']['hd'].append(hd)
        metrics['3d-multi-input-heart-locator']['all']['lv']['ed']['dc'].append(dc)
        metrics['3d-multi-input-heart-locator']['all']['lv']['ed']['hd'].append(hd)
        metrics['3d-multi-input-heart-locator']['all']['lv']['total']['dc'].append(dc)
        metrics['3d-multi-input-heart-locator']['all']['lv']['total']['hd'].append(hd)

    # total
    dc, hd, _, _, _ = compute_metrics(gt[:len(pred_multi_input_heart_locator_3d)//2], pred_multi_input_heart_locator_3d[:len(pred_multi_input_heart_locator_3d)//2], [1, 2, 3], pat_spacing[pat])
    if not np.isnan(dc) and not np.isnan(hd):
        metrics['3d-multi-input-heart-locator'][folder]['total']['ed']['dc'].append(dc)
        metrics['3d-multi-input-heart-locator'][folder]['total']['ed']['hd'].append(hd)
        metrics['3d-multi-input-heart-locator'][folder]['total']['total']['dc'].append(dc)
        metrics['3d-multi-input-heart-locator'][folder]['total']['total']['hd'].append(hd)
        metrics['3d-multi-input-heart-locator']['all']['total']['ed']['dc'].append(dc)
        metrics['3d-multi-input-heart-locator']['all']['total']['ed']['hd'].append(hd)
        metrics['3d-multi-input-heart-locator']['all']['total']['total']['dc'].append(dc)
        metrics['3d-multi-input-heart-locator']['all']['total']['total']['hd'].append(hd)

    # ## ES
    # RV
    dc, hd, _, _, _ = compute_metrics(gt[len(pred_multi_input_heart_locator_3d)//2:], pred_multi_input_heart_locator_3d[len(pred_multi_input_heart_locator_3d)//2:], [1], pat_spacing[pat])
    if not np.isnan(dc) and not np.isnan(hd):
        metrics['3d-multi-input-heart-locator'][folder]['rv']['es']['dc'].append(dc)
        metrics['3d-multi-input-heart-locator'][folder]['rv']['es']['hd'].append(hd)
        metrics['3d-multi-input-heart-locator'][folder]['rv']['total']['dc'].append(dc)
        metrics['3d-multi-input-heart-locator'][folder]['rv']['total']['hd'].append(hd)
        metrics['3d-multi-input-heart-locator']['all']['rv']['es']['dc'].append(dc)
        metrics['3d-multi-input-heart-locator']['all']['rv']['es']['hd'].append(hd)
        metrics['3d-multi-input-heart-locator']['all']['rv']['total']['dc'].append(dc)
        metrics['3d-multi-input-heart-locator']['all']['rv']['total']['hd'].append(hd)

    # MYO
    dc, hd, _, _, _ = compute_metrics(gt[len(pred_multi_input_heart_locator_3d)//2:], pred_multi_input_heart_locator_3d[len(pred_multi_input_heart_locator_3d)//2:], [2], pat_spacing[pat])
    if not np.isnan(dc) and not np.isnan(hd):
        metrics['3d-multi-input-heart-locator'][folder]['myo']['es']['dc'].append(dc)
        metrics['3d-multi-input-heart-locator'][folder]['myo']['es']['hd'].append(hd)
        metrics['3d-multi-input-heart-locator'][folder]['myo']['total']['dc'].append(dc)
        metrics['3d-multi-input-heart-locator'][folder]['myo']['total']['hd'].append(hd)
        metrics['3d-multi-input-heart-locator']['all']['myo']['es']['dc'].append(dc)
        metrics['3d-multi-input-heart-locator']['all']['myo']['es']['hd'].append(hd)
        metrics['3d-multi-input-heart-locator']['all']['myo']['total']['dc'].append(dc)
        metrics['3d-multi-input-heart-locator']['all']['myo']['total']['hd'].append(hd)

    # LV
    dc, hd, _, _, _ = compute_metrics(gt[len(pred_multi_input_heart_locator_3d)//2:], pred_multi_input_heart_locator_3d[len(pred_multi_input_heart_locator_3d)//2:], [3], pat_spacing[pat])
    if not np.isnan(dc) and not np.isnan(hd):
        metrics['3d-multi-input-heart-locator'][folder]['lv']['es']['dc'].append(dc)
        metrics['3d-multi-input-heart-locator'][folder]['lv']['es']['hd'].append(hd)
        metrics['3d-multi-input-heart-locator'][folder]['lv']['total']['dc'].append(dc)
        metrics['3d-multi-input-heart-locator'][folder]['lv']['total']['hd'].append(hd)
        metrics['3d-multi-input-heart-locator']['all']['lv']['es']['dc'].append(dc)
        metrics['3d-multi-input-heart-locator']['all']['lv']['es']['hd'].append(hd)
        metrics['3d-multi-input-heart-locator']['all']['lv']['total']['dc'].append(dc)
        metrics['3d-multi-input-heart-locator']['all']['lv']['total']['hd'].append(hd)

    # total
    dc, hd, _, _, _ = compute_metrics(gt[len(pred_multi_input_heart_locator_3d)//2:], pred_multi_input_heart_locator_3d[len(pred_multi_input_heart_locator_3d)//2:], [1, 2, 3], pat_spacing[pat])
    if not np.isnan(dc) and not np.isnan(hd):
        metrics['3d-multi-input-heart-locator'][folder]['total']['es']['dc'].append(dc)
        metrics['3d-multi-input-heart-locator'][folder]['total']['es']['hd'].append(hd)
        metrics['3d-multi-input-heart-locator'][folder]['total']['total']['dc'].append(dc)
        metrics['3d-multi-input-heart-locator'][folder]['total']['total']['hd'].append(hd)
        metrics['3d-multi-input-heart-locator']['all']['total']['es']['dc'].append(dc)
        metrics['3d-multi-input-heart-locator']['all']['total']['es']['hd'].append(hd)
        metrics['3d-multi-input-heart-locator']['all']['total']['total']['dc'].append(dc)
        metrics['3d-multi-input-heart-locator']['all']['total']['total']['hd'].append(hd)

    # ##### 2D Multi Input Weighted Average

    # ## ED
    # RV
    dc, hd, _, _, _ = compute_metrics(gt[:len(pred_multi_input_avg_2d)//2], pred_multi_input_avg_2d[:len(pred_multi_input_avg_2d)//2], [1], pat_spacing[pat])
    if not np.isnan(dc) and not np.isnan(hd):
        metrics['2d-multi-input-avg'][folder]['rv']['ed']['dc'].append(dc)
        metrics['2d-multi-input-avg'][folder]['rv']['ed']['hd'].append(hd)
        metrics['2d-multi-input-avg'][folder]['rv']['total']['dc'].append(dc)
        metrics['2d-multi-input-avg'][folder]['rv']['total']['hd'].append(hd)
        metrics['2d-multi-input-avg']['all']['rv']['ed']['dc'].append(dc)
        metrics['2d-multi-input-avg']['all']['rv']['ed']['hd'].append(hd)
        metrics['2d-multi-input-avg']['all']['rv']['total']['dc'].append(dc)
        metrics['2d-multi-input-avg']['all']['rv']['total']['hd'].append(hd)

    # MYO
    dc, hd, _, _, _ = compute_metrics(gt[:len(pred_multi_input_avg_2d)//2], pred_multi_input_avg_2d[:len(pred_multi_input_avg_2d)//2], [2], pat_spacing[pat])
    if not np.isnan(dc) and not np.isnan(hd):
        metrics['2d-multi-input-avg'][folder]['myo']['ed']['dc'].append(dc)
        metrics['2d-multi-input-avg'][folder]['myo']['ed']['hd'].append(hd)
        metrics['2d-multi-input-avg'][folder]['myo']['total']['dc'].append(dc)
        metrics['2d-multi-input-avg'][folder]['myo']['total']['hd'].append(hd)
        metrics['2d-multi-input-avg']['all']['myo']['ed']['dc'].append(dc)
        metrics['2d-multi-input-avg']['all']['myo']['ed']['hd'].append(hd)
        metrics['2d-multi-input-avg']['all']['myo']['total']['dc'].append(dc)
        metrics['2d-multi-input-avg']['all']['myo']['total']['hd'].append(hd)

    # LV
    dc, hd, _, _, _ = compute_metrics(gt[:len(pred_multi_input_avg_2d)//2], pred_multi_input_avg_2d[:len(pred_multi_input_avg_2d)//2], [3], pat_spacing[pat])
    if not np.isnan(dc) and not np.isnan(hd):
        metrics['2d-multi-input-avg'][folder]['lv']['ed']['dc'].append(dc)
        metrics['2d-multi-input-avg'][folder]['lv']['ed']['hd'].append(hd)
        metrics['2d-multi-input-avg'][folder]['lv']['total']['dc'].append(dc)
        metrics['2d-multi-input-avg'][folder]['lv']['total']['hd'].append(hd)
        metrics['2d-multi-input-avg']['all']['lv']['ed']['dc'].append(dc)
        metrics['2d-multi-input-avg']['all']['lv']['ed']['hd'].append(hd)
        metrics['2d-multi-input-avg']['all']['lv']['total']['dc'].append(dc)
        metrics['2d-multi-input-avg']['all']['lv']['total']['hd'].append(hd)

    # total
    dc, hd, _, _, _ = compute_metrics(gt[:len(pred_multi_input_avg_2d)//2], pred_multi_input_avg_2d[:len(pred_multi_input_avg_2d)//2], [1, 2, 3], pat_spacing[pat])
    if not np.isnan(dc) and not np.isnan(hd):
        metrics['2d-multi-input-avg'][folder]['total']['ed']['dc'].append(dc)
        metrics['2d-multi-input-avg'][folder]['total']['ed']['hd'].append(hd)
        metrics['2d-multi-input-avg'][folder]['total']['total']['dc'].append(dc)
        metrics['2d-multi-input-avg'][folder]['total']['total']['hd'].append(hd)
        metrics['2d-multi-input-avg']['all']['total']['ed']['dc'].append(dc)
        metrics['2d-multi-input-avg']['all']['total']['ed']['hd'].append(hd)
        metrics['2d-multi-input-avg']['all']['total']['total']['dc'].append(dc)
        metrics['2d-multi-input-avg']['all']['total']['total']['hd'].append(hd)

    # ## ES
    # RV
    dc, hd, _, _, _ = compute_metrics(gt[len(pred_multi_input_avg_2d)//2:], pred_multi_input_avg_2d[len(pred_multi_input_avg_2d)//2:], [1], pat_spacing[pat])
    if not np.isnan(dc) and not np.isnan(hd):
        metrics['2d-multi-input-avg'][folder]['rv']['es']['dc'].append(dc)
        metrics['2d-multi-input-avg'][folder]['rv']['es']['hd'].append(hd)
        metrics['2d-multi-input-avg'][folder]['rv']['total']['dc'].append(dc)
        metrics['2d-multi-input-avg'][folder]['rv']['total']['hd'].append(hd)
        metrics['2d-multi-input-avg']['all']['rv']['es']['dc'].append(dc)
        metrics['2d-multi-input-avg']['all']['rv']['es']['hd'].append(hd)
        metrics['2d-multi-input-avg']['all']['rv']['total']['dc'].append(dc)
        metrics['2d-multi-input-avg']['all']['rv']['total']['hd'].append(hd)

    # MYO
    dc, hd, _, _, _ = compute_metrics(gt[len(pred_multi_input_avg_2d)//2:], pred_multi_input_avg_2d[len(pred_multi_input_avg_2d)//2:], [2], pat_spacing[pat])
    if not np.isnan(dc) and not np.isnan(hd):
        metrics['2d-multi-input-avg'][folder]['myo']['es']['dc'].append(dc)
        metrics['2d-multi-input-avg'][folder]['myo']['es']['hd'].append(hd)
        metrics['2d-multi-input-avg'][folder]['myo']['total']['dc'].append(dc)
        metrics['2d-multi-input-avg'][folder]['myo']['total']['hd'].append(hd)
        metrics['2d-multi-input-avg']['all']['myo']['es']['dc'].append(dc)
        metrics['2d-multi-input-avg']['all']['myo']['es']['hd'].append(hd)
        metrics['2d-multi-input-avg']['all']['myo']['total']['dc'].append(dc)
        metrics['2d-multi-input-avg']['all']['myo']['total']['hd'].append(hd)

    # LV
    dc, hd, _, _, _ = compute_metrics(gt[len(pred_multi_input_avg_2d)//2:], pred_multi_input_avg_2d[len(pred_multi_input_avg_2d)//2:], [3], pat_spacing[pat])
    if not np.isnan(dc) and not np.isnan(hd):
        metrics['2d-multi-input-avg'][folder]['lv']['es']['dc'].append(dc)
        metrics['2d-multi-input-avg'][folder]['lv']['es']['hd'].append(hd)
        metrics['2d-multi-input-avg'][folder]['lv']['total']['dc'].append(dc)
        metrics['2d-multi-input-avg'][folder]['lv']['total']['hd'].append(hd)
        metrics['2d-multi-input-avg']['all']['lv']['es']['dc'].append(dc)
        metrics['2d-multi-input-avg']['all']['lv']['es']['hd'].append(hd)
        metrics['2d-multi-input-avg']['all']['lv']['total']['dc'].append(dc)
        metrics['2d-multi-input-avg']['all']['lv']['total']['hd'].append(hd)

    # total
    dc, hd, _, _, _ = compute_metrics(gt[len(pred_multi_input_avg_2d)//2:], pred_multi_input_avg_2d[len(pred_multi_input_avg_2d)//2:], [1, 2, 3], pat_spacing[pat])
    if not np.isnan(dc) and not np.isnan(hd):
        metrics['2d-multi-input-avg'][folder]['total']['es']['dc'].append(dc)
        metrics['2d-multi-input-avg'][folder]['total']['es']['hd'].append(hd)
        metrics['2d-multi-input-avg'][folder]['total']['total']['dc'].append(dc)
        metrics['2d-multi-input-avg'][folder]['total']['total']['hd'].append(hd)
        metrics['2d-multi-input-avg']['all']['total']['es']['dc'].append(dc)
        metrics['2d-multi-input-avg']['all']['total']['es']['hd'].append(hd)
        metrics['2d-multi-input-avg']['all']['total']['total']['dc'].append(dc)
        metrics['2d-multi-input-avg']['all']['total']['total']['hd'].append(hd)

    # ##### Multi Input Weighted Average

    # ## ED
    # RV
    dc, hd, _, _, _ = compute_metrics(gt[:len(pred_multi_input_avg)//2], pred_multi_input_avg[:len(pred_multi_input_avg)//2], [1], pat_spacing[pat])
    if not np.isnan(dc) and not np.isnan(hd):
        metrics['multi-input-avg'][folder]['rv']['ed']['dc'].append(dc)
        metrics['multi-input-avg'][folder]['rv']['ed']['hd'].append(hd)
        metrics['multi-input-avg'][folder]['rv']['total']['dc'].append(dc)
        metrics['multi-input-avg'][folder]['rv']['total']['hd'].append(hd)
        metrics['multi-input-avg']['all']['rv']['ed']['dc'].append(dc)
        metrics['multi-input-avg']['all']['rv']['ed']['hd'].append(hd)
        metrics['multi-input-avg']['all']['rv']['total']['dc'].append(dc)
        metrics['multi-input-avg']['all']['rv']['total']['hd'].append(hd)

    # MYO
    dc, hd, _, _, _ = compute_metrics(gt[:len(pred_multi_input_avg)//2], pred_multi_input_avg[:len(pred_multi_input_avg)//2], [2], pat_spacing[pat])
    if not np.isnan(dc) and not np.isnan(hd):
        metrics['multi-input-avg'][folder]['myo']['ed']['dc'].append(dc)
        metrics['multi-input-avg'][folder]['myo']['ed']['hd'].append(hd)
        metrics['multi-input-avg'][folder]['myo']['total']['dc'].append(dc)
        metrics['multi-input-avg'][folder]['myo']['total']['hd'].append(hd)
        metrics['multi-input-avg']['all']['myo']['ed']['dc'].append(dc)
        metrics['multi-input-avg']['all']['myo']['ed']['hd'].append(hd)
        metrics['multi-input-avg']['all']['myo']['total']['dc'].append(dc)
        metrics['multi-input-avg']['all']['myo']['total']['hd'].append(hd)

    # LV
    dc, hd, _, _, _ = compute_metrics(gt[:len(pred_multi_input_avg)//2], pred_multi_input_avg[:len(pred_multi_input_avg)//2], [3], pat_spacing[pat])
    if not np.isnan(dc) and not np.isnan(hd):
        metrics['multi-input-avg'][folder]['lv']['ed']['dc'].append(dc)
        metrics['multi-input-avg'][folder]['lv']['ed']['hd'].append(hd)
        metrics['multi-input-avg'][folder]['lv']['total']['dc'].append(dc)
        metrics['multi-input-avg'][folder]['lv']['total']['hd'].append(hd)
        metrics['multi-input-avg']['all']['lv']['ed']['dc'].append(dc)
        metrics['multi-input-avg']['all']['lv']['ed']['hd'].append(hd)
        metrics['multi-input-avg']['all']['lv']['total']['dc'].append(dc)
        metrics['multi-input-avg']['all']['lv']['total']['hd'].append(hd)

    # total
    dc, hd, _, _, _ = compute_metrics(gt[:len(pred_multi_input_avg)//2], pred_multi_input_avg[:len(pred_multi_input_avg)//2], [1, 2, 3], pat_spacing[pat])
    if not np.isnan(dc) and not np.isnan(hd):
        metrics['multi-input-avg'][folder]['total']['ed']['dc'].append(dc)
        metrics['multi-input-avg'][folder]['total']['ed']['hd'].append(hd)
        metrics['multi-input-avg'][folder]['total']['total']['dc'].append(dc)
        metrics['multi-input-avg'][folder]['total']['total']['hd'].append(hd)
        metrics['multi-input-avg']['all']['total']['ed']['dc'].append(dc)
        metrics['multi-input-avg']['all']['total']['ed']['hd'].append(hd)
        metrics['multi-input-avg']['all']['total']['total']['dc'].append(dc)
        metrics['multi-input-avg']['all']['total']['total']['hd'].append(hd)

    # ## ES
    # RV
    dc, hd, _, _, _ = compute_metrics(gt[len(pred_multi_input_avg)//2:], pred_multi_input_avg[len(pred_multi_input_avg)//2:], [1], pat_spacing[pat])
    if not np.isnan(dc) and not np.isnan(hd):
        metrics['multi-input-avg'][folder]['rv']['es']['dc'].append(dc)
        metrics['multi-input-avg'][folder]['rv']['es']['hd'].append(hd)
        metrics['multi-input-avg'][folder]['rv']['total']['dc'].append(dc)
        metrics['multi-input-avg'][folder]['rv']['total']['hd'].append(hd)
        metrics['multi-input-avg']['all']['rv']['es']['dc'].append(dc)
        metrics['multi-input-avg']['all']['rv']['es']['hd'].append(hd)
        metrics['multi-input-avg']['all']['rv']['total']['dc'].append(dc)
        metrics['multi-input-avg']['all']['rv']['total']['hd'].append(hd)

    # MYO
    dc, hd, _, _, _ = compute_metrics(gt[len(pred_multi_input_avg)//2:], pred_multi_input_avg[len(pred_multi_input_avg)//2:], [2], pat_spacing[pat])
    if not np.isnan(dc) and not np.isnan(hd):
        metrics['multi-input-avg'][folder]['myo']['es']['dc'].append(dc)
        metrics['multi-input-avg'][folder]['myo']['es']['hd'].append(hd)
        metrics['multi-input-avg'][folder]['myo']['total']['dc'].append(dc)
        metrics['multi-input-avg'][folder]['myo']['total']['hd'].append(hd)
        metrics['multi-input-avg']['all']['myo']['es']['dc'].append(dc)
        metrics['multi-input-avg']['all']['myo']['es']['hd'].append(hd)
        metrics['multi-input-avg']['all']['myo']['total']['dc'].append(dc)
        metrics['multi-input-avg']['all']['myo']['total']['hd'].append(hd)

    # LV
    dc, hd, _, _, _ = compute_metrics(gt[len(pred_multi_input_avg)//2:], pred_multi_input_avg[len(pred_multi_input_avg)//2:], [3], pat_spacing[pat])
    if not np.isnan(dc) and not np.isnan(hd):
        metrics['multi-input-avg'][folder]['lv']['es']['dc'].append(dc)
        metrics['multi-input-avg'][folder]['lv']['es']['hd'].append(hd)
        metrics['multi-input-avg'][folder]['lv']['total']['dc'].append(dc)
        metrics['multi-input-avg'][folder]['lv']['total']['hd'].append(hd)
        metrics['multi-input-avg']['all']['lv']['es']['dc'].append(dc)
        metrics['multi-input-avg']['all']['lv']['es']['hd'].append(hd)
        metrics['multi-input-avg']['all']['lv']['total']['dc'].append(dc)
        metrics['multi-input-avg']['all']['lv']['total']['hd'].append(hd)

    # total
    dc, hd, _, _, _ = compute_metrics(gt[len(pred_multi_input_avg)//2:], pred_multi_input_avg[len(pred_multi_input_avg)//2:], [1, 2, 3], pat_spacing[pat])
    if not np.isnan(dc) and not np.isnan(hd):
        metrics['multi-input-avg'][folder]['total']['es']['dc'].append(dc)
        metrics['multi-input-avg'][folder]['total']['es']['hd'].append(hd)
        metrics['multi-input-avg'][folder]['total']['total']['dc'].append(dc)
        metrics['multi-input-avg'][folder]['total']['total']['hd'].append(hd)
        metrics['multi-input-avg']['all']['total']['es']['dc'].append(dc)
        metrics['multi-input-avg']['all']['total']['es']['hd'].append(hd)
        metrics['multi-input-avg']['all']['total']['total']['dc'].append(dc)
        metrics['multi-input-avg']['all']['total']['total']['hd'].append(hd)

    Path('./samples/' + str(pat)).mkdir(parents=True, exist_ok=True)

    for i in range(len(gt)):
        input_img = Image.fromarray((input_to_image(data[i]) * 255).astype(np.uint8))
        input_img.save('./samples/' + str(pat) + '/input-' + str(i).zfill(3) + '.png', format='png')

        gt_img = Image.fromarray((mask_to_image(gt[i]) * 255).astype(np.uint8))
        gt_img.putalpha(255)
        new_img = Image.new('RGBA', (input_img.width, input_img.height), (0, 0, 0, 0))
        new_img.paste(gt_img, (0, 0), mask=gt_img)
        new_img.save('./samples/' + str(pat) + '/gt-' + str(i).zfill(3) + '.png', format='png')

        gt_img.putalpha(102)
        new_img = Image.new('RGBA', (input_img.width, input_img.height), (0, 0, 0, 0))
        new_img.paste(input_img, (0, 0))
        new_img.paste(gt_img, (0, 0), mask=gt_img)
        new_img.save('./samples/' + str(pat) + '/gt-input-' + str(i).zfill(3) + '.png', format='png')

        pred_2d_img = Image.fromarray((mask_to_image(pred_multi_input_heart_locator_2d[i]) * 255).astype(np.uint8))
        pred_2d_img.putalpha(102)
        new_img = Image.new('RGBA', (input_img.width, input_img.height), (0, 0, 0, 0))
        new_img.paste(input_img, (0, 0))
        new_img.paste(pred_2d_img, (0, 0), mask=pred_2d_img)
        new_img.save('./samples/' + str(pat) + '/2d-multi-input-heart-locator-' + str(i).zfill(3) + '.png', format='png')

        pred_2d_img = Image.fromarray((mask_to_image(pred_multi_input_heart_locator_3d[i]) * 255).astype(np.uint8))
        pred_2d_img.putalpha(102)
        new_img = Image.new('RGBA', (input_img.width, input_img.height), (0, 0, 0, 0))
        new_img.paste(input_img, (0, 0))
        new_img.paste(pred_2d_img, (0, 0), mask=pred_2d_img)
        new_img.save('./samples/' + str(pat) + '/3d-multi-input-heart-locator-' + str(i).zfill(3) + '.png', format='png')

        pred_2d_img = Image.fromarray((mask_to_image(pred_multi_input_heart_locator_2dpp[i]) * 255).astype(np.uint8))
        pred_2d_img.putalpha(102)
        new_img = Image.new('RGBA', (input_img.width, input_img.height), (0, 0, 0, 0))
        new_img.paste(input_img, (0, 0))
        new_img.paste(pred_2d_img, (0, 0), mask=pred_2d_img)
        new_img.save('./samples/' + str(pat) + '/2dpp-multi-input-heart-locator-' + str(i).zfill(3) + '.png', format='png')

        pred_2d_img = Image.fromarray((mask_to_image(pred_multi_input_avg[i]) * 255).astype(np.uint8))
        pred_2d_img.putalpha(102)
        new_img = Image.new('RGBA', (input_img.width, input_img.height), (0, 0, 0, 0))
        new_img.paste(input_img, (0, 0))
        new_img.paste(pred_2d_img, (0, 0), mask=pred_2d_img)
        new_img.save('./samples/' + str(pat) + '/multi-input-avg-' + str(i).zfill(3) + '.png', format='png')

        pred_2d_img = Image.fromarray((mask_to_image(pred_multi_input_avg_2d[i]) * 255).astype(np.uint8))
        pred_2d_img.putalpha(102)
        new_img = Image.new('RGBA', (input_img.width, input_img.height), (0, 0, 0, 0))
        new_img.paste(input_img, (0, 0))
        new_img.paste(pred_2d_img, (0, 0), mask=pred_2d_img)
        new_img.save('./samples/' + str(pat) + '/2d-multi-input-avg-' + str(i).zfill(3) + '.png', format='png')

for mt in model_types:
    print(mt)

    for mn in ['all']:
        print(mn)

        print('ED')
        print('LV')
        print('D: ', np.mean(metrics[mt][mn]['lv']['ed']['dc']))
        print('dH: ', np.mean(metrics[mt][mn]['lv']['ed']['hd']))
        print('RV')
        print('D: ', np.mean(metrics[mt][mn]['rv']['ed']['dc']))
        print('dH: ', np.mean(metrics[mt][mn]['rv']['ed']['hd']))
        print('MYO')
        print('D: ', np.mean(metrics[mt][mn]['myo']['ed']['dc']))
        print('dH: ', np.mean(metrics[mt][mn]['myo']['ed']['hd']))
        print('Total')
        print('D: ', np.mean(metrics[mt][mn]['total']['ed']['dc']))
        print('dH: ', np.mean(metrics[mt][mn]['total']['ed']['hd']))

        print('ES')
        print('LV')
        print('D: ', np.mean(metrics[mt][mn]['lv']['es']['dc']))
        print('dH: ', np.mean(metrics[mt][mn]['lv']['es']['hd']))
        print('RV')
        print('D: ', np.mean(metrics[mt][mn]['rv']['es']['dc']))
        print('dH: ', np.mean(metrics[mt][mn]['rv']['es']['hd']))
        print('MYO')
        print('D: ', np.mean(metrics[mt][mn]['myo']['es']['dc']))
        print('dH: ', np.mean(metrics[mt][mn]['myo']['es']['hd']))
        print('Total')
        print('D: ', np.mean(metrics[mt][mn]['total']['es']['dc']))
        print('dH: ', np.mean(metrics[mt][mn]['total']['es']['hd']))

        print('Total')
        print('LV')
        print('D: ', np.mean(metrics[mt][mn]['lv']['total']['dc']))
        print('dH: ', np.mean(metrics[mt][mn]['lv']['total']['hd']))
        print('RV')
        print('D: ', np.mean(metrics[mt][mn]['rv']['total']['dc']))
        print('dH: ', np.mean(metrics[mt][mn]['rv']['total']['hd']))
        print('MYO')
        print('D: ', np.mean(metrics[mt][mn]['myo']['total']['dc']))
        print('dH: ', np.mean(metrics[mt][mn]['myo']['total']['hd']))
        print('Total')
        print('D: ', np.mean(metrics[mt][mn]['total']['total']['dc']))
        print('dH: ', np.mean(metrics[mt][mn]['total']['total']['hd']))
