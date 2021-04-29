import numpy as np

from skimage.morphology import label

import SimpleITK as sitk
import os
from skimage.transform import resize

from dataset_utils import generate_test_patient_info
from pathlib import Path
from PIL import Image

DIR_2D = './pred-2dn/'
DIR_2DPP = './pred-2dnpp/'
DIR_3D = './pred-3dn/'

WEIGHT_MSE_2D_2DPP_3D = {
    1: [[0.33339201], [0.33334009], [0.33326789]],
    2: [[0.3333324], [0.3333339], [0.33333369]],
    3: [[0.33359929], [0.33353806], [0.33286265]],
    4: [[0.33349437], [0.33342641], [0.33307922]],
    5: [[0.33346418], [0.33347459], [0.33306123]],
}

WEIGHT_MODELS = [0.19980226, 0.1999886, 0.20004647, 0.20013502, 0.20002765]


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


def resize_pred(image, new_shape, n_channels=4, pad_value=0):
    data = np.zeros((new_shape[0], new_shape[1], new_shape[2]))
    data += pad_value
    s_offest = (new_shape[0] - image.shape[0]) // 2
    h_offest = (new_shape[1] - image.shape[1]) // 2
    w_offest = (new_shape[2] - image.shape[2]) // 2

    t_s_s = max(s_offest, 0)
    t_s_e = t_s_s + min(image.shape[0] + s_offest, image.shape[0]) - max(0, -s_offest)
    t_h_s = max(h_offest, 0)
    t_h_e = t_h_s + min(image.shape[1] + h_offest, image.shape[1]) - max(0, -h_offest)
    t_w_s = max(w_offest, 0)
    t_w_e = t_w_s + min(image.shape[2] + w_offest, image.shape[2]) - max(0, -w_offest)

    s_s_s = max(0, -s_offest)
    s_s_e = s_s_s + t_s_e - t_s_s
    s_h_s = max(0, -h_offest)
    s_h_e = s_h_s + t_h_e - t_h_s
    s_w_s = max(0, -w_offest)
    s_w_e = s_w_s + t_w_e - t_w_s

    data[t_s_s:t_s_e, t_h_s:t_h_e, t_w_s:t_w_e] = image[s_s_s:s_s_e, s_h_s:s_h_e, s_w_s:s_w_e]
    return data


def resize_image(image, old_spacing, new_spacing, order=3):
    new_shape = (int(np.round(old_spacing[0]/new_spacing[0]*float(image.shape[0]))),
                 int(np.round(old_spacing[1]/new_spacing[1]*float(image.shape[1]))),
                 int(np.round(old_spacing[2]/new_spacing[2]*float(image.shape[2]))))
    return resize(image, new_shape, order=order, mode='edge')


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


def postprocess_prediction(seg):
    # basically look for connected components and choose the largest one, delete everything else
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
            if sb >= 0:
                seg = postprocess_slice(seg, sb)
            if sa <= len(seg) - 1:
                seg = postprocess_slice(seg, sa)
            if sb <= 0 and sa >= len(seg) - 1:
                break
    return seg


dataset_base_dir = './testing'
patient_info = generate_test_patient_info(dataset_base_dir)

patients = np.arange(101, 151, 1)

for pat in patients:

    pred_combined_list = list()

    for i in range(5):
        model_n = i + 1
        folder = str(model_n)

        pred_multi_input_heart_locator_2d = np.load(
            DIR_2D + folder + '/' + str(pat).zfill(3) + '-multi-input-heart-locator.npy', allow_pickle=True)
        pred_multi_input_heart_locator_2dpp = np.load(
            DIR_2DPP + folder + '/' + str(pat).zfill(3) + '-multi-input-heart-locator.npy', allow_pickle=True)
        pred_multi_input_heart_locator_3d = np.load(
            DIR_3D + folder + '/' + str(pat).zfill(3) + '-multi-input-heart-locator.npy', allow_pickle=True)

        ed_idx = np.arange(0, len(pred_multi_input_heart_locator_2d), 2)
        es_idx = np.arange(1, len(pred_multi_input_heart_locator_2d), 2)

        images_idx = np.concatenate([ed_idx, es_idx])

        pred_combined_list.append(WEIGHT_MSE_2D_2DPP_3D[model_n][0] * pred_multi_input_heart_locator_2d
                                  + WEIGHT_MSE_2D_2DPP_3D[model_n][1] * pred_multi_input_heart_locator_2dpp
                                  + WEIGHT_MSE_2D_2DPP_3D[model_n][2] * pred_multi_input_heart_locator_3d)

    pred_combined = np.zeros(pred_combined_list[0].shape)

    for i in range(len(pred_combined_list)):
        pred_combined += WEIGHT_MODELS[i] * pred_combined_list[i]

    pred_combined_ed = pred_combined[:len(pred_combined) // 2]
    pred_combined_es = pred_combined[len(pred_combined) // 2:]

    raw_itk_ed = sitk.ReadImage(os.path.join(dataset_base_dir, "patient%03.0d" % pat,
                                             "patient%03.0d_frame%02.0d.nii.gz" %
                                             (pat, patient_info[pat]['ed'])))
    raw_ed = sitk.GetArrayFromImage(raw_itk_ed)
    
    old_spacing_ed = np.array(raw_itk_ed.GetSpacing())[[2, 1, 0]]
    target_spacing_ed = [old_spacing_ed[0], 1.25, 1.25]

    raw_itk_es = sitk.ReadImage(os.path.join(dataset_base_dir, "patient%03.0d" % pat,
                                             "patient%03.0d_frame%02.0d.nii.gz" %
                                             (pat, patient_info[pat]['es'])))
    raw_es = sitk.GetArrayFromImage(raw_itk_es)
    
    old_spacing_es = np.array(raw_itk_es.GetSpacing())[[2, 1, 0]]
    target_spacing_es = [old_spacing_es[0], 1.25, 1.25]
    
    print(raw_ed.shape, pred_combined_ed.shape)
    
    pred_combined_ed = resize_image(np.argmax(pred_combined_ed, axis=-1).astype(float), target_spacing_ed, old_spacing_ed, 3)
    pred_combined_es = resize_image(np.argmax(pred_combined_es, axis=-1).astype(float), target_spacing_es, old_spacing_es, 3)
    
    pred_combined_ed = np.round(pred_combined_ed).astype(np.int)
    pred_combined_es = np.round(pred_combined_es).astype(np.int)

    pred_combined_ed = resize_pred(pred_combined_ed, raw_ed.shape)
    pred_combined_es = resize_pred(pred_combined_es, raw_es.shape)
    
    pred_combined_ed = postprocess_prediction(pred_combined_ed)
    pred_combined_es = postprocess_prediction(pred_combined_es)
    
    Path('./test-output/' + str(pat)).mkdir(parents=True, exist_ok=True)

    for i in range(len(pred_combined_ed)):
        pred_combined_ed_img = Image.fromarray((mask_to_image(pred_combined_ed[i]) * 255).astype(np.uint8))
        pred_combined_ed_img.putalpha(102)
        input_img = Image.fromarray(raw_ed[i])
        new_img = Image.new('RGBA', (input_img.width, input_img.height), (0, 0, 0, 0))
        new_img.paste(input_img, (0, 0))
        new_img.paste(pred_combined_ed_img, (0, 0), mask=pred_combined_ed_img)
        new_img.save('./test-output/' + str(pat) + '/ed-' + str(i).zfill(3) + '.png', format='png')
    
    for i in range(len(pred_combined_es)):
        pred_combined_es_img = Image.fromarray((mask_to_image(pred_combined_es[i]) * 255).astype(np.uint8))
        pred_combined_es_img.putalpha(102)
        input_img = Image.fromarray(raw_es[i])
        new_img = Image.new('RGBA', (input_img.width, input_img.height), (0, 0, 0, 0))
        new_img.paste(input_img, (0, 0))
        new_img.paste(pred_combined_es_img, (0, 0), mask=pred_combined_es_img)
        new_img.save('./test-output/' + str(pat) + '/es-' + str(i).zfill(3) + '.png', format='png')
        
    itk_seg_ed = sitk.GetImageFromArray(pred_combined_ed.astype(np.uint8))
    itk_seg_ed.CopyInformation(raw_itk_ed)
    sitk.WriteImage(itk_seg_ed, os.path.join('./test-output', "patient%03.0d_%s.nii.gz" % (pat, 'ED')))

    itk_seg_es = sitk.GetImageFromArray(pred_combined_es.astype(np.uint8))
    itk_seg_es.CopyInformation(raw_itk_es)
    sitk.WriteImage(itk_seg_es, os.path.join('./test-output', "patient%03.0d_%s.nii.gz" % (pat, 'ES')))
