import os
from copy import deepcopy
from typing import Union, List

import numpy as np
import torch
from acvl_utils.cropping_and_padding.bounding_boxes import bounding_box_to_slice, insert_crop_into_image
from batchgenerators.utilities.file_and_folder_operations import load_json, isfile, save_pickle

from nnunetv2.configuration import default_num_processes
from nnunetv2.training.dataloading.nnunet_dataset import nnUNetDatasetBlosc2
from nnunetv2.utilities.label_handling.label_handling import LabelManager
from nnunetv2.utilities.plans_handling.plans_handler import PlansManager, ConfigurationManager

import threading
from filelock import FileLock
import pandas as pd
import cc3d

# # NOTE: Tianyu's code
thread_lock = threading.Lock()  # NOTE: all threads shared the same threading lock
class FileLockForAllUsers(FileLock):
    def _acquire(self):
        # 调用父类方法创建并加锁
        super()._acquire()
        # 将锁文件权限修改为 777，确保所有用户都能访问
        try:
            os.chmod(self.lock_file, 0o777)
        except Exception as e:
            print(f"\033[31mError found when giving lock file 777 permission: {e}\033[0m")
            # 这里可以记录日志，处理失败情况
            pass

# # NOTE: modified from Xinze's code
def keep_largest_component_multi_cls(seg_data, labels):
    largest_cc = np.zeros_like(seg_data)
    seg_data_mask = np.isin(seg_data, labels).astype(np.uint8)
    # print((np.isin(seg_data, labels)).shape)
    labeled, N = cc3d.connected_components(seg_data_mask==1, connectivity=26, return_N=True)
    max_component = None
    max_size = 0
    for i in range(1, N + 1):
        component = (labeled == i)
        size = np.sum(component)
        if size > max_size:
            max_component = component
            max_size = size
    if max_component is not None:
        # seg_data[seg_data_mask==1] = 0
        largest_cc[max_component] = 1
    return largest_cc


def get_shared_properties(pred_pancreas_tumor, probabilities_final, spacing, label_id):
    has_label = int((pred_pancreas_tumor==label_id).sum() > 0)
    # largest_cc = np.zeros_like(seg_data)
    # seg_data_mask = np.isin(seg_data, labels).astype(np.uint8)
    # # print((np.isin(seg_data, labels)).shape)
    labeled, N = cc3d.connected_components(pred_pancreas_tumor==label_id, connectivity=26, return_N=True)
    voxel_size_list = []
    volume_size_list = []
    largest_logits_list = []
    for i in range(1, N + 1):
        component = (labeled == i)
        voxel_size = int(np.sum(component))
        volume_size = round(float(voxel_size * spacing[0] * spacing[1] * spacing[2]), 2)
        if probabilities_final is not None:
            largest_logits = round(float(probabilities_final[label_id][component].max()), 2)
        else: 
            largest_logits = -1.    # `probabilities_final` is None
        voxel_size_list.append(voxel_size)
        volume_size_list.append(volume_size)
        largest_logits_list.append(largest_logits)
    # print(voxel_size_list, volume_size_list, largest_logits_list)

    if N > 0:
        sorted_info = sorted(zip(voxel_size_list, volume_size_list, largest_logits_list), reverse=True) # sort based on the first row
        voxel_size_list, volume_size_list, largest_logits_list = list(zip(*sorted_info))    # sort based on the fist list
        return has_label, N, list(voxel_size_list), list(volume_size_list), largest_logits_list[0]  # get only the largest logit in the largest cc
    else:
        return has_label, N, list(voxel_size_list), list(volume_size_list), 0.

def parse_prediction_new_row(pred_pancreas_tumor, probabilities_final, properties_dict, output_file_truncated, dataset_json_dict_or_file):
    bdmap_id = output_file_truncated.split("/")[-1]
    shape = (int(properties_dict["shape_before_cropping"][1]), int(properties_dict["shape_before_cropping"][2]), int(properties_dict["shape_before_cropping"][0]))  # (C H W) -> (H W C)
    spacing = (float(properties_dict["spacing"][1]), float(properties_dict["spacing"][2]), float(properties_dict["spacing"][0]))    # (C H W) -> (H W C)
    spcaing_for_volume_size = (spacing[2], spacing[0], spacing[1])      # (H W C) -> (C H W) tmp for calculating volume size

    pancreas, pancreas_component_count, pancreas_voxel_size, pancreas_volume_size, _ = get_shared_properties(pred_pancreas_tumor, probabilities_final, spcaing_for_volume_size, dataset_json_dict_or_file["labels"]["pancreas"])
    duct, duct_component_count, duct_voxel_size, duct_volume_size, _ = get_shared_properties(pred_pancreas_tumor, probabilities_final, spcaing_for_volume_size, dataset_json_dict_or_file["labels"]["pancreatic_duct"])
    PDAC, PDAC_component_count, PDAC_voxel_size, PDAC_volume_size, PDAC_largest_logit = get_shared_properties(pred_pancreas_tumor, probabilities_final, spcaing_for_volume_size, dataset_json_dict_or_file["labels"]["pancreatic_pdac"])
    cyst, cyst_component_count, cyst_voxel_size, cyst_volume_size, cyst_largest_logit = get_shared_properties(pred_pancreas_tumor, probabilities_final, spcaing_for_volume_size, dataset_json_dict_or_file["labels"]["pancreatic_cyst"])
    PNET, PNET_component_count, PNET_voxel_size, PNET_volume_size, PNET_largest_logit = get_shared_properties(pred_pancreas_tumor, probabilities_final, spcaing_for_volume_size, dataset_json_dict_or_file["labels"]["pancreatic_pnet"])

    return bdmap_id, shape, spacing, \
        pancreas,pancreas_component_count,pancreas_voxel_size,pancreas_volume_size, \
        duct,duct_component_count,duct_voxel_size,duct_volume_size, \
        PDAC,PDAC_component_count,PDAC_voxel_size,PDAC_volume_size,PDAC_largest_logit, \
        cyst,cyst_component_count,cyst_voxel_size,cyst_volume_size,cyst_largest_logit, \
        PNET,PNET_component_count,PNET_voxel_size,PNET_volume_size,PNET_largest_logit



def convert_predicted_logits_to_segmentation_with_correct_shape(predicted_logits: Union[torch.Tensor, np.ndarray],
                                                                plans_manager: PlansManager,
                                                                configuration_manager: ConfigurationManager,
                                                                label_manager: LabelManager,
                                                                properties_dict: dict,
                                                                return_probabilities: bool = False,
                                                                num_threads_torch: int = default_num_processes):
    old_threads = torch.get_num_threads()
    torch.set_num_threads(num_threads_torch)

    # resample to original shape
    spacing_transposed = [properties_dict['spacing'][i] for i in plans_manager.transpose_forward]
    current_spacing = configuration_manager.spacing if \
        len(configuration_manager.spacing) == \
        len(properties_dict['shape_after_cropping_and_before_resampling']) else \
        [spacing_transposed[0], *configuration_manager.spacing]
    predicted_logits = configuration_manager.resampling_fn_probabilities(predicted_logits,
                                            properties_dict['shape_after_cropping_and_before_resampling'],
                                            current_spacing,
                                            [properties_dict['spacing'][i] for i in plans_manager.transpose_forward])
    # return value of resampling_fn_probabilities can be ndarray or Tensor but that does not matter because
    # apply_inference_nonlin will convert to torch
    if not return_probabilities:
        # this has a faster computation path becasue we can skip the softmax in regular (not region based) trainig
        segmentation = label_manager.convert_logits_to_segmentation(predicted_logits)
    else:
        predicted_probabilities = label_manager.apply_inference_nonlin(predicted_logits)
        segmentation = label_manager.convert_probabilities_to_segmentation(predicted_probabilities)
    del predicted_logits

    # put segmentation in bbox (revert cropping)
    segmentation_reverted_cropping = np.zeros(properties_dict['shape_before_cropping'],
                                              dtype=np.uint8 if len(label_manager.foreground_labels) < 255 else np.uint16)
    segmentation_reverted_cropping = insert_crop_into_image(segmentation_reverted_cropping, segmentation, properties_dict['bbox_used_for_cropping'])
    del segmentation

    # segmentation may be torch.Tensor but we continue with numpy
    if isinstance(segmentation_reverted_cropping, torch.Tensor):
        segmentation_reverted_cropping = segmentation_reverted_cropping.cpu().numpy()

    # revert transpose
    segmentation_reverted_cropping = segmentation_reverted_cropping.transpose(plans_manager.transpose_backward)
    if return_probabilities:
        # revert cropping
        predicted_probabilities = label_manager.revert_cropping_on_probabilities(predicted_probabilities,
                                                                                 properties_dict[
                                                                                     'bbox_used_for_cropping'],
                                                                                 properties_dict[
                                                                                     'shape_before_cropping'])
        predicted_probabilities = predicted_probabilities.cpu().numpy()
        # revert transpose
        predicted_probabilities = predicted_probabilities.transpose([0] + [i + 1 for i in
                                                                           plans_manager.transpose_backward])
        torch.set_num_threads(old_threads)
        return segmentation_reverted_cropping, predicted_probabilities
    else:
        torch.set_num_threads(old_threads)
        return segmentation_reverted_cropping


def export_prediction_from_logits(predicted_array_or_file: Union[np.ndarray, torch.Tensor], properties_dict: dict,
                                  configuration_manager: ConfigurationManager,
                                  plans_manager: PlansManager,
                                  dataset_json_dict_or_file: Union[dict, str], output_file_truncated: str,
                                  save_probabilities: bool = False, output_csv_path: str = None,
                                  num_threads_torch: int = default_num_processes):
    # if isinstance(predicted_array_or_file, str):
    #     tmp = deepcopy(predicted_array_or_file)
    #     if predicted_array_or_file.endswith('.npy'):
    #         predicted_array_or_file = np.load(predicted_array_or_file)
    #     elif predicted_array_or_file.endswith('.npz'):
    #         predicted_array_or_file = np.load(predicted_array_or_file)['softmax']
    #     os.remove(tmp)

    if isinstance(dataset_json_dict_or_file, str):
        dataset_json_dict_or_file = load_json(dataset_json_dict_or_file)

    label_manager = plans_manager.get_label_manager(dataset_json_dict_or_file)
    ret = convert_predicted_logits_to_segmentation_with_correct_shape(
        predicted_array_or_file, plans_manager, configuration_manager, label_manager, properties_dict,
        return_probabilities=save_probabilities, num_threads_torch=num_threads_torch
    )
    del predicted_array_or_file

    # save
    if save_probabilities:
        segmentation_final, probabilities_final = ret
        del ret
    else:
        segmentation_final = ret
        del ret

    # NOTE: ePAI all-organ patch for PanTS annotation refinement.
    # Original behavior kept only pancreas/duct/tumor labels.
    # Default keeps all predicted labels; set EPAI_OUTPUT_LABEL_MODE=pancreas_only to restore original filtering.
    if os.environ.get("EPAI_OUTPUT_LABEL_MODE", "all_organs") == "pancreas_only":
        pancreas_and_tumor_labels = [
            dataset_json_dict_or_file["labels"]["pancreas"],
            dataset_json_dict_or_file["labels"]["pancreatic_duct"],
            dataset_json_dict_or_file["labels"]["pancreatic_pdac"],
            dataset_json_dict_or_file["labels"]["pancreatic_cyst"],
            dataset_json_dict_or_file["labels"]["pancreatic_pnet"]
        ]
        segmentation_final_pancreas_and_tumor = np.zeros_like(segmentation_final)
        largest_cc_pred_mask_pancreas_and_tumor = keep_largest_component_multi_cls(segmentation_final, pancreas_and_tumor_labels)
        segmentation_final_pancreas_and_tumor[largest_cc_pred_mask_pancreas_and_tumor==1] = segmentation_final[largest_cc_pred_mask_pancreas_and_tumor==1]
        segmentation_final = segmentation_final_pancreas_and_tumor

    # Always save the segmentation NIfTI (regardless of save_probabilities).
    rw = plans_manager.image_reader_writer_class()
    rw.write_seg(segmentation_final, output_file_truncated + dataset_json_dict_or_file['file_ending'],
                properties_dict)

    if save_probabilities:
        # NOTE: STEP 1 ==> extract the probabilities without saving the entire file!
        new_row = parse_prediction_new_row(segmentation_final, probabilities_final, properties_dict, output_file_truncated, dataset_json_dict_or_file)
        bdmap_id = new_row[0]
        writeable_new_row_wo_bdmap_id = tuple(map(lambda x: str(x) if isinstance(x, list) or isinstance(x, tuple) else x, new_row[1:])) # turn list and tuple to str
        lock_path = output_csv_path + ".lock"   # File Lock to prevent multi-processing problem
        with thread_lock:   # all threads shares the same threading lock
            with FileLockForAllUsers(lock_path, timeout=60):    # prevent `.lock` permission error when using multiple users
                result_csv_df = pd.read_csv(output_csv_path)
                row_index = result_csv_df["bdmap_id"].tolist().index(bdmap_id)
                result_csv_df.iloc[row_index, 1:] = writeable_new_row_wo_bdmap_id
                result_csv_df.to_csv(output_csv_path, index=False)

        del probabilities_final


def resample_and_save(predicted: Union[torch.Tensor, np.ndarray], target_shape: List[int], output_file: str,
                      plans_manager: PlansManager, configuration_manager: ConfigurationManager, properties_dict: dict,
                      dataset_json_dict_or_file: Union[dict, str], num_threads_torch: int = default_num_processes,
                      dataset_class=None) \
        -> None:
    old_threads = torch.get_num_threads()
    torch.set_num_threads(num_threads_torch)

    if isinstance(dataset_json_dict_or_file, str):
        dataset_json_dict_or_file = load_json(dataset_json_dict_or_file)

    spacing_transposed = [properties_dict['spacing'][i] for i in plans_manager.transpose_forward]
    # resample to original shape
    current_spacing = configuration_manager.spacing if \
        len(configuration_manager.spacing) == len(properties_dict['shape_after_cropping_and_before_resampling']) else \
        [spacing_transposed[0], *configuration_manager.spacing]
    target_spacing = configuration_manager.spacing if len(configuration_manager.spacing) == \
        len(properties_dict['shape_after_cropping_and_before_resampling']) else \
        [spacing_transposed[0], *configuration_manager.spacing]
    predicted_array_or_file = configuration_manager.resampling_fn_probabilities(predicted,
                                                                                target_shape,
                                                                                current_spacing,
                                                                                target_spacing)

    # create segmentation (argmax, regions, etc)
    label_manager = plans_manager.get_label_manager(dataset_json_dict_or_file)
    segmentation = label_manager.convert_logits_to_segmentation(predicted_array_or_file)
    # segmentation may be torch.Tensor but we continue with numpy
    if isinstance(segmentation, torch.Tensor):
        segmentation = segmentation.cpu().numpy()

    if dataset_class is None:
        nnUNetDatasetBlosc2.save_seg(segmentation.astype(dtype=np.uint8 if len(label_manager.foreground_labels) < 255 else np.uint16), output_file)
    else:
        dataset_class.save_seg(segmentation.astype(dtype=np.uint8 if len(label_manager.foreground_labels) < 255 else np.uint16), output_file)
    torch.set_num_threads(old_threads)
