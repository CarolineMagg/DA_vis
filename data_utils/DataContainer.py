########################################################################################################################
# DataContainer to load nifti files in patient data folder
########################################################################################################################
import copy
import json
import logging

import numpy as np
import cv2
import os.path
import nibabel as nib

__author__ = "c.magg"


class DataContainer:
    """
    DataContainer is a container for a nifti folders with different information like:
    * T1
    * T2
    * segmentation of VS/cochlea
    """

    def __init__(self, dir_path, alpha=0, beta=1):
        """
        Create a new DataContainer object.
        :param dir_path: path to nifti directory with t1, t2, vs and cochlea segmentation file
        """
        self._path_dir = dir_path
        files = os.listdir(dir_path)
        self._path_t1 = os.path.join(dir_path, f"vs_gk_t1_refT1_processed_{alpha}_{beta}.nii")
        self._path_t2 = os.path.join(dir_path, f"vs_gk_t2_refT1_processed_{alpha}_{beta}.nii")
        self._path_vs = os.path.join(dir_path, "vs_gk_struc1_refT1_processed.nii")
        self._data_t1 = None
        self._data_t2 = None
        self._data_vs = None
        self.load()

    def __len__(self):
        return self._data_t1.shape[2]

    @property
    def data(self):
        """
        Data dictionary with modality as key and data arrays as values.
        """
        return {'t1': self.t1_scan,
                't2': self.t2_scan,
                'vs': self.vs_segm}

    @property
    def shape(self):
        return self._data_t1.shape

    def load(self):
        """
        (Re)Load the data from nifti paths.
        """
        self._data_t1 = nib.load(self._path_t1)
        self._data_t2 = nib.load(self._path_t2)
        self._data_vs = nib.load(self._path_vs)

    def uncache(self):
        """
        Uncache the nifti container.
        """
        self._data_t1.uncache()
        self._data_t2.uncache()
        self._data_vs.uncache()

    @property
    def t1_scan(self):
        return np.asarray(self._data_t1.dataobj, dtype=np.float32)

    @property
    def t2_scan(self):
        return np.asarray(self._data_t2.dataobj, dtype=np.float32)

    @property
    def vs_segm(self):
        return np.asarray(self._data_vs.dataobj, dtype=np.int16)

    @property
    def vs_class(self):
        return [1 if np.sum(self.vs_segm[:, :, idx]) != 0 else 0 for idx in range(0, self.vs_segm.shape[2])]

    def t1_scan_slice(self, index=None):
        return np.asarray(self._data_t1.dataobj[..., index], dtype=np.float32)

    def t2_scan_slice(self, index=None):
        return np.asarray(self._data_t2.dataobj[..., index], dtype=np.float32)

    def vs_segm_slice(self, index=None):
        return np.asarray(self._data_vs.dataobj[..., index], dtype=np.int16)

    def vs_class_slice(self, index=None):
        return self.vs_class[index]

    @staticmethod
    def process_mask_to_contour(segm):
        contours = []
        if type(segm) == list or len(segm.shape) == 3:
            for i in range(len(segm)):
                contour, _ = cv2.findContours(segm[i].astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
                contour = [c.tolist() for c in contour]
                contours.append(contour)
        elif len(segm.shape) == 2:
            contours, _ = cv2.findContours(segm.astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        return contours

    @staticmethod
    def process_contour_to_mask(contour):
        pass
