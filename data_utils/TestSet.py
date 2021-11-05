########################################################################################################################
# TestSet containing information about test dataset
########################################################################################################################
import logging
import os
import time

import pandas as pd
import numpy as np
from natsort import natsorted
from sklearn.metrics import confusion_matrix

from data_utils.DataContainer import DataContainer

__author__ = "c.magg"


class TestSet:
    """
    TestSet is a container for the test set information:
    * T1 and T2 images
    * ground truth for VS segmentation
    * predictions for VS segmentation
    * pandas dataframe with information
    """

    MODELS_SIMPLE1 = ["XNet_T2_relu", "XNet_T2_leaky", "XNet_T2_selu"]
    MODELS_SIMPLE2 = ["XNet_T1_relu", "XNet_T1_leaky", "XNet_T1_selu"]
    MODELS_SIMPLE = [*MODELS_SIMPLE1, *MODELS_SIMPLE2]
    MODELS_CG = ["CG_XNet_T1_relu", "CG_XNet_T2_relu"]
    MODELS_DA = ["SegmS2T_GAN1_relu", "SegmS2T_GAN2_relu", "SegmS2T_GAN5_relu",
                 "CG_SegmS2T_GAN1_relu", "CG_SegmS2T_GAN2_relu", "CG_SegmS2T_GAN5_relu"]
    MODELS_GAN = ["GAN_1+XNet_T1_relu", "GAN_2+XNet_T1_relu", "GAN_5+XNet_T1_relu",
                  "GAN_1+CG_XNet_T1_relu", "GAN_2+CG_XNet_T1_relu", "GAN_5+CG_XNet_T1_relu"]
    MODELS = [*MODELS_SIMPLE, *MODELS_CG, *MODELS_GAN, *MODELS_DA]

    def __init__(self, path="/tf/workdir/data/VS_segm/VS_registered/test_processed/",
                 dsize=(256, 256), load=False, data_load=False, evaluation_load=True):
        self._path = path
        self._folders = natsorted(os.listdir(path))
        self._dsize = dsize
        self._data_container_0_1 = []
        self._data_container_1_1 = []
        self._df_evaluation = []
        self.patient_ids = []
        self._df_total = pd.DataFrame()

        if load:
            self.load_data(data_load, evaluation_load)

    @staticmethod
    def lookup_data_call():
        """
        Generate the mapping between data naming and data loading method in DataContainer.
        :return lookup dictionary with data name as key and method name as value
        """
        return {'t1': 't1_scan_slice',
                't2': 't2_scan_slice',
                'vs': 'segm_vs_slice',
                'cochlea': 'segm_cochlea_slice'}

    @property
    def all_models(self):
        return self.MODELS

    def load_data(self, data=False, evaluation=True):
        logging.info("Load data.")
        patient_ids = []
        if data or evaluation:
            for f in self._folders:
                folder = os.path.join(self._path, f)
                # load NIFTI data
                if data:
                    self._data_container_0_1.append(DataContainer(folder, alpha=0, beta=1))
                    self._data_container_1_1.append(DataContainer(folder, alpha=-1, beta=1))
                # load prediction information
                if evaluation:
                    self._df_evaluation.append(pd.read_json(os.path.join(folder, "evaluation.json")))
                patient_ids.append(f.split("/")[-1].split("_")[-1])
            self.patient_ids = patient_ids

        file_path = "/tf/workdir/DA_vis/data_utils/evaluation_all.json"
        if os.path.isfile(file_path):
            logging.info("Load all evaluation information.")
            self._df_total = pd.read_json(file_path)

    def get_patient_data(self, idx, alpha=0, beta=1):
        if alpha == 0 and beta == 1:
            return self._data_container_0_1[idx]
        elif alpha == -1 and beta == 1:
            return self._data_container_1_1[idx]

    @staticmethod
    def calculate_accuracy(conf_mat):
        if type(conf_mat) == dict:
            return (conf_mat["tp"] + conf_mat["tn"]) / (conf_mat["tp"] + conf_mat["tn"] + conf_mat["fp"] + conf_mat["fn"])
        else:
            return (conf_mat[3] + conf_mat[0]) / (conf_mat[3] + conf_mat[0] + conf_mat[1] + conf_mat[2])

    @staticmethod
    def calculate_tpr(conf_mat):
        if type(conf_mat) == dict:
            return (conf_mat["tp"]) / (conf_mat["tp"] + conf_mat["fn"])
        else:
            return (conf_mat[3]) / (conf_mat[3] + conf_mat[2])

    @staticmethod
    def calculate_tnr(conf_mat):
        if type(conf_mat) == dict:
            return (conf_mat["tn"]) / (conf_mat["tn"] + conf_mat["fp"]) if conf_mat["tn"] != 0 else 1.0
        else:
            return (conf_mat[0]) / (conf_mat[0] + conf_mat[1]) if conf_mat[0] != 0 else 1.0

    @property
    def df_total(self):
        if len(self._df_total) == 0:
            logging.info("Generate total df.")
            # init dataframes
            # all slices
            df_dice_all = pd.DataFrame(columns=["id", *self.MODELS])
            df_assd_all = pd.DataFrame(columns=["id", *self.MODELS])
            df_conf_mat_all = pd.DataFrame(columns=["id", *self.MODELS])
            df_acc_all = pd.DataFrame(columns=["id", *self.MODELS])
            df_tpr_all = pd.DataFrame(columns=["id", *self.MODELS])
            df_tnr_all = pd.DataFrame(columns=["id", *self.MODELS])
            df_dice_all["id"] = [*self.patient_ids, "DSC"]
            df_assd_all["id"] = [*self.patient_ids, "ASSD"]
            df_conf_mat_all["id"] = [*self.patient_ids, "conf_mat"]
            df_acc_all["id"] = [*self.patient_ids, "ACC"]
            df_tpr_all["id"] = [*self.patient_ids, "TPR"]
            df_tnr_all["id"] = [*self.patient_ids, "TNR"]
            # only tumor slices
            df_dice_only_tumor = pd.DataFrame(columns=["id", *self.MODELS])
            df_assd_only_tumor = pd.DataFrame(columns=["id", *self.MODELS])
            df_conf_mat_only_tumor = pd.DataFrame(columns=["id", *self.MODELS])
            df_acc_only_tumor = pd.DataFrame(columns=["id", *self.MODELS])
            df_tpr_only_tumor = pd.DataFrame(columns=["id", *self.MODELS])
            df_tnr_only_tumor = pd.DataFrame(columns=["id", *self.MODELS])
            df_dice_only_tumor["id"] = [*self.patient_ids, "DSC"]
            df_assd_only_tumor["id"] = [*self.patient_ids, "ASSD"]
            df_conf_mat_only_tumor["id"] = [*self.patient_ids, "conf_mat"]
            df_acc_only_tumor["id"] = [*self.patient_ids, "ACC"]
            df_tpr_only_tumor["id"] = [*self.patient_ids, "TPR"]
            df_tnr_only_tumor["id"] = [*self.patient_ids, "TNR"]

            for name in self.MODELS:
                dice_list = []
                dice_list_tumor = []
                assd_list = []
                assd_list_tumor = []
                tp_list = []
                tn_list = []
                fn_list = []
                fp_list = []
                tp_list_tumor = []
                tn_list_tumor = []
                fn_list_tumor = []
                fp_list_tumor = []
                cols_number = df_dice_all.columns.get_loc(name)
                for idx, df in enumerate(self._df_evaluation):
                    # all slices
                    df_dice_all.iloc[idx, cols_number] = np.mean(df[f"VS_segm_dice-{name}"].values)
                    df_assd_all.iloc[idx, cols_number] = np.mean(df[f"VS_segm_assd-{name}"].values)
                    dice_list += list(df[f"VS_segm_dice-{name}"].values)
                    assd_list += list(df[f"VS_segm_assd-{name}"].values)
                    tn, fp, fn, tp = confusion_matrix(df["VS_class_gt"].values,
                                                      df[f"VS_class_pred-{name}"].values,
                                                      labels=[0, 1]).ravel()
                    tp_list.append(tp)
                    tn_list.append(tn)
                    fn_list.append(fn)
                    fp_list.append(fp)
                    conf_mat_all = {"tp": tp, "tn": tn, "fn": fn, "fp": fp}
                    df_conf_mat_all.iloc[idx, cols_number] = str(conf_mat_all)
                    df_acc_all.iloc[idx, cols_number] = self.calculate_accuracy(conf_mat_all)
                    df_tpr_all.iloc[idx, cols_number] = self.calculate_tpr(conf_mat_all)
                    df_tnr_all.iloc[idx, cols_number] = self.calculate_tnr(conf_mat_all)
                    # only tumor slices
                    df_dice_only_tumor.iloc[idx, cols_number] = np.mean(
                        df[df["VS_class_gt"] == 1][f"VS_segm_dice-{name}"].values)
                    df_assd_only_tumor.iloc[idx, cols_number] = np.mean(
                        df[df["VS_class_gt"] == 1][f"VS_segm_assd-{name}"].values)
                    dice_list_tumor += list(df[df["VS_class_gt"] == 1][f"VS_segm_dice-{name}"].values)
                    assd_list_tumor += list(df[df["VS_class_gt"] == 1][f"VS_segm_assd-{name}"].values)
                    tn, fp, fn, tp = confusion_matrix(df[df["VS_class_gt"] == 1]["VS_class_gt"].values,
                                                      df[df["VS_class_gt"] == 1][f"VS_class_pred-{name}"].values,
                                                      labels=[0, 1]).ravel()
                    tp_list_tumor.append(tp)
                    tn_list_tumor.append(tn)
                    fn_list_tumor.append(fn)
                    fp_list_tumor.append(fp)
                    conf_mat = {"tp": tp, "tn": tn, "fn": fn, "fp": fp}
                    df_conf_mat_only_tumor.iloc[idx, cols_number] = str(conf_mat)
                    df_acc_only_tumor.iloc[idx, cols_number] = self.calculate_accuracy(conf_mat)
                    df_tpr_only_tumor.iloc[idx, cols_number] = self.calculate_tpr(conf_mat)
                    df_tnr_only_tumor.iloc[idx, cols_number] = self.calculate_tnr(conf_mat)
                df_dice_all.iloc[-1, cols_number] = np.mean(dice_list)
                df_assd_all.iloc[-1, cols_number] = np.mean(assd_list)
                conf_mat_all = {"tp": np.sum(tp_list), "tn": np.sum(tn_list),
                                "fn": np.sum(fn_list), "fp": np.sum(fp_list)}
                df_acc_all.iloc[-1, cols_number] = self.calculate_accuracy(conf_mat_all)
                df_tpr_all.iloc[-1, cols_number] = self.calculate_tpr(conf_mat_all)
                df_tnr_all.iloc[-1, cols_number] = self.calculate_tnr(conf_mat_all)
                df_conf_mat_all.iloc[-1, cols_number] = str(conf_mat_all)
                df_dice_only_tumor.iloc[-1, cols_number] = np.mean(dice_list_tumor)
                df_assd_only_tumor.iloc[-1, cols_number] = np.mean(assd_list_tumor)
                conf_mat = {"tp": np.sum(tp_list_tumor), "tn": np.sum(tn_list_tumor),
                            "fn": np.sum(fn_list_tumor), "fp": np.sum(fp_list_tumor)}
                df_conf_mat_only_tumor.iloc[-1, cols_number] = str(conf_mat)
                df_acc_only_tumor.iloc[-1, cols_number] = self.calculate_accuracy(conf_mat)
                df_tpr_only_tumor.iloc[-1, cols_number] = self.calculate_tpr(conf_mat)
                df_tnr_only_tumor.iloc[-1, cols_number] = self.calculate_tnr(conf_mat)
            collect_df = pd.DataFrame()
            collect_df = collect_df.append({"dice_all": df_dice_all,
                                            "dice_only_tumor": df_dice_only_tumor,
                                            "assd_all": df_assd_all,
                                            "assd_only_tumor": df_assd_only_tumor,
                                            "conf_mat_all": df_conf_mat_all,
                                            "conf_mat_only_tumor": df_conf_mat_only_tumor,
                                            "acc_all": df_acc_all,
                                            "acc_only_tumor": df_acc_only_tumor,
                                            "tpr_all": df_tpr_all,
                                            "tpr_only_tumor": df_tpr_only_tumor,
                                            "tnr_all": df_tnr_all,
                                            "tnr_only_tumor": df_tnr_only_tumor}, ignore_index=True)
            self._df_total = collect_df
        return self._df_total

    def __len__(self):
        return len(self._data_container_0_1)


if __name__ == "__main__":
    start = time.time()
    testset = TestSet("/tf/workdir/data/VS_segm/VS_registered/test_processed/", load=True,
                      data_load=True, evaluation_load=True)
    df_total = testset.df_total
    df_total.to_json("/tf/workdir/DA_vis/data_utils/evaluation_all.json")
    print(time.time() - start)
