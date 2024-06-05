import os
import numpy as np
import pandas as pd
from collections import OrderedDict
from typing import List
from common_tools.file_db_configs import BASE_PATH, CONST
import random
import string
import time


class TSFileDatabaseAPI:
    @staticmethod
    def find_related_paths(univ_type_, keys_list_):
        assert isinstance(keys_list_, (tuple,))
        pure_folders = list()
        mapping_folders = list()
        for root, dirs, files in os.walk(BASE_PATH + "\\" + univ_type_):
            if len(dirs) == 0 and "folder_map.csv" not in files:
                pure_folders.append(os.path.normpath(root))
            if "folder_map.csv" in files:
                mapping_folders.append(os.path.normpath(root))
        assert len(set(pure_folders).intersection(set(mapping_folders))) == 0
        rtn = list()
        for pf in pure_folders:
            pf_lower = pf.lower().split(os.sep)
            is_in = True
            for key in keys_list_:
                if key.lower() not in pf_lower:
                    is_in = False
                if not is_in:
                    break
            if is_in:
                rtn.append(("", pf))
        for mf in mapping_folders:
            mapping_df = pd.read_csv(os.path.join(mf, "folder_map.csv"), index_col=0)
            if not mapping_df.empty:
                for index, item in mapping_df.iterrows():
                    sub_fld = item["sub_fld"].lower()
                    sub_fld_list = sub_fld.replace('[', '.').replace(']', '.').replace(';', '.').replace(',', '.').split('.')
                    folder_name = item["folder_name"].lower()
                    is_in = True
                    for key in keys_list_:
                        if key.lower() not in sub_fld_list:
                            is_in = False
                        if not is_in:
                            break
                    if is_in:
                        rtn.append((sub_fld, os.path.join(mf, folder_name)))
        assert len(rtn) == len(set(rtn))
        rtn = pd.DataFrame(rtn, columns=["keyword", "path"])
        return rtn

    @staticmethod
    def find_cate_fld(univ_type_, cate_fld_, queried_dts_=None):
        path = FolderUtil.get_path(univ_type_, cate_fld_, False)
        is_path_exist = path is not None
        msg_info = dict()
        if is_path_exist:
            if queried_dts_ is None:
                rtn = True
            else:
                queried_univ_calcdates = list(queried_dts_.keys())
                queried_univ_calcdates.sort()
                root_start_dt = queried_dts_[queried_univ_calcdates[0]][0]
                root_end_dt = queried_dts_[queried_univ_calcdates[-1]][-1]
                file_dts = TSFileDatabaseAPI.select_datetimes(univ_type_, cate_fld_, root_start_dt, root_end_dt)

                unfound_univ_calc_dates = list(set(queried_univ_calcdates) - set(file_dts.keys()))
                if len(unfound_univ_calc_dates) == 0:
                    rtn = True
                    unfound_start_date = "2999-12-31 23:59:59"
                    unfound_end_date = "1900-01-01 00:00:00"
                    for univ_calcdate in queried_univ_calcdates:
                        fle_dts_in_1_calcdate = file_dts[univ_calcdate]
                        qrd_dts_in_1_calcdate = queried_dts_[univ_calcdate]
                        if len(set(qrd_dts_in_1_calcdate) - set(fle_dts_in_1_calcdate)) > 0:
                            unfound_start_date = min(unfound_start_date, qrd_dts_in_1_calcdate[0])
                            unfound_end_date = max(unfound_end_date, qrd_dts_in_1_calcdate[-1])
                            rtn = False
                    msg_info["not_found_start_datetime"] = unfound_start_date
                    msg_info["not_found_end_datetime"] = unfound_end_date
                else:
                    unfound_start_date = "2999-12-31 23:59:59"
                    unfound_end_date = "1900-01-01 00:00:00"
                    for u in queried_univ_calcdates:
                        qrd_dts_in_1_calcdate = queried_dts_[u]
                        if u not in unfound_univ_calc_dates:
                            fle_dts_in_1_calcdate = file_dts[u]
                            if len(set(qrd_dts_in_1_calcdate) - set(fle_dts_in_1_calcdate)) > 0:
                                unfound_start_date = min(unfound_start_date, qrd_dts_in_1_calcdate[0])
                                unfound_end_date = max(unfound_end_date, qrd_dts_in_1_calcdate[-1])
                        else:
                            unfound_start_date = min(unfound_start_date, qrd_dts_in_1_calcdate[0])
                            unfound_end_date = max(unfound_end_date, qrd_dts_in_1_calcdate[-1])
                    rtn = False
                    msg_info["not_found_start_datetime"] = unfound_start_date
                    msg_info["not_found_end_datetime"] = unfound_end_date
        else:
            rtn = False
        return rtn, msg_info

    @staticmethod
    def select_data(univ_type_, cate_fld_, start_datetime_, end_datetime_, univ_calc_date_=None, to_numeric_=True):
        assert start_datetime_ <= end_datetime_, cate_fld_ + " start_datetime:" + start_datetime_ + " end_datetime:" + end_datetime_
        path = FolderUtil.get_path(univ_type_, cate_fld_, mk_dirs_=False)
        assert path is not None
        rtn = TSFileDatabaseImpl_r.read_data(path, start_datetime_, end_datetime_, univ_calc_date_, to_numeric_)
        return rtn

    @staticmethod
    def select_datetimes(univ_type_, cate_fld_, start_datetime_, end_datetime_):
        assert False, "TODO:"
        assert start_datetime_ <= end_datetime_, cate_fld_ + " start_datetime:" + start_datetime_ + " end_datetime:" + end_datetime_
        path = FolderUtil.get_path(univ_type_, cate_fld_, mk_dirs_=False)
        rtn = TSFileDatabaseImpl_r.read_datetime_data(path, start_datetime_, end_datetime_)
        return rtn

    @staticmethod
    def replace_data(univ_type_, cate_fld_, univ_calcdate_: str, data_: list, datetime_list_: List[str], header_: List[str]):
        if len(data_) > 0:
            path = FolderUtil.get_path(univ_type_, cate_fld_, mk_dirs_=True)
            assert path is not None
            TSFileDatabaseImpl_w.write_data(path, univ_calcdate_, data_, datetime_list_,
                                            header_)


class TSFileDatabaseImpl_w:
    @staticmethod
    def write_data(path_, univ_calcdate_, to_insert_data_, datetime_list_, header_):
        to_write_data_file, to_write_index_file = TSFileDatabaseImpl_w._get_file_names(univ_calcdate_)
        if not os.path.exists(os.path.join(path_, to_write_index_file)):
            with open(os.path.join(path_, to_write_index_file), "w+") as f:
                f.write("datetime,start_index\n")
        if not os.path.exists(os.path.join(path_, to_write_data_file)):
            with open(os.path.join(path_, to_write_data_file), "w+") as f:
                f.write(",".join(["datetime"] + header_) + "\n")

        is_to_append, dt_keys, new_datetimes = TSFileDatabaseImpl_w._is_to_append(path_, to_write_index_file, datetime_list_)
        if not is_to_append:
            assert isinstance(datetime_list_, (list,)) and isinstance(dt_keys, (list,))
            assert len(set(datetime_list_) - set(dt_keys)) == 0
            to_insert_data = TSFileDatabaseImpl_w._reset_data(dt_keys,
                                                              path_, to_write_data_file, to_insert_data_,
                                                              datetime_list_, header_)
            with open(os.path.join(path_, to_write_data_file), "w+") as f:
                f.write(",".join(["datetime"] + header_) + "\n")
            with open(os.path.join(path_, to_write_index_file), "w+") as f:
                f.write("datetime,start_index\n")
        else:
            to_insert_data = list()
            for val in to_insert_data_:
                to_insert_data.append(",".join([str(x) for x in val]))
            dt_keys = datetime_list_
        if len(dt_keys) > 0:
            TSFileDatabaseImpl_w._append_data_to_file(
                path_, to_write_data_file, to_write_index_file, to_insert_data, dt_keys)

    @staticmethod
    def _append_data_to_file(path_, to_insert_data_file_, to_insert_index_file_, to_insert_data_: list, datetime_list_: list):
        assert len(to_insert_data_) == len(datetime_list_)
        index_data = list()
        last_cursor = TSFileDatabaseImplUtil.get_file_end(os.path.join(path_, to_insert_data_file_))
        with open(os.path.join(path_, to_insert_data_file_), "a") as f:
            for i in range(len(to_insert_data_)):
                line = datetime_list_[i] + "," + to_insert_data_[i] + "\n"
                f.write(line)
                index = ",".join([datetime_list_[i], str(last_cursor)]) + "\n"
                index_data.append(index)
                last_cursor += len(line) + CONST
        with open(os.path.join(path_, to_insert_index_file_), "a") as f:
            for row in index_data:
                f.write(row)

    @staticmethod
    def _reset_data(dt_keys_, path_, to_reset_data_file_, data_, datetime_list_, header_):
        # TODO: reset smarter
        assert len(data_) == len(datetime_list_)
        data_dict = OrderedDict()
        for dt in dt_keys_:
            data_dict[dt] = None

        with open(os.path.join(path_, to_reset_data_file_), "r") as f:
            existed_data = f.readlines()
        assert len(existed_data) > 1
        header = existed_data[0][:-1].split(",")
        assert header[1:] == header_, " ".join(header[1:]) + ":" + " ".join(header_)

        for row in existed_data[1:]:
            tmp = row.split(",")
            key = tmp[0]
            line = (",".join(tmp[1:]))[:-1]
            data_dict[key] = line
        for i in range(len(data_)):
            data_dict[datetime_list_[i]] = ",".join([str(item) for item in data_[i]])
        return list(data_dict.values())

    @staticmethod
    def _get_file_names(univ_calcdate_):
        return univ_calcdate_ + "_DATA.csv", univ_calcdate_ + "_INDEX.csv"

    @staticmethod
    def _is_to_append(path_, index_file_, datetime_keys_: list):
        all_datetimes = list()
        with open(os.path.join(path_, index_file_), "r") as f:
            lines = f.readlines()
            for i in range(1, len(lines)):
                all_datetimes.append(lines[i].split(",")[0])
        new_datetimes = [val for val in datetime_keys_ if val not in all_datetimes]
        all_datetimes.extend(datetime_keys_)
        uniq_datetimes = list(np.unique(all_datetimes))
        is_to_append = all_datetimes == uniq_datetimes
        return is_to_append, uniq_datetimes, new_datetimes


class TSFileDatabaseImpl_r:
    @staticmethod
    def read_datetime_data(path_, start_datetime_, end_datetime_):
        files = TSFileDatabaseImpl_r.get_files_to_read(path_, start_datetime_, end_datetime_)
        ix_files = [f[0] for f in files]
        rtn_dt = OrderedDict()
        for ix_file in ix_files:
            this_univ_calc_date = ix_file.replace("_INDEX.csv", "")
            rtn_dt[this_univ_calc_date] = TSFileDatabaseImplUtil.get_datetimes(os.path.join(path_, ix_file), start_datetime_, end_datetime_)
        return rtn_dt

    @staticmethod
    def read_data(path, start_datetime_, end_datetime_, file_key_=None, to_numeric_=True):
        assert file_key_ is None or isinstance(file_key_, (str,))
        files = TSFileDatabaseImpl_r.get_files_to_read(path, start_datetime_, end_datetime_, file_key_)
        rtn_data = OrderedDict()
        rtn_dt = OrderedDict()
        rtn_header = OrderedDict()
        for ix_file, data_file in files:
            stt_cursor, end_cursor = TSFileDatabaseImpl_r.get_start_and_end_cursor(
                path, ix_file, start_datetime_, end_datetime_)  # [,)
            assert stt_cursor is not None
            if end_cursor is not None:
                assert stt_cursor <= end_cursor
            if end_cursor is None or stt_cursor < end_cursor:
                this_univ_calc_date = ix_file.replace("_INDEX.csv", "")
                this_rtn_data, this_rtn_dt, this_rtn_header = TSFileDatabaseImpl_r.read_data_file(
                    path, data_file, stt_cursor, end_cursor, to_numeric_=to_numeric_)
                rtn_data[this_univ_calc_date] = this_rtn_data
                rtn_dt[this_univ_calc_date] = this_rtn_dt
                rtn_header[this_univ_calc_date] = this_rtn_header
        return rtn_data, rtn_dt, rtn_header

    @staticmethod
    def read_data_file(path_, file_, start_cursor_, end_cursor_, to_numeric_=True):
        if end_cursor_ is None:
            with open(os.path.join(path_, file_), "r") as f:
                end_cursor_ = f.seek(0, 2)
        assert start_cursor_ is not None and end_cursor_ is not None and (start_cursor_ < end_cursor_)
        data_list = list()
        dt_list = list()
        with open(os.path.join(path_, file_), "r") as f:
            headers = f.readline()
            headers = headers[:-1].split(",")
            f.seek(start_cursor_)
            while f.tell() < end_cursor_:
                line = f.readline()
                row_data = line[:-1].split(",")
                dt_list.append(row_data[0])
                if to_numeric_:
                    data_list.append(np.array(row_data[1:]).astype(np.float64))
                else:
                    data_list.append(row_data[1:])
        return data_list, dt_list, headers

    @staticmethod
    def get_start_and_end_cursor(path_, index_file_, start_datetime_, end_datetime_):
        assert start_datetime_ <= end_datetime_
        start_cursor = None
        end_cursor = None
        with open(os.path.join(path_, index_file_), "r") as f:
            f.readline() # skip header
            line = f.readline()
            info = line[:-1].split(",")
            if (len(info) > 0) and (info[0] <= end_datetime_):
                while line != "":
                    info = line[:-1].split(",")
                    if start_cursor is None and (info[0] >= start_datetime_):
                        start_cursor = int(info[1])
                    if end_cursor is None and (info[0] > end_datetime_):
                        end_cursor = int(info[1])
                    if start_cursor is not None and end_cursor is not None:
                        break
                    line = f.readline()
        assert start_cursor is not None
        return start_cursor, end_cursor

    @staticmethod
    def get_files_to_read(path_, start_datetime_, end_datetime_, file_tag_=None):
        file_list = os.listdir(path_)
        if file_tag_ is None:
            index_file_list = list()
            data_file_list = list()
            for file in file_list:
                if os.path.isfile(os.path.join(path_, file)):
                    if "DATA" in file:
                        data_file_list.append(file)
                    if "INDEX" in file:
                        index_file_list.append(file)
            assert len(index_file_list) == len(data_file_list)
            index_file_list = sorted(index_file_list)
            data_file_list = sorted(data_file_list)
        else:
            if (file_tag_ + "_INDEX.csv" in file_list) and (file_tag_ + "_DATA.csv" in file_list):
                index_file_list = [file_tag_ + "_INDEX.csv"]
                data_file_list = [file_tag_ + "_DATA.csv"]
            else:
                index_file_list = []
                data_file_list = []
        rtn = list()
        for i in range(len(index_file_list)):
            file_stt_dt, file_end_dt = TSFileDatabaseImplUtil.get_start_end_datetime(os.path.join(path_, index_file_list[i]))
            flag1 = (start_datetime_ <= file_stt_dt) and (end_datetime_ >= file_stt_dt)
            flag2 = (start_datetime_ <= file_end_dt) and (end_datetime_ >= file_end_dt)
            flag3 = (start_datetime_ >= file_stt_dt) and (end_datetime_ <= file_end_dt)
            flag4 = (start_datetime_ <= file_stt_dt) and (end_datetime_ >= file_end_dt)
            if flag1 or flag2 or flag3 or flag4:
                rtn.append((index_file_list[i], data_file_list[i]))
        return rtn


class TSFileDatabaseImplUtil:
    @staticmethod
    def get_start_end_datetime(index_file_):
        assert os.path.exists(index_file_)
        with open(index_file_, "rb") as f:
            f.readline()
            first_index_line = f.readline()
            assert len(first_index_line) > 0
            start_datetime = first_index_line.decode("utf-8").split(",")[0]

            f.seek(0, 2)
            end_cursor = f.tell()
            off = -5
            while True:
                if -off < end_cursor:
                    f.seek(off, 2)
                    lines = f.readlines()
                    if len(lines) >= 2:
                        last_index_line = lines[-1]
                        end_datetime = last_index_line.decode("utf-8").split(",")[0]
                        break
                    off *= 2
                else:
                    assert False
        return start_datetime, end_datetime

    @staticmethod
    def get_file_end(file_):
        with open(file_, "r") as f:
            rtn = f.seek(0, 2)
        return rtn

    @staticmethod
    def get_datetimes(index_file_, start_dt_, end_dt_):
        assert os.path.exists(index_file_)
        dts = list()
        with open(index_file_, "r") as f:
            contents = f.readlines()
            if len(contents) > 0:
                for line in contents[1:]:
                    dt = line.split(",")[0]
                    if dt >= start_dt_ and dt <= end_dt_:
                        dts.append(dt)
        return dts


class FolderUtil:
    @staticmethod
    def is_normal_path(cate_fld_):
        is_special = (len(cate_fld_) >= 7 and cate_fld_[:7] == "ftr.drv") or (
                len(cate_fld_) >= 9 and cate_fld_[:9] == "ftr.stats")
        return not is_special

    @staticmethod
    def get_path(univ_tag_, cate_fld_, mk_dirs_=False):
        is_normal = FolderUtil.is_normal_path(cate_fld_)
        if is_normal:
            cate_fld_terms = [t for t in
                              cate_fld_.replace("@", ".").replace('[', '.').replace(']', '.').replace(',', '.').replace(
                                  ';',
                                  '.').split(
                                  '.')
                              if t != '']
            cate_fld_key = '.'.join(cate_fld_terms)
            path = "\\".join([BASE_PATH, univ_tag_] + cate_fld_key.split("."))
            if os.path.exists(path):
                rtn = path
            else:
                if mk_dirs_:
                    os.makedirs(path)
                    rtn = path
                else:
                    rtn = None
        else:
            assert "@" in cate_fld_
            cate = cate_fld_.split('@')[0]
            fld = cate_fld_.split('@')[1]
            main_fld = fld.split('.')[0]
            sub_fld = '.'.join(fld.split('.')[1:])
            #
            main_path = "\\".join([BASE_PATH, univ_tag_] + cate.split('.') + [main_fld])
            if not os.path.exists(main_path):
                os.makedirs(main_path)
            folder_map_file = os.path.join(main_path, "folder_map.csv")
            if not os.path.exists(folder_map_file):
                pd.DataFrame(columns=["sub_fld", "folder_name"]).to_csv(folder_map_file)
            folder_map = pd.read_csv(folder_map_file, index_col=0)
            existed_folder_names = list(folder_map["folder_name"])
            assert len(set(existed_folder_names)) == len(existed_folder_names)
            existed_sub_flds = list(folder_map["sub_fld"])
            assert len(set(existed_sub_flds)) == len(existed_sub_flds)
            if sub_fld in existed_sub_flds:
                df = folder_map[folder_map["sub_fld"] == sub_fld]
                assert len(df) == 1
                sub_folder = df["folder_name"].iloc[0]
                path = os.path.join(main_path, sub_folder)
                assert os.path.exists(path), path
                rtn = path
            else:
                if mk_dirs_:
                    sub_folder = FolderUtil.generate_folder_name(existed_folder_names)
                    df = pd.concat(
                        (folder_map, pd.DataFrame([[sub_fld, sub_folder]], columns=["sub_fld", "folder_name"])),
                        axis=0)
                    path = os.path.join(main_path, sub_folder)
                    assert not os.path.exists(path)
                    os.makedirs(path)
                    df.to_csv(folder_map_file)
                    rtn = path
                else:
                    rtn = None
        return rtn

    @staticmethod
    def generate_folder_name(existed_folder_names_: list):
        existed_folder_names = [s[:10] for s in existed_folder_names_]
        rtn = random.sample(string.ascii_letters + string.digits, 10)
        while rtn in existed_folder_names:
            rtn = random.sample(string.ascii_letters + string.digits, 10)
        # return ''.join(rtn) + '_' + time.strftime("%Y%m%d%H%M")
        return time.strftime("%Y%m%d%H%M") + '_' + ''.join(rtn)


    # @staticmethod
    # def get_path(univ_tag_, cate_fld_: str, mk_dirs_=False):
    #     path = FolderUtil.get_path_name(univ_tag_, cate_fld_)
    #     if mk_dirs_ and not os.path.exists(path):
    #         os.makedirs(path)
    #     # if not mk_dirs_:
    #     #     assert os.path.exists(path), "invalid path: " + path
    #     # else:
    #     #     if not os.path.exists(path):
    #     #         os.makedirs(path)
    #     return path
    #
    # @staticmethod
    # def get_path_name(univ_tag_, cate_fld_: str):
    #     if (len(cate_fld_) >= 7 and cate_fld_[:7] == "ftr.drv") or (
    #             len(cate_fld_) >= 9 and cate_fld_[:9] == "ftr.stats"):
    #         assert "@" in cate_fld_
    #         cate = cate_fld_.split('@')[0]
    #         fld = cate_fld_.split('@')[1]
    #
    #         main_fld = fld.split('.')[0]
    #         sub_fld = '.'.join(fld.split('.')[1:])
    #         main_path = "\\".join([BASE_PATH, univ_tag_] + cate.split('.') + [main_fld])
    #         sub_path_df = os.path.join(main_path, "folder_map.csv")
    #         if os.path.exists(sub_path_df):
    #             folder_map = pd.read_csv(sub_path_df, index_col=0)
    #         else:
    #             folder_map = pd.DataFrame(columns=["sub_fld", "folder_name"])
    #         existed_folder_names = list(folder_map["folder_name"])
    #         assert len(set(existed_folder_names)) == len(existed_folder_names)
    #         existed_sub_flds = list(folder_map["sub_fld"])
    #         assert len(set(existed_sub_flds)) == len(existed_sub_flds)
    #         if sub_fld in existed_sub_flds:
    #             df = folder_map[folder_map["sub_fld"] == sub_fld]
    #             assert len(df) == 1
    #             sub_folder = df["folder_name"].iloc[0]
    #         else:
    #             sub_folder = FolderUtil.generate_folder_name(existed_folder_names)
    #             df = pd.concat((folder_map, pd.DataFrame([[sub_fld, sub_folder]], columns=["sub_fld", "folder_name"])),
    #                            axis=0)
    #             if not os.path.exists(os.path.dirname(sub_path_df)):
    #                 os.makedirs(os.path.dirname(sub_path_df))
    #             df.to_csv(sub_path_df)
    #         cate_fld_key = cate + '.' + main_fld + '.' + sub_folder
    #     else:
    #         cate_fld_terms = [t for t in
    #                           cate_fld_.replace("@", ".").replace('[', '.').replace(']', '.').replace(',', '.').replace(
    #                               ';',
    #                               '.').split(
    #                               '.')
    #                           if t != '']
    #         cate_fld_key = '.'.join(cate_fld_terms)
    #     path = "\\".join([BASE_PATH, univ_tag_] + cate_fld_key.split("."))
    #     return path

