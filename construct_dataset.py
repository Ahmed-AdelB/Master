import os
import pandas as pd
import numpy as np
from datetime import datetime

data_root = os.path.join('.', 'Machine learning/data/emg_dataset')
print(data_root)

data_sub_folders = ['sub1', 'sub3', 'sub4']  # TODO include sub2
print(data_sub_folders)

agg_classes_dict = {
    "Elbowing": 0,
    "Frontkicking": 1,
    "Hamering": 2,
    "Headering": 3,
    "Kneeing": 4,
    "Pulling": 5,
    "Punching": 6,
    "Pushing": 7,
    "Sidekicking": 8,
    "Slapping": 9
}
norm_classes_dict = {
    "Bowing": 10,
    "Clapping": 11,
    "Handshaking": 12,
    "Hugging": 13,
    "Jumping": 14,
    "Running": 15,
    "Seating": 16,
    "Standing": 17,
    "Walking": 18,
    "Waving": 19
}
col_names = ['R-Bic', 'R-Tri', 'L-Bic', 'L-Tri', 'R-Thi', 'R-Ham', 'L-Thi', 'L-Ham']
col_dtypes = dict()
for col in col_names:
    col_dtypes[col] = np.int32

data_classes_files_dict = dict()

df = pd.DataFrame()

for type_, category_dict in zip(["Aggressive", "Normal"],
                                [agg_classes_dict, norm_classes_dict]):
    for cls, id in category_dict.items():
        for folder in data_sub_folders:
            txt_file = os.path.join(data_root, folder, type_ + '/txt/', cls + '.txt')
            log_file = os.path.join(data_root, folder, type_ + '/log/', cls + '.log')
            print("Processing txt_file: {}".format(txt_file))

            with open(log_file, 'rb') as f:
                stri = f.read(100)
                _, s_t, e_t, _ = str(stri).split("\\r\\n")
                s_t = s_t[6:]
                e_t = e_t[4:]
                print(s_t)
                print(e_t)
                s_t = datetime.strptime(s_t, '%d/%m/%Y %H:%M:%S').timestamp() * 1000
                e_t = datetime.strptime(e_t, '%d/%m/%Y %H:%M:%S').timestamp() * 1000
                print(s_t)
                print(e_t)
                total_duration = e_t - s_t
                print(total_duration)

            df_cur = pd.read_csv(txt_file, delim_whitespace=True,
                                 header=None,
                                 names=col_names,
                                 dtype=col_dtypes)
            no_rows = df_cur.count()[0]
            print(no_rows)
            dur = total_duration / no_rows
            print(dur)
            df_cur = df_cur.assign(type=type_)
            df_cur = df_cur.assign(cls_id=id)
            df_cur = df_cur.assign(cls_name=cls)
            df_cur = df_cur.assign(sub_id=folder)
            df_cur = df_cur.assign(time_ms=pd.Series(np.arange(0, total_duration, dur, dtype=np.float32)))
            df = df.append(df_cur, ignore_index=True, verify_integrity=True)
print(df)
df.to_pickle('emg_dataset_pandas_dataframe.pkl')
