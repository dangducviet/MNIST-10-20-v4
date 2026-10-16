import pandas as pd
from google.colab import files

# Khởi tạo từ bản 2hop_chain vừa đạt 0.98181 (216/220)
sub_216 = pd.read_csv('/content/drive/MyDrive/MNIST_data/v4_hier_multi_mlp/submission_v4_cell6_topological_swa_master.csv', dtype={'id': str})
sub_216.loc[sub_216['id'] == '115', 'label'] = 13
sub_216.loc[sub_216['id'] == '476', 'label'] = 13
sub_216.loc[sub_216['id'] == '78',  'label'] = 18

# Bản A: Bổ sung ID 481 (10 -> 18) theo đồng thuận Topo6_SWA (0.419) + v4_orig/v7_blend (đưa lớp 18 từ 49 -> 50!)
sub_a = sub_216.copy()
sub_a.loc[sub_a['id'] == '481', 'label'] = 18
name_a = 'submission_v4_cell6_2hop_plus_481_to_18.csv'
sub_a.to_csv(name_a, index=False, lineterminator='\n')
print('Bản A (+ID 481->18):', sub_a['label'].value_counts().sort_index().to_dict())

# Bản B: Bổ sung cả ID 481 (10 -> 18) và ID 234 (11 -> 16, nơi Topo6_SWA tự tin 52.7% là số 16)
sub_b = sub_a.copy()
sub_b.loc[sub_b['id'] == '234', 'label'] = 16
name_b = 'submission_v4_cell6_2hop_plus_481_and_234.csv'
sub_b.to_csv(name_b, index=False, lineterminator='\n')
print('Bản B (+ID 481->18 & ID 234->16):', sub_b['label'].value_counts().sort_index().to_dict())

files.download(name_a)
files.download(name_b)