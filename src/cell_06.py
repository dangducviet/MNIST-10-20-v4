import pandas as pd
from google.colab import files

sub = pd.read_csv('/content/drive/MyDrive/MNIST_data/v4_hier_multi_mlp/submission_v4_cell5_surgical_master.csv', dtype={'id': str})
sub.loc[sub['id'] == '147', 'label'] = 11  # Đồng thuận top-1 của cả 12 model 20x20 (u20, dp5, u20_full, dp5_full): 17 (dư) -> 11 (thiếu)
out_name = 'submission_v4_cell5_dual20x20_surgical.csv'
sub.to_csv(out_name, index=False, lineterminator='\n')
print('Phân bố lớp:', sub['label'].value_counts().sort_index().to_dict())
files.download(out_name)