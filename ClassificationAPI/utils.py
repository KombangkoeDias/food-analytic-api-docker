from torchvision import transforms
import pandas as pd
import re
import math
from .config import config


transform = transforms.Compose([
    transforms.Resize(size= config['image_size']),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


def normalize_nutrient_name(nutrient_name: str):
    normalized_name = ""
    partitions = re.split('[,;]|\s-\s', nutrient_name)
    for partition in partitions:
        partition = partition.strip()
        if partition == 'total' or partition == 'by calculation':
            continue
        elif partition.startswith('total'):
            partition = " ".join(partition.split()[1:])
        elif partition == 'b-carotene':
            partition = 'Beta-carotene'
            
        if normalized_name == '': 
            normalized_name += partition
        else: 
            normalized_name += f" ({partition})"
    return normalized_name


def find_nutrients_dict(class_name: str):
    sub_class_names = class_name.split('-')
    for i in range(len(sub_class_names)):
        df_subclass = df[df['food_name']==sub_class_names[i]]
        if i==0: df_target = df_subclass
        else: df_target = pd.concat([df_target, df_subclass])
    # df_target.to_csv('output.csv', index=False)
    nutrients = df_target.mean(axis=0, numeric_only=True, skipna=True)

    nutrients_dict = {}
    for nutrient_name in nutrients.index.values:
        nutrient_value = nutrients[nutrient_name].item()
        if math.isnan(nutrient_value): 
            nutrient_value = None
        nutrients_dict[nutrient_name] = nutrient_value
    return nutrients_dict


df = pd.read_csv(config['food_nutrients_filepath'])
df.drop(['food_id','id','Source_id'], axis=1, inplace=True)
renamed_columns_map = {}
for column in df.columns.values:
    renamed_columns_map[column] = normalize_nutrient_name(column)
df.rename(columns=renamed_columns_map, inplace=True)