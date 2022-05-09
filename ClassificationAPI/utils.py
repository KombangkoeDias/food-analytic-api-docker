from torchvision import transforms
import pandas as pd
import re
import math
from .config import config


def normalize_nutrient_name(nutrient_name: str):
    normalized_name = ''
    partitions = re.split('[,;]|\s-\s', nutrient_name)
    for partition in partitions:
        partition = partition.strip()
        if partition == 'total' or partition == 'by calculation' or partition == 'available':
            continue
        elif partition.startswith('total'):
            partition = " ".join(partition.split()[1:])
        if normalized_name == '': 
            normalized_name += partition
        else: 
            normalized_name += f' ({partition})'
    return normalized_name


def find_nutrients_dict(class_name: str):
    df = pd.read_csv(config['food_nutrients_filepath'])
    sub_class_names = class_name.split('-')
    for i in range(len(sub_class_names)):
        df_subclass = df[df['food_name']==sub_class_names[i]]
        if i==0: df_target = df_subclass
        else: df_target = pd.concat([df_target, df_subclass])
    # df_target.to_csv('output.csv', index=False)
    nutrients = df_target.mean()
    nutrients.drop(['food_id','id','Source_id'], axis=0, inplace=True)

    nutrients_dict = {}
    for nutrient_name in nutrients.index.values:
        normalized_nutrient_name = normalize_nutrient_name(nutrient_name)
        nutrient_value = nutrients[nutrient_name].item()
        if math.isnan(nutrient_value): 
            nutrient_value = None
        nutrients_dict[normalized_nutrient_name] = nutrient_value
    return nutrients_dict


transform = transforms.Compose([
    transforms.Resize(size= config['image_size']),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])