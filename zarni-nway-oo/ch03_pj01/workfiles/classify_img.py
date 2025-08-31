import cnn_models
import pandas as pd
from keras.utils import load_img #type: ignore
import os
import time

def get_predictions(image_dir, k=1):
    model = cnn_models.cnnModels()
    model_name = ['ResNet50', 'VGGNet16', 'InceptionV3', 'ConvNeXt', 'EfficientNet']
    
    columns = []
    for name in model_name:
        for i in range(1, k+1):
            columns.append(f"{name}_top{i}")
            columns.append(f"{name}_top{i}_prob")
            
    timing_columns = [f"{name}_time" for name in model_name]
    columns.extend(timing_columns)
    columns.append('total_time') 

    result_rows = []
    labels = []

    for filename in os.listdir(image_dir):
        if filename.endswith('.jpeg') or filename.endswith('.png')or filename.endswith('.jpg'):
            image_path = os.path.join(image_dir, filename)
            img = load_img(image_path)   
            labels.append(filename.split('.')[0])
            
            row_data = []
            timing_data = []
            total_start_time = time.time()
            
            for name in model_name:
                model_start_time = time.time()
                preds = model.classify_image(name, img, top_k=k)[0]
                model_end_time = time.time()
                
                model_time = model_end_time - model_start_time
                timing_data.append(round(model_time, 2))
                for pred in preds:
                    row_data.append(pred[1]) #class name
                    row_data.append(pred[2]) #probability  
        
            row_data.extend(timing_data)
            total_time = time.time() - total_start_time
            row_data.append(round(total_time, 2))
            
            result_rows.append(row_data)

    result_df = pd.DataFrame(result_rows, columns=columns)
    result_df['label'] = labels      
    
    return result_df