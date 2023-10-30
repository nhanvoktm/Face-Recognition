import numpy as np
import keras.utils as image
import gradio as gr
import pickle
from keras.models import load_model

import os

# chạy load bộ weight cũng như bộ nhãn đã train trước đó
model= load_model('./train-weight/21out_v4.h5')

infile = open('./ResultsMap.pkl','rb')
ResultMap = pickle.load(infile)
infile.close()


#Hàm dự đoán giống như bên ipynb

def Predict(inp):
    
    out = image.load_img(inp,color_mode='grayscale',target_size=(64, 64))
    out = image.img_to_array(out)
    test_image=np.expand_dims(out,axis=0)
    
    result = model.predict(test_image,verbose=0)
    return ResultMap[np.argmax(result)]

# chạy giao diện
demo = gr.Interface(fn=Predict, 
            inputs=gr.Image(type="filepath"),
            outputs = "text" ,
            examples=[["./face1.jpg"]], 
        ).launch()

