from create_model import create_effnetB2_model
import gradio as gr
from timeit import default_timer as timer
from typing import Tuple, Dict
import torch
import os

effnet_B_model, effnetB2_transforms = create_effnetB2_model(101)

with open("class_names.txt", "r") as f:
  class_names = [class_name.strip() for class_name in f.readlines()]



def pred(img)-> Tuple[Dict, float]:


  Results_dict = {}
  img = effnetB2_transforms(img).unsqueeze(dim = 0)

  effnet_B_model.load_state_dict(torch.load(f = "effnetB2_model_big_20%.pth",map_location = torch.device("cpu")))

  model = effnet_B_model.eval()

  start_timer = timer()
  with torch.inference_mode():
    pred_logits = model(img)

  pred_probs = torch.softmax(pred_logits, dim = 1)

  pred_labels = torch.argmax(pred_probs, dim = 1)

  for x in range(101):
    Results_dict[class_names[x]] = pred_probs[0][x]

  end_timer = timer()

  pred_time = round(end_timer - start_timer, 2)

  return Results_dict, pred_time


title = "Food Recognition"
description ="For 101 types of food"

demo = gr.Interface(fn = pred,
                    inputs = gr.Image(type = "pil"),
                    outputs = [gr.Label(num_top_classes = 5, label = "Predictions"),
                               gr.Number(label = "Prediction time (seconds)")],
                    title = title,
                    description = description,
                    )

demo.launch(debug = True)
