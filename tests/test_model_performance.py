import pytest
import wandb
import os 
import time 
import torch
from dotenv import load_dotenv
from src.models.model import MyAwesomeModel

load_dotenv()

def load_model(model_checkpoint, logdir="models"):
    api = wandb.Api(
        api_key=os.getenv("WANDB_API_KEY"),
        overrides={"entity": os.getenv("WANDB_ENTITY"), "project": os.getenv("WANDB_PROJECT")},
    )
    artifact = api.artifact(model_checkpoint)
    artifact.download(root="models/")
    file_name = artifact.files()[0].name
    model = MyAwesomeModel(500, 250, 125, 60, 0.2)
    model.load_state_dict(torch.load(f"{logdir}/{file_name}", map_location=torch.device('cpu')))
    return model

def test_model_speed():
    model = load_model(os.getenv("MODEL_NAME"))
    model.eval()
    start = time.time()
    for _ in range(100):
        model(torch.rand(1, 1, 28, 28).view(-1, 28*28))
    end = time.time()
    assert end - start < 1

# model_checkpoint = os.getenv("MODEL_NAME")

# api = wandb.Api(
#     api_key=os.getenv("WANDB_API_KEY"),
#     overrides={"entity": os.getenv("WANDB_ENTITY"), "project": os.getenv("WANDB_PROJECT")}
# )

# artifact = api.artifact(model_checkpoint)
# artifact.download(root="models/")
# file_name = artifact.files()[0].name
# print(file_name)

