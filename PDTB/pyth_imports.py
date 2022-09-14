import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sentence_transformers import SentenceTransformer
import csv
import time

import pandas as pd
from collections import defaultdict

from torch.utils.data import Dataset, DataLoader
import torch
import torchvision
import torchvision.transforms as transforms

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn.modules.activation import Sigmoid
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report, precision_score, recall_score, f1_score
from transformers import BertTokenizer, TFBertModel, BertModel, RobertaTokenizer, GPT2Tokenizer, GPT2Model, TFRobertaModel, RobertaModel
# import random

torch.manual_seed(42)


if torch.cuda.is_available():
  device = torch.device("cuda")
  print("cuda available")
else:
  device = torch.device("cpu")