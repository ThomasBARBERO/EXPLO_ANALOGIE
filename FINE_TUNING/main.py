import os, sys
from STRUCTS.dataset import *
from xml.dom.expatbuilder import ParseEscape
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pyth_imports import *
from utils import *
from transformers import AutoModelForSequenceClassification, DistilBertForSequenceClassification

loc_train = "CORPUS/TOKENIZED/tokenized_train_400k.txt"
training_set = AnalogiesData(loc_train)

print("charge")
loc_valid = "CORPUS/TOKENIZED/tokenized_valid_40k.txt"
valid_set = AnalogiesData(loc_valid)


print("charge")
loc_test = "CORPUS/TOKENIZED/tokenized_test_40k.txt"
test_set = AnalogiesData(loc_test)


print("charge")

LR = 0.0001

train_params = {'batch_size': 20,
                'shuffle': True,
                'num_workers': 0
                }

training_loader = DataLoader(training_set, **train_params)


valid_params = {'batch_size': 20,
                'shuffle': True,
                'num_workers': 0
                }

valid_loader = DataLoader(valid_set, **valid_params)    


testing_loader = DataLoader(test_set, **valid_params)    

model = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=1)           
		
model.to(device)
#model.gradient_checkpointing_enable()


criterion = nn.BCEWithLogitsLoss().to(device)
optimizer = optim.Adam(model.parameters(), lr=LR)


EPOCHS = 10
results_loc="RESULTS/results.txt"
preds_loc=""
max_accu = 0
for epoch in range(EPOCHS):
    train(epoch, model, training_loader, optimizer, criterion)
    acc = valid(model, valid_loader, optimizer, results_loc, preds_loc)


acc = valid(model, testing_loader, optimizer, results_loc, preds_loc, True)


