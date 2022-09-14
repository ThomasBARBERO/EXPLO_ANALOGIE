import os, sys
from pickletools import float8
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pyth_imports import *
from STRUCTS.models import *
from statistics import median
from torch.special import expit

def calculate_accuracy(preds, targets):
    n_correct = (preds==targets).sum().item()
    return n_correct


def calculate_wrong(preds, targets):
    n_correct = (preds!=targets).sum().item()
    return n_wrong
	
def train(epoch, model, training_loader, criterion, optimizer):
    tr_loss = 0
    n_correct = 0
    nb_tr_steps = 0
    nb_tr_examples = 0
    targets_t = []

    model.train()
  
    for i, data in enumerate(training_loader, 0):
      inputs =  data['data'].to(device)
      targets =  data['targets'].to(device)
      targets = targets.unsqueeze(1).to(float)

      # zero the parameter gradients
      optimizer.zero_grad()

      # forward + backward + optimize
      outputs = model(inputs)
      #print(outputs)
      
      #big_idx = (outputs > 0.5).float()
      big_idx = (expit(outputs) > 0.5).float()
      #big_val, big_idx = torch.max(outputs.data)
      #loss = criterion(big_idx.float().requires_grad_(True), targets.float().requires_grad_(True))
      loss = criterion(outputs, targets)
      tr_loss += loss.item()
      n_correct += calculate_accuracy(big_idx, targets)

      nb_tr_steps += 1
      nb_tr_examples+=targets.size(0)
      
      if i%5000==0:
          loss_step = tr_loss/nb_tr_steps
          accu_step = (n_correct*100)/nb_tr_examples 
          #print(f"Training Loss per 5000 steps: {loss_step}")
          #print(f"Training Accuracy per 5000 steps: {accu_step}")

      loss.backward()
      optimizer.step()
      
      l0, l1 = matrix_test(targets.tolist(), outputs)

    """ print(f'The Total Accuracy for Epoch {epoch}: {(n_correct*100)/nb_tr_examples}')
    epoch_loss = tr_loss/nb_tr_steps
    epoch_accu = (n_correct*100)/nb_tr_examples
    print(f"Training Loss Epoch: {epoch_loss}")
    print(f"Training Accuracy Epoch: {epoch_accu}")"""

    return
	
	
def valid(model, testing_loader, criterion, optimizer, results_loc, preds_loc, test=False):
    model.eval()
    n_correct = 0; n_wrong = 0; total = 0; tr_loss=0; nb_tr_steps=0; nb_tr_examples=0
    f1_scores = []
    precisions = []
    recalls  = []
    accuracies = []

    targets_t = []
    preds_t = []

    with torch.no_grad():
        for i, data in enumerate(testing_loader, 0):
            inputs =  data['data'].to(device)
            targets =  data['targets'].to(device)
            targets = targets.unsqueeze(1).to(float)

            outputs = model(inputs)
            #print(outputs)
            #_, big_idx = torch.max(outputs.data)
            big_idx = (outputs > 0.5).float()
            big_idx = (expit(outputs) > 0.5).float()
            #print(big_idx)
            #big_val, big_idx = torch.max(outputs.data, dim=1)
            loss = criterion(outputs, targets)
            tr_loss += loss.item()
            n_correct += calculate_accuracy(big_idx, targets)
            #n_wrong += calculate_wrong(big_idx, targets)

            nb_tr_steps += 1
            nb_tr_examples+=targets.size(0)
            f1_scores.append(f1_score(big_idx.cpu(), targets.cpu()))
            precisions.append(precision_score(big_idx.cpu(), targets.cpu()))
            recalls.append(recall_score(big_idx.cpu(), targets.cpu()))

            targets_t = targets_t + targets.tolist()
            preds_t = preds_t + big_idx.tolist()

            if i%5000==0:
                loss_step = tr_loss/nb_tr_steps
                accu_step = (n_correct*100)/nb_tr_examples
                #print(f"Validation Loss per 100 steps: {loss_step}")
                #print(f"Validation Accuracy per 100 steps: {accu_step}")

    #epoch_loss = tr_loss/nb_tr_steps
    epoch_accu = (n_correct*100)/nb_tr_examples
    r0 = matrix_evaluation(targets_t, preds_t, evaluated_class=0)
    r1 = matrix_evaluation(targets_t, preds_t)
    #print(big_idx)


    with open(results_loc, 'a') as out :
        if (test):
            out.write("\n\Test report :\n")
        else:
            out.write("\n\Valid epoch report :\n")
        out.write("Class 0\n")
        out.write("Average f1 score : " + str(r0[2]) + '\n')
        out.write("Average precision : " + str(r0[1]) + '\n')
        out.write("Average recall : " + str(r0[0]) + '\n')
        out.write("Epoch accuracy : " + str(r0[3]) + '\n')
        out.write("Loss : " + str(tr_loss) + '\n')
        out.write("Class 1")
        out.write("Average f1 score : " + str(r1[2]) + '\n')
        out.write("Average precision : " + str(r1[1]) + '\n')
        out.write("Average recall : " + str(r1[0]) + '\n')
        out.write("Epoch accuracy : " + str(r1[3]) + '\n')
        out.write("Loss : " + str(tr_loss) + '\n')
        out.write("--END--\n")
        
    #with open(preds_loc, 'a') as out :
     #   out.write("Targets")

    return epoch_accu

def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0] #First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
    sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    return sum_embeddings / sum_mask

def matrix_evaluation(y_true, pred, threshold=0.5, evaluated_class=1):
    #print(y_true)
    
    #print(pred)

    if type(y_true[0]) == list:
        for i in range(len(y_true)):
            y_true[i] = y_true[i][0]
        
    if type(pred[0]) == list:
        for i in range(len(pred)):
            pred[i] = pred[i][0]

    print(y_true)
    
    print(pred)

    nb_dev = len(y_true)
    correct = 0
    acc = 0
    tp = 0 
    fn = 0 
    fp = 0
    p = 0 
    r = 0
    f1 = 0
    y_pred = []
    for j in range(nb_dev):
        if pred[j] >= threshold:
            y_pred.append(1)
        else:
            y_pred.append(0)
    #print("y_pred",y_pred)
    #print("y_true",y_true)
    for i in range(nb_dev):
        if y_pred[i] == y_true[i] :
            correct += 1 
            if y_true[i] == evaluated_class:
                tp += 1
        else :
            if y_true[i] == evaluated_class :
                fn += 1
            else :
                fp += 1
                
#         elif y_pred[i] == 0 and y_true[i] == 1:
#             fn += 1
#         elif y_pred[i] == 1 and y_true[i] == 0:
#             fp += 1
#         else:
#             pass
        
    if tp == 0:
        print("tp is 0")
        pass
    else:
        p = tp /float(tp + fp)
        r = tp /float(tp + fn)
        f1 = 2*p*r /(p + r)
        acc = float(correct / float(len(y_true)))


    return round(100 * p, 3) , round(100 * r, 3) , round(100 * f1, 3), round(100 * acc, 3)

def matrix_test(targets, preds):
    l0 = []
    l1 = []
    for i in range(0, len(targets)):
        if targets[i][0] == 0:
            l0.append(expit(preds[i][0]).item())
            #l0.append(expit(preds[i][0].item()))
        else:
            l1.append(expit(preds[i][0]).item())
            #l1.append(expit(preds[i][0].item()))

    return (l0, l1)


def findFirstDigit(chaine):
    c=0
    for i in chaine:
        if i.isdigit():
            break
        c = c+1
    return c

def findLastDigit(chaine):
    c=0
    for i in reversed(chaine):
        if i.isdigit():
            break
        c = c+1
    return c

def load_dic(location):
    with open(location, encoding = "ISO-8859-1") as filo:
        f = filo.read()
        items = f.split(')')[:-1]
        new_items = []
        for i, item in enumerate(items):
            temp = item.split('|')
            key = int(temp[0].strip())
            temp_encoded = temp[1]
            values = [float(x) for x in temp_encoded[findFirstDigit(temp_encoded):-findLastDigit(temp_encoded)].split(',')]
            new_items.append([key, torch.Tensor(values)])

        return dict(new_items)