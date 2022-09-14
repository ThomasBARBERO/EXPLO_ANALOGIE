from pyth_imports import *
from STRUCTS.dataset import *
from STRUCTS.models import *
from utils import *

ID = int(sys.argv[1])
ENCODER_NAME = str((sys.argv[2]))
CLASSIFIER_NAME = str((sys.argv[3]))

tdebut = time.perf_counter()

config_loc = 'RESULTS/EXPERIENCE_'+str(ID)+'/config.txt'
config = pd.read_csv(config_loc, sep="|", index_col=False)

NB_ANALOGIES_TRAIN = int(config['NB_TRAIN'])
NB_ANALOGIES_VALID = int(config['NB_VALID'])


TRAIN_BATCH_SIZE  = int(config['BATCH_TRAIN'])
VALID_BATCH_SIZE  = int(config['BATCH_VALID'])

LR = float(config['LR'])
DROPOUT = float(config['DROPOUT'])
EPOCHS = int(config['EPOCHS'])
WEIGHTS = int(config['WEIGHTS'])

tokenizer = None
SENTENCE_TRANSFORMER = False
if ENCODER_NAME.__contains__('SBert') :
    EMBEDDING_SIZE = 384
elif ENCODER_NAME.__contains__('bert-base-mean') :
    EMBEDDING_SIZE = 768
elif ENCODER_NAME.__contains__('bert-large-mean') :
    EMBEDDING_SIZE = 1024
elif ENCODER_NAME.__contains__('roberta-base-mean') :
    EMBEDDING_SIZE = 768
elif ENCODER_NAME.__contains__('roberta-large-mean') :
    EMBEDDING_SIZE = 1024
elif ENCODER_NAME.__contains__('GloVe-mean') :
    EMBEDDING_SIZE = 300
elif ENCODER_NAME.__contains__('gpt2') :
    EMBEDDING_SIZE = 768
else:
    print("Erreur dans le choix de l'encodeur\n Encodeurs disponibles : SBert")


if CLASSIFIER_NAME.__contains__('CNN'):
    model = Net(EMBEDDING_SIZE, DROPOUT)
elif CLASSIFIER_NAME.__contains__('MLP'):
    model = MLP(EMBEDDING_SIZE, DROPOUT)
else:
    print("Erreur dans le choix du Classifieur\n Classifieurs disponibles : CNN, MLP")

results_loc = 'RESULTS/EXPERIENCE_'+str(ID)+'/'+CLASSIFIER_NAME+"/results_"+ENCODER_NAME+".txt"
with open(results_loc, 'w+') as out :
    out.write('PARAMETRES EXPERIENCE : \nENCODER : ' + ENCODER_NAME +  '\t CLASSIFIER  : ' + CLASSIFIER_NAME + '\nLEARNING RATE : ' + str(LR) + '\t EPOCHS : ' + str(EPOCHS) +'\n\n')


t_debut_dataset = time.perf_counter()
"""
valid_data_file = 'CORPUS/analogies/analogies_paraphrases_valid_40k_60.csv'
test_data_file = 'CORPUS/analogies/analogies_paraphrases_test_40k_60.csv'
train_data_file = 'CORPUS/analogies/analogies_paraphrases_train_400k_60.csv'

valid_data_file = 'CORPUS/analogies/analogies_paraphrases_valid_400k.csv'
test_data_file = 'CORPUS/analogies/analogies_paraphrases_test_400k.csv'
train_data_file = 'CORPUS/analogies/analogies_paraphrases_train_4M.csv'
"""
valid_data_file = 'CORPUS/analogies/analogies_pdtb_valid_40k.csv'
test_data_file = 'CORPUS/analogies/analogies_pdtb_test_40k.csv'
train_data_file = 'CORPUS/analogies/analogies_pdtb_train_400k.csv'
"""
valid_data_file = 'msr-paraphrase-corpus/test1_ana.csv'
test_data_file = 'msr-paraphrase-corpus/test1_ana.csv'
train_data_file = 'msr-paraphrase-corpus/test1_ana.csv'


training_set = AnalogiesData(encoder_model, train_data_file, 'msr-paraphrase-corpus/msr_paraphrase_data.txt', tokenizer)
valid_set = AnalogiesData(encoder_model, valid_data_file, 'msr-paraphrase-corpus/msr_paraphrase_data.txt', tokenizer)
testing_set = AnalogiesData(encoder_model, test_data_file, 'msr-paraphrase-corpus/msr_paraphrase_data.txt', tokenizer)

training_set = AnalogiesData(train_data_file, 'CORPUS/ARCHIVE_ENCODED/encoded_phrases_SBert.pt')
valid_set = AnalogiesData(valid_data_file, 'CORPUS/ARCHIVE_ENCODED/encoded_phrases_SBert.pt')
testing_set = AnalogiesData(test_data_file, 'CORPUS/ARCHIVE_ENCODED/encoded_phrases_SBert.pt')
"""
training_set = AnalogiesData(train_data_file, 'CORPUS/ENCODED/encoded_phrases_'+ENCODER_NAME+'.txt')
valid_set = AnalogiesData(valid_data_file, 'CORPUS/ENCODED/encoded_phrases_'+ENCODER_NAME+'.txt')
testing_set = AnalogiesData(test_data_file, 'CORPUS/ENCODED/encoded_phrases_'+ENCODER_NAME+'.txt')


t_dataset = time.perf_counter()
print("Charglement des datasets fini en " + str(t_dataset - t_debut_dataset) + "s.")

train_params = {'batch_size': TRAIN_BATCH_SIZE,
                'shuffle': True,
                'num_workers': 0
                }

training_loader = DataLoader(training_set, **train_params)


valid_params = {'batch_size': VALID_BATCH_SIZE,
                'shuffle': True,
                'num_workers': 0
                }

valid_loader = DataLoader(valid_set, **valid_params)    


testing_loader = DataLoader(testing_set, **valid_params)    

            
		
model.to(device)

criterion = nn.BCEWithLogitsLoss().to(device)
optimizer = optim.Adam(model.parameters(), lr=LR)


preds_loc = 'RESULTS/EXPERIENCE_'+str(ID)+'/'+CLASSIFIER_NAME+"/predictions"+ENCODER_NAME+".txt"
max_accu = 0
for epoch in range(EPOCHS):
    PATH = 'RESULTS/EXPERIENCE_'+str(ID)+'/'+CLASSIFIER_NAME+"/MODELS/model_"+ENCODER_NAME+'_'+str(epoch)+".txt"
    t1 = time.perf_counter()
    train(epoch, model, training_loader, criterion, optimizer)
    acc = valid(model, valid_loader, criterion, optimizer, results_loc, preds_loc)
    t2 = time.perf_counter()
    
    torch.save({
            'model_state_dict': model.state_dict(),
            }, PATH)

    if max_accu > acc:
        model = model.load_state_dict(torch.load(BEST_PATH))
    else:
        BEST_PATH = PATH


    with open(results_loc, 'a') as out :
        out.write("Epoch "+ str(epoch)+" training and validation performed in "+ str((t2 - t1)/3600) +" hours\n")

    print("Epoch "+ str(epoch)+" training and validation performed in "+ str((t2 - t1)/3600) +" hours\n")


acc = valid(model, testing_loader, criterion, optimizer, results_loc, preds_loc, True)

tfin = time.perf_counter()
with open(results_loc, 'a') as out :
        out.write("Total running time : "+ str((tfin - tdebut)/3600) +" hours\n")
print("Total running time : "+ str((tfin - tdebut)/3600) +" hours\n")