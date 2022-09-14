import os, sys
from xml.dom.expatbuilder import ParseEscape
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pyth_imports import *
from transformers import AutoTokenizer


def tokenization(location_analogies, dic, descr):
	analogies_ID = pd.read_csv(location_analogies, names=['A', 'B', 'C', 'D', 'y'], sep='|')



	temp = (analogies_ID.A.map(dic)).tolist()
	data = np.array([temp])
	temp = (analogies_ID.B.map(dic)).tolist()
	data = np.concatenate((data, [temp]), axis=0)
	temp = (analogies_ID.C.map(dic)).tolist()
	data = np.concatenate((data, [temp]), axis=0)
	temp = (analogies_ID.D.map(dic)).tolist()
	data = np.concatenate((data, [temp]), axis=0)

	targets = analogies_ID.y.tolist()

	tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

	out_file = "TOKENIZED/tokenized_"+descr+".txt"
	with open(out_file, 'w+') as out:
		for i in range(len(data[0])):
			temp = [data[0][i]+data[1][i], data[2][i]+data[3][i]]
			with torch.no_grad():
				out.write(str(targets[i]) +'|' + str(tokenizer.encode_plus(temp, padding="max_length", truncation=True)) +'\n')

			#concaneted_phrases.append(temp)
			#tokenized_phrases.append(tokenizer.encode(temp))




data_file = 'msr-paraphrase-corpus/msr_paraphrase_data.txt'

phrases = pd.read_csv(data_file, names=['ID','String',	'Author',	'URL',	'Agency',	'Date',	'WebDate'], sep='\t')
phrases.drop(index=phrases.index[0], axis=0, inplace=True)
phrases = phrases.astype({"ID": int}, errors='raise') 
liste_phrases = phrases.String.tolist()
id_phrases = phrases.ID.tolist()

phrases = dict(zip(id_phrases, liste_phrases))
"""
loc ="analogies/analogies_paraphrases_test_40k.csv"
tokenization(loc, phrases, "test")

loc ="analogies/analogies_paraphrases_valid_40k.csv"
tokenization(loc, phrases, 'valid')

loc ="analogies/analogies_paraphrases_train_400k.csv"
tokenization(loc, phrases, 'train')
"""

tokenization("analogies/analogies_paraphrases_train_400k.csv", phrases, "train_400k_distil")
tokenization("analogies/analogies_paraphrases_valid_40k.csv", phrases, "valid_40k_distil")
tokenization("analogies/analogies_paraphrases_test_40k.csv", phrases, "test_40k_distil")
