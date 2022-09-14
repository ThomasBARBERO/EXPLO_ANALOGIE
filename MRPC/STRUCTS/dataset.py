import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pyth_imports import *
from utils import *
from tempfile import mkdtemp

class AnalogiesData(Dataset):
    def __init__(self, filename_analogies, filename_phrases, tokenizer = None):
      self.load_from_csv(filename_analogies, filename_phrases)
      #print temps
    
    def load_from_csv(self, location_analogies, location_phrases):
        t1 = time.perf_counter()
        self.analogies_ID = pd.read_csv(location_analogies, names=['A', 'B', 'C', 'D', 'y'], sep='|')
        self.encoded_phrases = load_dic(location_phrases)
        #temp = pd.read_csv(location_phrases, names=['ID','String'], sep="|")
        #self.encoded_phrases = dict(zip(temp.ID, temp.String))
        """s
        print((self.analogies_ID.A.map(self.encoded_phrases)).shape)
        print((self.analogies_ID.A.map(self.encoded_phrases))[0].shape)
        print((self.analogies_ID.A.map(self.encoded_phrases))[0][0].shape)
        print(type((self.analogies_ID.A.map(self.encoded_phrases))))
        print(type((self.analogies_ID.A.map(self.encoded_phrases))[0]))
        print(type((self.analogies_ID.A.map(self.encoded_phrases))[0][0]))
        #print(type((self.analogies_ID.A.map(self.encoded_phrases))[0][0][0]))
        """

        """print(len(list(self.encoded_phrases.values())))
        print(len(list(self.encoded_phrases.keys())))

        print(sum(type(x)==float for x in list(self.encoded_phrases.values())))
        print(sum(type(x)==float for x in list(self.analogies_ID.A.map(self.encoded_phrases))))"""
        #print(len(list(set(list(self.analogies_ID.A)) ^ set(list(self.encoded_phrases.keys())))))
        c = nc = 0
        for x in list(set(self.analogies_ID.A)):
            if x in self.encoded_phrases.keys():
                c = c+1
            else:
                nc = nc+1
        print(c)
        print(nc)
        print(len((list(self.encoded_phrases.keys()))))


        t2 = time.perf_counter()
        temp = (self.analogies_ID.A.map(self.encoded_phrases)).map(torch.Tensor.tolist).tolist()
        data = np.array([temp])
        temp = (self.analogies_ID.B.map(self.encoded_phrases)).map(torch.Tensor.tolist).tolist()
        data = np.concatenate((data, [temp]), axis=0)
        temp = (self.analogies_ID.C.map(self.encoded_phrases)).map(torch.Tensor.tolist).tolist()
        data = np.concatenate((data, [temp]), axis=0)
        temp = (self.analogies_ID.D.map(self.encoded_phrases)).map(torch.Tensor.tolist).tolist()
        data = np.concatenate((data, [temp]), axis=0)
        
        self.targets = self.analogies_ID.y.tolist()
        

        self.data = torch.Tensor(data)
        self.data = torch.transpose(self.data, 0, 1)
        self.data = torch.unsqueeze(self.data, 1)
        self.targets = torch.tensor(self.targets, dtype=torch.long)
        self.data.to(device)
        self.targets.to(device)

        t5 = time.perf_counter()
        #t_total = t4 - t2

        print(str(float(t5-t1)/3600) +'h : Dataset charge')


    def __len__(self):
        return len(self.targets)

    def __getitem__(self, index):
        targets_ = self.targets[index].clone().detach()
        return {'data': self.data[index].clone().detach(), 'targets': targets_ }