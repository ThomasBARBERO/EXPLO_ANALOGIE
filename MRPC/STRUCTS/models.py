import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pyth_imports import *
from STRUCTS.dataset import *

class Net(nn.Module):
    def __init__(self, EMBEDDING_SIZE, DROPOUT):
        super().__init__()
        out_channel1 = int((EMBEDDING_SIZE*4)/2)
        out_channel2 = int((out_channel1)/4)
        #print(out_channel1)
        #print(out_channel2)
        #print(out_channel2**2)
        #print("yyy")
        self.conv  = nn.Sequential(
          nn.Conv2d(in_channels=1, out_channels=out_channel1, kernel_size=(2,1), stride=(2,1)),
          nn.ReLU(),
          nn.Conv2d(in_channels=out_channel1, out_channels=out_channel2, kernel_size=(2,2), stride=(2,2)),
          nn.ReLU(),
          nn.Flatten(),
          nn.Dropout(DROPOUT),
          nn.Linear(out_channel2**2, 1)
        )

    def forward(self, x):
        x = self.conv(x)
        return x

class MLP(nn.Module):


  def __init__(self, EMBEDDING_SIZE, DROPOUT):
    super().__init__()
    self.layers = nn.Sequential(
      nn.Flatten(),
      nn.Linear(EMBEDDING_SIZE*4, 100),
      nn.ReLU(),
      nn.Dropout(DROPOUT),
      nn.Linear(100, 50),
      nn.ReLU(),
      nn.Dropout(DROPOUT),
      nn.Linear(50, 1)
    )


  def forward(self, x):

    return self.layers(x)