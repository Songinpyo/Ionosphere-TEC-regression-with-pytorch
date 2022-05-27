# Ionosphere-TEC-regression-with-pytorch
### Predict extended ionosphere total electron content with variable data

<img src="https://static.sciencelearn.org.nz/images/images/000/000/248/full/Layers-of-the-ionosphere20150924-22493-1th64qk.jpg?1522293304"/>

# Motivation
### This is an unofficial implementation of paper
[Extending Ionospheric Correction Coverage Area By Using A Neural 
Network Method](http://koreascience.or.kr/article/JAKO201614652759635.page)

##### Ionosphere's total electron content make confuse on wireless comunication especially GPS system

so, It's important to predict ionosphere's total electron content

# Framework used
<img src="https://img.shields.io/badge/PyTorch-EE4C2C?style=flat-square&logo=PyTorch&logoColor=white" height="30px" width="100px"/></a>

# Data Processing
Using variable 25 type of data, and they have diffent time scale.

##### So, U can get two versions of data that one is preprocessed data for same time scale(1), the other is original data(2)

1) Processed data is already interpolated into same time scale. All you have to do is modifying the regression model architecture

2) If you want to practice preprocessing data, select original data. Each data has it's own time scale. So, you have to preprocess the data.

# Example code
I built basic regression model, so you can easily modify the architecture

```python

class Regressor(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Sequential(
            nn.Linear(25, 64, bias=False),
            nn.BatchNorm1d(64, eps=1e-05, momentum=0.1),
            nn.ReLU()
        )
        
        self.layer2 = nn.Sequential(
            nn.Linear(64, 128, bias=False),
            nn.BatchNorm1d(128, eps=1e-05, momentum=0.1),
            nn.ReLU()
        )
        
        self.layer3 = nn.Sequential(
            nn.Linear(128, 256, bias=False),
            nn.BatchNorm1d(256, eps=1e-05, momentum=0.1),
            nn.ReLU()
        )
        
        self.layer4 = nn.Linear(256, 1, bias=False)
        
    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
      
        return x
        
```

# How to use?

##### 1) If you choose preprocessed data
You can do everyting in Ionosphere_regression.ipynb
Important thing is changing model architecture
Just set data path at start and make your own deep learning model architecture

##### 2) If you choose original data
You have to preprocess the data first, especally you can use interpolation.
After this all same with case 1)
