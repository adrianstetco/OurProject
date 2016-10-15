# USE SCIKIT-LEARN
from BaseModel import abstractmodel

class model(abstractmodel):
    def train(self): print("Linear train")
    def test(self): print("Linear test")
    
