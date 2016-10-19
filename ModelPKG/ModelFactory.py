## Factory for creating models by name
class Factory(object):
    def create(type):
        if type == "Logistic": import ModelPKG.Logistic; return ModelPKG.Logistic.model();
        if type == "Linear": import ModelPKG.Linear; return ModelPKG.model(); 
        if type == "DeepNet": import ModelPKG.DeepNet; return ModelPKG.DeepNet.model(); 
        assert 0, "Bad shape creation: " + type
    create = staticmethod(create)

def test():
    listOfModels = [Factory.create("Logistic"), Factory.create("Linear"), Factory.create("DeepNet")]
    for i in range(len(listOfModels)):
        listOfModels[i].train()  
        listOfModels[i].test() 
    
#test()