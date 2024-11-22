import os
from . import trainer
import json
import glob

def readjson(file):
    min_iter, min_l1 = None, 1e10
    with open(file, "r") as f:
        for i, line in enumerate(f):
            line = line.strip()
            if line:
                data = json.loads(line)
                l1 = data["val"]["current"]["L1"]
                if l1 < min_l1:
                    min_iter, min_l1 = i+1, l1
    return {"iter": min_iter,
            "L1": min_l1}
                
                
def getbestNN(files):
    min_exp, min_l1, min_res = None, 1e10, None
    for _file in files:
        res = readjson(_file)
        if res["L1"] < min_l1:
            min_exp, min_res = _file, res
    min_res["exp"] = min_exp
    return min_res


class Bermudan():
    def __init__(self, config):
        self.T = config["T"]
        self.num_ex = config["num_ex"]
    
    @staticmethod
    def get_NNpayoff(base_dir):
        pattern = os.path.join(base_dir, "*", "*", "*","*","result.json")
        files = glob.glob(pattern)
        bestnn = getbestNN(files)
        with open(os.path.join(base_dir, "bestNN.json"), "w") as f:
            json.dump(bestnn, f)
            
        pathparams = "/" + os.path.join(*(bestnn["exp"].split("/")[:-1] + ["params.json"]))
        with open(pathparams, "r") as f:
            config = json.load(f)
        config["checkpoint"] = True

        mytrainer = trainer.Trainer(config)
        pathmodel = "/" + os.path.join(*bestnn["exp"].split("/")[:-1]) + "/checkpoint_" + str(bestnn["iter"]).zfill(6) + "/model.pth"
        mytrainer.load_checkpoint(pathmodel)
        for param in mytrainer.model.parameters():
            param.requires_grad = False
        return mytrainer.model.module.forward
        

    def recursive_training(self, mode, base_dir):
        dt = self.T / self.num_ex
        for i in range(self.num_ex-1, -1, -1):
            parser = trainer.get_args()
            args = parser.parse_args()
            args.gpus = 1
            args.mode = mode
            if i == self.num_ex-1:
                payoff = None
            else:
                payoff = self.get_NNpayoff(exp_dir)
            exp_dir = os.path.join(base_dir, f"{str(i).zfill(2)}th_exercise_date")
            trainer.main(vars(args), dt, i, self.num_ex, exp_dir, payoff)

        
        


