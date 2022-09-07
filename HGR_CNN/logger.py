import os

class Logger:
    def __init__(self,dir,name):
        self.name = name
        self.dir = dir
    def log(self,result):
        dir_name = os.path.join(self.dir,"log.txt")
        f = open(dir_name,"a+")
        f.write(self.name +",")
        f.write(result+ "\n")
        f.close()