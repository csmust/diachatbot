#将控制台的输出，保存到log文件中
import sys

class Logger():
    def __init__(self, filename='train.log', stream=sys.stdout):
        self.terminal = stream
        self.log = open(filename, 'w',encoding="utf-8")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass



