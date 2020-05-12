class Logger():
    def __init__(self):
        self.itrs = []
        self.losses = []
        self.itr_times = []
        self.args = None

    def update(self, itr, loss, itr_time):
        self.itrs.append(itr)
        self.losses.append(loss)
        self.itr_times.append(itr_time)
    
    def update_args(self, args):
        self.args = args
    
    def print(self, output):
        print(output)