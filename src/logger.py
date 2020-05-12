class Logger():
    def __init__(self):
        self.itrs = []
        self.losses = []
        self.itr_times = []

        self.val_itrs = []
        self.val_losses = []
        self.args = None

    def update(self, itr, loss, itr_time):
        self.itrs.append(itr)
        self.losses.append(loss)
        self.itr_times.append(itr_time)
        
        self.print('Iteration {} -- Train loss: {}\t Iteration time: {}'.format(itr, loss, itr_time))
    
    def update_args(self, args):
        self.args = args

        self.print(self.args)

    def update_val_loss(self, itr, loss):
        self.val_itrs.append(itr)
        self.val_losses.append(loss)

        self.print('Iteration {} -- Validation loss: {}'.format(itr, loss))
    
    def print(self, output):
        print(output)