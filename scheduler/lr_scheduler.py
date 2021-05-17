class SimpleScheduler(object):
    def __init__(self, optimizer, lr):
        self.optimizer = optimizer
        self.lr = lr
        self.step_count = 0
        
    def step(self):
        self.optimizer.param_groups[0]['lr'] = self.get_lr()
        self.optimizer.step()
        self.step_count += 1
    
    def get_lr(self):
        if(step_count)
        return self.lr