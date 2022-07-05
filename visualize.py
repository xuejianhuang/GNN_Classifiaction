from visdom import  Visdom
class Visualize():
    def __init__(self,server,host,wins):
        self.server=server
        self.host=host
        self.wins=wins
        self.viz = Visdom (server=self.server, port=self.host)
        self.viz.line ([0], [0], name='train_loss', win=self.wins[0],
                  opts=dict (title='LOSS', xlabel='epoch', ylabel='loss', showlegend=True))
        self.viz.line ([0], [0], name='test_loss', win=self.wins[0], update='append')

        self.viz.line ([0], [0], name='train_acc', win=self.wins[1],
                  opts=dict (title='Accuracy', xlabel='epoch', ylabel='accuracy', showlegend=True))
        self.viz.line ([0], [0], name='test_acc', win=self.wins[1], update='append')

        self.viz.line ([0], [0], name='train_loss', win=self.wins[2],
                  opts=dict (title='step_loss', xlabel='step', ylabel='loss', showlegend=True))


    def append(self,y,x,name,win):
        self.viz.line(y,x,name=name,win=win,update='append')


