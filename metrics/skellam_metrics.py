#!/usr/bin/env python


class SkellamMetrics:

    def __init__(self, x, y, y_hat):
        self.x = x
        self.y = y
        self.y_hat = y_hat

    def sse(self):
        return ((self.y - self.y_hat)**2).sum()

    def y_bar(self):
        return self.y.mean()

    def sst(self):
        return ((self.y - self.y_bar())**2).sum()

    def r2(self):
        sse_sst = self.sse()/self.sst()
        return 1-sse_sst

