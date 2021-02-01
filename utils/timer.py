import time
import datetime

class Timer(object):
    def __init__(self):
        self.init_time = time.time()
        self.total_time = 0.
        self.calls = 0
        self.start_time = 0.
        self.diff = 0.
        self.average_time = 0.
        self.remain_time = 0.

    def tic(self):
        # 使用time.time而不是time.clock，因为time.clock在多线程下工作异常
        self.start_time = time.time()

    def toc(self, average=True):
        # 计算单次时间
        self.diff = time.time() - self.start_time
        # 计算总时间
        self.total_time += self.diff
        # 计算平均时间
        self.calls += 1
        self.average_time = self.total_time / self.calls
        if average:
            # 返回平均时间
            return self.average_time
        else:
            # 返回单次时间
            return self.diff

    def remain(self, iters, max_iters):
        if iters == 0:
            self.remain_time = 0
        else:
            # 预估训练还需要多长时间
            self.remain_time = (time.time() - self.init_time) * \
                (max_iters - iters) / iters
        return str(datetime.timedelta(seconds=int(self.remain_time)))
