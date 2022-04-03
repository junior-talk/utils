import os
import numpy as np
import random
import multiprocessing
import threading
import queue
import itertools
from tensorflow import keras
import logging

class ReaderProcess(multiprocessing.Process):
    def __init__(self, idx, files, parse, q, running, batch_size=64, cycle=False):
        super(ReaderProcess, self).__init__()
        self._name = 'Reader-{}'.format(idx)
        self._targets = files
        if cycle is True:
            self._targets = itertools.cycle(files)
        self._parse = parse
        self._running = running
        self._q = q
        self._batch_size = batch_size

    def run(self):
        batch = []
        logging.info('[{}] start'.format(self._name))
        for _ in self._targets:
            with open(_, 'r') as f:
                logging.info('[{}] reading {}'.format(self._name, _))
                for line in f:
                    result = self._parse(line)
                    if not result:
                        continue
                    if len(batch) == 0:
                        batch = [[] for _ in range(len(result))]
                    assert len(batch) == len(result)
                    for idx in range(len(result)):
                        batch[idx].extend(result[idx])
                    if len(batch[0]) >= self._batch_size:
                        self._q.put(batch, block=True)
                        logging.info('[{}] Put {} records to reader queue, new size {}'.format(self._name, len(batch[0]), self._q.qsize()))
                        batch = []
                    if not self._running.value:
                        break
            if not self._running.value:
                break
        if batch and batch[0]:
            self._q.put(batch, block=True)
            logging.info('[{}] Put {} reminder records to reader queue, new size {}'.format(self._name, len(batch[0]), self._q.qsize()))
        self._q.put(None)
        logging.info('[{}] end'.format(self._name))

class ParallelFileReader(keras.utils.Sequence):
    def __init__(self, path, parse, data_size=None,
                 batch_size=64, buffer_size=1024,
                 n_readers=4, cycle=False, shuffle=True):
        # parse-数据解析函数
        # 按行处理，line为原始文件行
        # 返回结果格式为 [[col_1], [col_2], [col_3], ..., [col_n]]

        assert buffer_size % batch_size == 0
        self._parse = parse

        assert os.path.isdir(path)
        self.data_filenames = [os.path.join(path, _) for _ in os.listdir(path)]
        assert not cycle or data_size is not None
        self._cycle = cycle
        self._data_size = data_size or 0
        if data_size is None:
            for _ in self.data_filenames:
                self._data_size += sum(1 for _ in open(_))
        logging.info("[Master] Total dataset size : " + str(self._data_size))

        self._batch_size = batch_size
        self._shuffle = shuffle
        self._buffer_size = buffer_size
        self._buffer = []
        self._idx = 0

        assert n_readers > 0
        self._running = multiprocessing.Value('i', False)
        self._n_readers = min(n_readers, len(self.data_filenames))
        logging.info("[Master] Reader number : " + str(self._n_readers))
        self._readers = []
        self._data_queues = []
        for _ in range(self._n_readers):
            q = multiprocessing.Queue(max(1, self._buffer_size // self._batch_size))
            self._data_queues.append(q)

        self._merge_queue = queue.Queue(2)
        self._merge_thread = None

    def _merge_readers(self):
        n_running_readers = self._n_readers
        buffer = []
        while True:
            for idx, _ in enumerate(self._data_queues):
                if not self._running.value:
                    return
                try:
                    data = _.get(False)
                except queue.Empty:
                    continue
                if data is None:
                    logging.info('[Merge] Got None from Reader-{} queue.'.format(idx))
                    n_running_readers -= 1
                    if n_running_readers <= 0:
                        if buffer and buffer[0]:
                            self._merge_queue.put(buffer)
                            logging.info('[Merge] Put {} reminder records to merge queue, new size {}.'.format(len(buffer[0]), self._merge_queue.qsize()))
                        return
                    continue
                logging.info('[Merge] Got {} records from Reader-{} queue, new size {}.'.format(len(data[0]), idx, _.qsize()))
                if len(buffer) == 0:
                    buffer = [[] for _ in range(len(data))]
                assert len(buffer) == len(data)
                for idx in range(len(data)):
                    buffer[idx].extend(data[idx])
                if len(buffer[idx]) < self._buffer_size:
                    continue
                self._merge_queue.put([_[:self._buffer_size] for _ in buffer])
                logging.info('[Merge] Put {} records to merge queue, new size {}.'.format(len(buffer[0]), self._merge_queue.qsize()))
                buffer = [_[self._buffer_size:] for _ in buffer]

    def start(self):
        assert not self._running.value, 'already running, call stop() first.'
        self._idx = 0
        self._running.value = True

        for _ in range(self._n_readers):
            targets = self.data_filenames[_::self._n_readers]
            reader = ReaderProcess(_, targets, self._parse,
                                  self._data_queues[_], self._running,
                                  self._batch_size, cycle=self._cycle)
            reader.daemon = True
            reader.start()
            self._readers.append(reader)

        self._merge_thread = threading.Thread(target=self._merge_readers)
        self._merge_thread.start()

    def _clear_queue(self, q, tries=5):
        while not q.empty():
            try:
                q.get(False)
            except queue.Empty:
                pass
        if tries > 0:
            self._clear_queue(q, tries - 1)

    # 注意执行顺序，防止进程/线程被block
    def stop(self):
        self._running.value = False
        self._idx = len(self)

        if self._merge_thread:
            self._clear_queue(self._merge_queue)
            self._merge_thread.join()
            self._merge_thread = None
        self._clear_queue(self._merge_queue, 1)

        if self._readers:
            for idx in range(self._n_readers):
                self._clear_queue(self._data_queues[idx])
                self._readers[idx].join()
                logging.info('[Master] Reader-{} join'.format(idx))
        self._readers = []

        for _ in self._data_queues:
            self._clear_queue(_, 1)

    def __len__(self):
        return (self._data_size + self._batch_size - 1) // self._batch_size

    def on_epoch_end(self):
        self.stop()
        self.start()

    def __getitem__(self, idx):
        assert self._running.value
        idx = self._idx
        assert idx < len(self), 'out of bounds'
        self._idx += 1

        idx = idx * self._batch_size % self._buffer_size
        if idx == 0:
            self._buffer = self._merge_queue.get()
            if self._shuffle:
                select_idx = list(range(len(self._buffer[0])))
                random.shuffle(select_idx)
                for col in range(len(self._buffer)):
                  self._buffer[col] = np.array(self._buffer[col])[select_idx]
            logging.info('[Master] Got {} records from merge queue, new size {}.'.format(len(self._buffer[0]), self._merge_queue.qsize()))

        batch = []
        for _ in self._buffer:
            batch.append(_[idx: idx + self._batch_size])

        return batch

if __name__ == "__main__":
    data_root = './data/test/'
    LOG_FORMAT = "%(message)s"
    logging.basicConfig(filename='reader.log', level=logging.INFO, format=LOG_FORMAT)

    def parse(line):
        return [[_] for _ in line.strip('\r\n').split('\t')]

    reader = ParallelFileReader(data_root, parse, shuffle=False, batch_size=2, buffer_size=4, n_readers=2)
    reader.start()

    for _ in range(len(reader)):
        print(_, reader[_])
    reader.stop()
    for _ in reader._data_queues:
        print(_.qsize())
