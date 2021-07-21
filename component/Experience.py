import sys
import math
import random
import numpy as np

import sys
import math

from collections import deque
import random

def list_to_dict(in_list):
    return dict((i, in_list[i]) for i in range(0, len(in_list)))


def exchange_key_value(in_dict):
    return dict((in_dict[i], i) for i in in_dict)


def main():
    pass

class BinaryHeap(object):

    def __init__(self, priority_size=100, priority_init=None, replace=True):
        self.e2p = {}
        self.p2e = {}
        self.replace = replace

        if priority_init is None:
            self.priority_queue = {}
            self.size = 0
            self.max_size = priority_size
        else:
            # not yet test
            self.priority_queue = priority_init
            self.size = len(self.priority_queue)
            self.max_size = None or self.size

            experience_list = list(map(lambda x: self.priority_queue[x], self.priority_queue))
            self.p2e = utility.list_to_dict(experience_list)
            self.e2p = utility.exchange_key_value(self.p2e)
            for i in range(int(self.size / 2), -1, -1):
                self.down_heap(i)

    def __repr__(self):
        """
        :return: string of the priority queue, with level info
        """
        if self.size == 0:
            return 'No element in heap!'
        to_string = ''
        level = -1
        max_level = int(math.floor(math.log(self.size, 2)))

        for i in range(1, self.size + 1):
            now_level = int(math.floor(math.log(i, 2)))
            if level != now_level:
                to_string = to_string + ('\n' if level != -1 else '') \
                            + '    ' * (max_level - now_level)
                level = now_level

            to_string = to_string + '%.2f ' % self.priority_queue[i][1] + '    ' * (max_level - now_level)

        return to_string

    def check_full(self):
        return self.size > self.max_size

    def _insert(self, priority, e_id):
        """
        insert new experience id with priority
        (maybe don't need get_max_priority and implement it in this function)
        :param priority: priority value
        :param e_id: experience id
        :return: bool
        """
        self.size += 1

        if self.check_full() and not self.replace:
            sys.stderr.write('Error: no space left to add experience id %d with priority value %f\n' % (e_id, priority))
            return False
        else:
            self.size = min(self.size, self.max_size)

        self.priority_queue[self.size] = (priority, e_id)
        self.p2e[self.size] = e_id
        self.e2p[e_id] = self.size

        self.up_heap(self.size)
        return True

    def update(self, priority, e_id):
        """
        update priority value according its experience id
        :param priority: new priority value
        :param e_id: experience id
        :return: bool
        """
        if e_id in self.e2p:
            p_id = self.e2p[e_id]
            self.priority_queue[p_id] = (priority, e_id)
            self.p2e[p_id] = e_id

            self.down_heap(p_id)
            self.up_heap(p_id)
            return True
        else:
            # this e id is new, do insert
            return self._insert(priority, e_id)

    def get_max_priority(self):
        """
        get max priority, if no experience, return 1
        :return: max priority if size > 0 else 1
        """
        if self.size > 0:
            return self.priority_queue[1][0]
        else:
            return 1

    def pop(self):
        """
        pop out the max priority value with its experience id
        :return: priority value & experience id
        """
        if self.size == 0:
            sys.stderr.write('Error: no value in heap, pop failed\n')
            return False, False

        pop_priority, pop_e_id = self.priority_queue[1]
        self.e2p[pop_e_id] = -1
        # replace first
        last_priority, last_e_id = self.priority_queue[self.size]
        self.priority_queue[1] = (last_priority, last_e_id)
        self.size -= 1
        self.e2p[last_e_id] = 1
        self.p2e[1] = last_e_id

        self.down_heap(1)

        return pop_priority, pop_e_id

    def up_heap(self, i):
        """
        upward balance
        :param i: tree node i
        :return: None
        """
        if i > 1:
            parent = int(math.floor(i / 2))
            if self.priority_queue[parent][0] < self.priority_queue[i][0]:
                tmp = self.priority_queue[i]
                self.priority_queue[i] = self.priority_queue[parent]
                self.priority_queue[parent] = tmp
                # change e2p & p2e
                self.e2p[self.priority_queue[i][1]] = i
                self.e2p[self.priority_queue[parent][1]] = parent
                self.p2e[i] = self.priority_queue[i][1]
                self.p2e[parent] = self.priority_queue[parent][1]
                # up heap parent
                self.up_heap(parent)

    def down_heap(self, i):
        """
        downward balance
        :param i: tree node i
        :return: None
        """
        if i < self.size:
            greatest = i
            left, right = i * 2, i * 2 + 1
            if left < self.size and self.priority_queue[left][0] > self.priority_queue[greatest][0]:
                greatest = left
            if right < self.size and self.priority_queue[right][0] > self.priority_queue[greatest][0]:
                greatest = right

            if greatest != i:
                tmp = self.priority_queue[i]
                self.priority_queue[i] = self.priority_queue[greatest]
                self.priority_queue[greatest] = tmp
                # change e2p & p2e
                self.e2p[self.priority_queue[i][1]] = i
                self.e2p[self.priority_queue[greatest][1]] = greatest
                self.p2e[i] = self.priority_queue[i][1]
                self.p2e[greatest] = self.priority_queue[greatest][1]
                # down heap greatest
                self.down_heap(greatest)

    def get_priority(self):
        """
        get all priority value
        :return: list of priority
        """
        return list(map(lambda x: x[0], self.priority_queue.values()))[0:self.size]

    def get_e_id(self):
        """
        get all experience id in priority queue
        :return: list of experience ids order by their priority
        """
        return list(map(lambda x: x[1], self.priority_queue.values()))[0:self.size]

    def balance_tree(self):
        """
        rebalance priority queue
        :return: None
        """
        sort_array = sorted(self.priority_queue.values(), key=lambda x: x[0], reverse=True)
        # reconstruct priority_queue
        self.priority_queue.clear()
        self.p2e.clear()
        self.e2p.clear()
        cnt = 1
        while cnt <= self.size:
            priority, e_id = sort_array[cnt - 1]
            self.priority_queue[cnt] = (priority, e_id)
            self.p2e[cnt] = e_id
            self.e2p[e_id] = cnt
            cnt += 1
        # sort the heap
        for i in range(int(math.floor(self.size / 2)), 1, -1):
            self.down_heap(i)

    def priority_to_experience(self, priority_ids):
        """
        retrieve experience ids by priority ids
        :param priority_ids: list of priority id
        :return: list of experience id
        """
        return [self.p2e[i] for i in priority_ids]


class Experience(object):

    def __init__(self, conf):
        self.size = conf['size']
        self.replace_flag = conf['replace_old'] if 'replace_old' in conf else True
        self.priority_size = conf['priority_size'] if 'priority_size' in conf else self.size

        self.alpha = conf['alpha'] if 'alpha' in conf else 0.7
        self.beta_zero = conf['beta_zero'] if 'beta_zero' in conf else 0.5
        self.batch_size = conf['batch_size'] if 'batch_size' in conf else 32
        self.learn_start = conf['learn_start'] if 'learn_start' in conf else 1000
        self.total_steps = conf['steps'] if 'steps' in conf else 100000
        # partition number N, split total size to N part
        self.partition_num = conf['partition_num'] if 'partition_num' in conf else 100

        self.index = 0
        self.record_size = 0
        self.isFull = False

        self._experience = {}
        self.priority_queue = BinaryHeap(self.priority_size)  #TODO
        self.distributions = self.build_distributions()  # TODO

        self.beta_grad = (1 - self.beta_zero) / (self.total_steps - self.learn_start)

    def build_distributions(self):
        """
        preprocess pow of rank
        (rank i) ^ (-alpha) / sum ((rank i) ^ (-alpha))
        """
        res = {}
        n_partitions = self.partition_num
        partition_num = 1
        # each part size
        partition_size = int(math.floor(self.size / n_partitions))

        for n in range(partition_size, self.size + 1, partition_size):
            if self.learn_start <= n <= self.priority_size:
                distribution = {}
                # P(i) = (rank i) ^ (-alpha) / sum ((rank i) ^ (-alpha))
                pdf = list(
                    map(lambda x: math.pow(x, -self.alpha), range(1, n + 1))
                )
                pdf_sum = math.fsum(pdf)
                distribution['pdf'] = list(map(lambda x: x / pdf_sum, pdf))
                # split to k segment, and than uniform sample in each k
                # set k = batch_size, each segment has total probability is 1 / batch_size
                # strata_ends keep each segment start pos and end pos
                cdf = np.cumsum(distribution['pdf'])
                strata_ends = {1: 0, self.batch_size + 1: n}
                step = 1.0 / self.batch_size
                index = 1
                for s in range(2, self.batch_size + 1):
                    while cdf[index] < step:
                        index += 1
                    strata_ends[s] = index
                    step += 1.0 / self.batch_size

                distribution['strata_ends'] = strata_ends

                res[partition_num] = distribution

            partition_num += 1

        return res

    def fix_index(self):
        """
        get next insert index
        :return: index, int
        """
        if self.record_size <= self.size:
            self.record_size += 1
        if self.index % self.size == 0:
            self.isFull = True if len(self._experience) == self.size else False
            if self.replace_flag:
                self.index = 1
                return self.index
            else:
                sys.stderr.write('Experience replay buff is full and replace is set to FALSE!\n')
                return -1
        else:
            self.index += 1
            return self.index

    def store(self, experience):
        """
        store experience, suggest that experience is a tuple of (s1, a, r, s2, t)
        so each experience is valid
        :param experience: maybe a tuple, or list
        :return: bool, indicate insert status
        """
        insert_index = self.fix_index()
        if insert_index > 0:
            if insert_index in self._experience:
                del self._experience[insert_index]
            self._experience[insert_index] = experience
            # add to priority queue
            priority = self.priority_queue.get_max_priority()
            self.priority_queue.update(priority, insert_index)
            return True
        else:
            sys.stderr.write('Insert failed\n')
            return False

    def retrieve(self, indices):
        """
        get experience from indices
        :param indices: list of experience id
        :return: experience replay sample
        """
        return [self._experience[v] for v in indices]

    def rebalance(self):
        """
        rebalance priority queue
        :return: None
        """
        self.priority_queue.balance_tree()

    def update_priority(self, indices, delta):
        """
        update priority according indices and deltas
        :param indices: list of experience id
        :param delta: list of delta, order correspond to indices
        :return: None
        """
        for i in range(0, len(indices)):
            self.priority_queue.update(math.fabs(delta[i]), indices[i])

    def sample(self, global_step):
        """
        sample a mini batch from experience replay
        :param global_step: now training step
        :return: experience, list, samples
        :return: w, list, weights
        :return: rank_e_id, list, samples id, used for update priority
        """
        if self.record_size < self.learn_start:
            sys.stderr.write('Record size less than learn start! Sample failed\n')
            return False, False, False

        dist_index = int(math.floor(self.record_size / self.size * self.partition_num))
        # issue 1 by @camigord
        partition_size = int(math.floor(self.size / self.partition_num))
        partition_max = dist_index * partition_size
        distribution = self.distributions[dist_index]
        rank_list = []
        # sample from k segments
        for n in range(1, self.batch_size + 1):
            if distribution['strata_ends'][n] + 1 <= distribution['strata_ends'][n + 1]:
                index = random.randint(distribution['strata_ends'][n] + 1,
                                       distribution['strata_ends'][n + 1])
            else:
                index = random.randint(distribution['strata_ends'][n + 1],
                                       distribution['strata_ends'][n] + 1)
            rank_list.append(index)

        # beta, increase by global_step, max 1
        beta = min(self.beta_zero + (global_step - self.learn_start - 1) * self.beta_grad, 1)
        # find all alpha pow, notice that pdf is a list, start from 0
        alpha_pow = [distribution['pdf'][v - 1] for v in rank_list]
        # w = (N * P(i)) ^ (-beta) / max w
        w = np.power(np.array(alpha_pow) * partition_max, -beta)
        w_max = max(w)
        w = np.divide(w, w_max)
        # rank list is priority id
        # convert to experience id
        rank_e_id = self.priority_queue.priority_to_experience(rank_list)
        # get experience id according rank_e_id
        experience = self.retrieve(rank_e_id)
        return experience, w, rank_e_id


class ReplayBuffer(object):

  def __init__(self, buffer_size):

    self.buffer_size = buffer_size
    self.num_experiences = 0
    #self.buffer = deque()
    conf = {'size': 100000,
            'learn_start': 0,
            'partition_num': 3,
            'total_step': 100000,
            'batch_size': 32}
    self.replay_memory = Experience(conf)

  def getBatch(self, batch_size):
    batch, w, e_id = self.replay_memory.sample(self.num_experiences)
    self.e_id = e_id
    self.w_id = w
    '''#state t
    self.state_t_batch = [item[0] for item in batch]
    self.state_t_batch = np.array(self.state_t_batch)
    #state t+1        
    self.state_t_1_batch = [item[1] for item in batch]
    self.state_t_1_batch = np.array( self.state_t_1_batch)
    self.action_batch = [item[2] for item in batch]
    self.action_batch = np.array(self.action_batch)
    self.action_batch = np.reshape(self.action_batch,[len(self.action_batch),self.num_actions])
    self.reward_batch = [item[3] for item in batch]
    self.reward_batch = np.array(self.reward_batch)
    self.done_batch = [item[4] for item in batch]
    self.done_batch = np.array(self.done_batch)'''
    return batch, self.w_id, self.e_id


  def size(self):
    return self.buffer_size

  def add(self, state, action, reward, next_state, done):#add(self, state, next_state, action, reward, done):
    #new_experience = (state, next_action, action, reward, done)#(state, action, reward, next_state, done)
    self.replay_memory.store((state, action, reward, next_state, done))
    #if self.num_experiences < self.buffer_size:
    #  self.buffer.append(new_experience)
    self.num_experiences += 1
    #else:
    #  self.buffer.popleft()
    #  self.buffer.append(new_experience)

  def count(self):
    # if buffer is full, return buffer size
    # otherwise, return experience counter
    return self.num_experiences

  #def erase(self):
  #  self.buffer = deque()
  #  self.num_experiences = 0
  def rebalance(self):
    self.replay_memory.rebalance()

  def update_priority(self, indices, delta):
    self.replay_memory.update_priority(indices, delta)