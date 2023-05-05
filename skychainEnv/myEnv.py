import numpy as np
import math
from scipy.special import comb, perm
import gym
from gym.spaces import Box
import copy
import time

'''state space'''
node_number = 100000
consensus_node_number = 8000
tx_queue = 2e8
hash_rate = 300 #kGH/s

'''action sapce'''
epoch_len = 10.0             
shard_size = 200           
MAX_SHARD_SIZE = 500
MAX_EPOCH_LEN = 10000
shard_number = 20
MAX_SHARD_NUMBER = 1
MAX_BLOCK_SIZE = 10*np.power(2,20)

fix_epoch = np.ceil(MAX_EPOCH_LEN*0.1)
fix_shard = np.power(2, np.round(0.6 * 10/2) + 2)
fix_block = np.ceil(MAX_BLOCK_SIZE*0.1)
fix_pow = 1
test_type = 1  #1:our, 2:fixe epoch, 3:fix shard, 4:fix block

'''poisson'''
tx_speed = 100000
node_speed = 100
poisson_size = 100000000

'''speed'''
trans_rate = 1 * np.power(2,20)    #1MBps
'''size'''
header_size = 548
'''time'''
route_trans_time = 0.0003     
valiate_time = 0.1
add_block_time = 0.2
randomness_time = 20  
state_block_time = 1
verification_time = 1
shuffle_time = 100.0     


'''others'''
gamma = 1/2
security_parameter = 5   
no_corrupt_bound = 0.9  
delay_latency_bound = 200
block_size = 1024       
tx_size = 512             
MAX_TIME = 2000000         
MAX_TX_QUEUE = 2e13       
system_tolerance = 1/4      
shard_tolerance = 1/3       
corrupt_ablity = 1.4e-6        
no_safe_punishment = 0
confirmation_latency_punishment = -1000
queue_full_punishment = -1000
illegal_punishment = -1000
assign_punishment = 0

#observation_space = Box(low=np.array([1, 1]),high=np.array([MAX_TX_QUEUE, node_number]))
observation_space = Box(low=0, high=1, shape = (26, ))    #改为二进制输入 1+15+10
action_space = Box(low=np.array([0, 0, 0]),high=np.array([MAX_EPOCH_LEN, MAX_SHARD_NUMBER, MAX_BLOCK_SIZE]))    #ours
#action_space = Box(low=np.array([0, 0]),high=np.array([MAX_SHARD_NUMBER, MAX_BLOCK_SIZE]))  #fix epoch lenth
#action_space = Box(low=np.array([0, 0]),high=np.array([MAX_EPOCH_LEN, MAX_BLOCK_SIZE]))    #fix shards number
#action_space = Box(low=np.array([0, 0]),high=np.array([MAX_EPOCH_LEN, MAX_SHARD_NUMBER]))    #fix block size



class myEnvClass(gym.Env):
    def __init__(self):
        self.metadata = {'render.modes': []}
        self.reward_range = (-float('inf'), float('inf'))
        self.spec = None
        
        # Set these in ALL subclasses
        self.action_space = action_space
        self.observation_space = observation_space
        
        #initialize state and action
        self.node_number = node_number
        self.consensus_node_number = consensus_node_number
        self.tx_queue = tx_queue
        self.hash_rate = hash_rate

        self.observation = np.array([self.tx_queue,self.consensus_node_number, self.hash_rate])
        self.epoch_len = epoch_len
        self.shard_size = shard_size
        self.hash_rate = hash_rate
        #self.action = [self.epoch_len, self.shard_size]
        
        #initialize hyper-parameter
        self.delay_latency_bound = delay_latency_bound
        self.security_parameter = security_parameter
        self.no_corrupt_bound = no_corrupt_bound
        self.block_size = block_size
        self.tx_size = tx_size
        self.randomness_time = randomness_time
        self.state_block_time = state_block_time
        self.verification_time = verification_time
        self.shuffle_time = shuffle_time
        self.MAX_TIME = MAX_TIME
        self.step_time = 0.
        self.MAX_SHARD_SIZE = MAX_SHARD_SIZE
        self.MAX_EPOCH_LEN = MAX_EPOCH_LEN
        self.trans_rate = trans_rate          
        self.header_size = header_size
        self.route_trans_time = route_trans_time
        self.valiate_time = valiate_time        
        self.add_block_time = add_block_time     
        self.tx_poisson = np.random.poisson(lam = tx_speed, size = poisson_size)
        self.gamma = gamma
#        self.node_poisson = np.random.poisson(lam = node_speed, size = poisson_size)
#        temp_index = np.random.randint(0, 2, poisson_size)
#        old_int = 0
#        new_int = -1
#        temp_i = (temp_index == old_int)
#        temp_index[temp_i] = new_int
#        self.node_poisson = self.node_poisson*temp_index    
        self.node_poisson = np.random.normal(loc=10, scale=100, size = poisson_size)
        self.hashrate_in = np.random.normal(loc=0, scale=20, size = poisson_size)
        self.system_tolerance = system_tolerance
        self.shard_tolerance = shard_tolerance
        self.step_number = 0        
        
        '''fix'''
        self.fix_epoch = fix_epoch
        self.fix_shard = fix_shard
        self.fix_block = fix_block
        self.fix_pow = fix_pow
        
        
    def reset(self):
        """Resets the state of the environment and returns an initial observation.
        Returns: 
            observation (object): the initial observation.
        """
        #initialize state and action
        self.node_number = node_number
        self.consensus_node_number = consensus_node_number
        self.tx_queue = tx_queue
        self.observation = np.array([self.tx_queue,self.consensus_node_number, self.hash_rate])
        self.epoch_len = epoch_len
        self.shard_size = shard_size
        self.step_time = 0.
        self.MAX_TIME = MAX_TIME
        self.MAX_EPOCH_LEN = MAX_EPOCH_LEN
        self.MAX_SHARD_SIZE = MAX_SHARD_SIZE
        self.trans_rate = trans_rate          
        self.header_size = header_size
        self.route_trans_time = route_trans_time
        self.valiate_time = valiate_time        
        self.add_block_time = add_block_time      
        #self.tx_poisson = np.random.poisson(lam = tx_speed, size = poisson_size)    
#        self.node_poisson = np.random.normal(loc=10, scale=100, size = poisson_size)   
        #self.node_poisson
        #raise NotImplementedError
        self.gamma = gamma
        self.system_tolerance = system_tolerance
        self.shard_tolerance = shard_tolerance
        self.step_number = 0        
        
        obs_len = 15
        obs_len_2 = 10
        obs_temp = int(np.round(self.observation[1]))     
        obs_computing_difficulty = int(np.round(self.observation[2]))
        bi_obs_str = '{:015b}'.format(obs_temp)
        bi_obs_2_str = '{:010b}'.format(obs_computing_difficulty)
        bi_obs = []
        bi_obs_2 = []
        for i in range(obs_len):
            bi_obs.append(int(bi_obs_str[i]))
        for i in range(obs_len_2):
            bi_obs_2.append(int(bi_obs_2_str[i]))
        return np.hstack(([self.observation[0]], bi_obs, bi_obs_2))    
    
    def step(self, action):
#        if test_type == 1 or test_type == 4:
#            action[1] = int(action[1] * 10/2) + 2
#            action[1] = np.power(2, action[1])
##            action[1] = 64
#        elif test_type == 2:
#            action[0] = int(action[0] * 10/2) + 2
#            action[0] = np.power(2, action[0])
#            
#        for i in range(len(action)):
#            action[i] = np.ceil(action[i])
#        if test_type == 1:
#            self.fix_epoch = action[0]
#            self.fix_shard = action[1]
#            self.fix_block = action[2]
#        elif test_type == 2:
#            self.fix_shard = action[0]
#            self.fix_block = action[1]
#        elif test_type == 3:
#            self.fix_epoch = action[0]
#            self.fix_block = action[1]
#        elif test_type == 4:
#            self.fix_epoch = action[0]
#            self.fix_shard = action[1]
        
        action[1] = int(action[1] * 8)
        action[1] = np.power(2, action[1])
        self.fix_epoch = action[0]
        self.fix_shard = action[1]
        self.fix_block = action[2]
        self.fix_pow = 10
            
        obs_len = 15
        obs_len_2 = 10
#        print("step_time:", self.step_time)
#        print("node_poisson:", self.node_poisson[:10])
#        print("action:", [self.fix_epoch, self.fix_shard, self.fix_block])
        info = {'observation':self.observation, 'action':action, 'step_time':self.step_time}
#        print("obs:", self.observation)
#        print("before action:", action)
        

        if self.fix_epoch >= self.MAX_TIME or self.fix_shard >= self.node_number:
            print("too large...")
            self.step_time += 100
            reward = illegal_punishment
            info['step_time'] = self.step_time
            new_obs = self.observation.copy()
            obs_temp = int(np.round(new_obs[1]))
            obs_computing_difficulty = int(np.round(new_obs[2]))
            bi_obs_str = '{:015b}'.format(obs_temp)
            bi_obs_2_str = '{:010b}'.format(obs_computing_difficulty)
            bi_obs = []
            bi_obs_2 = []
            for i in range(obs_len):
                bi_obs.append(int(bi_obs_str[i]))
            for i in range(obs_len_2):
                bi_obs_2.append(int(bi_obs_2_str[i]))
            return np.hstack(([self.observation[0]], bi_obs, bi_obs_2)), reward, False, info
        self.step_number += 1
        reward = 0
        done= 0
        illegal = False
        m = math.ceil(self.observation[1]/self.fix_shard)
        new_txs = self._compute_new_txs(self.step_time, self.step_time + self.fix_epoch)
        ref_time = self.fix_epoch-self.randomness_time-self.shuffle_time-self.state_block_time-self.verification_time
        tx_reduct = math.ceil(self.fix_shard*(ref_time/self._consensus_time(action)*self.fix_block/self.tx_size*(1-self._compute_redundancy_txs(action))))
#        print("new_txs:", new_txs)
#        print("ref_time/self._consensus_time(action):", ref_time/self._consensus_time(action))
#        print("self.fix_block/self.tx_size:", self.fix_block/self.tx_size)
#        print("tx_reduct:", tx_reduct)
        if self._shard_security_hyp(action) and self._corrupt_security(action) and self._consensus_time(action) < ref_time/6 and self._assign_security(action, self.gamma):    #self._shard_security(action)
            #print("shard_safe")
            if self.observation[0] + new_txs < tx_reduct and self.observation[0] + new_txs < MAX_TX_QUEUE:
                print("no tx...")
#                time.sleep(5)
                reward = self.observation[0] + new_txs
#                print("reward:", reward)
#                print("self.observation[0]:", self.observation[0])
#                print("new_txs:", new_txs)
                self.observation[0] = 1
                self.observation[2] = np.around(hash_rate+self._compute_hashrate(self.run_time))
#                done=1
                new_obs = self.observation.copy()
#                print("no tx new_obs:", new_obs)
                return new_obs, reward, done, info
                
            else:
                self.observation[0] += (new_txs-tx_reduct)
                if self.observation[0] > MAX_TX_QUEUE:
                    self.observation[0] = MAX_TX_QUEUE
                reward = tx_reduct
#                if self.observation[0] > MAX_TX_QUEUE:
##                    print("queue_full!!!")
#                    self.observation[0] = MAX_TX_QUEUE
#                    reward = queue_full_punishment * action[0]           
#                    #reward = 0
#                else:
#                    reward = tx_reduct
        else:   
            #print("punish...")
#            print("action:", action)
            if self._shard_security_hyp(action) == False:
#                print("shard_unsafe")
                reward = assign_punishment
#                a=1
            elif self._corrupt_security(action) == False:
#                a=1
#                print("corrpt_unsafe")
                reward = assign_punishment
            elif self._assign_security(action, self.gamma) == False:
#                assign_punishment
#                print("assign_unsafe")
                reward = assign_punishment
            else:
                reward = assign_punishment
#                a=1
#                confirmation_latency_punishment
#                print("latency too large...")
#            print("unsafe")  
            self.observation[0] += new_txs
#            reward = -np.log2(-no_safe_punishment * action[0])
#            print("-no_safe_punishment * action[0]:", -no_safe_punishment * action[0])
            
            if self.observation[0] + new_txs > MAX_TX_QUEUE:
                self.observation[0] = MAX_TX_QUEUE
        self.step_time += self.fix_epoch
        new_node = self._compute_new_node(self.step_time, self.step_time + self.fix_epoch)
        self.observation[1] += new_node
        #self.observation[1] += (action[0]*node_speed-node_reduct)
        #done = False
        
#        if self.step_time > self.MAX_TIME:
#            done = 1
        if self.observation[1] <= 0:    
            self.observation[1] = 1
            #done = True
        if self.observation[1] >= self.node_number:
            self.observation[1] = self.node_number
        new_obs = self.observation.copy()
        self.observation[2] = np.around(hash_rate+self._compute_hashrate(self.step_time))
#        print("reward:", reward/action[0])
#        print("step_next_state:", self.observation)
#        print("action:", action)
#        print("info_step_time:", info['step_time'])
        info['step_time'] = self.step_time
        obs_temp = int(np.round(new_obs[1]))
        obs_computing_difficulty = int(np.round(new_obs[2]))
        bi_obs_str = '{:015b}'.format(obs_temp)
        bi_obs_2_str = '{:010b}'.format(obs_computing_difficulty)
        bi_obs = []
        bi_obs_2 = []
        for i in range(obs_len):
            bi_obs.append(int(bi_obs_str[i]))
        for i in range(obs_len_2):
            bi_obs_2.append(int(bi_obs_2_str[i]))
        return np.hstack(([new_obs[0]], bi_obs, bi_obs_2)), reward, done, info
#        return new_obs, reward, done, info
        
        
    def _get_step_time(self):
        return self.step_time
    
    
    def _shard_security_bio(self, action):      
        pro = 0.0
        m = math.ceil(self.observation[1]/action[1])
        #print("corrupt node number ", int(self.observation[1]*self.tolerance))
        #print("shard size", int(cur_act[1]))
        for x in range(math.floor(m*self.shard_tolerance), m+1):
            pro += (comb(m, x)*math.pow(self.system_tolerance, x)*math.pow(1-self.system_tolerance, m-x))
        #print("pro:", pro)
        if math.isnan(pro) or math.isinf(pro):
            return True
            
        return pro < math.pow(2, -self.security_parameter)
    
    def _shard_security_hyp(self, action):     
        pro = 0.0
        m = math.ceil(self.observation[1]/self.fix_shard)
#        print("mmmmm:",m)
        for x in range(math.floor(m * self.shard_tolerance), m+1, 1):
#            test = comb(self.observation[1], m)
#            temp = comb(np.floor(self.observation[1]*self.system_tolerance), x, exact=True)*comb(np.floor(self.observation[1]-self.observation[1]*self.system_tolerance,), m-x,exact=True)/comb(np.floor(self.observation[1]), m, exact=True)
            temp = comb(np.floor(self.observation[1]*self.system_tolerance), x)*comb(np.floor(self.observation[1]-self.observation[1]*self.system_tolerance,), m-x)/comb(np.floor(self.observation[1]), m)
            if math.isnan(temp) :
                continue
            else:
                pro += temp
#            print("comb test:", test)
#            print("comb temp:", temp)
        if math.isnan(pro) or math.isinf(pro):
            return True
#        print("pro:", pro)
        return pro < math.pow(2, -self.security_parameter)
    
    def _corrupt_security(self, action):        

        c_t = self.fix_epoch-self.randomness_time-self.shuffle_time    
        n_c = self.observation[1] * self.system_tolerance * corrupt_ablity * c_t
#        print("n_c", n_c)
#        print("self.observation[1]:", self.observation[1])
#        print("action[1]:", action[1])
#        
#        print("corrupt number", math.ceil(self.observation[1]/action[1] * self.shard_tolerance))
        return math.ceil(n_c) <= math.ceil(self.observation[1]/self.fix_shard * self.shard_tolerance)
        
    
    def _consensus_time(self, action):    
        #print("shard-cross-time:", 0.01*math.log(self.observation[1], 2))
        #print("consensus_time:", 0.01*(math.log(self.observation[1], 2)+cur_act[1]*cur_act[1]/self.block_size))
        consensus_t = 3*self.valiate_time + self.add_block_time
        gossip_t = (self.fix_block / self.trans_rate) * math.log(self.observation[1]/self.fix_shard ,2) + 2 * (self.header_size / self.trans_rate)* math.log(self.observation[1]/self.fix_shard ,2)
        new_tx_consensus_t = consensus_t + gossip_t
#        print("new_tx_consensus_t:",new_tx_consensus_t)
        total_time = (self.route_trans_time*math.log(self.fix_shard+1, 2) + new_tx_consensus_t)
#        print("cross_shard_v time:", self.trans_time*math.log(self.observation[1]/action[1]+1, 2))
#        print("one tx time:", total_time)
#        print("total_time:", total_time)
        return total_time
    
    def _compute_new_txs(self, start_time, end_time):
        start_time = int(start_time)
        end_time = int(end_time)
#        if end_time > self.MAX_TIME:
#            end_time = self.MAX_TIME
        new_txs = 0
        for i in range(start_time, end_time):
            new_txs += self.tx_poisson[i]
        return new_txs
    
    def _compute_new_node(self, start_time, end_time):
        start_time = int(start_time)
        end_time = int(end_time)
#        new_node = 0
#        for i in range(start_time, end_time):
#            new_node += (self.node_poisson[i])
#        new_node += (self.node_poisson[end_time])
#        print('node_change:', self.node_poisson[end_time])
        return self.node_poisson[end_time]
#        return new_node
    
    def _compute_redundancy_txs(self, action):      
        k = math.ceil(self.fix_shard)
        rebundance_tx = 1 - math.pow(1/k, 2)
#        print("rebundance_tx/(rebundance_tx+1):", rebundance_tx/(rebundance_tx+1))
        return rebundance_tx/(rebundance_tx+1)
    def _compute_hashrate(self, end_time):
        end_time = int(end_time)
        #return 0
        #print("self.hashrate_in[end_time]:", self.hashrate_in[end_time])
        if self.hashrate_in[end_time] + hash_rate <= 0:
            return 1
        else:
            return self.hashrate_in[end_time]
        
    def _assign_security(self, action, _gamma):
        pow_times = np.around(self.observation[2]/self.fix_pow)
        return pow_times <= np.around(_gamma*self.fix_shard) and pow_times > 0
        
    def __get_node_poisson__(self):
        return self.node_poisson
    