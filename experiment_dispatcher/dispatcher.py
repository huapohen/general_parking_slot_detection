class Enumerate_params_dict():
    """
    get several param_pools get indexes
    pool_ls: list of param_list, e.g:[[0.1,0.01,0.001],[1e-4,1e-5,1e-6]] for lr and wd pairs
    task_thread: all the experiments are divided into several threads to do
 
    """
    def __init__(self, task_thread: int, if_single_id_task=False, **kwarg):
        def match_pairs(pair_ls, item_key):
            item_pool = kwarg[item_key]
            if type(item_pool) != list:
                raise ValueError('wrong type of the item pool:%s' % type(item_pool))
            if len(item_pool) == 0:
                raise ValueError('item_pool should not be temp, length of the item_pool: %s' % len(item_pool))
            new_data_ls = []
            for pair in pair_ls:
                for i in range(len(item_pool)):
                    new_data_ls.append(pair + ((item_key, i), ))
            return new_data_ls

        def divid_threads(ind_pairs, N: int):
            def get_param_dict(pair):
                temp = {}
                for p_key, ind in pair:
                    temp[p_key] = kwarg[p_key][ind]
                return temp

            data_ls = []
            for i in range(N):
                data_ls.append([])
            cnt = 0
            for pair in ind_pairs:
                param_dict = get_param_dict(pair)
                data_ls[cnt % len(data_ls)].append(param_dict)
                cnt += 1
            return data_ls

        data_ls = [
            (),
        ]
        item_key_ls = list(kwarg.keys())
        item_key_ls.sort()
        for item_key in item_key_ls:
            data_ls = match_pairs(data_ls, item_key)
        self.item_pool_dict = kwarg
        self.key_ind_pairs = data_ls
        if if_single_id_task:
            task_thread = len(data_ls)
        self.thread_pool = divid_threads(self.key_ind_pairs, task_thread)

    def get_thread(self, ind: int):
        thread_pool = self.thread_pool[ind % len(self.thread_pool)]
        return thread_pool

    @classmethod
    def demo(cls):
        test_id = 0
        param_pool_dict = {
            'a': [0, 1, 2],
            'b': [0, 1, 2],
        }
        task_manager = Enumerate_params_dict(task_thread=2, if_single_id_task=True, **param_pool_dict)
        param_pool = task_manager.get_thread(ind=test_id)
        print(param_pool)
        print('thread_pool', len(task_manager.thread_pool), task_manager.thread_pool)
