import os, pickle
from datetime import datetime

import numpy as np
import pandas as pd
import tensorflow as tf
import scipy.sparse as sp
from scipy.sparse import csr_matrix

# os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

data_path = '../datasets/old/'

f = open(data_path + "kb_nan.txt", "w", encoding='utf-8')
f1 = open(data_path + "ent_mapping.txt", "w", encoding='utf-8')


# reader


class BasicReader(object):

    def read(self, data_path=data_path):
        # read KGs
        def read_kb(path, names):
            return pd.read_csv(path, sep='\t', header=None, names=names)

        pd.set_option('display.max_rows', None)
        kb1 = read_kb(data_path + 'triples_1', names=['h_id', 'r_id', 't_id'])  # 读取第一个hn
        kb2 = read_kb(data_path + 'triples_2', names=['h_id', 'r_id', 't_id'])  # 读取第二个hn

        ent_mapping = read_kb(data_path + 'sup_ent_ids', names=['kb_1', 'kb_2'])  # 读取映射文档
        ent_testing = read_kb(data_path + 'ref_ent_ids', names=['kb_1', 'kb_2'])  #

        if not os.path.exists(data_path + 'sup_rel_ids'):
            os.mknod(data_path + 'sup_rel_ids')
        if not os.path.exists(data_path + 'rel_rel_ids'):
            os.mknod(data_path + 'rel_rel_ids')

        rel_mapping = read_kb(data_path + 'sup_rel_ids', names=['kb_1', 'kb_2'])
        rel_testing = read_kb(data_path + 'rel_rel_ids', names=['kb_1', 'kb_2'])

        ent_id_1 = read_kb(data_path + 'ent_ids_1', names=['id', 'e'])
        ent_id_2 = read_kb(data_path + 'ent_ids_2', names=['id', 'e'])
        ent_id_2.loc[:, 'e'] += ':KB2'
        i2el_1 = pd.Series(ent_id_1.e.values, index=ent_id_1.id.values)
        i2el_2 = pd.Series(ent_id_2.e.values, index=ent_id_2.id.values)

        rel_id_1 = read_kb(data_path + 'rel_ids_1', names=['id', 'r'])
        rel_id_2 = read_kb(data_path + 'rel_ids_2', names=['id', 'r'])
        rel_id_2.loc[:, 'r'] += ':KB2'
        i2rl_1 = pd.Series(rel_id_1.r.values, index=rel_id_1.id.values)
        i2rl_2 = pd.Series(rel_id_2.r.values, index=rel_id_2.id.values)

        # convert id
        def id2label(df, i2el, i2rl, is_kb=True):
            if is_kb:
                df['h'] = i2el.loc[df.h_id.values].values
                df['r'] = i2rl.loc[df.r_id.values].values
                df['t'] = i2el.loc[df.t_id.values].values

                return df
            else:
                df['kb_1'] = i2el.loc[df.kb_1.values].values
                df['kb_2'] = i2rl.loc[df.kb_2.values].values

                return df

        id2label(kb1, i2el_1, i2rl_1)
        #         print("kb1")
        #         print(kb1)
        id2label(kb2, i2el_2, i2rl_2)
        #         print("ent_mapping")
        #         print(ent_mapping)
        id2label(ent_mapping, i2el_1, i2el_2, is_kb=False)
        id2label(rel_mapping, i2rl_1, i2rl_2, is_kb=False)
        id2label(ent_testing, i2el_1, i2el_2, is_kb=False)
        id2label(rel_testing, i2rl_1, i2rl_2, is_kb=False)

        # add reverse edges
        kb = pd.concat([kb1, kb2], ignore_index=True)

        kb = kb[['h', 'r', 't']]

        rev_r = kb.r + ':reverse'
        rev_kb = kb.rename(columns={'h': 't', 't': 'h'})
        rev_kb['r'] = rev_r.values
        #         print("rev_kb")
        #         print(rev_kb)
        kb = pd.concat([kb, rev_kb], ignore_index=True)

        rev_rmap = rel_mapping + ':reverse'
        rel_mapping = pd.concat([rel_mapping, rev_rmap], ignore_index=True)

        # resort id in descending order of frequency, since we use log-uniform sampler for NCE loss
        def remap_kb(kb):
            es = pd.concat([kb.h, kb.t], ignore_index=True)
            rs = kb.r
            e_num = es.groupby(es.values).size().sort_values()[::-1]
            r_num = rs.groupby(rs.values).size().sort_values()[::-1]

            e_map = pd.Series(range(e_num.shape[0]), index=e_num.index)
            r_map = pd.Series(range(r_num.shape[0]), index=r_num.index)
            #             print("r_map")
            #             print(r_map)
            return e_map, r_map

        #         print("df")
        #         print(df)

        def index(df, e_map, r_map, is_kb=True):
            if is_kb:
                #                 print("e_map")
                #                 print(e_map)
                #                 print("r_map")
                #                 print(r_map)
                #                 print("df.h.values")
                #                 print(df.h.values)
                df['h_id'] = e_map.loc[df.h.values].values

                df['r_id'] = r_map.loc[df.r.values].values
                df['t_id'] = e_map.loc[df.t.values].values
            else:
                df['kb_1'] = e_map.loc[df.kb_1.values].values
                df['kb_2'] = e_map.loc[df.kb_2.values].values

        e_map, r_map = remap_kb(kb)
        #         print("kb")
        #         print(kb)

        index(kb, e_map, r_map)

        index(ent_mapping, e_map, None, is_kb=False)
        index(ent_testing, e_map, None, is_kb=False)
        index(rel_mapping, r_map, None, is_kb=False)
        index(rel_testing, r_map, None, is_kb=False)

        index(kb1, e_map, r_map)
        index(kb2, e_map, r_map)
        eid_1 = pd.unique(pd.concat([kb1.h_id, kb1.t_id], ignore_index=True))
        eid_2 = pd.unique(pd.concat([kb2.h_id, kb2.t_id], ignore_index=True))

        # add shortcuts
        self._eid_1 = pd.Series(eid_1)
        self._eid_2 = pd.Series(eid_2)

        self._ent_num = len(e_map)
        self._rel_num = len(r_map)
        self._ent_id = e_map
        self._rel_id = r_map

        self._ent_mapping = ent_mapping
        self._rel_mapping = rel_mapping
        self._ent_testing = ent_testing
        self._rel_testing = rel_testing

        self._kb = kb  # 此处kb没有nan

        # print("kb")
        #         f1.write(str(kb))
        # we first tag the entities that have algined entities according to entity_mapping
        self.add_align_infor()
        f.write(str(kb))
        # we then connect two KGs by creating new triples involving aligned entities.
        self.add_weight()

    def add_align_infor(self):  ###造成了空值   at_id这一列有空值ah_id有空值
        kb = self._kb

        ent_mapping = self._ent_mapping
        rev_e_m = ent_mapping.rename(columns={'kb_1': 'kb_2', 'kb_2': 'kb_1'})
        rel_mapping = self._rel_mapping
        rev_r_m = rel_mapping.rename(columns={'kb_1': 'kb_2', 'kb_2': 'kb_1'})

        ent_mapping = pd.concat([ent_mapping, rev_e_m], ignore_index=True)
        rel_mapping = pd.concat([rel_mapping, rev_r_m], ignore_index=True)

        ent_mapping = pd.Series(ent_mapping.kb_2.values, index=ent_mapping.kb_1.values)
        rel_mapping = pd.Series(rel_mapping.kb_2.values, index=rel_mapping.kb_1.values)

        self._e_m = ent_mapping
        self._r_m = rel_mapping

        kb['ah_id'] = kb.h_id
        #         for i in kb['ah_id']:
        #             if i == NaN:
        #                 i==2788
        #                 print(i)
        kb['ar_id'] = kb.r_id
        kb['at_id'] = kb.t_id
        #         for i in kb['at_id']:
        #             if i == None:
        #                 i==2788
        h_mask = kb.h_id.isin(ent_mapping)
        r_mask = kb.r_id.isin(rel_mapping)
        t_mask = kb.t_id.isin(ent_mapping)
        #         f1.write(str(h_mask))
        kb['ah_id'][h_mask] = ent_mapping.loc[kb['ah_id'][h_mask].values]
        kb['ar_id'][r_mask] = rel_mapping.loc[kb['ar_id'][r_mask].values]
        kb['at_id'][t_mask] = ent_mapping.loc[kb['at_id'][t_mask].values]
        #         print("kb的类型")
        #         print(type(kb))

        kb = kb.dropna()

        #         print(kb.notnull())
        f1.write(str(kb))
        self._kb = kb

    def add_weight(self):
        kb = self._kb[['h_id', 'r_id', 't_id', 'ah_id', 'ar_id', 'at_id']]

        kb['w_h'] = 0
        kb['w_r'] = 0
        kb['w_t'] = 0

        h_mask = ~(kb.h_id == kb.ah_id)
        r_mask = ~(kb.r_id == kb.ar_id)
        t_mask = ~(kb.t_id == kb.at_id)

        kb.loc[h_mask, 'w_h'] = 1
        kb.loc[r_mask, 'w_r'] = 1
        kb.loc[t_mask, 'w_t'] = 1

        akb = kb[['ah_id', 'ar_id', 'at_id', 'w_h', 'w_r', 'w_t']]
        #         akb.astype(int)
        akb = akb.rename(columns={'ah_id': 'h_id', 'ar_id': 'r_id', 'at_id': 't_id'})

        ahkb = kb[h_mask][['ah_id', 'r_id', 't_id', 'w_h', 'w_r', 'w_t']].rename(columns={'ah_id': 'h_id'})
        arkb = kb[r_mask][['h_id', 'ar_id', 't_id', 'w_h', 'w_r', 'w_t']].rename(columns={'ar_id': 'r_id'})
        atkb = kb[t_mask][['h_id', 'r_id', 'at_id', 'w_h', 'w_r', 'w_t']].rename(columns={'at_id': 't_id'})
        ahrkb = kb[h_mask & r_mask][['ah_id', 'ar_id', 't_id', 'w_h', 'w_r', 'w_t']].rename(
            columns={'ah_id': 'h_id', 'ar_id': 'r_id'})
        ahtkb = kb[h_mask & t_mask][['ah_id', 'r_id', 'at_id', 'w_h', 'w_r', 'w_t']].rename(
            columns={'ah_id': 'h_id', 'at_id': 't_id'})
        artkb = kb[r_mask & t_mask][['h_id', 'ar_id', 'at_id', 'w_h', 'w_r', 'w_t']].rename(
            columns={'ar_id': 'r_id', 'at_id': 't_id'})
        ahrtkb = kb[h_mask & r_mask & t_mask][['ah_id', 'ar_id', 'at_id', 'w_h', 'w_r', 'w_t']].rename(
            columns={'ah_id': 'h_id',
                     'ar_id': 'r_id',
                     'at_id': 't_id'})

        #         kb[['h_id','r_id','t_id', 'w_h', 'w_r', 'w_t']].astype(int)
        kb['w_h'] = 0
        kb['w_r'] = 0
        kb['w_t'] = 0

        kb = pd.concat(
            [akb, ahkb, arkb, atkb, ahrkb, ahtkb, artkb, ahrtkb, kb[['h_id', 'r_id', 't_id', 'w_h', 'w_r', 'w_t']]],
            ignore_index=True).drop_duplicates()  # hrt是小数需要改为整数
        kb.astype('int')
        #         print("kb的类型")
        #         print(type(kb))
        #         print("kb的hid的类型")
        kb = kb.astype(int)
        #         print(kb.astype(int))
        #         print(kb)
        self._kb = kb.reset_index(drop=True)


# sampler


class BasicSampler(object):

    def sample_paths(self, repeat_times=2):
        opts = self._options
        pd.set_option('display.max_rows', None)
        kb = self._kb.copy()
        #         f.write(str(kb))
        #         f.close()
        #         print("kb")
        #         print()
        kb = kb[['h_id', 'r_id', 't_id']]
        #         print("kb")
        #         print(kb)
        # sampling paths in the h_id-(r_id,t_id) form.

        rtlist = np.unique(kb[['r_id', 't_id']].values, axis=0)  # 存在空值
        #         print("rtlist")
        #         print(rtlist)
        rtdf = pd.DataFrame(rtlist, columns=['r_id', 't_id'])
        # assign tail=(r_id, t_id), we assign an id for each tail
        rtdf = rtdf.reset_index().rename({'index': 'tail_id'}, axis='columns')
        #         print("rtdf2")
        #         print(rtdf)
        # merge kb with rtdf, to get the (h_id, tail_id) dataframe
        rtkb = kb.merge(
            rtdf, left_on=['r_id', 't_id'], right_on=['r_id', 't_id'])
        htail = np.unique(rtkb[['h_id', 'tail_id']].values, axis=0)
        #         print("htail")
        #         print(htail)
        # save to the sparse matrix
        htailmat = csr_matrix((np.ones(len(htail)), (htail[:, 0], htail[:, 1])),
                              shape=(model._ent_num, rtlist.shape[0]))

        # calulate corss-KG bias at first, note that we use an approximate method:
        # if next entity e_{i+1} is in entity_mapping, e_i and e_{i+2} entity are believed in different KGs
        em = pd.concat(
            [model._ent_mapping.kb_1, model._ent_mapping.kb_2]).values

        rtkb['across'] = rtkb.t_id.isin(em)
        rtkb.loc[rtkb.across, 'across'] = opts.beta
        rtkb.loc[rtkb.across == 0, 'across'] = 1 - opts.beta

        rtailkb = rtkb[['h_id', 't_id', 'tail_id', 'across']]

        def gen_tail_dict(x):
            return x.tail_id.values, x.across.values / x.across.sum()

        # each item in rtailkb is in the form of (tail_ids, cross-KG biases)
        rtailkb = rtailkb.groupby('h_id').apply(gen_tail_dict)

        rtailkb = pd.DataFrame({'tails': rtailkb})

        # start sampling

        hrt = np.repeat(kb.values, repeat_times, axis=0)

        # for initial triples
        def perform_random(x):
            return np.random.choice(x.tails[0], 1, p=x.tails[1].astype(np.float))

        # else
        def perform_random2(x):
            # calculate depth bias
            pre_c = htailmat[np.repeat(x.pre, x.tails[0].shape[0]), x.tails[0]]

            pre_c[pre_c == 0] = opts.alpha
            pre_c[pre_c == 1] = 1 - opts.alpha

            # combine the biases
            p = x.tails[1].astype(np.float).reshape(
                [-1, ]) * pre_c.A.reshape([-1, ])
            p = p / p.sum()
            return np.random.choice(x.tails[0], 1, p=p)

        rt_x = rtailkb.loc[hrt[:, 2]].apply(perform_random, axis=1)
        rt_x = rtlist[np.concatenate(rt_x.values)]

        rts = [hrt, rt_x]
        c_length = 5
        pre = hrt[:, 0]
        print('current path length == %i' % c_length)
        while (c_length < opts.max_length):
            curr = rtailkb.loc[rt_x[:, 1]]

            # always using hrt[:, 0] as the previous entity is a stronger way to
            # generate deeper and cross-KG paths for the starting point.
            # use 'curr.loc[:, 'pre'] = pre' for 2nd-order sampling.
            curr.loc[:, 'pre'] = hrt[:, 0]

            rt_x = curr.apply(perform_random2, axis=1)
            rt_x = rtlist[np.concatenate(rt_x.values)]

            rts.append(rt_x)
            c_length += 2
            # pre = curr.index.values
            print('current path length == %i' % c_length)

        data = np.concatenate(rts, axis=1)
        data = pd.DataFrame(data)

        self._train_data = data
        data.to_csv('%spaths_%.1f_%.1f' % (opts.data_path, opts.alpha, opts.beta))


# model
class RSN4EA(BasicReader, BasicSampler):
    def __init__(self, options, session):
        self._options = options
        self._session = session

    def create_variables(self):
        options = self._options
        hidden_size = options.hidden_size

        self._entity_embedding = tf.get_variable(
            'entity_embedding',
            [self._ent_num, hidden_size],
            initializer=tf.contrib.layers.xavier_initializer(uniform=False)
        )
        self._relation_embedding = tf.get_variable(
            'relation_embedding',
            [self._rel_num, hidden_size],
            initializer=tf.contrib.layers.xavier_initializer(uniform=False)
        )

        self._rel_w = tf.get_variable(
            "relation_softmax_w",
            [self._rel_num, hidden_size],
            initializer=tf.contrib.layers.xavier_initializer(uniform=False)
        )
        self._rel_b = tf.get_variable(
            "relation_softmax_b",
            [self._rel_num],
            initializer=tf.constant_initializer(0)
        )
        self._ent_w = tf.get_variable(
            "entity_softmax_w",
            [self._ent_num, hidden_size],
            initializer=tf.contrib.layers.xavier_initializer(uniform=False)
        )
        self._ent_b = tf.get_variable(
            "entity_softmax_b",
            [self._ent_num],
            initializer=tf.constant_initializer(0)
        )
        self._lr = tf.Variable(options.learning_rate, trainable=False)

        self._optimizer = tf.train.AdamOptimizer(options.learning_rate)

    def bn(self, inputs, is_train=True, reuse=True):
        return tf.contrib.layers.batch_norm(inputs,
                                            center=True,
                                            scale=True,
                                            is_training=is_train,
                                            reuse=reuse,
                                            scope='bn',
                                            )

    def lstm_cell(self, drop=True, keep_prob=0.5, num_layers=2, hidden_size=None):
        if not hidden_size:
            hidden_size = self._options.hidden_size

        def basic_lstm_cell():
            return tf.contrib.rnn.GRUCell(
                num_units=hidden_size,
                kernel_initializer=tf.orthogonal_initializer,
                # forget_bias=1,
                reuse=tf.get_variable_scope().reuse,
                activation=tf.identity
            )

        def drop_cell():
            return tf.contrib.rnn.DropoutWrapper(
                basic_lstm_cell(),
                output_keep_prob=keep_prob
            )

        if drop:
            gen_cell = drop_cell
        else:
            gen_cell = basic_lstm_cell

        if num_layers == 0:
            return gen_cell()

        cell = tf.contrib.rnn.MultiRNNCell(
            [gen_cell() for _ in range(num_layers)],
            state_is_tuple=True,
        )
        return cell

    def sampled_loss(self, inputs, labels, w, b, weight=1, is_entity=False):
        num_sampled = min(self._options.num_samples, w.shape[0] // 3)

        labels = tf.reshape(labels, [-1, 1])

        losses = tf.nn.nce_loss(
            weights=w,
            biases=b,
            labels=labels,
            inputs=tf.reshape(inputs, [-1, int(w.shape[1])]),
            num_sampled=num_sampled,
            num_classes=w.shape[0],
            partition_strategy='div',
        )
        return losses * weight

    def logits(self, inputs, w, b):
        return tf.nn.bias_add(tf.matmul(inputs, tf.transpose(w)), b)

    # shuffle data
    def sample(self, data):
        choices = np.random.choice(len(data), size=len(data), replace=False)
        return data.iloc[choices]


# build tensorflow graph


# build an RSN of length l
def build_sub_graph(self, length=15, reuse=False):
    options = self._options
    hidden_size = options.hidden_size
    batch_size = options.batch_size

    seq = tf.placeholder(
        tf.int32, [batch_size, length], name='seq' + str(length))

    e_em, r_em = self._entity_embedding, self._relation_embedding

    # seperately read, and then recover the order
    ent = seq[:, :-1:2]
    rel = seq[:, 1::2]

    ent_em = tf.nn.embedding_lookup(e_em, ent)
    rel_em = tf.nn.embedding_lookup(r_em, rel)

    em_seq = []
    for i in range(length - 1):
        if i % 2 == 0:
            em_seq.append(ent_em[:, i // 2])
        else:
            em_seq.append(rel_em[:, i // 2])

    # 合作机制
    with tf.variable_scope('input_bn'):
        if not reuse:
            bn_em_seq = [tf.reshape(self.bn(em_seq[i], reuse=(
                    i is not 0)), [-1, 1, hidden_size]) for i in range(length - 1)]
        else:
            bn_em_seq = [tf.reshape(
                self.bn(em_seq[i], reuse=True), [-1, 1, hidden_size]) for i in range(length - 1)]

    bn_em_seq = tf.concat(bn_em_seq, axis=1)

    ent_bn_em = bn_em_seq[:, ::2]

    with tf.variable_scope('GRU', reuse=reuse):

        cell = self.lstm_cell(True, options.keep_prob, options.num_layers)

        outputs, state = tf.nn.dynamic_rnn(cell, bn_em_seq, dtype=tf.float32)

    rel_outputs = outputs[:, 1::2, :] + outputs[:, 1::2, :]
    outputs = [outputs[:, i, :] for i in range(length - 1)] + [outputs[:, i, :] for i in range(length - 1)]

    ent_outputs = outputs[::2] + outputs[::2]

    # GCAN
    res_rel_outputs = tf.contrib.layers.fully_connected(rel_outputs, hidden_size, biases_initializer=None,
                                                        activation_fn=None) + \
                      tf.contrib.layers.fully_connected(
                          ent_bn_em, hidden_size, biases_initializer=None, activation_fn=None)

    # recover the order
    res_rel_outputs = [res_rel_outputs[:, i, :] for i in range((length - 1) // 2)]
    outputs = []
    for i in range(length - 1):
        if i % 2 == 0:
            outputs.append(ent_outputs[i // 2])
        else:
            outputs.append(res_rel_outputs[i // 2])

    # output bn
    with tf.variable_scope('output_bn'):
        if reuse:
            bn_outputs = [tf.reshape(
                self.bn(outputs[i], reuse=True), [-1, 1, hidden_size]) for i in range(length - 1)]
        else:
            bn_outputs = [tf.reshape(self.bn(outputs[i], reuse=(
                    i is not 0)), [-1, 1, hidden_size]) for i in range(length - 1)]

    def cal_loss(bn_outputs, seq):
        losses = []

        decay = 0.8
        for i, output in enumerate(bn_outputs):
            if i % 2 == 0:
                losses.append(self.sampled_loss(
                    output, seq[:, i + 1], self._rel_w, self._rel_b, weight=decay ** (0), is_entity=i))
            else:
                losses.append(self.sampled_loss(
                    output, seq[:, i + 1], self._ent_w, self._ent_b, weight=decay ** (0), is_entity=i))
        losses = tf.stack(losses, axis=1)
        return losses

    seq_loss = cal_loss(bn_outputs, seq)

    losses = tf.reduce_sum(seq_loss) / batch_size

    return losses, seq


# build the main graph
def build_graph(self):
    options = self._options

    loss, seq = build_sub_graph(self, length=options.max_length, reuse=False)

    tvars = tf.trainable_variables()
    grads, _ = tf.clip_by_global_norm(tf.gradients(loss, tvars), 2.0)
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        train_op = self._optimizer.apply_gradients(
            zip(grads, tvars),
            global_step=tf.train.get_or_create_global_step()
        )

    self._seq, self._loss, self._train_op = seq, loss, train_op


# training procedure
def seq_train(self, data, choices=None, epoch=None):
    opts = self._options

    # shuffle data
    choices = np.random.choice(len(data), size=len(data), replace=True)
    batch_size = opts.batch_size

    num_batch = len(data) // batch_size

    fetches = {
        'loss': self._loss,
        'train_op': self._train_op
    }

    losses = 0
    for i in range(num_batch):
        one_batch_choices = choices[i * batch_size: (i + 1) * batch_size]
        one_batch_data = data.iloc[one_batch_choices]

        feed_dict = {}
        seq = one_batch_data.values[:, :opts.max_length]
        feed_dict[self._seq] = seq

        vals = self._session.run(fetches, feed_dict)

        del one_batch_data

        loss = vals['loss']
        losses += loss
        print('\r%i/%i, batch_loss:%f' % (i, num_batch, loss), end='')
    self._last_mean_loss = losses / num_batch

    return self._last_mean_loss


# eval_graph & eval method


def build_eval_graph(self, entity=True):
    options = self._options
    hidden_size = options.hidden_size
    batch_size = 2048

    e_em, r_em = self._entity_embedding, self._relation_embedding

    def em_lookup(indices, em):
        return tf.nn.embedding_lookup(em, indices)

    h, r = tf.placeholder(tf.int32, [None]), tf.placeholder(tf.int32, [None])

    he, re = em_lookup(h, e_em), em_lookup(r, r_em)

    he = tf.nn.l2_normalize(he, dim=-1)
    norm_e_em = tf.nn.l2_normalize(e_em, dim=-1)

    re = tf.nn.l2_normalize(re, dim=-1)
    norm_r_em = tf.nn.l2_normalize(r_em, dim=-1)

    aep = tf.matmul(he, tf.transpose(norm_e_em))
    arp = tf.matmul(re, tf.transpose(norm_r_em))

    if entity:
        return h, aep
    else:
        return r, arp


def eval_entity_align(model, data, kb_1to2=False):
    options = model._options
    batch_size = 16

    data, padding_num = padding_data(data, options, batch_size)

    h, aep = build_eval_graph(model)

    fetch = {'probs': aep, }

    num_batch = len(data) // batch_size

    probs = []
    for i in range(num_batch):
        one_batch_data = data.iloc[i * batch_size:(i + 1) * batch_size]

        feed_dict = {}
        if kb_1to2:
            feed_dict[h] = one_batch_data.kb_1.values
        else:
            feed_dict[h] = one_batch_data.kb_2.values

        vals = sess.run(fetch, feed_dict)
        probs.append(vals['probs'])

    probs = np.concatenate(probs)[:len(data) - padding_num]
    return probs


# some tools

def cal_ranks(probs, method, label):
    # in most cases, using method=='min' is safe and much faster than method=='average' or 'max'.
    # but note that, it will overestimate the results if the correct one has a same probability with others.
    #     print("probs的数据类型1")
    #     print(type(probs))
    #     print(probs)
    #     print("len(label)")
    #     print(type(probs))
    #     print(len(label))
    #     print("label")
    #     print(label)
    #     print("label的数据类型")
    #     print(type(label))
    label = label.dropna()
    label = label.astype(int)

    if method == 'min':
        probs = probs - probs[range(len(label)), label].reshape(len(probs), 1)
        ranks = (probs > 0).sum(axis=1) + 1
    else:
        ranks = pd.DataFrame(probs).rank(axis=1, ascending=False, method=method)
        ranks = ranks.values[range(len(label)), label]
    return ranks


# top-10 = hits@10
def cal_performance(ranks, top=10):
    m_r = sum(ranks) * 1.0 / len(ranks)
    h_10 = sum(ranks <= top) * 1.0 / len(ranks)
    mrr = (1. / ranks).sum() / len(ranks)
    return m_r, h_10, mrr


def padding_data(data, options, batch_size):
    padding_num = batch_size - len(data) % batch_size
    data = pd.concat([data, pd.DataFrame(np.zeros((padding_num, data.shape[1])), dtype=np.int32, columns=data.columns)],
                     ignore_index=True, axis=0)
    return data, padding_num


def in2d(arr1, arr2):
    """Generalisation of numpy.in1d to 2D arrays"""

    assert arr1.dtype == arr2.dtype

    arr1_view = np.ascontiguousarray(arr1).view(np.dtype((np.void,
                                                          arr1.dtype.itemsize * arr1.shape[1])))
    arr2_view = np.ascontiguousarray(arr2).view(np.dtype((np.void,
                                                          arr2.dtype.itemsize * arr2.shape[1])))
    intersected = np.in1d(arr1_view, arr2_view)
    return intersected.view(np.bool).reshape(-1)


# handle evaluation
def handle_evaluation(i=0, last_mean_loss=0, kb_1to2=True, method='min', valid=True):
    data_size = len(model._ent_testing)
    # we use 10% testing data for validation
    if valid:
        data = model._ent_testing.iloc[:data_size // 10]
    else:
        data = model._ent_testing.iloc[data_size // 10:]

    probs = eval_entity_align(model, data, kb_1to2=kb_1to2)
    #     print("probs的数据类型2")
    #     print(type(probs))
    #     print(probs)
    candi = model._ent_testing.kb_2 if kb_1to2 == True else model._ent_testing.kb_1
    mask = np.in1d(np.arange(probs.shape[1]), candi)
    # exclude known entities
    probs[:, ~mask] = probs.min() - 1

    label = data.kb_2 if kb_1to2 == True else data.kb_1
    ranks = cal_ranks(probs, method=method,
                      label=label)

    MR, H10, MRR = cal_performance(ranks, top=10)
    _, H1, _ = cal_performance(ranks, top=1)
    H1, MR, H10, MRR

    msg = 'epoch:%i, Hits@1:%.3f, Hits@10:%.3f, MR:%.3f, MRR:%.3f, mean_loss:%.3f' % (
        i, H1, H10, MR, MRR, last_mean_loss)
    print('\n' + msg)
    return msg, (i, H1, H10, MR, MRR, last_mean_loss)


def write_to_log(path, content):
    with open(path, 'a+', encoding='utf-8') as f:
        print(content, file=f)


class Options(object):
    pass


# set options
opts = Options()
opts.hidden_size = 256
opts.num_layers = 2
opts.batch_size = 512  # 512
opts.learning_rate = 0.001  # 0.003
opts.num_samples = 2048 * 4
opts.keep_prob = 0.5

opts.max_length = 15
opts.alpha = 0.9
opts.beta = 0.9

opts.data_path = data_path

opts.log_file_path = 'log/%s%dl_%s.log' % (opts.data_path.replace(
    '/', '-'), opts.max_length, datetime.now().strftime('%y-%m-%d-%H-%M'))

# and tensorflow config
config = tf.ConfigProto()
config.gpu_options.allow_growth = True

tf.reset_default_graph()

# initial model

sess = tf.InteractiveSession(config=config)

model = RSN4EA(options=opts, session=sess)

model.read(data_path=model._options.data_path)
model.create_variables()

sequence_datapath = '%spaths_%.1f_%.1f' % (
    model._options.data_path, model._options.alpha, model._options.beta)

if not os.path.exists(sequence_datapath):
    print('start to sample paths')
    model.sample_paths()
    train_data = model._train_data
else:
    print('load existing training sequences')
    train_data = pd.read_csv(sequence_datapath, index_col=0)

# build tensorflow graph and init all tensors
build_graph(model)
tf.global_variables_initializer().run()

# initial training settings

write_to_log(opts.log_file_path, opts.__dict__)
max_hits1, times, max_times = 0, 0, 3
epoch = 0

msg, r = handle_evaluation(0, 10000, valid=True)
write_to_log(opts.log_file_path, msg)

for i in range(epoch, 30):
    last_mean_loss = seq_train(model, train_data)
    epoch += 1

    # evaluation
    msg, r = handle_evaluation(i + 1, last_mean_loss, valid=True)
    write_to_log(opts.log_file_path, msg)

    # early stop
    hits1 = r[1]
    if hits1 > max_hits1:
        max_hits1 = hits1
        times = 0
    else:
        times += 1

    if times >= max_times:
        break

model_path = "../save/model.ckpt"
saver = tf.train.Saver()
print('----saving model----')
save_path = saver.save(sess, model_path, global_step=i)
print('----saved----')

# evaluation on testing data
print('final results:')
msg, r = handle_evaluation(i + 1, last_mean_loss, valid=False, method='average')
write_to_log(opts.log_file_path, msg)
