""" Differentiable Pattern Producing Network (DPPN) """

import numpy as np
import tensorflow as tf

def tf_gaussian(x):
    """ Gaussian function with TensorFlow operator. """
    with tf.name_scope("gaussian"):
        return tf.exp(tf.scalar_mul(0.5, tf.negative(tf.square(x))))

# Activation functions that are available for the DPPN.
ACTIV_FUNCS = {
    "sigmoid": tf.sigmoid,
    "tanh": tf.tanh,
    "relu": tf.nn.relu,
    "sin": tf.sin,
    "abs": tf.abs,
    "gaussian": tf_gaussian,
}

def rand_af_key():
    """ Return a random activation function key. """
    return np.random.choice(list(ACTIV_FUNCS.keys()))

def rand_weight():
    """ Return a random connection weight. """
    return np.random.normal(loc=0.0, scale=0.01)

def weight_variable(weights, name):
    """ Return a tensorflow.Variable given its initial values and name. """
    w_init = np.array(weights).reshape((len(weights), 1))
    return tf.Variable(w_init, name=name, dtype=tf.float32)

class Node(object):
    def __init__(self, node_id):
        self.id = node_id
        self.conn_nodes = []
        self.w_cont = []

    def __eq__(self, other):
        return self.id == other.id

    def __ne__(self, other):
        return not self.__eq__(other)

    def __str__(self):
        str_ = "Node(%d)" % self.id
        if len(self.conn_nodes):
            str_ += " {\n"
            tmpl = "\t<---[w=%.3f]---[src=%d]\n"
            for node, weight in zip(self.conn_nodes, self.w_cont):
                str_ += tmpl % (weight, node.id)
            str_ += "}"
        return str_

    def connect_from(self, node, weight):
        """ Connect a node to this node, given a connection weight. """
        self.conn_nodes.append(node)
        self.w_cont.append(weight)

class InputNode(object):
    def __init__(self, node_id):
        self.id = node_id
        self.input = tf.placeholder(tf.float32, shape=(None, 1))

    def __eq__(self, other):
        return self.id == other.id

    def __ne__(self, other):
        return not self.__eq__(other)

    def __str__(self):
        return "InputNode(%d)" % self.id

class LinearNode(Node):
    def __init__(self, node_id):
        super().__init__(node_id)
        self._built = False

    def __str__(self):
        if self._built:
            str_ = "LinearNode(%d)" % self.id
            if len(self.conn_nodes):
                str_ += "{\n\t%s\n\t%s\n}" % ([n.id for n in self.conn_nodes], self.w)
            return str_
        return "Linear" + super().__str__()

    def is_built(self):
        return self._built

    def connect_from(self, node, weight):
        """ Connect a node to this node, given a connection weight. """
        if self._built:
            raise RuntimeError("Node must be dissolved before adding a new connection.")
        super().connect_from(node, weight)

    def build(self):
        if not self._built:
            self.w = weight_variable(self.w_cont, "w_%d" % self.id)
            delattr(self, "w_cont")
            self._built = True

    def dissolve(self, sess):
        if self._built:
            self.w_cont = self.w.eval(sess).flatten().tolist()
            delattr(self, "w")
            if hasattr(self, "a"):
                delattr(self, "a")
            self._built = False

    def activate(self):
        if not self._built:
            raise RuntimeError("Node must be built before activation.")
        with tf.name_scope("node_%d" % self.id):
            input_mat = tf.concat(values=[src.input if type(src) is InputNode else src.a
                                          for src in self.conn_nodes], axis=1)
            self.a = tf.matmul(input_mat, self.w)

class NonlinearNode(LinearNode):
    def __init__(self, node_id, af_key):
        super().__init__(node_id)
        self.af_key = af_key

    def __str__(self):
        first_line = "NonlinearNode(%d, %s)" % (self.id, self.af_key)
        lines = super().__str__().split("\n")
        lines[0] = first_line
        return "\n".join(lines)

    def build(self):
        if not self._built:
            self.activ_func = ACTIV_FUNCS[self.af_key]
            super().build()

    def dissolve(self, sess):
        if self._built:
            delattr(self, "activ_func")
            super().dissolve(sess)

    def activate(self):
        super().activate()
        self.a = self.activ_func(self.a)

class DPPN(object):
    def __init__(self, n_in, n_out):
        self.input_nodes = [InputNode(i) for i in range(n_in)]
        self.output_nodes = [LinearNode(i) for i in range(n_in, n_in + n_out)]
        for output_node in self.output_nodes:
            for input_node in self.input_nodes:
                output_node.connect_from(input_node, rand_weight())
        self.hidden_nodes = []
        self._next_id = n_in + n_out
        self._built = False

    def __str__(self):
        str_ = "DPPN(input=%d, hidden=%d, output=%d) {\n" %\
            (len(self.input_nodes), len(self.hidden_nodes), len(self.output_nodes))
        str_ += "\tInput:\n"
        for node in self.input_nodes:
            str_ += "\t\t%s\n" % str(node).replace("\n", "\n\t\t")
        str_ += "\tHidden:\n"
        for node in self.hidden_nodes:
            str_ += "\t\t%s\n" % str(node).replace("\n", "\n\t\t")
        str_ += "\tOutput:\n"
        for node in self.output_nodes:
            str_ += "\t\t%s\n" % str(node).replace("\n", "\n\t\t")
        return str_ + "}"

    def __call__(self):
        return self.forward()

    def nodes(self):
        """ Return a complete list of nodes in the DPPN. """
        return self.input_nodes + self.output_nodes + self.hidden_nodes

    def buildable_nodes(self):
        """ Return a list of buildable nodes (at least LinearNode). """
        return [node for node in self.nodes() if isinstance(node, LinearNode)]

    def build(self):
        """ Build the DPPN and all of its nodes. """
        if not self._built:
            self.sess = tf.Session()
            for node in self.buildable_nodes():
                node.build()
            self._built = True

    def dissolve(self):
        """ Dissolve the DPPN and all of its nodes. """
        if self._built:
            for node in self.buildable_nodes():
                node.dissolve(self.sess)
            self.sess.close()
            delattr(self, "sess")
            self._built = False

    def inputs(self):
        """ Return a list of tensorflow.Placeholders for all input nodes in order. """
        return [node.input for node in self.input_nodes]

    def forward(self):
        if not self._built:
            raise RuntimeError("DPPN must be built before activation.")
        with tf.name_scope("dppn"):
            for node in self.hidden_nodes + self.output_nodes:
                node.activate()
        return tf.concat([node.a for node in self.output_nodes], axis=1)

    def add_node(self):
        """ Add a new random node to the DPPN. """
        if self._built:
            raise RuntimeError("DPPN must be dissolved before adding a new node.")
        new = NonlinearNode(self._next_id, rand_af_key())
        pos = (np.random.randint(0, len(self.hidden_nodes)) if len(self.hidden_nodes) else 0)
        self.hidden_nodes.insert(pos, new)
        self._next_id += 1
        src = np.random.choice(self.input_nodes + self.hidden_nodes[:pos])
        dst = np.random.choice(self.hidden_nodes[pos + 1:] + self.output_nodes)
        new.connect_from(src, rand_weight())
        dst.connect_from(new, rand_weight())

    def add_conn(self):
        """ Add a new connection between two random nodes. """
        if self._built:
            raise RuntimeError("DPPN must be dissolved before adding a new connection.")
        valid_dsts = [(node, i + len(self.input_nodes))
                      for i, node in enumerate(self.hidden_nodes + self.output_nodes)
                      if len(node.conn_nodes) < i + len(self.input_nodes)]
        if len(valid_dsts) > 0:
            dst, pos = valid_dsts[np.random.randint(len(valid_dsts))]
            valid_srcs = [node for node in (self.input_nodes + self.hidden_nodes)[:pos]
                          if node not in dst.conn_nodes]
            if len(valid_srcs) > 0:
                src = np.random.choice(valid_srcs)
                dst.connect_from(src, rand_weight())

    def switch(self):
        """ Switch the random node's activation function to another random function. """
        if self._built:
            raise RuntimeError("DPPN must be dissolved before switching an activation function.")
        if len(self.hidden_nodes):
            node = np.random.choice(self.hidden_nodes)
            node.af_key = rand_af_key()

    def mutate(self, r_add_node, r_add_conn, r_switch):
        """ Mutate the DPPN by adding a node, adding a connection, or switching a random
        hidden node to a random activation function. """
        np.random.choice([self.add_node, self.add_conn, self.switch],
                         p=[r_add_node, r_add_conn, r_switch])()
