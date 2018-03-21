import tensorflow as tf

class RWACell(tf.nn.rnn_cell.RNNCell):
    def __init__(self, num_units):
        self.num_units = num_units

    def __call__(self, inputs, state, scope=None):
        '''
        print '#############################################'
        print 'yo'
        print '#############################################'
        print inputs.shape
        print '#############################################'
        #print state.shape
        print '#############################################'
        '''

        x = inputs
        h = state

        h = tf.reshape(h, [1, 32, 51, 51])
        x = tf.reshape(x, [1, 2, 51, 51])

        h = tf.transpose(x, [0, 2, 3, 1])
        x = tf.transpose(x, [0, 2, 3, 1])

        padded_input = tf.pad(x, [[0, 0], [3, 3], [3, 3], [0, 0]], "CONSTANT")
        net = tf.layers.conv2d(padded_input, 16, 7, activation=tf.nn.sigmoid)
        neth = tf.concat([net, h], 3)              
        padded_input = tf.pad(neth, [[0, 0], [3, 3], [3, 3], [0, 0]], "CONSTANT")
        hid1 = tf.layers.conv2d(padded_input, 32, 7, activation=tf.nn.sigmoid)
        padded_input = tf.pad(hid1, [[0, 0], [3, 3], [3, 3], [0, 0]], "CONSTANT")
        out1 = tf.layers.conv2d(padded_input, 1, 7, activation=tf.nn.sigmoid)

        '''
        print '################################################################'
        print 'boo'
        print '################################################################'
        print out1.shape
        print '################################################################'
        print hid1.shape
        print '################################################################'  
        '''  

        '''
        global dictt, coun
        print '###############################'
        print coun
        print '###############################'
        tuplee = []
        tuplee.append(h)
        tuplee.append(x)
        dictt[coun] = tuplee
        coun += 1 

        c = tf.constant([[0]])
        d = tf.constant([[0]])
        '''
        
        h = tf.reshape(hid1, [1, 51*51*32])
        x = tf.reshape(out1, [1, 51*51*1])   

        return (x, h)

    @property
    def state_size(self):
        #op = (51, 51, 32)
        #print 'output_size' 
        return (51 * 51 * 32)

    @property
    def output_size(self):
        #op = (51, 51, 32)
        #print 'output_size' 
        return (51 * 51 * 1)