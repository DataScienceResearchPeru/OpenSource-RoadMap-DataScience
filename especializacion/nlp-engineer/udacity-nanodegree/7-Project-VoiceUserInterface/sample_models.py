from keras import backend as K
from keras.models import Model
from keras.layers import (BatchNormalization, Conv1D, Dense, Input, 
    TimeDistributed, Activation, Bidirectional, SimpleRNN, GRU, LSTM, Dropout)

def simple_rnn_model(input_dim, output_dim=29):
    """ Build a recurrent network for speech 
    """
    # Main acoustic input
    input_data = Input(name='the_input', shape=(None, input_dim))
    # Add recurrent layer
    simp_rnn = GRU(output_dim, return_sequences=True, 
                 implementation=2, name='rnn')(input_data)
    # Add softmax activation layer
    y_pred = Activation('softmax', name='softmax')(simp_rnn)
    # Specify the model
    model = Model(inputs=input_data, outputs=y_pred)
    model.output_length = lambda x: x
    print(model.summary())
    return model

def rnn_model(input_dim, units, activation, output_dim=29):
    """ Build a recurrent network for speech 
    """
    # Main acoustic input
    input_data = Input(name='the_input', shape=(None, input_dim))
    # Add recurrent layer
    simp_rnn = GRU(units, activation=activation,
                   return_sequences=True, implementation=2, name='rnn', dropout=0.3)(input_data)
    # TODO: Add batch normalization 
    bn_rnn = BatchNormalization()(simp_rnn)
    # TODO: Add a TimeDistributed(Dense(output_dim)) layer
    time_dense = TimeDistributed(Dense(units=output_dim))(bn_rnn)
    # Add softmax activation layer
    y_pred = Activation('softmax', name='softmax')(time_dense)
    # Specify the model
    model = Model(inputs=input_data, outputs=y_pred)
    model.output_length = lambda x: x
    print(model.summary())
    return model


def cnn_rnn_model(input_dim, filters, kernel_size, conv_stride,
                  conv_border_mode, units, output_dim=29):
    """ Build a recurrent + convolutional network for speech 
    """
    # Main acoustic input
    input_data = Input(name='the_input', shape=(None, input_dim))
    # Add convolutional layer
    conv_1d = Conv1D(filters, kernel_size, 
                     strides=conv_stride, 
                     padding=conv_border_mode,
                     activation='relu',
                     name='conv1d')(input_data)
    # Add batch normalization
    bn_cnn = BatchNormalization(name='bn_conv_1d')(conv_1d)
    # Add a recurrent layer
    simp_rnn = SimpleRNN(units, activation='relu',
        return_sequences=True, implementation=2, name='rnn')(bn_cnn)
    # TODO: Add batch normalization
    bn_rnn = BatchNormalization()(simp_rnn)
    # TODO: Add a TimeDistributed(Dense(output_dim)) layer
    time_dense = TimeDistributed(Dense(units=output_dim))(bn_rnn)
    # Add softmax activation layer
    y_pred = Activation('softmax', name='softmax')(time_dense)
    # Specify the model
    model = Model(inputs=input_data, outputs=y_pred)
    model.output_length = lambda x: cnn_output_length(
        x, kernel_size, conv_border_mode, conv_stride)
    print(model.summary())
    return model

def cnn_output_length(input_length, filter_size, border_mode, stride,
                       dilation=1):
    """ Compute the length of the output sequence after 1D convolution along
        time. Note that this function is in line with the function used in
        Convolution1D class from Keras.
    Params:
        input_length (int): Length of the input sequence.
        filter_size (int): Width of the convolution kernel.
        border_mode (str): Only support `same` or `valid`.
        stride (int): Stride size used in 1D convolution.
        dilation (int)
    """
    if input_length is None:
        return None
    assert border_mode in {'same', 'valid'}
    dilated_filter_size = filter_size + (filter_size - 1) * (dilation - 1)
    if border_mode == 'same':
        output_length = input_length
    elif border_mode == 'valid':
        output_length = input_length - dilated_filter_size + 1
    return (output_length + stride - 1) // stride




def deep_rnn_model(input_dim, units, recur_layers, output_dim=29):
    """ Build a deep recurrent network for speech 
    """
    # Main acoustic input
    input_data = Input(name='the_input', shape=(None, input_dim))
    # TODO: Add recurrent layers, each with batch normalization
    bn_rnn = input_data
    for i in range(recur_layers):
        layer_name='rnn_'+str(i)
        simp_rnn = GRU(units, activation='relu',
                       return_sequences=True, implementation=2, name=layer_name)(bn_rnn)
        bn_rnn = BatchNormalization()(simp_rnn)       
    # TODO: Add a TimeDistributed(Dense(output_dim)) layer
    time_dense = TimeDistributed(Dense(units=output_dim))(bn_rnn)
    # Add softmax activation layer
    y_pred = Activation('softmax', name='softmax')(time_dense)
    # Specify the model
    model = Model(inputs=input_data, outputs=y_pred)
    model.output_length = lambda x: x
    print(model.summary())
    return model

def bidirectional_rnn_model(input_dim, units, output_dim=29):
    """ Build a bidirectional recurrent network for speech
    """
    # Main acoustic input
    input_data = Input(name='the_input', shape=(None, input_dim))
    # TODO: Add bidirectional recurrent layer
    bidir_rnn =  Bidirectional(GRU(output_dim, return_sequences=True,
                                   implementation=2, name='rnn'), 
                               merge_mode='concat')(input_data)
    # TODO: Add a TimeDistributed(Dense(output_dim)) layer
    time_dense = TimeDistributed(Dense(units=output_dim))(bidir_rnn)
    # Add softmax activation layer
    y_pred = Activation('softmax', name='softmax')(time_dense)
    # Specify the model
    model = Model(inputs=input_data, outputs=y_pred)
    model.output_length = lambda x: x
    print(model.summary())
    return model




def final_model(input_dim,  
                # CNN parameters
                filters=200, kernel_size=11, conv_stride=2, conv_border_mode='same', dilation=1,
                cnn_layers=1,
                cnn_implementation='BN-DR-AC',
                cnn_dropout=0.2,
                cnn_activation='relu',
                # RNN parameters
                reccur_units=29,
                recur_layers=2,
                recur_type='GRU',
                recur_implementation=2,
                reccur_droput=0.2,
                recurrent_dropout=0.2,
                reccur_merge_mode='concat',
                # Fully Connected layer parameters
                fc_units=[50],
                fc_dropout=0.2,
                fc_activation='relu'):
    """ Build a deep network for speech  
    """
    
    # Checks literal parameters values
    assert cnn_implementation in {'BN-DR-AC', 'AC-DR-BN'}
    assert cnn_activation in {'relu', 'selu'} 
    assert recur_type in {'GRU', 'LSTM'}
    assert reccur_merge_mode in {'sum', 'mul', 'concat', 'ave' }
    assert fc_activation in {'relu', 'selu'} 

    
    # Main acoustic input
    input_data = Input(name='the_input', shape=(None, input_dim))
    nn=input_data
    
    # Add convolutional layers
    for i in range(cnn_layers):
        layer_name='cnn_'+str(i)
        nn = Conv1D(filters,
                    kernel_size,
                    strides=conv_stride,
                    padding=conv_border_mode,
                    dilation_rate=dilation,
                    activation=None,
                    name=layer_name)(nn)

        if cnn_implementation=='BN-DR-AC':
            # Add (in order) Batch Normalization,Dropout and Activation
            nn = BatchNormalization(name='bn_'+layer_name)(nn)
            nn = Dropout(cnn_dropout, name='drop_'+layer_name)(nn)
            nn = Activation(cnn_activation, name='act_'+layer_name)(nn)
        else:
            # Add (in order) Activation,D ropout and Batch Normalization
            nn = Activation(cnn_activation, name='act_'+layer_name)(nn)
            nn = Dropout(cnn_dropout, name='drop_'+layer_name)(nn)
            nn = BatchNormalization(name='bn_'+layer_name)(nn)

    
    # TODO: Add bidirectional recurrent layers
    for i in range(recur_layers):
        layer_name='rnn_'+str(i)
        if  recur_type=='GRU':
            nn =  Bidirectional(GRU(reccur_units, return_sequences=True,
                                    implementation=recur_implementation,
                                    name=layer_name,
                                    dropout=reccur_droput,
                                    recurrent_dropout=recurrent_dropout),
                                merge_mode=reccur_merge_mode)(nn)
        else:
            nn =  Bidirectional(LSTM(reccur_units, return_sequences=True,
                                     implementation=recur_implementation,
                                     name=layer_name,
                                     dropout=reccur_droput,
                                     recurrent_dropout=recurrent_dropout),
                                merge_mode=reccur_merge_mode)(nn)
            
        nn = BatchNormalization(name='bn_'+layer_name)(nn) 
        
        
    # TODO: Add a Fully Connected layers
    fc_layers = len(fc_units)
    for i in range(fc_layers):
        layer_name='fc_'+str(i)
        nn = TimeDistributed(Dense(units=fc_units[i], name=layer_name))(nn)
        nn = Dropout(fc_dropout, name='drop_'+layer_name)(nn)
        nn = Activation(fc_activation, name='act_'+layer_name)(nn)
        
    nn = TimeDistributed(Dense(units=29, name='fc_out'))(nn)  
        
    # TODO: Add softmax activation layer
    y_pred = Activation('softmax', name='softmax')(nn)
    
    # TODO: Specify the model
    model = Model(inputs=input_data, outputs=y_pred)
    
    # TODO: Specify model.output_length: select custom or Udacity version
    model.output_length = lambda x: multi_cnn_output_length(x, kernel_size, conv_border_mode, conv_stride,
                                                            cnn_layers=cnn_layers)
    
    
    print(model.summary(line_length=110))
    return model


def multi_cnn_output_length(input_length, filter_size, border_mode, stride,
                            dilation=1, cnn_layers=1):
    """ Compute the length of the output sequence after 1D convolution along
        time. Note that this function is in line with the function used in
        Convolution1D class from Keras.
    Params:
        input_length (int): Length of the input sequence.
        filter_size (int): Width of the convolution kernel.
        border_mode (str): Only support `same` or `valid`.
        stride (int): Stride size used in 1D convolution.
        dilation (int)
    """
       
    if input_length is None:
        return None
    
    # Stacking several convolution layers only works with 'same' padding in this implementation
    if cnn_layers>1:
        assert border_mode in {'same'}
    else:
        assert border_mode in {'same', 'valid'}
    
    length = input_length
    for i in range(cnn_layers):
    
        dilated_filter_size = filter_size + (filter_size - 1) * (dilation - 1)
        if border_mode == 'same':
            output_length = length
        elif border_mode == 'valid':
            output_length = length - dilated_filter_size + 1
                
        length = (output_length + stride - 1) // stride
        
    return length
              

