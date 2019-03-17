import numpy as np
from wave import open as open_wave
import math
import warnings
import datetime
import copy
import tensorflow as tf
import matplotlib.pyplot as plt
import midi
import os


# how many time points does the AI inspect and predict?
N_STEPS = 64

# a single channel, means vector.  linear rhythms.
# extending to chords or beats with multiple drums playing at the same time is easy,
# but will be too high-dimensional for plotting
N_INPUTS = 1 



# re-drum midi notes are in an interval slightly greater than between 37 and 41.

def midi_to_array(filename,n_quanta = 64, quanta_per_qn = 4, pitch_offset= 34):
    pattern = midi.read_midifile(filename)
    note_vector = np.zeros(n_quanta)#pattern.resolution*n_quarter_notes )
    cum_ticks = 0
    ticks_per_quanta = pattern.resolution/quanta_per_qn  # ticks per quarter note * quarter note per quanta
    for event in pattern[-1]:
        cum_ticks += event.tick
        if type(event) == midi.events.NoteOnEvent:
            quanta_index = int(cum_ticks/ticks_per_quanta)
            note_vector[quanta_index] = event.pitch - pitch_offset
    return(note_vector)


def array_to_midi(output_filename, note_array):
    ticks_per_quanta = 3840 # generalize this later
    pattern = midi.Pattern()
    track = midi.Track()
    pattern.append(track)
    pattern.resolution = 15360
    collect_zeros = 0

    for onset in range(0,len(note_array)):
        print(collect_zeros)
        current_pitch = int(note_array[onset])
        if note_array[onset] > 0: # if note not rest
            track.append(midi.NoteOnEvent(tick=int(ticks_per_quanta*collect_zeros), velocity=100, pitch=current_pitch))
            track.append(midi.NoteOffEvent(tick=ticks_per_quanta,  pitch=current_pitch))
            collect_zeros = 0
        else:
            collect_zeros += 1  # increment the count of consecutive rests.

    # TO DO:  make sure total ticks adds up to ticks per quanta times n_quanta

    # Add the end of track event, append it to the track
    eot = midi.EndOfTrackEvent(tick=1)
    track.append(eot)
    # Print out the pattern
    print pattern
    # Save the pattern to disk
    midi.write_midifile(output_filename, pattern)



def array_to_midi_old(output_filename, note_array):
    ticks_per_quanta = 3840 # generalize this later
    pattern = midi.Pattern()
    track = midi.Track()
    pattern.append(track)
    pattern.resolution = 15360
    last_pitch = 69
    collect_zeros = 0
    for onset in range(0,len(note_array)):
        print(collect_zeros)
        if note_array[onset] > 0:
            if collect_zeros != 0:
               # note off
               track.append(midi.NoteOffEvent(tick=int(ticks_per_quanta*collect_zeros), pitch=last_pitch))
            last_pitch = int(note_array[onset])
            track.append(midi.NoteOnEvent(tick=ticks_per_quanta, velocity=100, pitch=last_pitch))
        else:
            collect_zeros += 1


    # Add the end of track event, append it to the track
    eot = midi.EndOfTrackEvent(tick=1)
    track.append(eot)
    # Print out the pattern
    print pattern
    # Save the pattern to disk
    midi.write_midifile(output_filename, pattern)






def load_train_data_from_3_sketches():
    # load my recorded piano rhythms

    module_dir = os.path.dirname(os.path.realpath(__file__))
    print(module_dir)

    train_files_dir = module_dir + '/training_midis/2018_linear_rhythms/'
    train_input1 = midi_to_array(train_files_dir + "midi_in_01.mid")
    train_target1 = midi_to_array(train_files_dir + "midi_target_01.mid")

    train_input2 = midi_to_array(train_files_dir + "midi_in_02.mid")
    train_target2 = midi_to_array(train_files_dir + "midi_target_02.mid")

    train_input3 = midi_to_array(train_files_dir + "midi_in_03.mid")
    train_target3 = midi_to_array(train_files_dir + "midi_target_03.mid")
    
    train_input_array = np.append(train_input1, train_input2)
    train_input_array = np.append(train_input_array, train_input3)

    # make master train target.    
    train_target_array = np.append(train_target1, train_target2)
    train_target_array = np.append(train_target_array, train_target3)

    return train_input_array, train_target_array

def data_aug_sparsify(tr_input, tr_target, sparse_factor = 4):

    # data augmentation
    # randomly delete onsets to make rhythms more sparse
    # random boolean mask for which values will be changed
    mask = np.random.randint(0, 2, size=len(tr_input)).astype(np.bool)
    mask1 = np.random.randint(0, 2, size=len(tr_input)).astype(np.bool)
    mask2 = np.random.randint(0, 2, size=len(tr_input)).astype(np.bool)
    mask3 = np.random.randint(0, 2, size=len(tr_input)).astype(np.bool)
    r_zeros = np.zeros(len(tr_input))
    augmented_input = tr_input.copy()
    augmented_target = tr_target.copy()
    # make sparse
    if sparse_factor > 0:
        augmented_input[mask] = r_zeros[mask]
        augmented_target[mask] = r_zeros[mask]
        if sparse_factor > 1:
            augmented_input[mask1] = r_zeros[mask1]
            augmented_target[mask1] = r_zeros[mask1]
            if sparse_factor > 2:
                augmented_input[mask2] = r_zeros[mask2]
                augmented_target[mask2] = r_zeros[mask2]
                if sparse_factor > 3:
                    augmented_input[mask3] = r_zeros[mask3]
                    augmented_target[mask3] = r_zeros[mask3]

    return augmented_input, augmented_target


def append_aug(input, target, sparseness = 2):
    aug_i, aug_t = data_aug_sparsify(input, target, sparse_factor=sparseness)
    new_input = np.append(input, aug_i )
    new_target = np.append(target, aug_t )
    return(new_input, new_target)


# make training input and training target
train_input, train_target = load_train_data_from_3_sketches()
a_i, a_t = append_aug(train_input, train_target)

def plot_rhythm_basic(input, target_or_output):
    # inspect input and target/output rhythms
    plt.plot(input)
    plt.plot(target_or_output)
    plt.show()

# hide plot for now
#plot_rhythm_basic(train_input, train_target)

t_min, t_max = 0, len(a_i)

# Set up subsampling input and train_target sequences
def next_batch(batch_size, n_steps):
    '''
    This function randomly selects sub-sequences of number of contiguous steps equal to  n_steps
    The number of sub-sequences selected per batch is the batch_size.
    :param batch_size:
    :param n_steps:
    :return:
    '''
    t0s = np.random.randint(0, t_max - n_steps, batch_size)
    t0s = np.expand_dims(t0s, axis=1)
    instance_indices = t0s + np.arange(0, n_steps )
    # return two values:  one is set of input sub-timeseries,
    # the other is corresponding set of train_target sub-timeseries!
    # note that the -1 of reshape means "figure it out, numpy"
    ys = a_i[instance_indices].reshape(-1, n_steps, 1), a_t[instance_indices].reshape(-1, n_steps, 1)
    # return a tuple called ys
    return(ys)


def batch_visualize():
    ## let's make sure batches are working!!!
    asdf = next_batch(batch_size = 8,n_steps = 40)
    plt.plot(asdf[0][0])
    plt.plot(asdf[1][0])
    plt.show()
    # fantastic!

### TENSORFLOW SECTION


def reset_graph(seed=42):
    # useful for reproducibility and for managing tensorflow's computational graphs.
    tf.reset_default_graph()
    tf.set_random_seed(seed)
    np.random.seed(seed)

def add_tail(wave_samples, n_steps):
    # add the beginning of the waveform to the end so that the waveform length is
    # divisible by n_inputs.  Makes the waveform briefly periodic.
    _, remainder = divmod(len(wave_samples), n_steps)
    tail_length = n_steps - remainder
    # make the tail
    tail = wave_samples[:tail_length]
    series_plus_tail = np.append(wave_samples, tail)
    return series_plus_tail, tail_length




def compute_test_mse(y_predicted,  y_true):
    '''
    This calculates the mean of squared errors.
    :param test_output: this is output from the neural net
    :param ground_truth:  this is what the true effect sounds like when applied to the input.
    :return:
    '''
    return(np.sum(np.square(y_true - y_predicted))/len(y_true))

def hyperparams_string(n_steps, n_neurons, depth, learning_rate, optimizer, n_iterations, batch_size):
    model_details_string = ('n_time_steps, ' + str(n_steps) + '\nn_neurons, ' + str(n_neurons) +
        '\ndepth, ' + str(depth) + '\nlearning_rate, ' + str(learning_rate) + '\noptimizer, ' + str(optimizer) +
        '\nn_iterations, ' + str(n_iterations) + '\nbatch_size, ' + str(batch_size))
    date_string = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M")
    filename_string = date_string + '_' + str(n_steps) + '-timesteps_' + str(n_neurons) + '-neurons_' + str(depth) + '-layers' + '.csv'
    return(model_details_string, filename_string)


def save_model_hyperparams_and_results(model_filename, hyperparameter_string, training_mse_list, mse_per_n_steps, test_error, test_mse):
    '''
    This function writes a file containing the information from hyperparameter_string plus the test_error and test_mape
    :param model_filename:
    :param hyperparameter_string:
    :param training_mse_list:
    :param mse_per_n_steps:
    :param test_error:
    :param test_mape:
    :return:
    '''

    output_string = (hyperparameter_string + '\ntraining_error_per_' + str(mse_per_n_steps) + '_steps, ' + str(training_mse_list) +
              '\ntest_error, ' + str(test_error) + '\ntest_mse, ' + str(test_mse))

    model_details_file = open(model_filename, "w")
    model_details_file.write(output_string)
    model_details_file.close()


def plot_input_output(input, predicted_output, true_output, start_sample = 0, n_samples = 64):
    '''
    Handy plotter for looking at the INPUT from the test set,
    the generated OUTPUT (aka y hat / y_pred), and the GROUND TRUTH
    :param input:
    :param predicted_output:
    :param true_output:
    :param start_sample:
    :param n_samples:
    :return:
    '''
    plt.plot(input[start_sample:start_sample + n_samples], label='input')
    plt.plot(predicted_output[start_sample:start_sample + n_samples], label='neural network output')
    plt.plot(true_output[start_sample:start_sample + n_samples], label = 'true midi target output')
    plt.legend()
    plt.show()
    plt.savefig(model_filename.replace('.csv', ('_ts-' + str(start_sample) + '-' + str(start_sample + n_samples) + '.jpg') ) )
    plt.gcf().clear()


# TO DO: make this non-64 beat dependent!!
# TO DO: make this more than 1d. 

def clean_up_array(nn_output_array):
    # make array of floats an array of ints
    int_array = nn_output_array.T.astype(int)
    # make 1d.
    array_1d = np.squeeze(int_array.reshape(1, 64))
    return(array_1d)


def train_network():    
    reset_graph()
    
    
    ### NEURAL NETWORK HYPERPARAMETERS
    n_steps = 64 #400  # time steps!   # number of recurrent neurons
    n_inputs = 1  # number of channels.
    n_neurons = 200 #400 #200 #100 # neurons per node
    n_outputs = 1
    
    X = tf.placeholder(tf.float32, [None, n_steps, n_inputs], name='X')
    y = tf.placeholder(tf.float32, [None, n_steps, n_outputs])
    
    # this will give 100 outputs per cell
    #cell = tf.contrib.rnn.BasicRNNCell(num_units=n_neurons, activation=tf.nn.relu)
    # this will give one output per cell
    cell = tf.contrib.rnn.OutputProjectionWrapper(
        tf.contrib.rnn.BasicRNNCell(num_units=n_neurons, activation=tf.nn.relu),
        output_size=n_outputs)
    
    n_layers = 1
    
    #layers = [tf.contrib.rnn.BasicRNNCell(num_units=n_neurons)
    #          for layer in range(n_layers)]
    #
    #multi_layer_cell = tf.contrib.rnn.MultiRNNCell(layers)
    #outputs, states = tf.nn.dynamic_rnn(multi_layer_cell, X, dtype=tf.float32)
    
    
    outputs, states = tf.nn.dynamic_rnn(cell, X, dtype=tf.float32)  #grrr wish I could change the name of this variable.

    # I will want to reference the outputs for generating music.
    # Currently the outputs op is named 'rnn/transpose:0' which is uninformative
    # If you want to "rename" an op, there is no way to do that directly, because a tf.Operation (or tf.Tensor) 
    # is immutable once it has been created.   The workaround is to create an identity variable.

    outputs = tf.identity(outputs, name="outputs")
    # print('here\'s the name of the \'outputs\' variable:')
    # print(outputs.name)
    # print('here\'s the shape of the \'outputs\' variable:')
    # print(outputs.shape)
    
    
    ### TRAINING HYPERPARAMETERS
    learning_rate = 0.001
    
    report_mse_per_n_steps = 100
    
    loss = tf.reduce_mean(tf.square(outputs - y)) # MSE
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
    training_op = optimizer.minimize(loss)
    
    init = tf.global_variables_initializer()
    
    saver = tf.train.Saver()
    #n_iterations = 1500
    #n_iterations = 300
    n_iterations = 600
    
    batch_size = 100
    
    model_hyperparams_string, model_filename = hyperparams_string(n_steps, n_neurons, n_layers, learning_rate, optimizer, n_iterations, batch_size)
    
    # save the session with filename prefix.  append 'RNN-model'.
    first_session_filename = './' + model_filename.replace('.csv','-RNN_model')
    
    ### BEGIN TRAINING SESSION
    training_mse_list = []
    with tf.Session() as sess:
        init.run()
        for iteration in range(n_iterations):
            X_batch, y_batch = next_batch(batch_size, n_steps)
            sess.run(training_op, feed_dict={X: X_batch, y: y_batch})
            if iteration % report_mse_per_n_steps == 0:
                mse = loss.eval(feed_dict={X: X_batch, y: y_batch})
                training_mse_list.append(mse)
                print(iteration, "\tMSE:", mse)
    
        saver.save(sess, first_session_filename)  
    
    ### END TRAINING
    
    
### BEGIN TESTING!

def make_eg_midi_inputs(example_number):

    if example_number == 0:

        ## FIRST EXAMPLE:  "RANDOM" NOTES
        new_midi_00 = np.zeros(64)
        new_midi_00[32] = 4.
        new_midi_00[26] = 4.
        new_midi_00[18] = 4.
        new_midi_00[16] = 3.
        new_midi_00[30] = 3.
        new_midi_00[20] = 3.
        new_midi_00[12] = 3.
        new_midi_00[8] = 3.

        # save
        array_to_midi("test_midi_npgen.mid", new_midi_00)

        return(new_midi_00)

    elif example_number == 1:

        ## ANOTHER EXAMPLE:  SINGLE DRUM HIT
        new_midi_01 = np.zeros(64)
        new_midi_01[0] = 4.
    
        # save
        array_to_midi("test_midi_npgen_01.mid", new_midi_01)
    
        return new_midi_01

    elif example_number == 2:

        ##  all the same note
        new_midi_02 = np.zeros(64) + 3.
        array_to_midi("test_midi_npgen_02.mid", new_midi_02)
    
        return new_midi_02
    
    elif example_number == 3:

        ##  all the same note, different note
        new_midi_03 = np.zeros(64) + 4.
        array_to_midi("test_midi_npgen_03.mid", new_midi_03)

        return new_midi_03

    elif example_number == 4:

        ##  alternating 4s and 3s the same note
        new_midi_04 = np.zeros(64)
    
        for i in range(0, 64):
            if (i%16 == 0) | (i%16 == 1) | (i%16 == 2) | (i%16 == 3):
                new_midi_04[i] = 3.
            elif (i%16 == 8) | (i%16 == 9) | (i%16 == 10) | (i%16 == 11):
                new_midi_04[i] = 4.
        return new_midi_04

    else:

        print('number is out of bounds!')
        return




def predict_and_save(new_midifile, session_filename, output_filename, n_steps = N_STEPS, n_inputs = N_INPUTS):
    print(session_filename)

    with tf.Session() as sess:
        saver = tf.train.import_meta_graph(session_filename + '.meta')
        saver.restore(sess, session_filename)
        
        # later make a NN class and do self.n_steps
        X_new = new_midifile.reshape(-1, n_steps, n_inputs)
        #all_vars = [n.name for n in tf.get_default_graph().as_graph_def().node]
        #print(all_vars)

        outputs_var = tf.get_default_graph().get_tensor_by_name('outputs:0')
        X_var = tf.get_default_graph().get_tensor_by_name('X:0')

        y_pred = sess.run(outputs_var, feed_dict={X_var: X_new})

        ai_generated_midi = clean_up_array(y_pred)
        array_to_midi(output_filename, ai_generated_midi)
        

def concert_with_four_midis(session_filename):

    # uses 'make_eg_midi_inputs' and 'predict_and_save' to 
    # make 4 input rhythms and pass them through the AI.
    for i in range(0,5):
        # make input midi file, midi riff i
        new_midi = make_eg_midi_inputs(i)

        int_string = "%02d" % (i,) 
        ai_output_filename = "ai_gen_midi_" + int_string +  ".mid"

        # predict, produce ai-generated midi riff i
        predict_and_save(new_midi, session_filename,  ai_output_filename)
