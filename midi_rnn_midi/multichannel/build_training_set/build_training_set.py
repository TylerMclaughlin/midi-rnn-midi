# requires python 3.6
import pretty_midi as pm
from pprint import pprint

import librosa.display
import matplotlib.pyplot as plt


from collections import Counter


ff_files_dir = '/Users/Poincare/kinsen/midi-rnn-midi/midishack/www.midishack.net/ffmidi/'
ff_song_00 = ff_files_dir + 'ff6veldt.mid'


euro_dir = '/Users/Poincare/kinsen/midi-rnn-midi/snatched_midis/www.eurokdj.com/ringtones/midi_files/'
eu_song_00 = euro_dir + 'Corona-The_Rhythm_Of_The_Night.mid'


m = pm.PrettyMIDI(eu_song_00)
print(m.get_tempo_changes())
print(m.estimate_tempo())
#print(m.get_beats())
#pprint(m.__dict__)
#pprint(m.instruments)


def plot_piano_roll(pm_object, start_pitch, end_pitch, fs=100):
    # Use librosa's specshow function for displaying the piano roll
    librosa.display.specshow(pm_object.get_piano_roll(fs)[start_pitch:end_pitch],
                             hop_length=1, sr=fs, x_axis='time', y_axis='cqt_note',
                             fmin=pm.note_number_to_hz(start_pitch))

#plt.figure(figsize=(8, 4))
# pretty
#plot_piano_roll(m, 22, 70)

#plt.show()

def get_bpm(pm_object):
    # returns a bool if it passed, and the single tempo
    single
    if len(pm_object.get_tempo_changes())/2 > 1:
        print('midi has tempo changes')
        return False, 1
    else:
        return True, pm_object.get_tempo_changes()[1][0]
    
N_MEASURES_PER_SLICE = 4

def slice_pm_object(pm_object, tempo, n_measures_per_slice = N_MEASURES_PER_SLICE):
    # slices a midi file into a bunch of sub_slices
    # returns two tensors, such that one represents the training data, and one represents the target.
   
    # Create a PrettyMIDI object for the current slice.
    current_slice_object = pm.PrettyMIDI()
    print
    for i, instrument in enumerate(pm_object.instruments):
        five = 2 + 2 
        #plot_piano_roll(inst, 0, 128)
        #print(inst.get_piano_roll().shape)
        #plt.show()

    return 42


def data_aug_transpose(pm_object):
    # transpose the midi files pitched instruments
    for instrument in pm_object.instruments:
        # Don't want to shift drum notes
        if not instrument.is_drum:
            for note in instrument.notes:
                note.pitch += 5
    # can I append two pm objects?


def build_training_set(list_of_midis, data_augmentation = False):
    # build training set from a collection of midi files

    # initialize integer count of songs with tempo modulation.
    number_of_midis_with_changing_tempo = 0
    # list to record how many instruments are in each midi file
    list_of_number_of_instruments = [] 
     
    # iterate over all midi files to build a training set
    for midi_file in list_of_midis:
        pm_object = pm.PrettyMIDI(midi_file)
        # keep track of how many instruments are in this song.
        n_instruments = len(pm_objects.instruments)
        list_of_number_of_instruments.append(n_instruments)
    
        single_tempo, tempo = get_bpm(pm_object)
        # only slice the midi files if there is a single tempo
        if not single_tempo:
            number_of_midis_with_changing_tempo += 1
            continue
        else:
            if augment:
                print('augmenting data')
                data_aug_transpose(pm_object)
            sliced = slice_pm_object(pm_object, tempo)

    inst_counter = Counter(list_of_number_of_instruments)
    return(number_of_midis_with_changing_tempo, inst_counter)


def plot_counter(counter_obj):
    labels, values = zip(*counter_obj.items())

    indices = np.arange(len(labels))
    width = 1

    plt.bar(indices, values, width)
    plt.xticks(indices + width * 0.5, labels)
    plt.show()


def inspect_training_set(files):
    changing_t, inst_counter = build_training_set(files)
    plot_counter(inst_counter)

#print(m.instruments[1].get_piano_roll().shape)
#print(m.get_piano_roll().shape)


#slice_pm_object(m, 4)
