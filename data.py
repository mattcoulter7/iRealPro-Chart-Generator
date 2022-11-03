import re
from pychord import Chord, QualityManager
from pychord.constants import NOTE_VAL_DICT
from pychord.utils import note_to_val,val_to_note
from pyRealParser import Tune
import numpy as np
import os
import pickle
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from progress.bar import Bar

# https://www.irealpro.com/ireal-pro-file-format/

class iRealDataProcessor():
    def __init__(self,file_name,lag,target_key,test_proportion):
        self.register_qualities()
        self.pickle_file = f'{file_name}-{lag}-{target_key}-{test_proportion}.data'
        
        if self.load_cache() is None:
            self.file_name = file_name
            self.lag = lag
            self.target_key = target_key
            self.test_proportion = test_proportion
            self.data = []

            self.tunes = self.get_tunes()
            self.get_encoders()
            self.get_scalers()

    def load_cache(self):
        if os.path.exists(self.pickle_file):
            f = open(self.pickle_file, 'rb')
            obj = pickle.load(f)
            self.__dict__.update(obj.__dict__)
            f.close()
            return obj
        return None

    def cache(self):
        f = open(self.pickle_file, 'wb')
        pickle.dump(self,f)
        f.close()

    def get_tunes(self):
        file = open(self.file_name, 'r').read()
        return Tune.parse_ireal_url(file)

    def get_encoders(self):
        self.styles = np.array([t.style for t in self.tunes])
        self.styles_encoder = LabelEncoder()
        self.styles_encoded = self.styles_encoder.fit_transform(self.styles.reshape(-1,1)).reshape(-1,1)

        self.composers = np.array([t.composer for t in self.tunes])
        self.composers_encoder = LabelEncoder()
        self.composers_encoded = self.composers_encoder.fit_transform(self.composers.reshape(-1,1)).reshape(-1,1)

        self.qualities = np.array(list(QualityManager()._qualities.keys()))
        self.qualities_encoder = LabelEncoder()
        self.qualities_encoded = self.qualities_encoder.fit_transform(self.qualities.reshape(-1,1)).reshape(-1,1)

        self.notes = np.array(list(NOTE_VAL_DICT.values()))

    def get_scalers(self):
        self.styles_scalar = MinMaxScaler(feature_range=(0,1)).fit(self.styles_encoded)
        self.styles_scaled = self.styles_scalar.transform(self.styles_encoded).reshape(1,-1)[0]

        self.composers_scalar = MinMaxScaler(feature_range=(0,1)).fit(self.composers_encoded)
        self.composers_scaled = self.composers_scalar.transform(self.composers_encoded).reshape(1,-1)[0]

        self.qualities_scalar = MinMaxScaler(feature_range=(0,1)).fit(self.qualities_encoded)

        self.note_scalar = MinMaxScaler(feature_range=(0,1)).fit(self.notes.reshape(-1,1))

    def transform_value(self,raw_value,encoder,scalar):
        encoded_value = encoder.transform([raw_value]) if encoder != None else np.array([raw_value])
        normalised_value = scalar.transform(encoded_value.reshape(-1,1))
        return normalised_value[0][0]

    def inverse_transform_value(self,value,encoder,scalar):
        inverse_normalised_value = np.rint(scalar.inverse_transform([[value]])).astype(int)
        deencoded_value = encoder.inverse_transform(inverse_normalised_value.reshape(-1,1)) if encoder != None else inverse_normalised_value[0]
        return deencoded_value[0]

    def generate_training_data(self):
        if len(self.data) > 0: return self

        # map chord objects
        bar = Bar('Processing iReal Pro Charts', max=len(self.tunes))
        for (tune_index,tune) in enumerate(self.tunes):
            # get al the chords for the song
            measures = self.get_measures(tune)
            chords = np.concatenate(measures).flat
            
            # normalise chords into a common tune for assisted pattern recorgnition
            self.tranpose_chords(chords,tune.key,self.target_key)
            
            composer = self.composers_scaled[tune_index]
            style = self.styles_scaled[tune_index]

            # add chords
            for i in range(len(chords) - self.lag - 1):
                row = []
                for j in range(self.lag):
                    c = chords[i+j]
                    note_index = self.transform_value(note_to_val(c.root),None,self.note_scalar)
                    quality = self.transform_value(c.quality.quality,self.qualities_encoder,self.qualities_scalar)
                    row.append(note_index)
                    row.append(quality)
                
                output_chord = chords[i+self.lag]
                row.append(composer)
                row.append(style)
                row = row + self.raw_to_output(output_chord.root,output_chord.quality.quality)

                self.data.append(row)
            bar.next()

        # generate the training and testing data
        self.data = np.array(self.data)
        np.random.shuffle(self.data)
        test_splice_index = len(self.data) - int(len(self.data) * self.test_proportion)
        train = self.data[:test_splice_index]
        test = self.data[test_splice_index:]
        self.X_train = train[:, :-self.output_size]
        self.y_train = train[:, -self.output_size:]
        self.X_test = test[:, :-self.output_size]
        self.y_test = test[:, -self.output_size:]

        # save the data to a file
        self.cache()

        return self

    def raw_to_output(self,note,quality):
        note_output = [0 for _ in range(int(self.note_scalar.data_max_[0]) + 1)]
        note_index = note_to_val(note)
        note_output[note_index] = 1

        quality_output = [0 for _ in range(int(self.qualities_scalar.data_max_[0]) + 1)]
        quality_index = self.qualities_encoder.transform([quality])[0]
        quality_output[quality_index] = 1

        combined = note_output + quality_output
        self.output_size = len(combined)

        return combined

    def output_to_raw(self,y_pred):
        splice_point = int(self.note_scalar.data_max_[0]) + 1
        note_output = y_pred[:splice_point]
        note_index = note_output.argmax()
        note = val_to_note(note_index,scale=self.target_key)

        quality_output = y_pred[splice_point:]
        quality_index = quality_output.argmax()
        quality = self.qualities_encoder.inverse_transform([[quality_index]])[0]

        return note,quality

    # register new qualities
    def register_qualities(self):
        QualityManager().set_quality('7b9sus2', (0, 2, 7, 10, 13))
        QualityManager().set_quality('7b9sus4', (0, 5, 7, 10, 13))
        QualityManager().set_quality('7b9sus', (0, 5, 7, 10, 13))
        QualityManager().set_quality('13sus4', (0, 5, 7, 10, 14, 21))
        QualityManager().set_quality('13sus', (0, 5, 7, 10, 14, 21))
        QualityManager().set_quality('+', QualityManager().get_quality('aug').components)
        QualityManager().set_quality('m11', (0, 3, 7, 10, 14, 17))
        QualityManager().set_quality('7b59', QualityManager().get_quality('9b5').components)
        QualityManager().set_quality('mb6', (0, 3, 7, 8))
        QualityManager().set_quality('m#5', (0, 3, 8))
        QualityManager().set_quality('7sus4add3', (0, 4, 5, 7, 10))
        QualityManager().set_quality('7b13sus4', (0, 5, 7, 10, 20))

    def split_chord_string(self,chord_string):
        chord_string = chord_string.replace("r", "")
        chord_string = chord_string.replace("n", "")
        chord_string = chord_string.replace("x", "")
        chord_string = chord_string.replace("U", "")
        chord_string = chord_string.replace("W", "")
        chord_string = chord_string.replace("S", "")
        chord_string = chord_string.replace("p", "")

        split_chord_regex = re.compile("(?P<note>(^[A-G]|(?<!\/)[A-G]))")
        # ensure each chord is seperated
        chord_string = split_chord_regex.sub(r' \g<note>', chord_string)
        chords = chord_string.split(" ")
        chords = list(filter(None, chords))
        return chords

    def pyreal_to_pychord(self,chord):
        chord = chord.replace("-", "m")
        chord = chord.replace("^", "")
        chord = chord.replace("h7", "7b5")
        chord = chord.replace("h", "7b5")  # diminished
        chord = chord.replace("o7", "dim7")
        chord = chord.replace("o6", "dim6")
        chord = chord.replace("o", "dim")
        chord = chord.replace("sus", "sus4")
        chord = chord.replace("at", "")  # altered chord (not specific enough)

        # N1 First ending
        chord = chord.replace("N1", "")
        # N2 Second Ending
        chord = chord.replace("N2", "")
        # N3 Third Ending
        chord = chord.replace("N3", "")
        # N0 No text Ending
        chord = chord.replace("N0", "")
        return chord


    def get_measures(self,tune):
        measures = []
        for measure in tune.measures_as_strings:
            # split chords up into list of chords
            chord_list = self.split_chord_string(measure)
            if len(chord_list) == 0:
                continue

            # convert pyRealParser chord syntax to pychord syntax
            chord_list = [self.pyreal_to_pychord(chord) for chord in chord_list]

            # pyreal_to_pychord may return an empty string
            chord_list = list(filter(None, chord_list))

            # map chord strings to chord objects
            chords = [Chord(chord) for chord in chord_list]

            measures.append(chords)
        return measures

    def tranpose_chords(self,chords,current_key,transposed_key):
        # normalise the chords into a common key signature
        key_root_note = current_key.split('-')[0]
        key_is_minor = current_key.endswith('-')

        transposed_key_index = note_to_val(transposed_key)

        if key_is_minor:
            transposed_key_index = (transposed_key_index - 3) % 12

        current_key_index = note_to_val(key_root_note)

        offset = transposed_key_index - current_key_index

        for c in chords:
            c.transpose(offset)