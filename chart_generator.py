import warnings
import os
import numpy as np
from pyrealpro import Song, Measure, TimeSignature
from data import iRealDataProcessor
from keras.models import load_model
from pychord import Chord
from pychord.utils import note_to_val
warnings.filterwarnings("ignore")

def pychord_to_pyreal(chord):
    #TODO Make this function work for all chord cases!!
    chord = chord.replace("m", "-")
    chord = chord.replace("7b5", "h7")
    chord = chord.replace("7b5", "h")
    chord = chord.replace("dim7", "o7")
    chord = chord.replace("dim6", "o6")
    chord = chord.replace("dim", "o")
    chord = chord.replace("sus4", "sus")
    return chord

def generate_chord_seq(starting_chords,key,style,composer,chord_count = 20):
    file = os.path.join(os.path.dirname(__file__),'jazz1400.txt')
    dp = iRealDataProcessor(file,4,'C',0.3)
    model = load_model(os.path.join(os.path.dirname(__file__),'model',f'gru.h5'))
    
    chord_sequence = [Chord(c) for c in starting_chords]
    dp.tranpose_chords(chord_sequence,key,dp.target_key)

    style_transformed = dp.transform_value(style,dp.styles_encoder,dp.styles_scalar) if style is not None else 0
    composer_transformed = dp.transform_value(composer,dp.composers_encoder,dp.composers_scalar) if composer is not None else 0

    for _ in range(chord_count):
        X = []
        for c in chord_sequence[-dp.lag:]:
            note_index = dp.transform_value(note_to_val(c.root),None,dp.note_scalar)
            quality = dp.transform_value(c.quality.quality,dp.qualities_encoder,dp.qualities_scalar)
            X.append(note_index)
            X.append(quality)
        X.append(style_transformed)
        X.append(composer_transformed)

        X = np.array(X)
        X = np.reshape(X, (1,X.shape[0],1))

        y_pred = model.predict(X)[0]
        note_name,quality = dp.output_to_raw(y_pred)
        chord_name = f"{note_name}{quality}"
        chord = Chord(chord_name)
    
        chord_sequence.append(chord)
        
    dp.tranpose_chords(chord_sequence,dp.target_key,key)
    return chord_sequence

def generate_irealpro_chart(chord_seq,props):
    tune = Song(title=props.get("title"), composer="ChartGeneratorAI", key=props.get("key"), style=props.get("style"),
                 composer_name_last="M.C.",
                 composer_name_first="ChartGeneratorAI")
    
    for i,c in enumerate(chord_seq):
        tune.measures.append(
            Measure(
                chords=pychord_to_pyreal(c._chord),
                barline_open=('[' if i == 0 else None),
                barline_close=('Z' if i == len(chord_seq) - 1 else None),
            )
        )

    return tune

def main():
    props = {
        "title":"AI Generated Chart #1",
        "composer":"ChartGeneratorAI",
        "key":"C",
        "starting_chords":['C69','C#dim7','Dm7','G13b9'],
        "style":"Latin",
        "influenced_composer":None,
        "generated_chord_count":12
    }

    sequence = generate_chord_seq(
        props.get("starting_chords"),
        props.get("key"),
        props.get("style"),
        props.get("influenced_composer"),
        props.get("generated_chord_count")
    )

    chart = generate_irealpro_chart(sequence,props)

    print(chart.url())


if __name__ == '__main__':
    main()
