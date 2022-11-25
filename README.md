# iRealPro-Chart-Generator
Generating jazz music charts for the iRealPro mobile app.

## Dependencies
1. numpy
2. pandas
3. keras
4. sklearn
5. pychord
6. pyRealParser
7. pyrealpro
8. progress

## How to run
1. Open `chart_generator.py` 
2. Modify `props` at the bottom of the file, specifying:
    1. `key` The key of the chart you want to generate
    2. `starting_chords` The first four starting chords of the chart
    3. `style` The Style of the Chart or None
    4. `influenced_composer` The Composer of the chart or None
    5. `generated_chord_count` The number of generated chords, on top of the provided 4
3. Run the file, and check the console for the iRealPro url. This can be opened in Safari to load the chart into your app.

## Sample Output
irealbook://AI%20Generated%20Chart%20%231=M.C.%20ChartGeneratorAI=Latin=C=n=%5BT44C69%2C%20%2C%20%2C%20%7CDbdi-7%2C%20%2C%20%2C%20%7CD-7%2C%20%2C%20%2C%20%7CG13b9%2C%20%2C%20%2C%20%7CF6%2C%20%2C%20%2C%20%7CE-7%2C%20%2C%20%2C%20%7CA7b9%2C%20%2C%20%2C%20%7CD-7%2C%20%2C%20%2C%20%7CD-7%2C%20%2C%20%2C%20%7CF-7%2C%20%2C%20%2C%20%7CBb7%2C%20%2C%20%2C%20%7CEb7%2C%20%2C%20%2C%20%7CDb7%2311%2C%20%2C%20%2C%20%7CC7%2C%20%2C%20%2C%20%7CC7%2C%20%2C%20%2C%20%7CC7%2C%20%2C%20%2C%20Z
![IMG_4632](https://user-images.githubusercontent.com/53892067/203913349-f5f1ae4a-baec-4e1d-a20c-c3291aa7d1c5.jpg)

## How it works
### Model Input
The model works by trying to predict the fifth chord based on the previous 4 chords. This accumulates to 8 inputs in the Neural Network as each chords is identified by its root note, and its quality. Another 2 inputs are added, the style (i.e. Latin, Medium Swing) and the composer.

### Model Output
The outputs works best by using a categorical approach. the total output layer is comprised of 114 neurons (12 notes 102 Qualities).

### Dataset
This structure results in a training dataset size of 43575, and a test dataset size of 18675.
The provided model is trained based on the jazz 1400 library from the [iRealPro Forum](https://www.irealb.com/forums/showthread.php?12753-Jazz-1400-Standards), but this could be used for any style of music that iRealPro supports.

### Improvements
1. When training, all of the pieces are transposed to a common key of C Major (or A minor), as it makes the model only cares about the chords relations, not the absolute chords. This means that when generating a chart, the provided chords are transposed, then output charts is inverse transposed back to your desired key.

2. I originaly encountered an issue where the generated chords will get stuck in a loop, and then the same chord will get generated over and over again. A solution to fix this is (yet to implement):
    1. Don't train the neural network on repeated chords
    2. Create Multiple models with different lags (default 4), then a lag 6, lag 8, lag 10 and so on. When the chart is being generated, use the appropriate model based on how many chords already exist in the chart.

## Supported Styles
inspect dp.styles_encoder.classes_

## Supported Composers
inspect dp.composers_encoder.classes_
