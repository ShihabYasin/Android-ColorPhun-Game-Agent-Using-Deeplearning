# Android-ColorPhun-Game-Agent-Using-Deeplearning
## An intelligent game agent to play autonomously a simple game idea namely ColorPhun. 
https://github.com/prakhar1989/ColorPhun

## Screen Shot: Games Playing by Agent
<br>
<img height="700" src="https://github.com/ShihabYasin/Android-ColorPhun-Game-Agent-Using-Deeplearning/blob/master/Screenshot_2017-06-06-11-20-17.png"/>
<br>

## Prerequisites
1. An Android device or emulator that runs API level 21 or higher
2. About 200 MB of internal storage space free.
3. Android Studio 2.2 or newer

## Usage
After installing this game application:
1. Tap on (1)CREATE NET OR RESET button. It will create a Neural Network with some initialization twist.
2. Then Tap on (2) TRAIN NET which will train this net with 1000 samples randomly generated.
3. To test Net performance now press (3) CHANGE COLOR button that will change color and transparency level of above two boxes.
4. This game agent will select the colored box that has maximum transparency. Tap (4) TEST game agent will Toast Up its prediction as shown in the image above with a confidence value.

## Tune
1. You may optimize Net performance changing Threshold Value to match appropriate color selection or decesion boundary. Below threshold value will indicate to select LEFT box and above predicted value will be used for selecting RIGHT box.
2. Also you can change transparency of these boxes on real time before testing using sliders attached to each box( see image above )

## Library Information
Sample Android Studio project that shows you how to use Deeplearning4J in Android apps.
Read full tutorial here: [How to Use Deeplearning4J in Android Apps](http://progur.com/2017/01/how-to-use-deeplearning4j-on-android.html)
