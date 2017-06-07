package com.yasin.IntelligentGameAgent;

import android.app.ProgressDialog;
import android.graphics.Color;
import android.graphics.drawable.ColorDrawable;
import android.support.v7.app.AppCompatActivity;
import android.os.Bundle;
import android.util.Log;
import android.view.Gravity;
import android.view.View;
import android.widget.LinearLayout;
import android.widget.SeekBar;
import android.widget.TextView;
import android.widget.Toast;

import org.deeplearning4j.earlystopping.saver.LocalFileModelSaver;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.factory.Nd4j;

import java.io.IOException;

public class MainActivity extends AppCompatActivity {

    // 0 = LeftView , 1 = RightView
    public int mColor;
    public TextView seekBarValueLeft, seekBarValueRight, seekBarThresholdValue;
    String localDirToSaveNet = "/data/user/0/com.progur.IntelligentGameAgent/files";
    public DenseLayer inputLayer;
    public DenseLayer hiddenLayer ;
    public OutputLayer outputLayer ;
    public MultiLayerConfiguration conf;
    public MultiLayerNetwork myNetwork;
    public INDArray trainingInputs;
    public INDArray trainingOutputs;
    public DataSet myData;
    public INDArray actualInput; // Create input
    public INDArray actualOutput;
    public LocalFileModelSaver m; // Saving Net
    public MultiLayerNetwork mm = null; // Retrieved Net
    public double thresholdValue = 0.00000;

    private void iniNet()
    {
        Log.d("myNetwork Output", "In createAndUseNetwork");
        inputLayer = new DenseLayer.Builder()
                .nIn(2)
                .nOut(3)
                .name("Input")
                .activation(Activation.SIGMOID)
                .build();

        hiddenLayer = new DenseLayer.Builder()
                .nIn(3)
                .nOut(2)
                .name("Hidden")
                .activation(Activation.SIGMOID)
                .build();

        outputLayer = new OutputLayer.Builder()
                .nIn(2)
                .nOut(2)
                .name("Output")
                .activation(Activation.SOFTMAX)
                .build();

        NeuralNetConfiguration.Builder nncBuilder = new NeuralNetConfiguration.Builder();
        nncBuilder.iterations(600);
        nncBuilder.learningRate(0.1);

        NeuralNetConfiguration.ListBuilder listBuilder = nncBuilder.list();
        listBuilder.layer(0, inputLayer);
        listBuilder.layer(1, hiddenLayer);
        listBuilder.layer(2, outputLayer);

        listBuilder.backprop(true);

        conf = listBuilder.build();

        myNetwork = new MultiLayerNetwork(conf);
        myNetwork.init();
    }

    private void createTrainingsamples()
    {
        final int NUM_SAMPLES = 1000;
        trainingInputs = Nd4j.zeros(NUM_SAMPLES, inputLayer.getNIn());
        trainingOutputs = Nd4j.zeros(NUM_SAMPLES, outputLayer.getNOut());
        //////   Sample Creating Here
        for (int i = 0 ; i<NUM_SAMPLES;i+=2) {
            java.util.Random rnd = new java.util.Random();

            int rnd1  = rnd.nextInt(125);
            int a1 = 129+rnd1;
            int a2 = rnd.nextInt(126);
            int r = rnd.nextInt(256);
            int g = rnd.nextInt(256);
            int b = rnd.nextInt(256);

            int colorLeft = Color.argb(a1, r, g, b); /// -
            int colorRight = Color.argb(a2, r, g, b); /// +

            int outputAnsNeuron =0;
            int outputAntiAnsNeuron =0;

            if (colorLeft > 0 && colorRight > 0) {
                if (colorLeft <= colorRight) outputAnsNeuron = 0;
                else outputAnsNeuron = 1;
            } else if (colorLeft < 0 && colorRight < 0) {
                int temp1 = -colorLeft;
                int temp2 = -colorRight;
                if (temp1 >= temp2) outputAnsNeuron = 0;
                else outputAnsNeuron = 1;
            } else if (colorLeft < 0 && colorRight > 0) {
                outputAnsNeuron = 1;
            } else if (colorLeft > 0 && colorRight < 0) {
                outputAnsNeuron = 0;
            }

            outputAntiAnsNeuron = 1 - outputAnsNeuron;
            // 0 = LeftView , 1 = RightView
            trainingInputs.putScalar(new int[]{i, 0}, colorLeft);
            trainingInputs.putScalar(new int[]{i, 1}, colorRight);
            trainingOutputs.putScalar(new int[]{i, 0}, outputAnsNeuron);
            trainingOutputs.putScalar(new int[]{i, 1}, outputAntiAnsNeuron);

            trainingInputs.putScalar(new int[]{i, 0},  colorRight );
            trainingInputs.putScalar(new int[]{i, 1},  colorLeft);
            trainingOutputs.putScalar(new int[]{i, 0}, outputAntiAnsNeuron );
            trainingOutputs.putScalar(new int[]{i, 1},   outputAnsNeuron);

            Log.d("TrainingSamples",""+colorLeft + " "+ colorRight + " " + outputAnsNeuron  + " "+ outputAntiAnsNeuron);
            Log.d("TrainingSamples",""+ colorRight + " "+  colorLeft+ " " + outputAntiAnsNeuron  + " "+outputAnsNeuron );

        }
        //////
        myData = new DataSet(trainingInputs, trainingOutputs);
    }
private void trainNet()
{
    myNetwork.fit(myData);
}
private void createTestingSamples()
{
    final int NUM_SAMPLES = 1;
    trainingInputs = Nd4j.zeros(NUM_SAMPLES, inputLayer.getNIn());
    trainingOutputs = Nd4j.zeros(NUM_SAMPLES, outputLayer.getNOut());

    View left = (LinearLayout) findViewById(R.id.layoutLeft);
    View right = (LinearLayout) findViewById(R.id.layoutRight);

    ColorDrawable cd = (ColorDrawable) left.getBackground();
    int COLOR = cd.getColor();
    int a = cd.getAlpha();
    int r = Color.red(COLOR);
    int g = Color.green(COLOR);
    int b = Color.blue(COLOR);
    int colorLeft = Color.argb( a,r,g,b);

    cd = (ColorDrawable) right.getBackground();
    COLOR = cd.getColor();
    a = cd.getAlpha();
    r = Color.red(COLOR);
    g = Color.green(COLOR);
    b = Color.blue(COLOR);
    int colorRight = Color.argb( a,r,g,b);

    // 0 = LeftView , 1 = RightView
    int outputAnsNeuron =0 ;
    int outputAntiAnsNeuron=0;

    if(colorLeft > 0 && colorRight > 0)
    {
        if(colorLeft <= colorRight ) outputAnsNeuron = 0;
        else outputAnsNeuron = 1;
    }
    else if(colorLeft < 0 && colorRight < 0)
    {
        int temp1 = -colorLeft;
        int temp2 = -colorRight;
        if(temp1 >= temp2) outputAnsNeuron = 0;
        else outputAnsNeuron = 1;
    }

    else if(colorLeft < 0 && colorRight > 0)
    {
        outputAnsNeuron = 1;
    }
    else if(colorLeft > 0 && colorRight < 0)
    {
        outputAnsNeuron = 0;
    }

    outputAntiAnsNeuron = 1- outputAnsNeuron;

    trainingInputs.putScalar(new int[]{0, 0}, colorLeft);
    trainingInputs.putScalar(new int[]{0, 1}, colorRight);
    trainingOutputs.putScalar(new int[]{0, 0}, outputAnsNeuron);
    trainingOutputs.putScalar(new int[]{0, 1}, outputAntiAnsNeuron);

    myData = new DataSet(trainingInputs, trainingOutputs);
}
private void saveNetwork()
{
    m = new LocalFileModelSaver(localDirToSaveNet);
    //Good Directory =>    "/data/user/0/com.progur.IntelligentGameAgent/files"

    Log.d("myNetwork Output", getBaseContext().getFilesDir().toString());
    //
    try {
        m.saveLatestModel(myNetwork, .1);
    } catch (IOException e) {
        Log.d("myNetwork Output", "Error 9");
        e.printStackTrace();
    }
}
private void retrieveNetwork()
{
    try {
        mm = m.getLatestModel();
    } catch (IOException e) {
        Log.d("myNetwork Output", "Error 3");
        e.printStackTrace();
    }
}
private void trainSaveNetwork() {
        //retrieveNetwork();
        //myNetwork = (MultiLayerNetwork) mm;
        //createTrainingsamples(4);
    createTrainingsamples();
    trainNet();

        //saveNetwork();
        //createTestingSamples();

        //generateOutputAfterTraining();

        //retrieveNetwork();
        //generateOutputForTESTINGdata();
    }
    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);
        /////////////////////// Left Seek Bar /////////////////////////////// 
        SeekBar seekBarL = (SeekBar)findViewById(R.id.seekBarLeft);
        seekBarValueLeft = (TextView)findViewById(R.id.seekBarLeftValue);
        seekBarL.setOnSeekBarChangeListener(new SeekBar.OnSeekBarChangeListener(){
            @Override
            public void onProgressChanged(SeekBar seekBar, int progress,boolean fromUser) {
                // TODO Auto-generated method stub
                seekBarValueLeft.setText(String.valueOf(progress));
                View left= (LinearLayout)findViewById(R.id.layoutLeft);
                ColorDrawable cd = (ColorDrawable) left.getBackground();
                int COLOR = cd.getColor();
                int a = progress;
                int r = Color.red(COLOR);
                int g = Color.green(COLOR);

                int b = Color.blue(COLOR);

                int colorOne = Color.argb( a,r,g,b);
                left.setBackgroundColor(colorOne);
                TextView tvLeft = (TextView) findViewById(R.id.textViewLeft);
                tvLeft.setText(String.valueOf(colorOne));
            }
            @Override
            public void onStartTrackingTouch(SeekBar seekBar){
                // TODO Auto-generated method stub 
            }
            @Override
            public void onStopTrackingTouch(SeekBar seekBar) {
                // TODO Auto-generated method stub
            }
        });

        ///////////////////////End  Left  Seek Bar /////////////////////////////// 
        /////////////////////// Right Seek Bar ///////////////////////////////

        SeekBar seekBarR = (SeekBar)findViewById(R.id.seekBarRight);
        seekBarValueRight = (TextView)findViewById(R.id.seekBarRightValue);
        seekBarR.setOnSeekBarChangeListener(new SeekBar.OnSeekBarChangeListener(){
            @Override
            public void onProgressChanged(SeekBar seekBar, int progress, boolean fromUser){
                // TODO Auto-generated method stub
                seekBarValueRight.setText(String.valueOf(progress));
                View left= (LinearLayout)findViewById(R.id.layoutRight);
                ColorDrawable cd = (ColorDrawable) left.getBackground();
                int COLOR = cd.getColor();
                int a = progress;
                int r = Color.red(COLOR);
                int g = Color.green(COLOR);
                int b = Color.blue(COLOR);
                int colorOne = Color.argb( a,r,g,b);
                left.setBackgroundColor(colorOne);
                TextView tvLeft = (TextView) findViewById(R.id.textViewRight);
                tvLeft.setText(String.valueOf(colorOne));
            }
            @Override
            public void onStartTrackingTouch(SeekBar seekBar){
                // TODO Auto-generated method stub
            }
            @Override
            public void onStopTrackingTouch(SeekBar seekBar) {
                // TODO Auto-generated method stub
            }
        });
        ///////////////////////End  Right Seek Bar ///////////////////////////////
        /////////////////////// Threshold Seek Bar /////////////////////////////// 
        SeekBar seekBarThreshold = (SeekBar)findViewById(R.id.seekBarThresholdSet);
        seekBarThresholdValue = (TextView) findViewById(R.id.textViewThresholdValue);
        seekBarThreshold.setOnSeekBarChangeListener(new SeekBar.OnSeekBarChangeListener(){
            @Override
            public void onProgressChanged(SeekBar seekBar, int progress,boolean fromUser) {
                // TODO Auto-generated method stub
                double temp = (double)progress;
                thresholdValue = temp/10000000.0000;
                seekBarThresholdValue.setText(String.valueOf(thresholdValue));
            }
            @Override
            public void onStartTrackingTouch(SeekBar seekBar){
                // TODO Auto-generated method stub 
            }
            @Override
            public void onStopTrackingTouch(SeekBar seekBar) {
                // TODO Auto-generated method stub
            }
        });
        ///////////////////////End  Threshold  Seek Bar /////////////////////////////// 
    }
    @Override
    protected void onResume() {
        super.onResume();
    }
    void randomClorChanger() {
        java.util.Random rnd = new java.util.Random();

        int a = 255;
        int r = rnd.nextInt(256);
        int g = rnd.nextInt(256);
        int b = rnd.nextInt(256);

        int colorOne = Color.argb(a, r, g, b);
        int colorTwo = Color.argb(110, r, g, b);

        View left = (LinearLayout) findViewById(R.id.layoutLeft);
        View right = (LinearLayout) findViewById(R.id.layoutRight);

        TextView tvRight, tvLeft;

        tvLeft = (TextView) findViewById(R.id.textViewLeft);
        tvRight = (TextView) findViewById(R.id.textViewRight);
        int arrayColor[] = {colorOne, colorTwo};

        int rndLeftindex = rnd.nextInt(2);
        int rndRightindex = 1 - rndLeftindex;


        tvLeft.setText(String.valueOf(arrayColor[rndLeftindex]));
        tvRight.setText(String.valueOf(arrayColor[rndRightindex]));


        left.setBackgroundColor(arrayColor[rndLeftindex]);
        right.setBackgroundColor(arrayColor[rndRightindex]);


        Log.d("myNetwork Output", "end of randomcolorchanger");
        return;
    }
    public void trainNettoSave(View view) {
        final ProgressDialog progressDialog = ProgressDialog.show(this, "", "Please Wait...");
        new Thread() { public void run(){
            try
            {
                trainSaveNetwork();
            }
            catch (Exception e) { Log.e("tag", e.getMessage()); }
            // dismiss the progress dialog
            progressDialog.dismiss();
        } }.start();
    }
    public void changeColorNow(View view) {
        randomClorChanger();
    }
    public void test(View view) {
        //retrieveNetwork();
        //myNetwork = (MultiLayerNetwork) mm;

        createTestingSamples();
        //createTrainingsamplesUPDATED();
        // Generate output
        // 0 = LeftView , 1 = RightView
        actualInput = trainingInputs;
        Toast.makeText(this, actualInput.toString(), Toast.LENGTH_SHORT).show();
        actualOutput = myNetwork.output(actualInput);

        double ans = actualOutput.getDouble(0,0);
        String ansSelected = null;
        if(  ans*100000 < thresholdValue*100000 )
        {
            ansSelected = "LEFT";
        }else {
            ansSelected = "RIGHT";
        }

        Toast toastMessage = Toast.makeText(getBaseContext(),ansSelected+" "+ans, Toast.LENGTH_SHORT);
        toastMessage.setGravity(Gravity.CENTER,0,0);
        toastMessage.show();

        Log.d("myNetwork Output", actualOutput.toString());
    }
    public void resetNet(View view) {

        Toast toastMessage = Toast.makeText(getBaseContext(),"Creating/Resetting Net", Toast.LENGTH_SHORT);
        toastMessage.setGravity(Gravity.CENTER,0,0);
        toastMessage.show();
        iniNet();
        //createTrainingsamplesUPDATED();
        //trainNet();
        //saveNetwork();
    }
}
