package com.tencent.yolov5ncnn;


import android.content.Context;
import android.content.res.AssetFileDescriptor;
import android.content.res.AssetManager;
import android.graphics.Bitmap;
import android.graphics.Matrix;
import android.widget.ImageView;

import java.io.File;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStream;
import java.nio.ByteBuffer;
import java.nio.FloatBuffer;
import java.nio.MappedByteBuffer;
import java.nio.channels.FileChannel;
import java.util.HashMap;
import org.tensorflow.lite.Interpreter;
import org.tensorflow.lite.gpu.CompatibilityList;
import org.tensorflow.lite.gpu.GpuDelegate;

import ai.onnxruntime.*;

public class ClassifierONNX {
    OrtEnvironment env;
    OrtSession session;

    int DIM_BATCH_SIZE = 1;
    int DIM_PIXEL_SIZE = 3;
    int IMAGE_SIZE_X = 48;
    int IMAGE_SIZE_Y = 48;

    Interpreter interpreter;


    private MappedByteBuffer loadModelFile(AssetManager assetManager, String modelPath) {
        AssetFileDescriptor fileDescriptor = null;
        try {
            fileDescriptor = assetManager.openFd(modelPath);
        } catch (IOException e) {
            e.printStackTrace();
        }
        FileInputStream inputStream = new FileInputStream(fileDescriptor.getFileDescriptor());
        FileChannel fileChannel = inputStream.getChannel();
        long startOffset = fileDescriptor.getStartOffset();
        long declaredLength = fileDescriptor.getDeclaredLength();
        try {
            return fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength);
        } catch (IOException e) {
            e.printStackTrace();
        }
        return null;
    }

    public ClassifierONNX(Context context) {
        AssetManager assetManager = context.getAssets();
        Interpreter.Options options = new Interpreter.Options();
        CompatibilityList compatList = new CompatibilityList();

        if(compatList.isDelegateSupportedOnThisDevice()){
            // if the device has a supported GPU, add the GPU delegate
            GpuDelegate.Options delegateOptions = compatList.getBestOptionsForThisDevice();
            GpuDelegate gpuDelegate = new GpuDelegate(delegateOptions);
            options.addDelegate(gpuDelegate);
        } else {
            // if the GPU is not supported, run on 4 threads
            options.setNumThreads(4);
        }

        interpreter = new Interpreter(loadModelFile(assetManager, "rummi_936.tflite"), options);
    }

    FloatBuffer preProcess(Bitmap bitmap) {
        FloatBuffer imgData = FloatBuffer.allocate(
                DIM_BATCH_SIZE
                        * DIM_PIXEL_SIZE
                        * IMAGE_SIZE_X
                        * IMAGE_SIZE_Y
        );

        imgData.rewind();
        int stride = IMAGE_SIZE_X * IMAGE_SIZE_Y;
        int[] bmpData = new int[stride];
        bitmap.getPixels(bmpData, 0, bitmap.getWidth(), 0, 0, bitmap.getWidth(), bitmap.getHeight());
        for (int i = 0; i < IMAGE_SIZE_X; i++) {
            for (int j = 0; j < IMAGE_SIZE_Y; j++) {
                int idx = IMAGE_SIZE_Y * j + i;
                int pixelValue = bmpData[idx];
                imgData.put(3*idx+0, ((pixelValue >> 16) & 0xFF) / 127.5f-1);
                imgData.put(3*idx+1, ((pixelValue >>  8) & 0xFF) / 127.5f-1);
                imgData.put(3*idx+2, ((pixelValue >>  0) & 0xFF) / 127.5f-1);
            }
        }

        imgData.rewind();
        return imgData;
    }


    private static Bitmap getResizedBitmap(Bitmap bitmap, int newWidth, int newHeight, boolean isNecessaryToKeepOrig) {
        Matrix m = new Matrix();
        m.setScale(0.5F, 0.5F);
        while (bitmap.getWidth() > 2*newWidth && bitmap.getHeight() > 2*newHeight)
        {
            Bitmap bitmap_half = Bitmap.createBitmap(bitmap, 0, 0, bitmap.getWidth(), bitmap.getHeight(), m, true);
            bitmap.recycle();
            bitmap = bitmap_half;
        }
        m.setScale(((float) newWidth) / bitmap.getWidth(), ((float) newHeight) / bitmap.getHeight());
        return Bitmap.createBitmap(bitmap, 0, 0, bitmap.getWidth(), bitmap.getHeight(), m, false);
    }

    public void predict(Bitmap image, YoloV5Ncnn.Obj object)
    {
        Bitmap crop = Bitmap.createBitmap(image, (int)object.x, (int)object.y, (int)object.w, (int)object.h);
        Bitmap scaled = getResizedBitmap(crop, IMAGE_SIZE_X, IMAGE_SIZE_Y, true);
        FloatBuffer sourceArray = preProcess(scaled);
        OnnxTensor t1 = null;
        try {
            t1 = OnnxTensor.createTensor(env,sourceArray, new long[]{DIM_BATCH_SIZE, IMAGE_SIZE_X, IMAGE_SIZE_Y, DIM_PIXEL_SIZE});
        } catch (OrtException e) {
            e.printStackTrace();
        }
        OnnxTensor finalT = t1;
        HashMap<String, OnnxTensor> input = new HashMap<String, OnnxTensor>() {{
            put("input_1", finalT);
        }};

        float[][] outputs = new float[1][54];
        interpreter.run(sourceArray, outputs);

        float[][] results;
        try {
            long startTime2 = System.nanoTime();
            OrtSession.Result output = session.run(input);
            System.out.println("session.run: " + (System.nanoTime()-startTime2));

            results = (float[][]) output.get(0).getValue();
        } catch (Exception e) {
            e.printStackTrace();
            return;
        }

        float max_prob = 0.0f;
        int max_label = -1;
        for (int i = 0; i < results[0].length; i++)
        {
            if (results[0][i] > max_prob)
            {
                max_prob = results[0][i];
                max_label = i;
            }
        }
//        System.out.println("max_prob " + max_prob + " detection prob " + object.prob);
        object._color = max_label % 4;
        object._value = max_label / 4;
        object.prob = 2 * (max_prob * object.prob) / (max_prob + object.prob);
//        System.out.println("new_prob " + object.prob);
    }
}
