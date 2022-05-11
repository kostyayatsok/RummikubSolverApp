package com.tencent.yolov5ncnn;


import android.content.Context;
import android.graphics.Bitmap;
import android.graphics.Matrix;
import android.widget.ImageView;

import java.io.IOException;
import java.io.InputStream;
import java.nio.FloatBuffer;
import java.util.HashMap;

import ai.onnxruntime.*;

public class ClassifierONNX {
    OrtEnvironment env;
    OrtSession session;

    int DIM_BATCH_SIZE = 1;
    int DIM_PIXEL_SIZE = 3;
    int IMAGE_SIZE_X = 32;
    int IMAGE_SIZE_Y = 32;


    public ClassifierONNX(Context context) {
        env = OrtEnvironment.getEnvironment();
        byte[] buffer = null;
        try {
            InputStream iS = context.getAssets().open("rummi.onnx");
            buffer = new byte[iS.available()];
            iS.read(buffer);
        } catch (IOException e) {
            e.printStackTrace();
        }
        try {
            session = env.createSession(buffer, new OrtSession.SessionOptions());
        } catch (OrtException e) {
            e.printStackTrace();
        }

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
//                imgData.put(idx, 0.f);
//                imgData.put(idx + stride, 0.f);
//                imgData.put(idx + stride * 2, 0.f);
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
//        int width = bm.getWidth();
//        int height = bm.getHeight();
//        float scaleWidth = ((float) newWidth) / width;
//        float scaleHeight = ((float) newHeight) / height;
//        // CREATE A MATRIX FOR THE MANIPULATION
//        Matrix matrix = new Matrix();
//        // RESIZE THE BIT MAP
//        matrix.postScale(scaleWidth, scaleHeight);
//
//        // "RECREATE" THE NEW BITMAP
//        Bitmap resizedBitmap = Bitmap.createBitmap(bm, 0, 0, width, height, matrix, false);
//        if(!isNecessaryToKeepOrig){
//            bm.recycle();
//        }
//        return resizedBitmap;
    }

    public void predict(Bitmap image, YoloV5Ncnn.Obj object)
    {
//        int sz = (int) Math.max(object.h, object.w);
//        int x = (int)(object.x-(sz-object.w)/2);
//        int y = (int)(object.y-(sz-object.h)/2);
//        if (x < 0) x = 0;
//        if (y < 0) y = 0;
//        if (x+sz > image.getWidth()) sz = image.getWidth()-x;
//        if (y+sz > image.getHeight()) sz = image.getHeight()-y;

        Bitmap crop = Bitmap.createBitmap(image, (int)object.x, (int)object.y, (int)object.w, (int)object.h);
//        Bitmap scaled = Bitmap.createScaledBitmap(crop, IMAGE_SIZE_X, IMAGE_SIZE_Y, true);
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

        float[][] results = null;
        try {
            OrtSession.Result output = session.run(input);
            results = (float[][]) output.get(0).getValue();
        } catch (Exception e) {
            e.printStackTrace();
            return;
        }

        float max_prob = 0;
        int max_label = -1;
        for (int i = 0; i < results[0].length; i++)
        {
            if (results[0][i] > max_prob)
            {
                max_prob = results[0][i];
                max_label = i;
            }
        }
        System.out.println("max_prob "+max_prob);
        object._color = max_label % 4;
        object._value = max_label / 4;
        object.prob = max_prob;
    }
}
