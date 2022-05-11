// Tencent is pleased to support the open source community by making ncnn available.
//
// Copyright (C) 2020 THL A29 Limited, a Tencent company. All rights reserved.
//
// Licensed under the BSD 3-Clause License (the "License"); you may not use this file except
// in compliance with the License. You may obtain a copy of the License at
//
// https://opensource.org/licenses/BSD-3-Clause
//
// Unless required by applicable law or agreed to in writing, software distributed
// under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
// CONDITIONS OF ANY KIND, either express or implied. See the License for the
// specific language governing permissions and limitations under the License.

package com.tencent.yolov5ncnn;

import android.app.Activity;
import android.app.AlertDialog;
import android.content.Intent;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.Paint;
import android.media.ExifInterface;
import android.graphics.Matrix;
import android.net.Uri;
import android.os.Bundle;
import android.os.Environment;
import android.provider.MediaStore;
import android.util.Log;
import android.view.Gravity;
import android.view.View;
import android.view.WindowManager;
import android.widget.Button;
import android.widget.ImageView;
import android.widget.LinearLayout;
import android.widget.PopupWindow;
import android.widget.TableLayout;
import android.widget.TableRow;
import android.widget.TextView;

import androidx.core.content.FileProvider;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.text.SimpleDateFormat;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Date;
import java.util.Map;


public class MainActivity extends Activity
{
    private static final int SELECT_IMAGE = 1;
    private static final int TAKE_IMAGE = 2;

    public LinearLayout boardView, handView;
    private Button solveButton;
    private TextView scoreView;
    private Bitmap bitmap = null;
    private Bitmap yourSelectedImage = null;
    private Uri lastUri = null;
    private boolean onBoard = false;


    private YoloV5Ncnn yolov5ncnn = new YoloV5Ncnn();
    private Solver solver = new Solver();
    private ClassifierONNX classifier;
    private Bitmap background;

    private int tileWidth, tileHeight, tileSpace;

    private Integer[] colors = {Color.RED, Color.parseColor("#FF7800"), Color.BLUE, Color.BLACK,};
    private String[] values = {"j", "1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12", "13",};

    /** Called when the activity is first created. */
    @Override
    public void onCreate(Bundle savedInstanceState)
    {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.main);

        boolean ret_init = yolov5ncnn.Init(getAssets());
        if (!ret_init)
        {
            Log.e("MainActivity", "yolov5ncnn Init failed");
        }


//        boolean ret_init = classifierNcnn.Init(getAssets());
//        if (!ret_init)
//        {
//            Log.e("MainActivity", "classifier Init failed");
//        }
        boardView = findViewById(R.id.board);
        handView = findViewById(R.id.hand);
        scoreView = findViewById(R.id.score);
        solveButton = findViewById(R.id.solve);

        boardView.setOnClickListener(v -> scan(true));
        handView.setOnClickListener(arg0 -> scan(false));
//        solutionView.setOnClickListener(v -> displayPopupWindow(solutionView));
//        Button buttonSolve = findViewById(R.id.solve);
//        buttonSolve.setOnClickListener(arg0 -> {

//        });
        classifier = new ClassifierONNX(this);

        boardView.setBackground(getDrawable(R.drawable.background));
        handView.setBackground(getDrawable(R.drawable.background));

        solveButton.setOnClickListener(arg -> {
            YoloV5Ncnn.Obj[][] solution = solve();
            if (solution.length > 0)
                showObjects(solution, boardView);
        });
    }

    @Override
    public void onWindowFocusChanged(boolean hasFocus) {
        super.onWindowFocusChanged(hasFocus);
        tileWidth = Math.min(boardView.getWidth()/14, boardView.getHeight()/14);
        tileHeight = 4*tileWidth/3;
        tileSpace = tileWidth/4;
    }

    private void displayPopupWindow(TextView anchorView, int[][] tiles) {
        anchorView.setBackground(getDrawable(R.drawable.broder_magenta));

        PopupWindow popup = new PopupWindow(this);
        View changeTileView = getLayoutInflater().inflate(R.layout.change_tile, null);

        LinearLayout layout = changeTileView.findViewById(R.id.chooseValue);
        TextView btnTag;
        LinearLayout.LayoutParams params = new LinearLayout.LayoutParams(tileWidth, tileHeight);
        params.weight = 1;
        for (String val : values) {
            btnTag = new TextView(this);
            btnTag.setLayoutParams(params);
            btnTag.setText(val);
            btnTag.setGravity(Gravity.CENTER);
            btnTag.setClickable(true);
            btnTag.setBackground(getDrawable(R.drawable.broder_gray));
            btnTag.setOnClickListener(arg0->{
                int v = Arrays.asList(values).indexOf(anchorView.getText());
                int c = Arrays.asList(colors).indexOf(anchorView.getCurrentTextColor());
                tiles[v][c]--;

                anchorView.setText(val);

                v = Arrays.asList(values).indexOf(val);
                tiles[v][c]++;
            });
            layout.addView(btnTag);

        }
        layout = changeTileView.findViewById(R.id.chooseColor);
        for (int col : colors) {
            btnTag = new Button(this);
            btnTag.setLayoutParams(params);
            btnTag.setBackgroundColor(col);
            btnTag.setOnClickListener(arg0->{
                int v = Arrays.asList(values).indexOf(anchorView.getText());
                int c = Arrays.asList(colors).indexOf(anchorView.getCurrentTextColor());
                tiles[v][c]--;

                anchorView.setTextColor(col);

                c = Arrays.asList(colors).indexOf(col);
                tiles[v][c]++;
            });
            layout.addView(btnTag);

        }

        popup.setContentView(changeTileView);
        popup.setHeight(WindowManager.LayoutParams.WRAP_CONTENT);
        popup.setWidth(WindowManager.LayoutParams.WRAP_CONTENT);
        // Closes the popup window when touch outside of it - when looses focus
        popup.setOutsideTouchable(true);
//        popup.setFocusable(true);
        popup.showAsDropDown(anchorView);
    }

    private YoloV5Ncnn.Obj[][] solve()
    {
        int[] runsHashes = new int[solver.K];
        int score = solver.maxScore(0, runsHashes);
        if (score < 0) {
            scoreView.setText("Board is incorrect");
        }
        int handScore = score;
        for (int[] row : solver.board)
            for (int c : row)
                handScore -= c;
        scoreView.setText("You can put " + handScore + " tiles");
        System.out.println("Your possible score: " + handScore + " " + score);
        if (score > 0) {
            ArrayList<ArrayList<Solver.Pair<Integer, Integer>>> rows = solver.restore();
            YoloV5Ncnn.Obj[][] rowsObj = new YoloV5Ncnn.Obj[rows.size()][];
            int[][] tiles = Arrays.stream(solver.board).map(int[]::clone).toArray(int[][]::new);
            for (int i = 0; i < rows.size(); i++)
            {
                rowsObj[i] = new YoloV5Ncnn.Obj[rows.get(i).size()];
                for (int j = 0; j < rowsObj[i].length; j++)
                {
                    YoloV5Ncnn.Obj obj = yolov5ncnn.new Obj();
                    obj.prob = 1;
                    obj._value = rows.get(i).get(j).getFirst();
                    obj._color = rows.get(i).get(j).getSecond();
                    obj.fromHand = tiles[obj._value][obj._color] <= 0;
                    rowsObj[i][j] = obj;

                    tiles[obj._value][obj._color]--;
                }

            }
            System.out.println("Rows: " + rows);
            return rowsObj;
        }
        return new YoloV5Ncnn.Obj[0][];
    }

    private void scan(boolean _onBoard) {
        onBoard=_onBoard;

        final CharSequence[] options = { "Choose from Photos", "Take Picture", "Cancel" };
        AlertDialog.Builder builder = new AlertDialog.Builder(MainActivity.this);
        builder.setTitle("New Test Image");

        builder.setItems(options, (dialog, item) -> {
            if (options[item].equals("Take Picture")) {
                // start default camera
                Intent takePicture = new Intent(MediaStore.ACTION_IMAGE_CAPTURE);
                if (takePicture.resolveActivity(getPackageManager()) != null) {
                    File photoFile = null;
                    try {
                        photoFile = createImageFile();
                    } catch (IOException ex) {
                    }
                    if (photoFile != null) {
                        lastUri = FileProvider.getUriForFile(this,
                                "com.tencent.yolov5ncnn.fileprovider",
                                photoFile);
                        takePicture.putExtra(MediaStore.EXTRA_OUTPUT, lastUri);
                        startActivityForResult(takePicture, TAKE_IMAGE);
                    }
                }
            }
            else if (options[item].equals("Choose from Photos")) {
                Intent pickPhoto = new Intent(Intent.ACTION_PICK, MediaStore.Images.Media.INTERNAL_CONTENT_URI);
                startActivityForResult(pickPhoto , SELECT_IMAGE);
            }
            else if (options[item].equals("Cancel")) {
                dialog.dismiss();
            }
        });
        builder.show();
    }

    private File createImageFile() throws IOException {
        // Create an image file name
        String timeStamp = new SimpleDateFormat("yyyyMMdd_HHmmss").format(new Date());
        String imageFileName = "JPEG_" + timeStamp + "_";
        File storageDir = getExternalFilesDir(Environment.DIRECTORY_PICTURES);
        File image = File.createTempFile(
                imageFileName,  /* prefix */
                ".jpg",         /* suffix */
                storageDir      /* directory */
        );

        return image;
    }

    private LinearLayout createRow()
    {
        LinearLayout row = new LinearLayout(this);
        row.setOrientation(LinearLayout.HORIZONTAL);
        row.setPadding(0,0,0,tileHeight/10);
        TableRow.LayoutParams rowParams = new TableRow.LayoutParams();
//        rowParams.setMargins(0, 0, 0, tileSpace);
        row.setLayoutParams(rowParams);
        return row;
    }

    private TextView createTile(YoloV5Ncnn.Obj obj)
    {
        if (obj == null) {
            TextView button = new TextView(this);
            System.out.println("tileSpace: " + tileSpace);
            button.setLayoutParams(new TableRow.LayoutParams(tileSpace, tileHeight));
            return button;
        }
        TextView button = new TextView(this);
        button.setLayoutParams(new TableRow.LayoutParams(tileWidth, tileHeight));
        button.setText(values[obj._value]);
        button.setTextColor(colors[obj._color]);
        if (obj.fromHand) {
            button.setBackground(getDrawable(R.drawable.broder_green));
        } else {
            button.setBackground(getDrawable(R.drawable.broder_gray));
        }
        button.setGravity(Gravity.CENTER);
        button.setClickable(true);

        button.setOnClickListener(v->displayPopupWindow(button, onBoard ? solver.board : solver.hand));

        return button;
    }

    private void showObjects(YoloV5Ncnn.Obj[][] objects, LinearLayout anchorView)
    {
        anchorView.removeAllViews();

        LinearLayout row = createRow();
        row.addView(createTile(null));

        int position = tileSpace;
        for (YoloV5Ncnn.Obj[] objRow : objects) {
            if (position + objRow.length * tileWidth > anchorView.getWidth()) {
                anchorView.addView(row);
                row = createRow();
                row.addView(createTile(null));
                position = tileSpace;
            }
            for (YoloV5Ncnn.Obj obj : objRow) {
                row.addView(createTile(obj));
                position += tileWidth;
            }
            row.addView(createTile(null));
            position += tileSpace;
        }
        anchorView.addView(row);
    }

    private void showObjects_old(YoloV5Ncnn.Obj[] objects, ImageView view, Bitmap bitmap)
    {
        if (objects == null)
        {
            view.setImageBitmap(bitmap);
            return;
        }

        // draw objects on bitmap
        Bitmap rgba = bitmap.copy(Bitmap.Config.ARGB_8888, true);

        Canvas canvas = new Canvas(rgba);

        Paint paint = new Paint();
        paint.setStyle(Paint.Style.FILL);

        Paint borderpaint = new Paint();
        borderpaint.setStyle(Paint.Style.STROKE);
        borderpaint.setStrokeWidth(4);
        borderpaint.setColor(Color.BLACK);

        Paint textpaint = new Paint();
        textpaint.setTextAlign(Paint.Align.CENTER);

//        int[] colors = {Color.BLUE, Color.BLACK, Color.YELLOW, Color.RED};
//        String[] values = {"1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12", "13", "j"};
        for (int i = 0; i < objects.length; i++)
        {
            if (objects[i].fromHand)
                paint.setColor(Color.GREEN);
            else
                paint.setColor(Color.WHITE);
            paint.setAlpha(150);

            canvas.drawRect(
                    objects[i].x, objects[i].y,
                    objects[i].x + objects[i].w,
                    objects[i].y + objects[i].h,
                    paint);
            canvas.drawRect(
                    objects[i].x, objects[i].y,
                    objects[i].x + objects[i].w,
                    objects[i].y + objects[i].h,
                    borderpaint);

            // draw filled text inside image
            {
                textpaint.setColor(colors[objects[i]._color]);
                textpaint.setAlpha(150);
                textpaint.setTextSize(objects[i].w);// * getResources().getDisplayMetrics().scaledDensity);

                String text = values[objects[i]._value];//label + " = " + String.format("%.1f", objects[i].prob * 100) + "%";
//                text += "\n"+rows;
                float text_width = textpaint.measureText(text);
                float text_height = - textpaint.ascent() + textpaint.descent();

                float x = objects[i].x+objects[i].w/2;
                float y = objects[i].y+objects[i].h/2 - text_height/2;
                if (y < 0)
                    y = 0;
                if (x + text_width > rgba.getWidth())
                    x = rgba.getWidth() - text_width;

//                canvas.drawRect(x, y, x + text_width, y + text_height, textbgpaint);
                canvas.drawText(text, x, y - textpaint.ascent(), textpaint);
            }
        }

//        textpaint.setTextSize(50);
//        textpaint.setColor(Color.BLACK);
//        textpaint.setTextAlign(Paint.Align.LEFT);
//        paint.setColor(Color.MAGENTA);
//
//        String text = "Your max possible score: " + score;
//        float text_width = textpaint.measureText(text) + 10;
//        float text_height = (-textpaint.ascent() + textpaint.descent()) + 10;
//        canvas.drawRect(10, 10, text_width, text_height, paint);
//        canvas.drawText(text, 10, 10 - textpaint.ascent(), textpaint);

        view.setImageBitmap(rgba);
    }

    @Override
    protected void onActivityResult(int requestCode, int resultCode, Intent data)
    {
        super.onActivityResult(requestCode, resultCode, data);

        if (resultCode == RESULT_OK) {
            try
            {
                Uri selectedImage = null;
                if (requestCode == SELECT_IMAGE) {
                    if (null != data) {
                        selectedImage = data.getData();
                    }
                } else if (requestCode == TAKE_IMAGE) {
                    selectedImage = lastUri;
                }
                if (selectedImage == null)
                    return;
                bitmap = decodeUri(selectedImage);

                if (bitmap == null)
                    return;
                yourSelectedImage = bitmap.copy(Bitmap.Config.ARGB_8888, true);

                YoloV5Ncnn.Obj[] objects = yolov5ncnn.Detect(yourSelectedImage, false);
                for (int i = 0; i < objects.length; i++) {
                    classifier.predict(yourSelectedImage, objects[i]);
                }
                YoloV5Ncnn.Obj[][] objects_ = new YoloV5Ncnn.Obj[objects.length][1];
                for (int i = 0; i < objects.length; i++)
                    objects_[i][0] = objects[i];
                showObjects(objects_, onBoard ? boardView : handView);
                solver.setTiles(objects, onBoard);
                YoloV5Ncnn.Obj[][] solution = solve();
                if (solution.length > 0)
                    showObjects(solution, boardView);
            }
            catch (FileNotFoundException e)
            {
                Log.e("MainActivity", "FileNotFoundException");
                return;
            }
        }
    }

    private Bitmap decodeUri(Uri selectedImage) throws FileNotFoundException
    {
        // Decode image size
        BitmapFactory.Options o = new BitmapFactory.Options();
        o.inJustDecodeBounds = true;
        BitmapFactory.decodeStream(getContentResolver().openInputStream(selectedImage), null, o);

        // The new size we want to scale to
//        final int REQUIRED_SIZE = 640;
        final int REQUIRED_SIZE = 1280;

        // Find the correct scale value. It should be the power of 2.
        int width_tmp = o.outWidth, height_tmp = o.outHeight;
        int scale = 1;
        while (true) {
            if (width_tmp / 2 < REQUIRED_SIZE
               || height_tmp / 2 < REQUIRED_SIZE) {
                break;
            }
            width_tmp /= 2;
            height_tmp /= 2;
            scale *= 2;
        }

        // Decode with inSampleSize
        BitmapFactory.Options o2 = new BitmapFactory.Options();
        o2.inSampleSize = scale;
        Bitmap bitmap = BitmapFactory.decodeStream(getContentResolver().openInputStream(selectedImage), null, o2);

        // Rotate according to EXIF
        int rotate = 0;
        try
        {
            ExifInterface exif = new ExifInterface(getContentResolver().openInputStream(selectedImage));
            int orientation = exif.getAttributeInt(ExifInterface.TAG_ORIENTATION, ExifInterface.ORIENTATION_NORMAL);
            switch (orientation) {
                case ExifInterface.ORIENTATION_ROTATE_270:
                    rotate = 270;
                    break;
                case ExifInterface.ORIENTATION_ROTATE_180:
                    rotate = 180;
                    break;
                case ExifInterface.ORIENTATION_ROTATE_90:
                    rotate = 90;
                    break;
            }
        }
        catch (IOException e)
        {
            Log.e("MainActivity", "ExifInterface IOException");
        }

        Matrix matrix = new Matrix();
        matrix.postRotate(rotate);
        return Bitmap.createBitmap(bitmap, 0, 0, bitmap.getWidth(), bitmap.getHeight(), matrix, true);
    }

}
