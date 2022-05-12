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
import android.content.DialogInterface;
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
import android.view.ViewGroup;
import android.view.Window;
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
    public static float probThresh = 0.3f;

    public LinearLayout boardView, handView;
    private Button solveButton;
    private Bitmap bitmap = null;
    private Bitmap yourSelectedImage = null;
    private Uri lastUri = null;
    private boolean onBoard = false;


    private YoloV5Ncnn yolov5ncnn = new YoloV5Ncnn();
    private Solver solver = new Solver();
    private ClassifierONNX classifier;

    private int tileWidth, tileHeight, tileSpace;

    private Integer[] colors = {Color.RED, Color.parseColor("#FF7800"), Color.BLUE, Color.BLACK,};
    private String[] values = {"j", "1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12", "13",};

    /** Called when the activity is first created. */
    @Override
    public void onCreate(Bundle savedInstanceState)
    {
        super.onCreate(savedInstanceState);

        requestWindowFeature(Window.FEATURE_NO_TITLE);
//        getWindow().setFlags(WindowManager.LayoutParams.FLAG_FULLSCREEN,
//                WindowManager.LayoutParams.FLAG_FULLSCREEN);

        setContentView(R.layout.main);

        boolean ret_init = yolov5ncnn.Init(getAssets());
        if (!ret_init)
        {
            Log.e("MainActivity", "yolov5ncnn Init failed");
        }


        boardView = findViewById(R.id.board);
        handView = findViewById(R.id.hand);
        solveButton = findViewById(R.id.solve);

        findViewById(R.id.scanBoard).setOnClickListener(v -> scan(true));
        findViewById(R.id.scanHand).setOnClickListener(arg0 -> scan(false));

        classifier = new ClassifierONNX(this);

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
        int v = Arrays.asList(values).indexOf(anchorView.getText());
        int c = Arrays.asList(colors).indexOf(anchorView.getCurrentTextColor());

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
                anchorView.setText(val);
            });
            layout.addView(btnTag);

        }
        layout = changeTileView.findViewById(R.id.chooseColor);
        for (int col : colors) {
            btnTag = new Button(this);
            btnTag.setLayoutParams(params);
            btnTag.setBackgroundColor(col);
            btnTag.setOnClickListener(arg0->{
                anchorView.setTextColor(col);
            });
            layout.addView(btnTag);

        }

        popup.setContentView(changeTileView);
        popup.setHeight(WindowManager.LayoutParams.WRAP_CONTENT);
        popup.setWidth(WindowManager.LayoutParams.WRAP_CONTENT);
        // Closes the popup window when touch outside of it - when looses focus
        popup.setOutsideTouchable(false);
        popup.setFocusable(true);
        popup.setOnDismissListener(() -> {
            tiles[v][c]--;
            int new_v = Arrays.asList(values).indexOf(anchorView.getText());
            int new_c = Arrays.asList(colors).indexOf(anchorView.getCurrentTextColor());
            tiles[new_v][new_c]++;
            anchorView.setBackground(MainActivity.this.getDrawable(R.drawable.broder_gray));
        });
        popup.showAsDropDown(anchorView);
    }

    public void popupMessage(){
        AlertDialog.Builder alertDialogBuilder = new AlertDialog.Builder(this);
        alertDialogBuilder.setMessage("The board is incorrect!\n Fix it manually or take a new photo.");
//        alertDialogBuilder.setIcon(R.drawable.ic_no_internet);
        alertDialogBuilder.setTitle("Fail to solve");
        alertDialogBuilder.setNegativeButton("OK", (dialogInterface, i) -> {
//            Log.d("internet","Ok btn pressed");
//            // add these two lines, if you wish to close the app:
//            finishAffinity();
//            System.exit(0);
        });
        AlertDialog alertDialog = alertDialogBuilder.create();
        alertDialog.show();
    }

    private YoloV5Ncnn.Obj[][] solve()
    {
        int[] runsHashes = new int[solver.K];
        int score = solver.maxScore(0, runsHashes);
        if (score < 0) {
            popupMessage();
            return new YoloV5Ncnn.Obj[0][];
        } else {
            int handScore = score;
            for (int[] row : solver.board)
                for (int c : row)
                    handScore -= c;
            System.out.println("Your possible score: " + handScore + " " + score);

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
                        ex.printStackTrace();
                        return;
                    }
                    if (photoFile != null) {
                        lastUri = FileProvider.getUriForFile(this,
                                "com.tencent.yolov5ncnn.fileprovider",
                                photoFile);
                        takePicture.putExtra(MediaStore.EXTRA_OUTPUT, lastUri);
                        takePicture.putExtra("android.intent.extra.quickCapture",true);
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
        row.setLayoutParams(rowParams);
        return row;
    }

    private TextView createTile(YoloV5Ncnn.Obj obj)
    {
        if (obj == null) {
            TextView button = new TextView(this);
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
        button.setOnLongClickListener((view) -> {
            TextView but = (TextView) view;
            int v = Arrays.asList(values).indexOf(but.getText());
            int c = Arrays.asList(colors).indexOf(but.getCurrentTextColor());
            if (onBoard)
                solver.board[v][c]--;
            else
                solver.hand[v][c]--;

            ViewGroup parentView = (ViewGroup) view.getParent();
            parentView.removeView(view);
            return true;
        });
        return button;
    }

    private void showObjects(YoloV5Ncnn.Obj[][] objects, LinearLayout anchorView)
    {
        anchorView.removeAllViews();

        LinearLayout row = createRow();
        row.addView(createTile(null));

        int position = tileSpace;
        for (YoloV5Ncnn.Obj[] objRow : objects) {
            if (objRow == null) continue;
            if (position + objRow.length * tileWidth > anchorView.getWidth()) {
                anchorView.addView(row);
                row = createRow();
                row.addView(createTile(null));
                position = tileSpace;
            }
            for (int i = 0; i < objRow.length; i++) {
                if (objRow[i] == null) continue;
                row.addView(createTile(objRow[i]));
                position += tileWidth;
                if (i == objRow.length) {
                    row.addView(createTile(null));
                    position += tileSpace;
                }
            }

        }
        anchorView.addView(row);
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
                for (int i = 0; i < objects.length; i++) {
                    objects_[i][0] = objects[i].prob >= probThresh ? objects[i] : null;
                }
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
        BitmapFactory.Options o = new BitmapFactory.Options();
        Bitmap bitmap = BitmapFactory.decodeStream(getContentResolver().openInputStream(selectedImage), null, o);

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
