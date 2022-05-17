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

import static java.util.Arrays.sort;

import android.app.Activity;
import android.app.AlertDialog;
import android.content.Intent;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.Color;
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
import android.widget.LinearLayout;
import android.widget.PopupWindow;
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


public class MainActivity extends Activity
{
    private static final int BOARD_SCAN = 1;
    private static final int HAND_SCAN = 2;
    public static float probThresh = 0.1f;

    private LinearLayout boardView, handView;
    private Uri lastUri = null;

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
//        int[][] hand = new int[][] {
//                new int[] { 1,2,3,4,5,6,7,8,9,10,11,12,13,1,2,3,4,5,6,7,8,9,10,11,12,13,0 },
//                new int[] { 1,2,3,4,5,6,7,8,9,10,11,12,13,1,2,3,4,5,6,7,8,9,10,11,12,13,0 },
//                new int[] { 1,2,3,4,5,6,7,8,9,10,11,12,13,1,2,3,4,5,6,7,8,9,10,11,12,13 },
//                new int[] { 1,2,3,4,5,6,7,8,9,10,11,12,13,1,2,3,4,5,6,7,8,9,10,11,12,13 },
//        };
//
//        int[][] board = new int[][] {
//                new int[] {  },
//                new int[] {  },
//                new int[] {  },
//                new int[] {  },
//        };
//        long startTime = System.nanoTime();
//        new SolverNew().solve(hand, board);
//        long stopTime = System.nanoTime();
//        System.out.println("NewSolver time: " + (stopTime - startTime));

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
        Button solveButton = findViewById(R.id.solve);

        findViewById(R.id.scanBoard).setOnClickListener(v -> scan(true));
        findViewById(R.id.scanHand).setOnClickListener(arg0 -> scan(false));

        classifier = new ClassifierONNX(this);

        solveButton.setOnClickListener(arg -> {
            YoloV5Ncnn.Obj[][] solution = solve();
            showObjects(solution, true);
        });

        YoloV5Ncnn.Obj plus = yolov5ncnn.new Obj();
        plus.addTile = true;
        boardView.addView(createTile(plus, true));
        handView.addView(createTile(plus, false));
    }

    @Override
    public void onWindowFocusChanged(boolean hasFocus) {
        super.onWindowFocusChanged(hasFocus);
        tileWidth = Math.min(boardView.getWidth()/14, boardView.getHeight()/14);
        tileHeight = 4*tileWidth/3;
        tileSpace = tileWidth/4;
    }

    private void changeTile(TextView anchorView, boolean onBoard) {
        int[][] tiles = onBoard ? solver.board : solver.hand;

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
            btnTag.setOnClickListener(arg0-> anchorView.setText(val));
            layout.addView(btnTag);

        }
        layout = changeTileView.findViewById(R.id.chooseColor);
        for (int col : colors) {
            btnTag = new Button(this);
            btnTag.setLayoutParams(params);
            btnTag.setBackgroundColor(col);
            btnTag.setOnClickListener(arg0-> anchorView.setTextColor(col));
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

    private void addTileManually(TextView anchorView, boolean onBoard) {
        int[][] tiles = onBoard ? solver.board : solver.hand;
        LinearLayout view = onBoard ? boardView : handView;

        anchorView.setLayoutParams(new TableRow.LayoutParams(tileWidth, tileHeight));
        anchorView.setBackground(getDrawable(R.drawable.broder_magenta));
        anchorView.setText("?");
        anchorView.setTextColor(Color.LTGRAY);

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
            btnTag.setOnClickListener(arg0-> anchorView.setText(val));
            layout.addView(btnTag);

        }
        layout = changeTileView.findViewById(R.id.chooseColor);
        for (int col : colors) {
            btnTag = new Button(this);
            btnTag.setLayoutParams(params);
            btnTag.setBackgroundColor(col);
            btnTag.setOnClickListener(arg0-> anchorView.setTextColor(col));
            layout.addView(btnTag);

        }

        popup.setContentView(changeTileView);
        popup.setHeight(WindowManager.LayoutParams.WRAP_CONTENT);
        popup.setWidth(WindowManager.LayoutParams.WRAP_CONTENT);
        // Closes the popup window when touch outside of it - when looses focus
        popup.setOutsideTouchable(false);
        popup.setFocusable(true);
        popup.setOnDismissListener(() -> {
            int new_v = Arrays.asList(values).indexOf(anchorView.getText());
            int new_c = Arrays.asList(colors).indexOf(anchorView.getCurrentTextColor());
            if (new_v >= 0 && new_v < values.length && new_c >= 0 && new_c < colors.length) {
                tiles[new_v][new_c]++;

                LinearLayout row = (LinearLayout) anchorView.getParent();
                row.removeView(anchorView);

                YoloV5Ncnn.Obj obj = yolov5ncnn.new Obj();
                obj._value = new_v;
                obj._color = new_c;
                obj.prob = 1;

                row.addView(createTile(obj, onBoard));
                row.addView(createTile(null, onBoard));

                int position = row.getChildCount() / 2 * tileWidth + (row.getChildCount() / 2 + 1) * tileSpace;
//                for (int i = 0; i < row.getChildCount(); i++) {
//                    position += row.getChildAt(i).getWidth();
//                }
                YoloV5Ncnn.Obj plus = yolov5ncnn.new Obj();
                plus.addTile = true;
                if (position + tileWidth >= view.getWidth()) {
                    row = createRow();
                    row.addView(createTile(null, onBoard));
                    row.addView(createTile(plus, onBoard));
                    view.addView(row);
                }
                else {
                    row.addView(createTile(plus, onBoard));
                }
            } else {
                anchorView.setLayoutParams(new TableRow.LayoutParams(tileWidth, tileWidth));
                anchorView.setBackground(getDrawable(R.drawable.plus));
                anchorView.setText("+");
                anchorView.setTextColor(Color.GREEN);
            }
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
            return getBoard();
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

    private YoloV5Ncnn.Obj[][] getBoard() {
        ArrayList<YoloV5Ncnn.Obj[]> solution_ = new ArrayList<>();
        for (int val = 0; val < solver.board.length; val++) {
            for (int col = 0; col < solver.board[val].length; col++) {
                for (int i = 0; i < solver.board[val][col]; i++) {
                    YoloV5Ncnn.Obj obj = yolov5ncnn.new Obj();
                    obj._color = col;
                    obj._value = val;
                    solution_.add(new YoloV5Ncnn.Obj[]{obj});
                }
            }
        }
        return solution_.toArray(new YoloV5Ncnn.Obj[solution_.size()][1]);
    }

    private void scan(boolean onBoard) {
        Intent takePicture = new Intent(MediaStore.ACTION_IMAGE_CAPTURE);
        if (takePicture.resolveActivity(getPackageManager()) != null) {
            File photoFile;
            try {
                photoFile = createImageFile();
            } catch (IOException ex) {
                ex.printStackTrace();
                return;
            }
            lastUri = FileProvider.getUriForFile(this,
                    "com.tencent.yolov5ncnn.fileprovider",
                    photoFile);
            takePicture.putExtra(MediaStore.EXTRA_OUTPUT, lastUri);
            takePicture.putExtra("android.intent.extra.quickCapture",true);
            startActivityForResult(takePicture, onBoard ? BOARD_SCAN : HAND_SCAN);
        }
    }

    private File createImageFile() throws IOException {
        // Create an image file name
        String timeStamp = new SimpleDateFormat("yyyyMMdd_HHmmss").format(new Date());
        String imageFileName = "JPEG_" + timeStamp + "_";
        File storageDir = getExternalFilesDir(Environment.DIRECTORY_PICTURES);

        return File.createTempFile(
                imageFileName,  /* prefix */
                ".jpg",         /* suffix */
                storageDir      /* directory */
        );
    }

    private LinearLayout createRow()
    {
        LinearLayout row = new LinearLayout(this);
        row.setOrientation(LinearLayout.HORIZONTAL);
        row.setPadding(0,tileHeight/20,0,tileHeight/20);
        TableRow.LayoutParams rowParams = new TableRow.LayoutParams();
        row.setLayoutParams(rowParams);
        return row;
    }

    private TextView createTile(YoloV5Ncnn.Obj obj, boolean boardTile)
    {
        if (obj == null) {
            TextView button = new TextView(this);
            button.setLayoutParams(new TableRow.LayoutParams(tileSpace, tileHeight));
            return button;
        }
        TextView button = new TextView(this);
        if (obj.fromHand) {
            button.setLayoutParams(new TableRow.LayoutParams(tileWidth, tileHeight));
            button.setGravity(Gravity.CENTER);
            button.setText(values[obj._value]);
            button.setTextColor(colors[obj._color]);
            button.setBackground(getDrawable(R.drawable.broder_green));
        } else if (obj.addTile) {
            button.setLayoutParams(new TableRow.LayoutParams(tileWidth, tileWidth));
            button.setGravity(Gravity.CENTER);
            button.setClickable(true);
            button.setOnClickListener(v->addTileManually(button, boardTile));
            button.setText("+");
            button.setTextColor(Color.GREEN);
            button.setBackground(getDrawable(R.drawable.plus));
        } else {
            button.setLayoutParams(new TableRow.LayoutParams(tileWidth, tileHeight));
            button.setGravity(Gravity.CENTER);
            button.setClickable(true);
            button.setOnClickListener(v->changeTile(button, boardTile));


            button.setBackground(getDrawable(R.drawable.broder_gray));
            button.setText(values[obj._value]);
            button.setTextColor(colors[obj._color]);
            button.setOnLongClickListener((view) -> {
                TextView but = (TextView) view;
                int v = Arrays.asList(values).indexOf(but.getText());
                int c = Arrays.asList(colors).indexOf(but.getCurrentTextColor());
                if (boardTile)
                    solver.board[v][c]--;
                else
                    solver.hand[v][c]--;

                ViewGroup parentView = (ViewGroup) view.getParent();
                parentView.removeView(view);
                return true;
            });

        }

        return button;
    }

    private void showObjects(YoloV5Ncnn.Obj[][] objects, boolean onBoard)
    {
        if (objects == null) return;

        LinearLayout view = onBoard ? boardView : handView;
        view.removeAllViews();

        LinearLayout row = createRow();
        row.setGravity(Gravity.CENTER);
        row.addView(createTile(null, onBoard));

        int position = tileSpace;
        for (YoloV5Ncnn.Obj[] objRow : objects) {
            if (objRow == null) continue;
            if (position + objRow.length * tileWidth > view.getWidth()) {
                view.addView(row);
                row = createRow();
                row.addView(createTile(null, onBoard));
                position = tileSpace;
            }
            for (int i = 0; i < objRow.length; i++) {
                if (objRow[i] == null) continue;
                row.addView(createTile(objRow[i], onBoard));
                position += tileWidth;
                if (i + 1 == objRow.length) {
                    row.addView(createTile(null, onBoard));
                    position += tileSpace;
                }
            }

        }
        if (position + tileWidth > view.getWidth()) {
            view.addView(row);
            row = createRow();
            row.addView(createTile(null, onBoard));
        }
        YoloV5Ncnn.Obj plus = yolov5ncnn.new Obj();
        plus.addTile = true;
        row.addView(createTile(plus, onBoard));
        view.addView(row);
    }

    @Override
    protected void onActivityResult(int requestCode, int resultCode, Intent data)
    {
        super.onActivityResult(requestCode, resultCode, data);

        if (resultCode == RESULT_OK) {
            try
            {
                Uri selectedImage = lastUri;
                if (selectedImage == null)
                    return;
                Bitmap bitmap = decodeUri(selectedImage);

                if (bitmap == null)
                    return;

                long startTime = System.nanoTime();
                YoloV5Ncnn.Obj[] objects = yolov5ncnn.Detect(bitmap, false);
                System.out.println("Detect time: " + (System.nanoTime() - startTime));
                long startTime2 = System.nanoTime();
                for (YoloV5Ncnn.Obj object : objects) {
                    classifier.predict(bitmap, object);
                }
                System.out.println("Class time: " + (System.nanoTime() - startTime2));
                System.out.println("Class time avg: " + (System.nanoTime() - startTime2)/objects.length);
                System.out.println("Total time: " + (System.nanoTime() - startTime));

                sort(objects, (a, b)->{
                    if (a._value == b._value) return Integer.compare(a._color, b._color);
                    return Integer.compare(a._value, b._value);
                });
                YoloV5Ncnn.Obj[][] objects_ = new YoloV5Ncnn.Obj[objects.length][1];
                for (int i = 0; i < objects.length; i++) {
                    objects_[i][0] = objects[i].prob >= probThresh ? objects[i] : null;
                }

                showObjects(objects_, requestCode==BOARD_SCAN);
                solver.setTiles(objects, requestCode==BOARD_SCAN);
                YoloV5Ncnn.Obj[][] solution = solve();
                showObjects(solution, true);
            }
            catch (FileNotFoundException e)
            {
                Log.e("MainActivity", "FileNotFoundException");
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
