package com.tencent.yolov5ncnn;

import static java.util.Arrays.fill;
import static java.util.Arrays.sort;

import java.lang.reflect.Array;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collection;
import java.util.Collections;
import java.util.Comparator;
import java.util.HashSet;
import java.util.List;

public class Solver {

    int INF = (int) 1e9;

    int N = 14; // # values
    int K = 4;  // # colors
    int M = 2;  // # copies
    int S = 3;  // # min len

    // n×k×f(M) multi-dimensional array which contains the maximum
    // score that can be obtained given this state of the puzzle
    int[][] scores;
    int[][] scoresMemory;

    int[][] hand = new int[N][K];
    int[][] board = new int[N][K];

    int[][] allRuns;
    int maxHash;

    void setTiles(YoloV5Ncnn.Obj[] objects, boolean onBoard)
    {
        int[][] tiles;
        if (onBoard)
            tiles = board;
        else
            tiles = hand;

        Arrays.stream(tiles).forEach(a -> Arrays.fill(a, 0));
        for (int i = 0; i < objects.length; i++)
        {
            if (objects[i]._value == 13) continue;
            tiles[objects[i]._value][objects[i]._color]++;
        }
    }

    // 4 * K
    int hashRun(int[] run)
    {
        int hash = 0;
        int pow = 1;
        sort(run);
        for (int i = 0; i < run.length; i++)
        {
            hash += pow * run[i];
            pow *= (S + 1);
        }
        return hash;
    }

    // 4 * K
    int[] unhashRun(int hash)
    {
        int[] runs = new int[M];
        for (int i = 0; i < runs.length; i++)
        {
            runs[i] = hash % (S + 1);
            hash /= (S + 1);
        }
        return runs;
    }

    int combineHashes(int[] hashes)
    {
        int pow = 1;
        int result = 0;
        for (int hash : hashes)
        {
            result += pow * hash;
            pow *= maxHash;
        }
        return result;
    }

    int[] splitHashes(int totalHash)
    {
        int[] hashes = new int[K];
        for (int i = 0; i < K; i++)
        {
            hashes[i] = totalHash % maxHash;
            totalHash /= maxHash;
        }
        return hashes;
    }

    public Solver()
    {
        maxHash = 1;
        for (int i = 0; i < M; i++)
        {
            maxHash *= (S + 1);
        }
        allRuns = new int[maxHash][];
        for (int hash = 0; hash < maxHash; hash++)
        {
            allRuns[hash] = unhashRun(hash);
        }
        int maxTotalHash = 1;
        for (int i = 0; i < K; i++)
        {
            maxTotalHash *= maxHash;
        }
        scores = new int[N][maxTotalHash];
        Arrays.stream(scores).forEach(a -> Arrays.fill(a, -INF));
        scoresMemory = new int[N][maxTotalHash];
    }


    private int makeGroups(int[] groups) {
        int rest = 0;
        int j;
        for (j = 0; j < M; j++)
        {
            int groupSize = 0;
            int[] usedGroups = new int[K];
            for (int i = 0; i < K; i++)
            {
                if (groups[i] - usedGroups[i] > 0)
                {
                    groupSize++;
                    usedGroups[i]++;
                }
            }
            if (groupSize + rest < S)
            {
                break;
            }
            rest += groupSize - S;
            for (int i = 0; i < K; i++)
            {
                groups[i] -= usedGroups[i];
            }
        }
        return j;
    }


    ArrayList<Pair<int[], Integer>> makeRuns(int[] runsHashes, int value)
    {
        int[][] runs = new int[runsHashes.length][];
        for (int i = 0; i < runsHashes.length; i++)
        {
            runs[i] = unhashRun(runsHashes[i]);
        }
        ArrayList<Pair<Integer, Integer>[]> newRunsCombinations = new ArrayList<>();
        for (int color = 0; color < K; color++)
        {
            HashSet<Pair<Integer, Integer>> newRuns = new HashSet<>();
            for (int[] newRun : allRuns)
            {
                boolean correct = true;
                int count = hand[value][color] + board[value][color];
                for (int i = 0; i < newRun.length && count >= 0 && correct; i++)
                {
                    if (newRun[i] == runs[color][i] + 1) // continue run
                    {
                        count--;
                    }
                    else if (runs[color][i] == 3 && newRun[i] == 3) // continue run with 3 or more tiles
                    {
                        count--;
                    }
                    else if (runs[color][i] == 3 && newRun[i] == 0) // finish run with 3 or more tiles
                    {
                        // not use tiles
                    }
                    else if (runs[color][i] == 0 && newRun[i] == 0) // do nothing with empty run
                    {
                        // not use tiles
                    }
                    else // else achive newRun from runs[color] is impossible
                    {
                        correct = false;
                    }
                }
                if (correct && count >= 0)
                {
                    newRuns.add(new Pair(hashRun(newRun), count));
                }
            }

            if (newRuns.isEmpty())
            {
                return new ArrayList<>();
            }

            if (color == 0)
            {
                for (Pair e : newRuns)
                {
                    newRunsCombinations.add(new Pair[K]);
                    newRunsCombinations.get(newRunsCombinations.size() - 1)[color] = e;
                }
                continue;
            }
            ArrayList<Pair<Integer, Integer>[]> newRunsCombinations_ = new ArrayList<>(); //ooof
            for (Pair[] combination : newRunsCombinations)
            {
                for (Pair e : newRuns)
                {
                    combination[color] = e;
                    newRunsCombinations_.add(combination.clone());
                }
            }
            newRunsCombinations = newRunsCombinations_; // oooof
        }
        ArrayList<Pair<int[], Integer>> result = new ArrayList<>(); // newRunsHashes / scores
        for (Pair<Integer, Integer>[] combination : newRunsCombinations)
        {
            int[] newRunsHashes = new int [K];
            int[] groups = new int [K];
            for (int i = 0; i < K; i++)
            {
                newRunsHashes[i] = combination[i].first;
                groups[i] = combination[i].second;
            }

            makeGroups(groups);
            boolean useAllBoard = true;
            int score = 0;
            for (int i = 0; i < K; i++)
            {
                int usedCnt = board[value][i] + hand[value][i] - groups[i];
                if (board[value][i] > usedCnt)
                {
                    useAllBoard = false;
                    break;
                }
                score += usedCnt; // * (value + 1);
            }
            if (useAllBoard)
            {
                result.add(new Pair(newRunsHashes, score));
            }
        }
        return result;
    }

    boolean checkRunsFinished(int[] runsHashes)
    {
        for (int runHash : runsHashes)
        {
            int[] run = unhashRun(runHash);
            for (int e : run)
            {
                if (e > 0 && e < S)
                {
                    return false;
                }
            }
        }
        return true;
    }

    int maxScore(int value, int[] runsHashes)
    {
        if (value == 0)
        {
            Arrays.stream(scores).forEach(a -> Arrays.fill(a, -INF));
            Arrays.stream(scoresMemory).forEach(a -> Arrays.fill(a, -INF));
        }
        else if (value >= N)
        {
            if (checkRunsFinished(runsHashes))
            {
                return 0;
            }
            else
            {
                return -INF;
            }
        }

        int totalRunsHash = combineHashes(runsHashes);
        if (scores[value][totalRunsHash] > -INF)
        {
            return scores[value][totalRunsHash];
        }

        ArrayList<Pair<int[], Integer>> newRunsHashes = makeRuns(runsHashes, value);
        for (int i = 0; i < newRunsHashes.size(); i++)
        {
            int curScore = newRunsHashes.get(i).second;
            int nextScore = maxScore(value+1, newRunsHashes.get(i).first);
            if (nextScore > -INF)
            {
                if (scores[value][totalRunsHash] < curScore+nextScore)
                {
                    scores[value][totalRunsHash] = curScore+nextScore;
                    scoresMemory[value][totalRunsHash] = combineHashes(newRunsHashes.get(i).first);
                }
            }

        }
        return scores[value][totalRunsHash];
    }

    ArrayList<ArrayList<Pair<Integer, Integer>>> restore()
    {
        ArrayList<ArrayList<Pair<Integer, Integer>>> rows = new ArrayList<>();

        Integer[][] runs = new Integer[K][M];
        Arrays.stream(runs).forEach(a -> Arrays.fill(a, 0));
        int totalRunsHash = 0;
        for (int value = 0; value < N; value++)
        {
            int[] groups = new int[K];
            totalRunsHash = scoresMemory[value][totalRunsHash];
            int[] runsHashes = splitHashes(totalRunsHash);
            for (int i = 0; i < K; i++)
            {
                int[] run = unhashRun(runsHashes[i]);

                int cnt = 0;
                for (int r : run)
                {
                    if (r > 0)
                    {
                        cnt++;
                    }
                }
                sort(runs[i], (a, b) -> a == 0 ? 1 : a < b ? -1 : a == b ? 0 : 1);
                for (int j = 0; j < cnt; j++)
                {
                    runs[i][j]++;
                    if (value == N-1)
                    {
                        int len_run = runs[i][j];
                        rows.add(new ArrayList<>(len_run));
                        for (int k = 0; k < len_run; k++)
                        {
                            rows.get(rows.size() - 1).add(new Pair(value - len_run + 1 + k, i));
                        }
                        runs[i][j] = 0;
                    }
                }
                for (int j = cnt; j < M && runs[i][j] > 0; j++)
                {
                    int len_run = runs[i][j];
                    rows.add(new ArrayList<>(len_run));
                    for (int k = 0; k < len_run; k++)
                    {
                        rows.get(rows.size() - 1).add(new Pair(value - len_run + k, i));
                    }
                    runs[i][j] = 0;
                }

                groups[i] = hand[value][i] + board[value][i] - cnt;
            }
            int n_groups = makeGroups(groups.clone());
            if (n_groups == 0)
                continue;
            for (int i = 0; i < n_groups; i++)
                rows.add(new ArrayList<>());
            for (int i = 0; i < K; i++)
            {
                Collections.sort(rows.subList(rows.size()-n_groups, rows.size()), Comparator.comparingInt(ArrayList::size));
                for (int j = 0; j < groups[i] && j < n_groups; j++)
                {
                    rows.get(rows.size() - 1 - j).add(new Pair(value, i));
                }
            }
        }

        return rows;
    }

    public class Pair<A, B> {
        private A first;
        private B second;

        public Pair(A first, B second) {
            super();
            this.first = first;
            this.second = second;
        }

        public int hashCode() {
            int hashFirst = first != null ? first.hashCode() : 0;
            int hashSecond = second != null ? second.hashCode() : 0;

            return (hashFirst + hashSecond) * hashSecond + hashFirst;
        }

        public boolean equals(Object other) {
            if (other instanceof Pair) {
                Pair otherPair = (Pair) other;
                return
                        ((  this.first == otherPair.first ||
                                ( this.first != null && otherPair.first != null &&
                                        this.first.equals(otherPair.first))) &&
                                (  this.second == otherPair.second ||
                                        ( this.second != null && otherPair.second != null &&
                                                this.second.equals(otherPair.second))) );
            }

            return false;
        }

        public String toString()
        {
            return "(" + first + ", " + second + ")";
        }

        public A getFirst() {
            return first;
        }

        public void setFirst(A first) {
            this.first = first;
        }

        public B getSecond() {
            return second;
        }

        public void setSecond(B second) {
            this.second = second;
        }
    }
}

