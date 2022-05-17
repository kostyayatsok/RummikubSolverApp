package com.tencent.yolov5ncnn;

import java.util.HashMap;
import java.util.Map;
import java.util.ArrayList;
import java.util.Arrays;

import java.util.function.IntBinaryOperator;
import java.util.stream.IntStream;
import java.util.Random;

public class SolverNew {

    public static class State {

        public int[] ln;
        public int[] bitmap;
        public int[] grps;
        State() { ln = new int[9]; bitmap = new int[8]; grps = new int[14];  }

        State(State rhs) { ln = rhs.ln.clone(); bitmap = rhs.bitmap.clone(); grps = rhs.grps.clone(); hash_internal = rhs.hash_internal; }
        int ord(int v1, int v2) { return (v1 < v2) ?  v1*10+v2 : v2*10+v1; }
        public boolean equals(Object o) { return o.hashCode() == this.hashCode(); }

        int hash_internal;
        public int updateHash() {
            hash_internal = ln[0] * 100000000 + ord(ln[1],ln[2])*1000000+ ord(ln[3],ln[4])*10000+ ord(ln[5],ln[6])*100 + ord(ln[7],ln[8]);
            return hash_internal;
        }

        public int hashCode() {
            return hash_internal;
        }


        void update_states(int[] q, int bit){
            for (int cl = 0; cl < 4; cl++) {
                for (int i = 0; i < 2; i++) {
                    if(ln[cl*2 + i + 1] == 3){
                        if(q[cl] != 0) {
                            bitmap[cl*2 + i] |= (1<<bit);
                            q[cl]--;
                        }
                        else
                            ln[cl*2 + i + 1] = 0;
                    }

                    if(ln[cl*2 + i + 1] < 3 && ln[cl*2 + i + 1] > 0) {
                        ln[cl*2 + i + 1]++;
                        bitmap[cl*2 + i] |= (1<<bit);
                    }
                }

                for (int i = 0; i < 2; i++) {
                    if(ln[cl*2 + i + 1] == 0 && q[cl] != 0){
                        ln[cl*2 + i + 1] = 1;
                        bitmap[cl*2 + i ] |= (1<<bit);
                        q[cl]--;
                    }
                }
            }
        }
    }

    public static  class Value {

        public Value(int _points, State _prev_state) {
            points = _points;
            prev_state = _prev_state;
        }
        public int points;
        public State prev_state;
    }

    public static int calc_hand_board(int[][] _hand, int[][] _board, int[][] hand, int[][] board) {
        for (int i = 0; i < 4; i++) {
            for (int j = 0; j < _hand[i].length; j++) {
                hand[_hand[i][j]][i] ++;
            }
        }
        for (int i = 0; i < 4; i++) {
            for (int j = 0; j < _board[i].length; j++) {
                board[_board[i][j]][i] ++;
            }
        }
        int dj = 0;
        for (int i = 0; i < 4; i++) {
            dj += board[0][i] + hand[0][i];
        }
        return dj;
    }

    public static boolean is_state_better(State st, State rhs)
    {
        if(st.ln[0] > rhs.ln[0]) return false;
        for (int i = 1; i <= 8; i++)
            if(st.ln[i] < rhs.ln[i] || (st.ln[i] != 0 && rhs.ln[i] == 0)) return false;
        return true;
    }

    public static boolean is_there_state_better(Map<State, Value> states, State cur, int points) {
        cur = new State(cur);
        if(cur.ln[0] > 0) {
            cur.ln[0]--;
            cur.updateHash();
            Value bs = states.get(cur);
            if(bs != null && bs.points >= points)
                return true;
            cur.ln[0]++;
        }


        for (int i = 1; i <= 8; i++) {
            if(cur.ln[i] < 3 && cur.ln[i] > 0) {
                cur.ln[i]++;
                cur.updateHash();
                Value bs = states.get(cur);
                if(bs != null && bs.points >= points)
                    return true;
                cur.ln[i]--;
            }
        }
        return false;

        /*for(Map.Entry<State, Value> entry : states.entrySet()) {
            if(entry.getValue().points < points)
                continue;
            if(entry.getKey().hashCode() == cur.hashCode())
                continue;
            if(is_state_better(entry.getKey(), cur))
                return true;
        }
        return false;*/
    }

    public static int restore_from_state(State st, int[][] hand, int[][] board, int num_jockers) {
        String[] cols = {"K","O","C","Ч","J","J"};
        ArrayList<String> res = new ArrayList<String>();
        int add_pnts = 0;
        num_jockers -= st.ln[0];
        if (num_jockers != 0){
            for (int i = 0; i < 4 && num_jockers > 0; i++) {
                for(int j = 5; j <=13 && num_jockers > 0; j++) {
                    int fish = board[j][i]+hand[j][i];
                    int vist = ((st.bitmap[2*i]&(1<<j)) != 0?1:0)
                            + ((st.bitmap[2*i+1]&(1<<j)) != 0?1:0)
                            + ((st.grps[j]&(1<<i)) != 0?1:0)
                            + ((st.grps[j]&(1<<(i+8))) != 0?1:0)
                            + ((st.grps[j]&(1<<(i+16))) != 0?1:0)
                            + ((st.grps[j]&(1<<(i+24))) != 0?1:0);
                    if(vist < fish)  {
                        if(
                                (st.bitmap[2*i]&(1<<j)) == 0
                                        &&
                                        (st.bitmap[2*i]&(1<<(j-1))) == 0
                                        &&
                                        (st.bitmap[2*i]&(1<<(j-2))) != 0
                        )                        {
                            num_jockers--;
                            st.bitmap[2*i] |= (1<<j);
                            st.bitmap[2*i] |= (1<<j-1);
                            add_pnts++;
                        }

                    }
                }
            }
        }


        for (int i = 1; i < 13; i++) {
            for (int j = 0; j < 3; j++) {
                int cr = (st.grps[i]>>(j*8))&255;
                if(cr != 0) {
                    String cs = new String(Integer.toString(i));
                    for (int k = 0; k < 6; k++) {
                        if( (cr&(1<<k)) != 0) {
                            cs += "_" + cols[k];
                            if(board[i][j] != 0)
                                board[i][j]--;
                            else if(hand[i][j] != 0)
                                hand[i][j]--;
                        }
                    }
                    res.add(cs);
                }
            }
        }

        for (int i = 0; i < 8; i++) {
            int len = 0;
            String cs = new String(cols[i/2]);
            for(int j = 1; j <=13; j++) {
                if((st.bitmap[i]&(1<<j))!=0) {
                    if(board[j][i/2] != 0) {
                        board[j][i/2]--;
                        cs += "_"+Integer.toString(j);
                    }
                    else if(hand[j][i/2] != 0) {
                        hand[j][i/2]--;
                        cs += "_"+Integer.toString(j);
                    }
                    else {
                        cs += "_J";
                    }
                    len++;
                }
                else if(len != 0)
                {
                    res.add(cs);
                    len = 0;
                    cs = new String(cols[i/2]);
                }
            }
            if(len != 0 && len < 3)
            {
                // Shoudnot be there - resolved on previous step
                cs += "_J";
                st.bitmap[i] |= (1<<12);
                st.bitmap[i] |= (1<<11);
            }
            if(len != 0)
                res.add(cs);
        }
        for (String it : res) {
            System.out.println(it);
        }
        return add_pnts;
    }
    public static int solve(int[][] _hand, int[][] _board) {

        int[][] hand = new int[14][4];
        int[][] board = new int[14][4];
        int num_jockers = calc_hand_board(_hand, _board, hand, board);

        Map<State, Value>[] states = (Map<State, Value>[])new Map[14];
        states[0] = new HashMap<>();
        states[0].put(new State(),new Value(0,null));

        for (int i = 1; i < states.length; i++) {
            states[i] = new HashMap<State, Value>();

            int[] hnd = hand[i];
            int[] brd = board[i];
            int [] sum = IntStream.range(0, 4).map(index -> hnd[index]+brd[index]).toArray();


            for(Map.Entry<State, Value> entry : states[i-1].entrySet()) {
                int [] sl = sum.clone();
                State st = entry.getKey();
                Value vl = entry.getValue();
                int udj = st.ln[0];

                // find is there better config
                if(is_there_state_better(states[i-1],st,vl.points)) continue;


                Boolean bValid = true;
                for (int j = 1; j <= 8 && bValid; j++) {
                    int cl = (j-1)/2;
                    if(st.ln[j] < 3 && st.ln[j] > 0) {
                        if(sl[cl] == 0){
                            if(num_jockers <= udj)
                                bValid = false;
                            else
                                udj++;
                        }
                        else
                            sl[cl]--;
                    }
                }
                if(!bValid)
                    continue;

                int[] c = sl.clone();
                while (true) {
                    State ns = new State(st);
                    ns.update_states(c.clone(),i);
                    for(int nj = 0; nj <= num_jockers-udj; nj++) {
                        int [] r = IntStream.range(0, 4).map(index -> sl[index]-c[index]).toArray();
                        //long notz1 = Arrays.stream(r).filter(v -> v == 1).count();
                        //long notz2 = Arrays.stream(r).filter(v -> v == 2).count();

                        //    0 1 2 3 4 5 6
                        // 0  0 0 0 3 4 4 6
                        // 1  0 0 3 4 6 7
                        // 2  0 3 6 7 8
                        // 3  6 7 8 9 *
                        // 4  8 9 10* *

                        int n4 = 0, cj = nj;
                        int[] grps = new int[6];
                        int ngr = 0;
                        while(Arrays.stream(r).filter(v -> v >= 1).count() + cj + n4 >= 3) {
                            int count = 0;
                            for (int j = 0; j < 4; j++) {
                                if(r[j] > 0) {
                                    r[j]--;
                                    count++;
                                    grps[ngr] |= (1<<j);
                                }
                            }
                            while(count < 3 && n4 > 0) {
                                count ++;
                                n4--;
                                for (int j = 0; j < 4; j++)
                                    if( (grps[ngr]&(1<<j)) == 0) {
                                        grps[ngr] |= (1<<j);
                                        grps[0] &= ~(1<<j);
                                        break;
                                    }
                            }
                            int sh = 0;
                            while(count < 3 && cj > 0) {
                                count ++;
                                cj--;
                                sh++;
                                grps[ngr] |= (1<<(4+sh));
                            }
                            if(count == 4)
                                n4++;
                            ngr++;
                        }

                        bValid = true;
                        int vist = vl.points;
                        for (int j = 0; j < 4 && bValid; j++) {
                            if(r[j] > hand[i][j])
                                bValid = false;
                            vist += sum[j] - r[j];
                        }
                        if(!bValid)
                            continue;

                        State nsj = new State(ns);
                        nsj.ln[0] = udj + nj - cj;
                        nsj.grps[i] = grps[0]+(grps[1]<<8)+(grps[2]<<16)+(grps[3]<<24);
                        nsj.updateHash();
                        if(states[i].get(nsj) == null || states[i].get(nsj).points < vist){
                            states[i].remove(nsj);
                            states[i].put(nsj, new Value(vist, st));
                        }
                        if(Arrays.stream(r).filter(v -> v >= 1).count() == 0)
                            break;
                    }
                    for (int j = 0; j < 4; j++) {
                        c[j]--;
                        if(c[j] >= 0) break;
                        else if(j < 3)
                            c[j] = sl[j];
                    }
                    if(c[3] < 0)  break;
                }
            }
        }

        State best = null;
        int best_points = 0;
        for(Map.Entry<State, Value> entry : states[13].entrySet()) {

            State st = entry.getKey();
            Value vl = entry.getValue();
            if(vl.points*4 - st.ln[0] <= best_points)
                continue;
            int free_jockers = num_jockers - st.ln[0];
            Boolean bValid = true;
            for (int j = 1; j <= 8 && bValid; j++) {
                if(st.ln[j] > 0) {
                    if(st.ln[j] < 3 - free_jockers) bValid = false;
                    else
                    {
                        st.bitmap[j-1] |= (1<<12);
                        st.bitmap[j-1] |= (1<<13);
                        free_jockers -= 3 - st.ln[j];
                        st.ln[0] += 3 - st.ln[j];
                    }
                }
            }

            if(bValid && vl.points*4 - (num_jockers - free_jockers) > best_points)
            {
                best_points = vl.points*4 - st.ln[0];
                best = st;
            }
        }
        best_points += 4*restore_from_state(best, hand, board, num_jockers);
        System.out.println("Points: " + (best_points+3)/4);
        return (best_points+3)/4;
    }

    final static Random random = new Random();
    static int fill_random_hand(int[][] hand) {
        int num3l = random.nextInt(10)+3;
        int num3r = random.nextInt(10)+3;
        int numj = random.nextInt(3);

        ArrayList<Integer> [] arrays = new ArrayList[] { new ArrayList<Integer>(),new ArrayList<Integer>(),new ArrayList<Integer>(),new ArrayList<Integer>()};

        for (int i = 0; i < num3l; i++) {
            int num1 = random.nextInt(10)+1;
            int col = random.nextInt(4);
            for (int j = 0; j < 3; j++)
                arrays[col].add(j+num1);
        }

        for (int i = 0; i < num3r; i++) {
            int num1 = random.nextInt(13)+1;
            int col = random.nextInt(4);
            for (int j = 0; j < 3; j++)
                arrays[(col+j)%4].add(num1);
        }
        for (int i = 0; i < numj; i++) {
            int col = random.nextInt(4);
            int ind = random.nextInt(arrays[col].size());
            if(0 == (int)arrays[col].get(ind))
                i--;
            else
                arrays[col].set(ind,0);
        }
        for (int i = 0; i < 4; i++) {
            hand[i] = new int[arrays[i].size()];
            for (int j = 0; j < arrays[i].size(); j++)
                hand[i][j] = (int)arrays[i].get(j);
        }
        return num3l*3+num3r*3-numj;
    }


    public static void main(String args[]) {
        int[][] hand = new int[][] {
                new int[] { 1,2,4,5,6,7,8,9,10,11,12,13,1,2,4,5,6,7,8,9,10,11,12,13,0,0 },
                new int[] { 1,2,4,5,6,7,8,9,10,11,12,13,1,2,3,4,5,6,7,8,9,10,11,12,13 },
                new int[] { 1,2,3,4,5,6,7,8,9,10,11,12,13,1,2,3,4,5,6,7,8,9,10,11,12,13 },
                new int[] { 1,2,3,4,5,6,7,8,9,10,11,12,13,1,2,3,4,5,6,7,8,9,10,11,12,13 },
                //new int[] { 12,13,0 },
                new int[] {  },
                new int[] {  },
                new int[] {  },
                new int[] { },
        };

        int[][] board = new int[][] {
                new int[] {  },
                new int[] {  },
                new int[] {  },
                new int[] {  },
        };

        solve(hand, board);

        for(int i = 0; i<100; i++)
        {
            int score1 = fill_random_hand(hand);
            int score2 = solve(hand, board);
            if(score1 != score2)
            {
                int score4 = solve(hand, board);
                System.out.println("Socres: " + score1 + " " + score2);

            }
        }
    }
}
