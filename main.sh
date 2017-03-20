#!/bin/sh

#CNN-rand
./rand_1.py > rand_result_1.txt
./rand_2.py > rand_result_2.txt
./rand_vader.py > rand_vader_result.txt

#CNN-w2v
./w2v_1.py > w2v_result_1.txt
./w2v_2.py > w2v_result_2.txt
./w2v_vader.py > w2v_vader_result.txt
