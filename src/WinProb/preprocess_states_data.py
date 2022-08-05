"""Package the data into test and train folders for easy retrieval"""
import os
import random
import shutil

root_dir = '../state_tensors/states'
test_dir = '../state_tensors/test'

percent = 0.2  # the percentage of states to be moved to the test folder

win_dir = os.path.join(os.path.abspath(root_dir), 'win')
lose_dir = os.path.join(os.path.abspath(root_dir), 'lose')

test_path_win = os.path.join(os.path.abspath(test_dir), 'win')  # win directory in the test folder
test_path_lose = os.path.join(os.path.abspath(test_dir), 'lose')

win_dir_list = os.listdir(win_dir)
lose_dir_list = os.listdir(lose_dir)

win_shift = int(len(win_dir_list) * percent)
lose_shift = int(len(lose_dir_list) * percent)

random.shuffle(win_dir_list)
random.shuffle(lose_dir_list)

test_win = win_dir_list[:win_shift]
test_lose = lose_dir_list[:lose_shift]

# move test_win and test_lose to the test directory
for f in test_win:
    shutil.move(os.path.join(win_dir, f), test_path_win)

for f in test_lose:
    shutil.move(os.path.join(lose_dir, f), test_path_lose)
