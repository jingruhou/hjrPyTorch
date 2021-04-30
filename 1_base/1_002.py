# coding: utf-8
import torch
import torchaudio

"""
    @Time    : 2021/4/27 14:23
    @Author  : houjingru@semptian.com
    @FileName: 1_002.py
    @Software: PyCharm
"""
print(torch.__version__)

print(torchaudio.__version__)
# torchaudio.datasets.YESNO(
#     root,
#     url="http://www.openslr.org/resources/1/waves_yesno.tar.gz",
#     folder_in_archive='waves_yesno',
#     download=False,
#     transform=None,
#     target_transform = None)

# A data point in Yesno is a tuple (waveform, sample_rate, labels) where labels
# is a list of integers with 1 for yes and 0 for no.
yesno_data = torchaudio.datasets.YESNO('./', download=True)

# Pick data point number 3 to see an example of the the yesno_data:
n = 3
waveform, sample_rate, labels = yesno_data[n]
print("Waveform: {}\nSample rate: {}\nLabels: {}".format(waveform, sample_rate, labels))