# aaltoasr-commands
Easy to use command line tools for the AaltoASR system
==================================

Dependencies
-------------------------
AaltoASR, pre-trained acoustic and language models
sox
lavennin
python2.7


Recognition
-------------------------
Basic recognition example of short audio file: output to text file
```bash
./aaltoasr-rec -o asr_output.txt test.wav
```

Recognition example of long audio file: audio is split in 10 second segments and multiprocessor system (if available) is used for recognition, output also into TextGrid and ELAN formats.
```
./aaltoasr-rec --split 10 -n 4 -r --output asr_output.txt --tg asr_output.tg -E asr_output.eaf --mode trans,segword test.wav
```

Supervised in-document adaptation
-------------------------
Recognition models are adapted based on user edited part (60 seconds) of previous ASR output. New output saved into same file.
```
./aaltoasr-rec -C asr_output_edited.eaf -c 60 test.wav
```

Alignment
-------------------------
Align text with audio file.
```
./aaltoasr-align -t test.txt test.wav
```

Output is in the following format [start-in-seconds end-in-seconds word]:
```
0.104 1.048 norjan
1.088 1.568 poliisi
1.576 1.696 on
1.704 2.256 julkaissut
..
..
```
