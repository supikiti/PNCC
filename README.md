# PNCC implementation in Python
Based on 
- http://www.cs.cmu.edu/~robust/Papers/OnlinePNCC_V25.pdf
- https://www.eurasip.org/Proceedings/Eusipco/Eusipco2015/papers/1570104069.pdf

put a slide about PNCC
- https://www.slideshare.net/ssuser3f97dd/dl-hucks-pncc

## Installation:
Clone and install requirements.
```bash
cd ~
git clone https://github.com/supikiti/PNCC.git
cd PNCC
pip install -r requirements.txt
```

# PNCC features
If you want to change the PNCC parameters, the following parameters are supported.
```python
def pncc(signal=audio_wave, n_fft=512, sr=16000, winlen=0.020, winstep=0.010,
         n_mels=128, n_pncc=13):
```
|Parameter|Description|
|---|---|
|signal|the audio signal from which to compute features. Should be an (N, 1) array|
|n_fft|the FFT size. Default is 512.|
|sr|the samplerate of the signal we are working with.|
|winlen|the length of the analysis window in seconds. Default is 0.020s.(25 milliseconds)|
|winstep|the step between successive windows in seconds. Default is 0.010.(10 milliseconds)|
|n_mels|the number of filters in the filterbank, default 128.|
|n_pncc|the number of cepstrum to return, default 13.|

## Licence

[MIT](https://github.com/tcnksm/tool/blob/master/LICENCE)

## Author

[supikiti](https://github.com/supikiti)
