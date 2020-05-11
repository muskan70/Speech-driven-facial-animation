import cv2
import math
import pathlib as plb
import librosa
import csv
import numpy as np


def write_csv(filename, data):
    with open(filename, 'w', newline="") as csvfile:
        writer = csv.writer(csvfile)
        for arow in data:
            writer.writerow(arow)

def get_fps(videofile):
    cap = cv2.VideoCapture(videofile)
    fps = cap.get(cv2.CAP_PROP_FPS)
    nFrame = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return (nFrame, fps)

def extract_one_frame_data(data, curPosition, nFrameSize, nSamPerFrame):
    frameData = np.zeros(nFrameSize, dtype=np.float32)
    if curPosition < 0:
        startPos = -curPosition
        frameData[startPos:nFrameSize] = data[0:(curPosition+nFrameSize)]
    else:
        frameData[:] = data[curPosition:(curPosition+nFrameSize)]
    nextPos = curPosition + nSamPerFrame
    return (frameData, nextPos)
        

def extract_one_file(videofile, audiofile):
    print (" --- " + videofile)
    # get video FPS
    nFrames, fps = get_fps(videofile)
    # load audio
    data, sr = librosa.load(audiofile, sr=44100) 
    # number of audio samples per video frame
    nSamPerFrame = int(math.floor(float(sr) / fps))
    # number of samples per 0.025s
    n25sSam = int(math.ceil(float(sr) * 0.025))
    # number of sample per step
    nSamPerStep = 512 
    # number of steps per frame
    nStepsPerFrame = 3 
    nFrameSize = (nStepsPerFrame - 1) * nSamPerStep + n25sSam
    
    curPos = nSamPerFrame - nFrameSize
    mfccs = []
    melspecs = []
    chromas = []
    for f in range(0,nFrames):
        # extract features
        frameData, nextPos = extract_one_frame_data(data, curPos, nFrameSize, nSamPerFrame)
        curPos = nextPos
        S = librosa.feature.melspectrogram(frameData, sr, n_mels=128, hop_length=nSamPerStep)
        # 1st is log mel spectrogram
        log_S = librosa.amplitude_to_db(S)
        # 2nd is MFCC and its deltas
        mfcc = librosa.feature.mfcc(y=frameData, sr=sr, hop_length=nSamPerStep, n_mfcc=13)
        delta_mfcc = librosa.feature.delta(mfcc,mode='nearest')
        delta2_mfcc = librosa.feature.delta(delta_mfcc,mode='nearest')
        # 3rd is chroma
        chroma = librosa.feature.chroma_cqt(frameData, sr, hop_length=nSamPerStep)        

        full_mfcc = np.concatenate([mfcc[:,0:3].flatten(), delta_mfcc[:,0:3].flatten(), delta2_mfcc[:,0:3].flatten()])
        mfccs.append(full_mfcc.tolist())
        melspecs.append(log_S[:,0:3].flatten().tolist())
        chromas.append(chroma[:,0:3].flatten().tolist())

    return (mfccs, melspecs, chromas)

video_root = "../video"
audio_root = "../speech"
feat_root ="../feat_new"

def process_all():
    video_dir = plb.Path(video_root)
    feat_dir = plb.Path(feat_root)
    feat_dir.mkdir(parents=True, exist_ok=True)
    for actor in video_dir.iterdir():
        for video_file in actor.iterdir():
            if video_file.name[len(video_file.name)-4:] != '.mp4' or video_file.name[0:2] != '01':
                continue 
            ar_dir=  plb.Path(feat_root + "/" + actor.name)
            if not ar_dir.exists():
                try:
                    ar_dir.mkdir()
                except FileExistsError:
                    print ("Directory " + actor.name + "  existed")
                    
            seq_dir = plb.Path( feat_root + "/" + actor.name + "/" + video_file.stem )
            if seq_dir.exists():
                continue     
            try:
                seq_dir.mkdir()
            except FileExistsError:
                print ("feat Root Directory existed")
            except FileNotFoundError:
                print ("Parent directory not found")    
            video_path = str(video_file)
            audio_path = audio_root + "/" + actor.name + "/" + video_file.stem + ".wav"
            mfccs, melspecs, chromas = extract_one_file(video_path, audio_path)
            mfcc_path = feat_root + "/" + actor.name + "/" + video_file.stem + "/mfcc_2.csv"
            mel_path = feat_root + "/" + actor.name + "/" + video_file.stem + "/log_mel.csv"
            chroma_path = feat_root + "/" + actor.name + "/" + video_file.stem + "/chroma_cqt.csv"
            write_csv(mfcc_path, mfccs)
            write_csv(mel_path, melspecs)
            write_csv(chroma_path, chromas)

if __name__ == "__main__":
    process_all()
       
        
    