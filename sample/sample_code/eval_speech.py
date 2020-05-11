import numpy as np
import cntk as C
import cv2
from scipy.signal import medfilt
from SysUtils import make_dir, get_items


def load_image(path):
    img = cv2.imread(path, 1)
    if img.shape[0] != 100 or img.shape[1] != 100:
        img = cv2.resize(img, (100, 100), interpolation=cv2.INTER_CUBIC)
    return img

def load_image_stack(paths):
    imgs = [load_image(path) for path in paths]
    return np.stack(imgs)


def load_exp_sequence(path, use_medfilt=False, ksize=3):
    exp = np.load(path).astype(np.float32)
    if use_medfilt:
        exp = medfilt(exp, kernel_size=(ksize,1)).astype(np.float32)
    return exp


# estimate audio sequence -> exp
def is_recurrent(model):
    names = ["RNN", "rnn", "LSTM", "LSTM1", "GRU", "gru", "fwd_rnn", "bwd_rnn"]
    isrnn = False
    for name in names:
        if model.find_by_name(name) is not None:
            isrnn = True
    return isrnn


def estimate_one_audio_seq(model, audio_seq, small_mem=False):
    if isinstance(model, str):
        model = C.load_model(model)
    # set up 2 cases: if the model is recurrent or static
    if is_recurrent(model):
        n = audio_seq.shape[0]
        NNN = 125
        if n > NNN and small_mem:
            nseqs = n//NNN + 1
            indices = []
            for i in range(nseqs-1):
                indices.append(NNN*i + NNN)
            input_seqs = np.vsplit(audio_seq, indices)
            outputs = []
            for seq in input_seqs:
                output = model.eval({model.arguments[0]:[seq]})[0]
                outputs.append(output)
            output = np.concatenate(outputs)
        else:
            output = model.eval({model.arguments[0]:[audio_seq]})[0]
    else:
        output = model.eval({model.arguments[0]: audio_seq})
    return output



def visualize_one_audio_seq(model, video_frame_list, audio_csv_file, exp_npy_file, save_dir):
    if isinstance(model, str):
        model = C.load_model(model)
    # evaluate model with given audio data
    audio = np.loadtxt(audio_csv_file, dtype=np.float32, delimiter=",")
    audio_seq = np.reshape(audio, (audio.shape[0], 1, 128, 32))
    e_fake = estimate_one_audio_seq(model, audio_seq)
    if e_fake.shape[1] != 46:
        if e_fake.shape[1] == 49:
            e_fake = e_fake[:,3:]
        else:
            raise ValueError("unsupported output of audio model")
    # load true labels with optional median filter to smooth it (not used in training)
    e_real = load_exp_sequence(exp_npy_file, use_medfilt=True)

    if e_real.shape[0] != e_fake.shape[0]:
        raise ValueError("number of true labels and number of outputs do not match")
    
    # create directory to store output frames
    if video_frame_list:
        video = load_image_stack(video_frame_list)
        if video.shape[0] != e_real.shape[0]:
            print("number of frames and number of labels do not match. Not using video")

    # make folder to store generated frames
    make_dir(save_dir)

    lf=[]
    lr=[]
    limg=[]
    n = e_real.shape[0]
    for i in range(0,n):
        if video is not None:
            img = video[i,:,:,:]
        else:
            img = None # not include input video in the output  
        ef = e_fake[i,:]
        er = e_real[i,:]
        lf.append(ef)
        lr.append(er)
        limg.append(img)
        print(type(lf),type(lr),type(img))
        
    np.save('../speech_dir/fake.npy',np.array(lf,dtype=np.float32))
    np.save('../speech_dir/real.npy',np.array(lr,dtype=np.float32))
    np.save('../speech_dir/img.npy',np.array(limg,dtype=np.float32))
    


def test_one_seq():
    # directory to store output video. It will be created if it doesn't exist
    save_dir = "../speech_dir/test_output_single"
    model_file = "../speech_dir/model_audio2exp_2019-03-24-21-03/model_audio2exp_2019-03-24-21-03.dnn"
    # video directory holding separate frames of the video. Each image should be square.
    video_dir = "../FrontalFaceData"
    # spectrogram sequence is stored in a .csv file
    audio_file = "../speech_dir/RAVDESS_feat/Actor_21/01-01-07-02-01-01-21/dbspectrogram.csv"
    # AU labels are stored in an .npy file
    exp_file = "../ExpLabels/RAVDESS/Actor_21/01-01-07-02-01-01-21.npy"

    video_list = get_items(video_dir, "full") # set to None if video_dir does not exist
    model = C.load_model(model_file)

    visualize_one_audio_seq(model, video_list, audio_file, exp_file, save_dir)

if __name__ == "__main__":
    test_one_seq()
    