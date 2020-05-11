import numpy as np
import cv2
from scipy.signal import medfilt
import pathlib as plb
import ShapeUtils as SU
from SysUtils import make_dir, get_items


def visualize_one_audio_seq(visualizer,lf,lr,limg):
    ar_dir = plb.Path("../speech_dir/test_output_single8")
    try:
        ar_dir.mkdir()
    except FileExistsError:
        plb.Path(ar_dir).rmdir()
    for i in range(122):
        ef=lf[i]
        er=lr[i]
        img=limg[i]
        ret = visualizer.visualize(img, er, ef)
        # draw plot
        plot = SU.draw_error_bar_plot(er, ef, (ret.shape[1],200))
        ret = np.concatenate([ret, plot], axis=0)
        save_path = str(ar_dir) + "/result{:06d}.jpg".format(i)
        cv2.imwrite(save_path, ret)
        # can call cv2.imshow() here

def process_all(actor,file):
    visualizer = SU.Visualizer()
    path='../speech_dir/npy_output/'+ actor + "/" + file
    lf= np.load(path+'/fake.npy')
    lr =  np.load(path+'/real.npy')
    limg=np.load(path+'/img.npy')
    visualize_one_audio_seq(visualizer,lf,lr,limg)
    
    
process_all("Actor","01-02-03-01-02-02-24")    
    