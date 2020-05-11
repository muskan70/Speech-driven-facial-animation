import numpy as np
import cv2
from scipy.signal import medfilt
import ShapeUtils as SU
from SysUtils import make_dir, get_items

def visualize_one_audio_seq(visualizer,lf,lr,limg):
    save_dir = "../speech_dir/test_output_single"
    make_dir(save_dir)
    for i in range(122):
        ef=lf[i]
        er=lr[i]
        img=limg[i]
        ret = visualizer.visualize(img, er, ef)
        # draw plot
        print("wow4")
        plot = SU.draw_error_bar_plot(er, ef, (ret.shape[1],200))
        ret = np.concatenate([ret, plot], axis=0)
        save_path = save_dir + "/result{:06d}.jpg".format(i)
        cv2.imwrite(save_path, ret)
        # can call cv2.imshow() here


if __name__ == "__main__":
    visualizer = SU.Visualizer()
    lf= np.load('../speech_dir/fake.npy')
    lr =  np.load('../speech_dir/real.npy')
    limg=np.load('../speech_dir/img.npy')
    print(type(lf),type(lr),type(limg))
    visualize_one_audio_seq(visualizer,lf,lr,limg)
    