import pathlib as plb
import subprocess as sup

video_path = "../sample_video/01-02-01-01-01-01-01.mp4"
audio_path = "../sample_video/01-02-01-01-01-01-01.wav"

def extract_one_video(video_file, audio_file):
    command = "ffmpeg -i " + video_file + " -y -ab 160k -ac 2 -ar 44100 -vn " + audio_file + " -loglevel quiet"
    sup.call(command, shell=True)    
    
def make_dirs(video_root, audio_root):
    vr_dir = plb.Path(video_root)
    ar_dir = plb.Path(audio_root)

def convert_all(video_root, audio_root):
    vr_dir = plb.Path(video_root)
    for actor in vr_dir.iterdir():
        for video_file in actor.iterdir():
            if video_file.name[len(video_file.name)-4:] == '.mp4':
                video_path=video_root+"/"+actor.name+"/" +video_file.stem+".mp4"
                audio_path = audio_root + "/" + actor.name + "/" + video_file.stem + ".wav"
                

if __name__ == "__main__":
    extract_one_video(video_path, audio_path)
    
    