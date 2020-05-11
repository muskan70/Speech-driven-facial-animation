import pathlib as plb
import subprocess as sup

video_root = "../video"
audio_root = "../speech"

def extract_one_video(video_file, audio_file):
    command = "ffmpeg -i " + video_file + " -y -ab 160k -ac 2 -ar 44100 -vn " + audio_file + " -loglevel quiet"
    sup.call(command, shell=True)    
    
def make_dirs(video_root, audio_root):
    vr_dir = plb.Path(video_root)
    ar_dir = plb.Path(audio_root)
    try:
        ar_dir.mkdir()
    except FileExistsError:
        print ("Audio Root Directory existed")
    except FileNotFoundError:
        print ("Parent directory not found")
    for actor in vr_dir.iterdir():
        if actor.is_dir():
            try:
                plb.Path(audio_root + "/" + actor.name).mkdir()
            except FileExistsError:
                print ("Directory " + actor.name + "  existed")

def convert_all(video_root, audio_root):
    vr_dir = plb.Path(video_root)
    for actor in vr_dir.iterdir():
        for video_file in actor.iterdir():
            if video_file.name[len(video_file.name)-4:] == '.mp4':
                video_path=video_root+"/"+actor.name+"/" +video_file.stem+".mp4"
                audio_path = audio_root + "/" + actor.name + "/" + video_file.stem + ".wav"
                extract_one_video(video_path, audio_path)

if __name__ == "__main__":
    make_dirs(video_root, audio_root)
    convert_all(video_root, audio_root)
    
    