import numpy as np
import pathlib as plb
from SysUtils import get_items, make_dir, get_path

def convert_vector_to_string(np_vec):
    a = list(np_vec)
    ret = " ".join(["{:.6f}".format(x) for x in a])
    return ret


def write_seq(nSeq, audio, exp, fout):
    n = audio.shape[0]
    # if nSeq < 0: write line without sequence ID
    if nSeq >= 0:
        for i in range(n):
            rowstr = "{:d} |features {:s} |labels {:s}\n".format(nSeq, convert_vector_to_string(audio[i,:]), convert_vector_to_string(exp[i,:]))
            fout.write(rowstr)
    else:
        for i in range(n):
            rowstr = "|features {:s} |labels {:s}\n".format(convert_vector_to_string(audio[i,:]), convert_vector_to_string(exp[i,:]))
            fout.write(rowstr)


def create_ctf_file(seq_root, exp_root, output_file, is_training=True):
    databases = ["RAVDESS","VIDTIMIT","SAVEE"]

    nSeq = 0
    fout = open(output_file, "w")
    
    for i, db in enumerate(databases):
        # each database
        db_dir = seq_root + "/" + db
        actors = get_items(db_dir)
        if is_training:
            if db == "RAVDESS":
                actors = actors[:20]
            elif db == "SAVEE":
                actors = actors[:3]
            elif db == "VIDTIMIT":
                actors = actors[:40]
            else:
                raise IOError("folder does not exist!")
        else:
            if db == "RAVDESS":
                actors = actors[20:]
            elif db == "SAVEE":
                actors = actors[3:]
            elif db == "VIDTIMIT":
                actors = actors[40:]
            else:
                raise IOError("folder does not exist!")
        for actor in actors:
            seq_dir = db_dir + "_feat/" + actor
            exp_dir = exp_root + "/" + db + "/" + actor
            items = get_items(seq_dir)
            for seq in items:
                print(seq_dir + "/" + seq)
                seq_path = seq_dir + "/" + seq + "/dbspectrogram.csv"
                exp_path = exp_dir + "/" + seq + ".npy"
                if plb.Path(seq_path).exists() and plb.Path(exp_path).exists():
                    # load data
                    audio = np.loadtxt(seq_path, dtype=np.float32, delimiter=',')
                    exp = np.load(exp_path)
                    if exp.shape[0] != audio.shape[0]:
                        print("length not matched")
                        continue
                    # write
                    write_seq(nSeq, audio, exp, fout)
                    nSeq += 1
                else:
                    print("file not exist")
    fout.close()


def create_ctf_file_noseq(seq_root, exp_root, output_file, is_training=True):
    databases = ["RAVDESS","VIDTIMIT","SAVEE"]

    fout = open(output_file, "w")
    
    for i, db in enumerate(databases):
        # each database
        db_dir = seq_root + "/" + db
        actors = get_items(db_dir)
        if is_training:
            if db == "RAVDESS":
                actors = actors[:20]
            elif db == "SAVEE":
                actors = actors[:3]
            elif db == "VIDTIMIT":
                actors = actors[:40]
            else:
                raise IOError("folder does not exist!")
        else:
            if db == "RAVDESS":
                actors = actors[20:]
            elif db == "SAVEE":
                actors = actors[3:]
            elif db == "VIDTIMIT":
                actors = actors[40:]
            else:
                raise IOError("folder does not exist!")
        for actor in actors:
            seq_dir = db_dir + "_feat/" + actor
            exp_dir = exp_root + "/" + db + "/" + actor
            items = get_items(seq_dir)
            for seq in items:
                print(seq_dir + "/" + seq)
                seq_path = seq_dir + "/" + seq + "/dbspectrogram.csv"
                exp_path = exp_dir + "/" + seq + ".npy"
                if plb.Path(seq_path).exists() and plb.Path(exp_path).exists():
                    # load data
                    audio = np.loadtxt(seq_path, dtype=np.float32, delimiter=',')
                    exp = np.load(exp_path)
                    if exp.shape[0] != audio.shape[0]:
                        print("length not matched")
                        continue
                    # write
                    write_seq(-1, audio, exp, fout)
                else:
                    print("file not exist")
    fout.close()
    

def main():
    seq_root = "../speech_dir"
    exp_root = "../ExpLabels"

    # create CTF data to train CNN
    ctf_train = seq_root + "/audio_exp_train_noseq.ctf"
    create_ctf_file_noseq(seq_root, exp_root, ctf_train)
    ctf_test = seq_root + "/audio_exp_test_noseq.ctf"
    create_ctf_file_noseq(seq_root, exp_root, ctf_test, False)

    # create CTF data to train recurrent models
    ctf_train = seq_root + "/audio_exp_train.ctf"
    create_ctf_file(seq_root, exp_root, ctf_train)
    ctf_test = seq_root + "/audio_exp_test.ctf"
    create_ctf_file(seq_root, exp_root, ctf_test, False)


if __name__ == "__main__":
    main()
    