import pathlib as plb
import csv
import sys

def list_all_sequences(root_path):
    root_dir = plb.Path(root_path)
    actors = []
    for actor_dir in root_dir.iterdir():
        if(actor_dir.is_dir()):
            seqs = []
            for seq_dir in actor_dir.iterdir():
                if seq_dir.name[:2] == "01":
                    seqs.append(str(seq_dir))
            actors.append(seqs)
    return actors

def convert_one_sequence(seq_id, seq_dir, output):
    #read features & labels
    try:
        mfccsinput = csv.reader(open(seq_dir + "/mfcc_normed.csv"), delimiter=',', quotechar='|')
        melsinput = csv.reader(open(seq_dir + "/log_mel_normed.csv"), delimiter=',', quotechar='|')
        chromasinput = csv.reader(open(seq_dir + "/chroma_normed.csv"), delimiter=',', quotechar='|')
        labelsinput = csv.reader(open(seq_dir + "/Explabels.csv"), delimiter=',', quotechar='|')

    except IOError as e:
        (errno,strerror) = e.args
        print ("I/O error({0}): {1}".format(errno,strerror))
        sys.exit(1)
    except:
        print ("Unexpected Error: ",sys.exc_info()[0])
        raise
    mfccs = list(mfccsinput)
    mels = list(melsinput)
    chromas = list(chromasinput)
    labels2 = list(labelsinput)
    
    labels = []
    i = 0
    for label in labels2:
        if len(label) > 49:
            if label[49] == '':
                print(seq_dir)
                label = label[:49]
        if len(label) > 0:
            labels.append(label)
            if len(label) != 49:
                print ("labels at " + str(i) + " of " + seq_dir + " have less than 49 values")
                return False
            i += 1
    if len(labels) != len(mfccs):
        print ("Lengths of features and labels of " + seq_dir + " are different")
        return False
    # write sequence
    seq_len = min([len(labels), len(mfccs)])
    for i in range(0,seq_len):
        if len(labels[i]) != 49:
            print ("labels at " + str(i) + " of " + seq_dir + " have less than 49 value")
            return False
        output.write(str(seq_id))
        # features
        ft_str = "\t|features "
        # mel spectro
        for ft in mels[i]:
            ft_str += str(ft) + " "
        # chroma
        for ft in chromas[i]:
            ft_str += str(ft) + " "
        # MFCC
        for ft in mfccs[i]:
            ft_str += str(ft) + " "
        output.write(ft_str)
        #labels
        lb_str = "\t|labels "
        for lb in labels[i]:
            lb_str += str(lb) + " "
        output.write(lb_str)
        output.write("\n")
    return True
        
def convert_one_actor(actor, output, last_seq_id):
    seq_id = last_seq_id
    for seq_dir in actor:
        seq_id += 1
        if not convert_one_sequence(seq_id, seq_dir, output):
            seq_id -= 1
    return seq_id
       

def convert_all(ctf_train, ctf_test, actors):
    # use first 20 actors for training and last 4 actors for testing
    train_num = 0
    test_num = 0
    for i in range(0,20):
       train_num += len(actors[i])
    for i in range(20, 24):
       test_num += len(actors[i])
    print ("train num: " , train_num, " test num: ", test_num)
    train_output = open(ctf_train, "w")
    last_id = -1
    for i in range(20):
        print(" --- Converting actor" + str(i+1) + "\n")
        last_id = convert_one_actor(actors[i], train_output, last_id)
    test_output = open(ctf_test, "w")
    for i in range(20, 24):
        print(" --- Converting actor" + str(i+1) + "\n")
        last_id = convert_one_actor(actors[i], test_output, last_id)


if __name__ == '__main__':
    root_path = "../feat_root"
    ctf_path="../speech_dir"
    actors = list_all_sequences(root_path)
    ctf_train = ctf_path + "/train_1_20_mfcc_0406.ctf"
    ctf_test = ctf_path + "/test_21_24_mfcc_0406.ctf"
    convert_all(ctf_train, ctf_test, actors)
    print ("Done.")
