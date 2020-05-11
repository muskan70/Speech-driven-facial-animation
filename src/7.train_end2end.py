import numpy as np
import cntk as C
import sys
import argparse
from LayerUtils import conv_bn_lrelu, bi_recurrence, flatten
from SysUtils import get_current_time_string, is_Win32, make_dir, ArgParser

C.cntk_py.set_fixed_random_seed(1)

F_DIM = 128
T_DIM = 32

input_dim_model = (1, F_DIM, T_DIM)
input_dim = F_DIM*T_DIM
label_dim = 46

# the audio CNN encoder subnetwork
def audio_encoder(input):
    # F-convolution
    # 1x128x32
    h = conv_bn_lrelu(input, filter_shape=(5,1), num_filters=32, strides=(2,1), name="conv1")
    # 32x64x32
    h = conv_bn_lrelu(h, filter_shape=(3,1), num_filters=64, strides=(2,1), name="conv2")
    # 64x32x32
    h = conv_bn_lrelu(h, filter_shape=(3,1), num_filters=128, strides=(2,1), name="conv3")
    # 128x16x32
    h = conv_bn_lrelu(h, filter_shape=(3,1), num_filters=256, strides=(2,1), name="conv4")
    # 256x8x32
    h = conv_bn_lrelu(h, filter_shape=(3,1), num_filters=512, strides=(2,1), name="conv5")
    # 512x4x32

    # T-convolution
    h = conv_bn_lrelu(h, filter_shape=(1,3), num_filters=512, strides=(1,2), name="t_conv1")
    # 512 x 4 x 16
    h = conv_bn_lrelu(h, filter_shape=(1,3), num_filters=512, strides=(1,2), name="t_conv2")
    # 512 x 4 x 8
    h = conv_bn_lrelu(h, filter_shape=(1,3), num_filters=512, strides=(1,2), name="t_conv3")
    # 512 x 4 x 4
    return h

def audio_encoder_2(input):
    # F-convolution
    # 1x128x32
    h = conv_bn_lrelu(input, filter_shape=(5,1), num_filters=64, strides=(2,1), name="conv1")
    # 64x64x32
    h = conv_bn_lrelu(h, filter_shape=(3,1), num_filters=128, strides=(2,1), name="conv2")
    # 128x32x32
    h = conv_bn_lrelu(h, filter_shape=(3,1), num_filters=256, strides=(2,1), name="conv3")
    # 256x16x32
    h = conv_bn_lrelu(h, filter_shape=(3,1), num_filters=512, strides=(2,1), name="conv4")
    # 512x8x32
    h = conv_bn_lrelu(h, filter_shape=(3,1), num_filters=1024, strides=(2,1), name="conv5")
    # 1024x4x32
    
    # T-convolution
    h = conv_bn_lrelu(h, filter_shape=(1,3), num_filters=1024, strides=(1,2), name="t_conv1")
    # 1024 x 4 x 16
    h = conv_bn_lrelu(h, filter_shape=(1,3), num_filters=1024, strides=(1,2), name="t_conv2")
    # 1024 x 4 x 8
    h = conv_bn_lrelu(h, filter_shape=(1,3), num_filters=1024, strides=(1,2), name="t_conv3")
    # 1024 x 4 x 4
    return h

def audio_encoder_3(input, model_file, cloning=False):
    # Load and freeze pre-trained encoder
    last_layer_name = "t_conv3"
    model = C.load_model(model_file)
    input_node = model.find_by_name("input")
    last_conv = model.find_by_name(last_layer_name)
    if not last_conv:
        raise ValueError("the layer does not exist")
    h = C.combine([last_conv.owner]).clone(C.CloneMethod.clone if cloning else C.CloneMethod.freeze, {input_node: input})
    return h

def create_model(input, net_type="gru", encoder_type=1, model_file=None, e3cloning=False):
    if encoder_type == 1:
        h = audio_encoder(input)
        if net_type.lower() is not "cnn":
            h = flatten(h)
    elif encoder_type == 2:
        h = audio_encoder_2(input)
        h = C.layers.GlobalAveragePooling(name="avgpool")(h)
        h = C.squeeze(h)
    elif encoder_type == 3:
        h = audio_encoder_3(input, model_file, e3cloning)
        if net_type.lower() is not "cnn":
            h = flatten(h)
    else:
        raise ValueError("encoder type {:d} not supported".format(encoder_type))

    if net_type.lower() == "cnn":
        h = C.layers.Dense(1024, init=C.he_normal(), activation=C.tanh)(h)
    elif net_type.lower() == "gru":
        h = C.layers.Recurrence(step_function=C.layers.GRU(256), go_backwards=False, name="rnn")(h)
    elif net_type.lower() == "lstm":
        h = C.layers.Recurrence(step_function=C.layers.LSTM(256), go_backwards=False, name="rnn")(h)
    elif net_type.lower() == "bigru":
        h = bi_recurrence(h, C.layers.GRU(128), C.layers.GRU(128), name="bigru")
    elif net_type.lower() == "bilstm":
        h = bi_recurrence(h, C.layers.LSTM(128), C.layers.LSTM(128), name="bilstm")
    h = C.layers.Dropout(0.2)(h)
    # output
    y = C.layers.Dense(label_dim, activation=C.sigmoid, init=C.he_normal(), name="output")(h)
    return y

def l2_loss(output, target):
    return C.reduce_mean(C.square(output - target))

def std_normalized_l2_loss(output, target):
    std_inv = np.array([6.6864805402, 5.2904440280, 3.7165409939, 4.1421640454, 8.1537399389, 7.0312877415, 2.6712380967,
                        2.6372177876, 8.4253649884, 6.7482162880, 9.0849960354, 10.2624412692, 3.1325531319, 3.1091179819,
                        2.7337937590, 2.7336441031, 4.3542467871, 5.4896293687, 6.2003761588, 3.1290341469, 5.7677042738,
                        11.5460919611, 9.9926451700, 5.4259818848, 20.5060642486, 4.7692101480, 3.1681517575, 3.8582905289,
                        3.4222250436, 4.6828286809, 3.0070785113, 2.8936539301, 4.0649030157, 25.3068458731, 6.0030623160,
                        3.1151977458, 7.7773542649, 6.2057372469, 9.9494258692, 4.6865422850, 5.3300697628, 2.7722027974,
                        4.0658663003, 18.1101618617, 3.5390113731, 2.7794520068], dtype=np.float32)
    weights = C.constant(value=std_inv) #.reshape((1, label_dim)))
    dif = output - target
    ret = C.reduce_mean(C.square(C.element_times(dif, weights)))
    return ret

def l1_reg_loss(output):
    return C.reduce_mean(output)

# create computational graph and learner
def build_graph(config):
    assert(config['type'] in ["cnn", "lstm", "gru", "bilstm", "bigru"])
    if config["type"] == "cnn":
        # static model
        features = C.input_variable(input_dim_model, name="input")
        labels = C.input_variable(label_dim, name="label")
    else:
        # recurrent model
        features = C.sequence.input_variable(input_dim_model, name="input")
        labels = C.sequence.input_variable(label_dim, name="label")
    netoutput = create_model(features, config["type"], config["encoder"], config["pretrained_model"], config["e3_clone"])

    if config["l2_loss_type"] == 1:
        print("Use standard l2 loss")
        ce = l2_loss(netoutput, labels)
    elif config["l2_loss_type"] == 2:
        print("Use variance normalized l2 loss")
        ce = std_normalized_l2_loss(netoutput, labels)
    else:
        raise ValueError("Unsupported loss type")

    if config["l1_reg"] > sys.float_info.epsilon:
        ce = ce + config["l1_reg"] * l1_reg_loss(netoutput)
    
    # performance metrics
    pe = C.squared_error(netoutput, labels)

    if config["constlr"]:
        lr_schedule = config["lr"]
    else:
        if config["lr_list"] is not None:
            print("use learning rate schedule from file")
            lr_schedule = config["lr_list"]
        else:
            if config["type"] != "cnn": # default learning rate for recurrent model
                lr_schedule = [0.005] + [0.0025]*2 + [0.001]*4 + [0.0005]*8 + [0.00025]*16 + [0.0001]*1000 + [0.00005]*1000 + [0.000025]
            elif config["lr_schedule"] == 1: # learning rate for CNN
                lr_schedule = [0.005] + [0.0025]*2 + [0.00125]*3 + [0.0005]*4 + [0.00025]*5 + [0.0001]
            elif config["lr_schedule"] == 2:
                lr_schedule = [0.005] + [0.0025]*2 + [0.00125]*3 + [0.0005]*4 + [0.00025]*5 + [0.0001]*100 + [0.00005]*50 + [0.000025]*50 + [0.00001]
            else:
                raise ValueError("unknown learning rate")
    learning_rate = C.learning_parameter_schedule_per_sample(lr_schedule, epoch_size=config["epoch_size"])
    momentum_schedule = C.momentum_schedule(0.9, minibatch_size=300)
    
    learner = C.adam(netoutput.parameters, lr=learning_rate, momentum=momentum_schedule,
                        l2_regularization_weight=0.0001,
                        gradient_clipping_threshold_per_sample=3.0, gradient_clipping_with_truncation=True)
    trainer = C.Trainer(netoutput, (ce, pe), [learner])

    return features, labels, netoutput, trainer


# training procedure
def create_reader(path, is_training=True):
    return C.io.MinibatchSource(C.io.CTFDeserializer(path, C.io.StreamDefs(
        features = C.io.StreamDef(field='features', shape=input_dim, is_sparse=False),
        labels = C.io.StreamDef(field='labels', shape=label_dim, is_sparse=False)
    )), randomize=is_training, max_sweeps = C.io.INFINITELY_REPEAT if is_training else 1)


def train(config):
    features, labels, netoutput, trainer = build_graph(config)

    C.logging.log_number_of_parameters(netoutput) ; print()

    #training config
    epoch_size = config["epoch_size"]
    
    progress_printer = C.logging.ProgressPrinter(freq=200, tag='Training') # more detailed logging

    minibatch_size = config["minibatch_size"]
    max_epochs = config["num_epochs"]
    model_file = config["modelfile"]
    log_file = config["logfile"]

    reader = create_reader(config["datafile"])
    input_map = {features: reader.streams.features, labels: reader.streams.labels}
    
    t = 0
    for epoch in range(max_epochs):      
        epoch_end = (epoch+1) * epoch_size
        while t < epoch_end:             
            data = reader.next_minibatch(min(minibatch_size, epoch_end-t), input_map=input_map)
            trainer.train_minibatch(data)                                  
            t += trainer.previous_minibatch_sample_count                    
            progress_printer.update_with_trainer(trainer, with_metric=True) # log progress
        loss, metric, actual_samples = progress_printer.epoch_summary(with_metric=True)
        with open(log_file, 'a') as csvfile:
            csvfile.write("Epoch:  "+ str(epoch) + " Loss: " + str(loss) + " Metric: " + str(metric) + "\n")
    # save
    netoutput.save(model_file)
    return features, labels, netoutput, trainer


def evaluate(datafile, features, labels, trainer, log_file):
    progress_printer = C.logging.ProgressPrinter(tag="Evaluation", num_epochs=0)
    minibatch_size = 200

    reader = create_reader(datafile, is_training=False)
    input_map = {features: reader.streams.features, labels: reader.streams.labels}

    while True:
        data = reader.next_minibatch(minibatch_size, input_map=input_map)
        if not data:                               
            break
        metric = trainer.test_minibatch(data)
        progress_printer.update(0, data[labels].num_samples, metric) # log progress
    loss, metric, actual_samples = progress_printer.epoch_summary(with_metric=True)
    with open(log_file, 'a') as csvfile:
        csvfile.write("\n\n --- Test error: " + str(metric) + "\n")


def process_args():
    class ThisArgParser(ArgParser):
        def __init__(self):
            super(ThisArgParser, self).__init__()
            self.config["minibatch_size"] = 300
            self.config["lr_schedule"] = 1
            self.config["type"] = "gru"
            self.config["encoder"] = 1
            self.config["pretrained_model"] = None
            self.config["e3_clone"] = False

        def prepare(self):
            super(ThisArgParser, self).prepare()
            self.parser.add_argument("--gru", action="store_true")
            self.parser.add_argument("--lstm", action="store_true")
            self.parser.add_argument("--cnn", action="store_true")
            self.parser.add_argument("--bigru", action="store_true")
            self.parser.add_argument("--bilstm", action="store_true")
            self.parser.add_argument("--l2type", type=int, default=2)
            self.parser.add_argument("--l1reg", type=float, default=0.1)
            self.parser.add_argument("--lrschd", type=int, default=self.config["lr_schedule"])
            self.parser.add_argument("--encoder", type=int, default=self.config["encoder"])
            self.parser.add_argument("--pretrained_model", type=str)
            self.parser.add_argument("--e3clone", action="store_true")
            

        def parse(self):
            super(ThisArgParser, self).parse()
            self.config["lr_schedule"] = self.args.lrschd
            self.config["l2_loss_type"] = self.args.l2type
            self.config["l1_reg"] = self.args.l1reg
            self.config["lr_schedule"] = self.args.lrschd
            self.config["encoder"] = self.args.encoder
            if self.args.pretrained_model:
                self.config["pretrained_model"] = self.args.pretrained_model
    
            if self.args.cnn:
                self.config["type"] = "cnn"
            if self.args.lstm:
                self.config["type"] = "lstm"
            if self.args.gru:
                self.config["type"] = "gru"
            if self.args.bigru:
                self.config["type"] = "bigru"
            if self.args.bilstm:
                self.config["type"] = "bilstm"

            if self.args.e3clone:
                self.config["e3_clone"] = True


    parser = ThisArgParser()
    parser.prepare()
    parser.parse()
    config = parser.config
    return config


def main():
    config = process_args()
    print("training type: {:s}".format(config["type"]))
    print("max epoch: {:d}".format(config["num_epochs"]))

    current_time = get_current_time_string()
    
    # set main directory path
    data_dir = "../speech_dir"

    # set proper paths
    if config["type"] == "cnn":
        train_file = data_dir + "/audio_exp_train_noseq.ctf"
        test_file = data_dir + "/audio_exp_test_noseq.ctf"
    else:
        train_file = data_dir + "/audio_exp_train.ctf"
        test_file = data_dir + "/audio_exp_test.ctf"

    model_dir = data_dir + "/model_audio2exp_" + current_time
    make_dir(model_dir)
    print("directory created")
    
    model_filename = model_dir + "/model_audio2exp_" + current_time
    model_file = model_filename + ".dnn"
    log_file = model_dir + "/training_log.txt"
    
    if config["encoder"] == 3:
        if not config["pretrained_model"]:
            config["pretrained_model"] = data_dir + "/model_audio2exp_2019-03-24-21-03.dnn"
    else:
        config["pretrained_model"] = None

    config["datafile"] = train_file
    config["modelfile"] = model_file
    config["logfile"] = log_file

    features, labels, netoutput, trainer = train(config)
    print ("Training done!")
    # test
    evaluate(test_file, features, labels, trainer, log_file)
    print ("Testing done!")

if __name__=='__main__':
    main()