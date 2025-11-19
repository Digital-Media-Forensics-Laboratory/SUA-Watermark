class Config(object):
    env = 'default'
    backbone = 'resnet18'
    classify = 'softmax'
    num_classes = 13938
    metric = 'arc_margin'
    easy_margin = False
    use_se = False
    loss = 'focal_loss'

    display = False
    finetune = False

    train_root = './Datasets/webface/CASIA-maxpy-clean/'
    train_list = './Datasets/webface/cleaned_list.txt'
    val_list = './Datasets/webface/val_data_13938.txt'

    # test_root = '/data1/Datasets/anti-spoofing/test/data_align_256'
    # test_list = 'test.txt'

    lfw_root = './Datasets/lfw/lfw-align-128'
    lfw_test_list = './Datasets/lfw/lfw_test_pair.txt'
    # lfw_root = './Datasets/test_data/test_data'
    # lfw_test_list = './Datasets/test_data/test_data.txt'

    checkpoints_path = './checkpoints'
    load_model_path = './models/resnet50.pth'
    test_model_path = './checkpoints/resnet18_50.pth'
    save_interval = 10

    train_batch_size = 8  # batch size
    test_batch_size = 10

    input_shape = (3, 256, 256)
    # input_shape = (1, 112, 112)

    optimizer = 'sgd'

    use_gpu = True  # use GPU or not
    gpu_id = '0, 1'
    num_workers =1  # how many workers for loading data
    print_freq = 100  # print info every N batch

    debug_file = '/tmp/debug'  # if os.path.exists(debug_file): enter ipdb
    result_file = 'result.csv'

    max_epoch = 100
    lr = 1e-1  # initial learning rate
    lr_step = 10
    lr_decay = 0.95  # when val_loss increase, lr = lr*lr_decay
    weight_decay = 5e-4
