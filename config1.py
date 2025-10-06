import argparse

def get_config_1mask_2d():
    parse = argparse.ArgumentParser()
    
    # train
    parse.add_argument("--batch_size", type=int, default=1)     # todo 一般生成任务的bs较小，分类等简单任务的bs则可以设为较大值
    parse.add_argument("--epoch_num", type=int, default=100)
    parse.add_argument("--num_workers", type=int, default=4)
    parse.add_argument("--lr", type=float, default=0.0001)      # todo 初始学习率，可以配合学习率衰减策略
                                                                            # todo 初始一般设为 1e-3 or 1e-4
    
    # data
    parse.add_argument("--img_size", type=int, default=512)     # mayo_ldct > 512
    
    # path  todo 最好使用相对路径，绝对路径可能会有中文路径无法识别的问题
    # dataset path
    parse.add_argument("--data_root", type=str, default="../data")  #/mayo_ldct
    # save path
    parse.add_argument("--weight_path", type=str, default="../Result_1mask_2d/weight.pth")
    parse.add_argument("--weight_save_root", type=str, default="../Result_1mask_2d/weights")
    parse.add_argument("--train_img_root", type=str, default="../Result_1mask_2d/train_images")
    parse.add_argument("--test_img_root", type=str, default="../Result_1mask_2d/test_images")
    parse.add_argument("--metric_path", type=str, default="../Result_1mask_2d/metric.xls")
    
    # regularization term weight
    parse.add_argument("--lambda_l1", type=int, default=20)
    # todo loss = loss_main + lambda1 * loss_reg1 + lambda2 * loss_reg2
    
    config = parse.parse_args()
    return config


def get_config_3mask_2d():
    parse = argparse.ArgumentParser()

    # train
    parse.add_argument("--batch_size", type=int, default=1)  # todo 一般生成任务的bs较小，分类等简单任务的bs则可以设为较大值
    parse.add_argument("--epoch_num", type=int, default=100)
    parse.add_argument("--num_workers", type=int, default=4)
    parse.add_argument("--lr", type=float, default=0.0001)  # todo 初始学习率，可以配合学习率衰减策略
    # todo 初始一般设为 1e-3 or 1e-4

    # data
    parse.add_argument("--img_size", type=int, default=512)  # mayo_ldct > 512

    # path  todo 最好使用相对路径，绝对路径可能会有中文路径无法识别的问题
    # dataset path
    parse.add_argument("--data_root", type=str, default="../data")  # /mayo_ldct
    # save path
    parse.add_argument("--weight_path", type=str, default="../Result_3mask_2d/weight.pth")
    parse.add_argument("--weight_save_root", type=str, default="../Result_3mask_2d/weights")
    parse.add_argument("--train_img_root", type=str, default="../Result_3mask_2d/train_images")
    parse.add_argument("--test_img_root", type=str, default="../Result_3mask_2d/test_images")
    parse.add_argument("--metric_path", type=str, default="../Result_3mask_2d/metric.xls")

    # regularization term weight
    parse.add_argument("--lambda_l1", type=int, default=20)
    # todo loss = loss_main + lambda1 * loss_reg1 + lambda2 * loss_reg2

    config = parse.parse_args()
    return config

def get_config_1mask_3d():
    parse = argparse.ArgumentParser()

    # train
    parse.add_argument("--batch_size", type=int, default=1)  # todo 一般生成任务的bs较小，分类等简单任务的bs则可以设为较大值
    parse.add_argument("--epoch_num", type=int, default=100)
    parse.add_argument("--num_workers", type=int, default=4)
    parse.add_argument("--lr", type=float, default=0.0001)  # todo 初始学习率，可以配合学习率衰减策略
    # todo 初始一般设为 1e-3 or 1e-4

    # data
    parse.add_argument("--img_size", type=int, default=512)  # mayo_ldct > 512

    # path  todo 最好使用相对路径，绝对路径可能会有中文路径无法识别的问题
    # dataset path
    parse.add_argument("--data_root", type=str, default="../data_3d_3label_128")  # /mayo_ldct
    # save path
    parse.add_argument("--weight_path", type=str, default="../Result_1mask_3d/weight.pth")
    parse.add_argument("--weight_save_root", type=str, default="../Result_1mask_3d/weights")
    parse.add_argument("--train_img_root", type=str, default="../Result_1mask_3d/train_images")
    parse.add_argument("--test_img_root", type=str, default="../Result_1mask_3d/test_images")
    parse.add_argument("--metric_path", type=str, default="../Result_1mask_3d/metric.xls")

    # regularization term weight
    parse.add_argument("--lambda_l1", type=int, default=20)
    # todo loss = loss_main + lambda1 * loss_reg1 + lambda2 * loss_reg2

    config = parse.parse_args()
    return config

def get_config_3mask_3d():
    parse = argparse.ArgumentParser()

    # train
    parse.add_argument("--batch_size", type=int, default=1)  # todo 一般生成任务的bs较小，分类等简单任务的bs则可以设为较大值
    parse.add_argument("--epoch_num", type=int, default=200)
    parse.add_argument("--num_workers", type=int, default=4)
    parse.add_argument("--lr", type=float, default=0.0001)  # todo 初始学习率，可以配合学习率衰减策略
    # todo 初始一般设为 1e-3 or 1e-4

    # data
    parse.add_argument("--img_size", type=int, default=512)  # mayo_ldct > 512

    # path  todo 最好使用相对路径，绝对路径可能会有中文路径无法识别的问题
    # dataset path
    parse.add_argument("--data_root", type=str, default="../data_3d_3label_128")  # /mayo_ldct
    # save path
    parse.add_argument("--weight_path", type=str, default="../Result_3mask_3d/weight.pth")
    parse.add_argument("--weight_save_root", type=str, default="../Result_3mask_3d/weights")
    parse.add_argument("--train_img_root", type=str, default="../Result_3mask_3d/train_images")
    parse.add_argument("--test_img_root", type=str, default="../Result_3mask_3d/test_images")
    parse.add_argument("--metric_path", type=str, default="../Result_3mask_3d/metric.xls")

    # regularization term weight
    parse.add_argument("--lambda_l1", type=int, default=20)
    # todo loss = loss_main + lambda1 * loss_reg1 + lambda2 * loss_reg2

    config = parse.parse_args()
    return config

def get_config_9mask_3d():
    parse = argparse.ArgumentParser()

    # train
    parse.add_argument("--batch_size", type=int, default=1)  # todo 一般生成任务的bs较小，分类等简单任务的bs则可以设为较大值
    parse.add_argument("--epoch_num", type=int, default=200)
    parse.add_argument("--num_workers", type=int, default=4)
    parse.add_argument("--lr", type=float, default=0.0001)  # todo 初始学习率，可以配合学习率衰减策略
    # todo 初始一般设为 1e-3 or 1e-4

    # data
    parse.add_argument("--img_size", type=int, default=512)  # mayo_ldct > 512

    # path  todo 最好使用相对路径，绝对路径可能会有中文路径无法识别的问题
    # dataset path
    parse.add_argument("--data_root", type=str, default="../data_3d_9label")  # /mayo_ldct
    # save path
    parse.add_argument("--weight_path", type=str, default="../Result_9mask_3d/weight.pth")
    parse.add_argument("--weight_save_root", type=str, default="../Result_9mask_3d/weights")
    parse.add_argument("--train_img_root", type=str, default="../Result_9mask_3d/train_images")
    parse.add_argument("--test_img_root", type=str, default="../Result_9mask_3d/test_images")
    parse.add_argument("--metric_path", type=str, default="../Result_9mask_3d/metric.xls")

    # regularization term weight
    parse.add_argument("--lambda_l1", type=int, default=20)
    # todo loss = loss_main + lambda1 * loss_reg1 + lambda2 * loss_reg2

    config = parse.parse_args()
    return config














