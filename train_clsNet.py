import argparse
from data.dataset import *
from classification import clsNet

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--database_dir', default="D:\Lu\data_thesis")
    parser.add_argument('--id', type=str, default="ResNet_cls_polyp")
    parser.add_argument('--epoch_num', type=int, default=30)
    parser.add_argument('--batch_size', type=int, default=128)

    args = parser.parse_args()

    id = args.id
    device = "cuda" if torch.cuda.is_available() else "cpu" 

    train_dataset = prepare_data(["Galar-tech"], 
                            used_classes=None,
                            phase="train",
                            resize=(224,224),
                            augment=True,
                            normalize_mean=[0.485, 0.456, 0.406],
                            normalize_std=[0.229, 0.224, 0.225],
                            database_dir=args.database_dir)
    test_dataset = prepare_data(["Galar-tech"], 
                            used_classes=None,
                            phase="test",
                            resize=(224,224),
                            augment=False,
                            normalize_mean=[0.485, 0.456, 0.406],
                            normalize_std=[0.229, 0.224, 0.225],
                            database_dir=args.database_dir)
    
    check_points_dir = os.path.join(os.path.abspath(os.getcwd()),"checkpoints", id)
    clsNet.train(device, train_dataset, test_dataset, 
                 classes=Const.Text_Annotation["technical"],
                 run_id=id,
                 model_name='ResNet',
                 freeze_num= 7,
                 lr = 1e-4,
                 weight_decay= 0,
                 ckp_dir=check_points_dir,
                 batch_size_train=args.batch_size,
                 num_epochs=args.epoch_num)