from pre_process.split import PreProcess, SplitDataset
from pre_process.rebuild_dataset import RebuildDataset
from i3d.i3d import I3DInterface
from utils.debug import Debug
from model.model import *
from utils.dataloader import MDataset, MSampler, MTestSampler
from torch.utils.data import DataLoader

if __name__ == "__main__":


    # Example usage of PreProcess to split videos into snippets
    #
    # DATASET = "D:/HLCH/Works/BatchNorm_Based_VAD/Dataset/UCF-Crime-Rebuild"
    # OUTPUT = "D:/HLCH/Works/BatchNorm_Based_VAD/Dataset/UCF-Crime-Splited"
    # PreProcess.batch_process(DATASET, OUTPUT, size=(224, 224), snippet_len=18*3, snippet_cnt=200, max_frames=60*10*30, sample_cnt=3)


    # Example usage of RebuildDataset to organize dataset into Train and Test sets
    #
    # DATASET = "D:/HLCH/Works/BatchNorm_Based_VAD/Dataset/UCF-Crime-Splited"
    # OUTPUT = "D:/HLCH/Works/BatchNorm_Based_VAD/Dataset/UCF-Crime-Features"
    # i3d = I3DInterface()
    # i3d.batch_predict(DATASET, OUTPUT, 18, 16)


    # Example usage of divide dataset into Train and Val sets
    #
    # DATASET = "D:/HLCH/Works/BatchNorm_Based_VAD/Dataset/UCF-Crime-Features"
    # OUTPUT = "D:/HLCH/Works/BatchNorm_Based_VAD/Dataset/UCF-Crime-TrainVal"
    # SplitDataset.split(DATASET, OUTPUT, 0.2)


    # Example usage of Training the model
    #
    TRAIN_SET = "D:/HLCH/Works/BatchNorm_Based_VAD/Dataset/UCF-Crime-Features/Train"
    VAL_SET = "D:/HLCH/Works/BatchNorm_Based_VAD/Dataset/UCF-Crime-Features/Test"
    EPOCHS = 50
    BATCH_SIZE = 64
    train_set = MDataset(TRAIN_SET)
    val_set = MDataset(VAL_SET)
    train_sampler = MSampler(train_set.label_list, batch_size=BATCH_SIZE)
    val_sampler = MSampler(val_set.label_list, batch_size=BATCH_SIZE)
    test_sampler = MTestSampler(val_set.label_list, batch_size=BATCH_SIZE)
    train_dataloader = DataLoader(train_set, batch_sampler=train_sampler, pin_memory=True)
    val_dataloader = DataLoader(val_set, batch_sampler=val_sampler, pin_memory=True)
    test_dataloader = DataLoader(val_set, batch_sampler=test_sampler, pin_memory=True)
    model = Backbone().cuda().float()
    enhancer = Enhancer().cuda().float()
    classifier1 = Classifier(32).cuda().float()
    classifier2 = Classifier(16).cuda().float()
    optimizer = torch.optim.Adam(
            list(model.parameters()) + list(enhancer.parameters()) + list(classifier1.parameters()) + list(classifier2.parameters()), lr=1e-4, weight_decay=5e-5
        )
    train(
        model=model,
        enhancer=enhancer,
        classifier1=classifier1,
        classifier2=classifier2,
        train_loader=train_dataloader,
        val_loader=val_dataloader,
        test_loader=test_dataloader,
        criterion=MLoss(5, 20),
        optimizer=optimizer,
        scheduler=torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.5),
        epochs=EPOCHS,
        alpha=0.1,
        ps=0.1,
        pb=0.2,
        device=torch.device("cuda"),
        w1=5,
        w2=20
    )