from yoro.utils.train_util import YOROTrain

if __name__ == '__main__':

    tc = YOROTrain('config/example.yaml')
    tc.import_model('coating_epoch_30000.zip')
    tc.valid()
