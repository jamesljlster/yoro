from yoro.utils.train_util import YOROTrain

if __name__ == '__main__':

    tc = YOROTrain('config/example.yaml')
    tc.restore()
    tc.valid()
    tc.train()
    tc.export_model()
