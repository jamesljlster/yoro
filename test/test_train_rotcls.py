from yoro.utils.train_util import RotClassifierTrain

if __name__ == '__main__':

    tc = RotClassifierTrain('config/rotation_classifier.yaml')
    tc.restore()
    tc.valid()
    tc.train()
    tc.export_model()
