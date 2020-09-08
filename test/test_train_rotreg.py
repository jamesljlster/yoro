from yoro.utils.train_util import RotRegressorTrain

if __name__ == '__main__':

    tc = RotRegressorTrain('config/rotation_regressor.yaml')
    tc.restore()
    tc.valid()
    tc.train()
    tc.export_model()
