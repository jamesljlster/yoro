from yoro.utils.train_util import RotAnchorTrain

if __name__ == '__main__':

    tc = RotAnchorTrain('config/rotation_anchor.yaml')
    tc.restore()
    tc.valid()
    tc.train()
    tc.export_model()
