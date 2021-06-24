from yoro.backbone.darknet.config_parser import parse_config

if __name__ == '__main__':
    config = parse_config('/home/james/api/darknet/cfg/yolov3.cfg')
    for elem in config:
        print('%d: %s' % (elem['layer_idx'], elem['type']))
        print('  from:', elem['from'])
        print('  to:', elem['to'])
        print('  param:')
        for key in elem['param']:
            print('    %s = %s' % (key, elem['param'][key]))
        print()
