comment_char = '#'
split_char = '='


def parse_config(cfg_file):

    # Read lines and strip whitespace
    lines = []
    with open(cfg_file, 'r') as f:
        for line in f.read().splitlines():
            line = line.strip()
            if comment_char in line:
                line = line[:line.find(comment_char)]
            if len(line) > 0:
                lines.append(line)

    # Parse config
    config = []
    layerCounter = -1
    for line in lines:

        if line.startswith('['):
            blockName = line.strip('[ ]')

            layerIdx = layerCounter
            layerCounter += 1

            config.append({
                'type': blockName,
                'layer_idx': layerIdx,
                'from': [],
                'to': [],
                'param': {}
            })

        elif split_char in line:
            key, value = line.split(split_char)
            key = key.strip()

            # Parse value
            def parse_value(string):
                if '.' in string:
                    ret = float(string)
                else:
                    try:
                        ret = int(string)
                    except:
                        ret = string.strip()
                return ret

            if ',' in value:
                value = [parse_value(string) for string in value.split(',')]
            else:
                value = parse_value(value)

            config[-1]['param'][key] = value

    config = config[1:]  # Remove [net] section

    # Construct layer linking relationship
    for layerCfg in config:

        layerType = layerCfg['type']
        layerIdx = layerCfg['layer_idx']

        # Get source layer indices
        fromInd = [-1] if layerIdx > 0 else []

        if layerType == 'shortcut':
            fromInd = layerCfg['param'].pop('from')
            if not isinstance(fromInd, list):
                fromInd = [fromInd]
            fromInd = [-1] + fromInd

        elif layerType == 'route':
            fromInd = layerCfg['param'].pop('layers')
            if not isinstance(fromInd, list):
                fromInd = [fromInd]

        for i in range(len(fromInd)):
            if fromInd[i] < 0:
                fromInd[i] += layerIdx

        # Construct linking relationship
        layerCfg['from'] = fromInd
        for idx in fromInd:
            config[idx]['to'].append(layerIdx)

    return config
