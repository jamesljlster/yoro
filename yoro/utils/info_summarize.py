
def tensor_simplify(src):
    out = src.tolist()
    if len(out) == 1:
        out = out[0]

    return out


def elem_assign(src):

    if type(src).__name__ == 'Tensor':
        src = tensor_simplify(src)

    out = None
    srcType = type(src).__name__
    if srcType == 'dict':
        out = {}
        for key in src.keys():
            out[key] = elem_assign(src[key])

    elif srcType == 'list':
        out = []
        for elem in src:
            out.append(elem_assign(elem))

    else:
        out = src

    return out


def info_assign(src):

    out = {}
    for key in src.keys():
        out[key] = [elem_assign(src[key][0]), src[key][1]]

    return out


def elem_add(out, src):

    if type(src).__name__ == 'Tensor':
        src = tensor_simplify(src)

    srcType = type(src).__name__
    if srcType == 'dict':
        for key in src.keys():
            out[key] = elem_add(out[key], src[key])

    elif srcType == 'list':
        for i in range(len(src)):
            out[i] = elem_add(out[i], src[i])

    else:
        out += src

    return out


def info_add(out, src):

    if out is None:
        out = info_assign(src)

    else:
        for key in src.keys():
            out[key][0] = elem_add(out[key][0], src[key][0])
            out[key][1] += src[key][1]

    return out


def elem_div(src, divisor):

    if type(src).__name__ == 'Tensor':
        src = tensor_simplify(src)

    out = None
    srcType = type(src).__name__
    if srcType == 'dict':
        out = {}
        for key in src.keys():
            out[key] = elem_div(src[key], divisor)

    elif srcType == 'list':
        out = []
        for elem in src:
            out.append(elem_div(elem, divisor))

    else:
        out = src / divisor

    return out


def info_simplify(src):

    out = {}
    for key in src.keys():
        out[key] = elem_div(src[key][0], src[key][1])

    return out


def elem_repr_recursive(reprStr, src, fmt):

    if type(src).__name__ == 'Tensor':
        src = tensor_simplify(src)

    srcType = type(src).__name__
    if srcType == 'dict':
        strList = []
        for key in src.keys():
            strList.append(elem_repr_recursive(key + ': ', src[key], fmt))
        subStr = '{' + ', '.join(strList) + '}'

    elif srcType == 'list':
        strList = []
        for i in range(len(src)):
            strList.append(elem_repr_recursive('', src[i], fmt))
        subStr = '[' + ', '.join(strList) + ']'

    else:
        subStr = fmt.format(src)

    return reprStr + subStr


def info_represent(src, decimal=3):

    info = info_simplify(src)
    fmt = '{0:.%df}' % decimal

    return elem_repr_recursive('', info, fmt)
