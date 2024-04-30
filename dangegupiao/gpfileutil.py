import os


def getmaxzql(gpdmmc, path='./data'):
    file_list = os.listdir(path)
    zqll = []
    for filename in file_list:
        if filename.endswith('_model.pth'):
            name = os.path.splitext(filename)[0].split('_')
            zql = name[1]
            file_gpdmmc = name[0]
            if gpdmmc == file_gpdmmc:
                zqll.append(zql)
    if len(zqll) > 0:
        zqlx = max([float(x) for x in zqll])
        return zqlx
    return 0

