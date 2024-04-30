import trainclass
import predictclass

epochs = 5000
batches = 64
lr = 0.01
input_size = 45
output_size = 3
hidden_size = 100
train_or_predict = 'predict'
gpdic = {'1': '万润科技_2654', '2': '奥拓电子_2587', '3': '海源复材_2529', '4': '东晶电子_2199',
         '5': '清源股份_603628', '6': '中远海控_601919', '7': '中公教育_2607', '8': '第一医药_600833',
         '9': '金正大_002470', '10': '常熟汽饰_603035', '11': '安妮股份_2235', '12': '合兴包装_2228',
         '13': '中国银行_601988', '14': '索菱股份_2766', '15': '老白干_600559', '16': '凯瑞德_2072',
         '17': '勤上股份_2638', '18': '联明股份_603006', '19': '美克家居_600337', '20': '沪电股份_2463',
         '21': '同方股份_600100', '22': '北汽蓝谷_600733', '23': '万科A_000002', '24': '光启技术_2625',
         '25': '唐人神_2567', '26': '赛力斯_601127',  '27': '中信银行_601998',
         '28': '中信海直_99', '29': '文一科技_600520'}
p_tdsdate='20240425'
p_tdedate='20240429'

if train_or_predict == 'train':
    for key in gpdic:
        gpmc = gpdic[key].split('_')[0]
        gpdm = str(gpdic[key].split('_')[1]).zfill(6)
        print(gpmc, gpdm)
        trainclass.train(epochs=epochs, batches=batches, lr=lr, input_size=input_size,
                         output_size=output_size, hidden_size=hidden_size,gpmc=gpmc, gpdm=gpdm)

if train_or_predict == 'predict':
    for key in gpdic:
        gpmc = gpdic[key].split('_')[0]
        gpdm = str(gpdic[key].split('_')[1]).zfill(6)
        predictclass.predict(input_size=input_size, output_size=output_size, hidden_size=hidden_size,
                              gpmc=gpmc, gpdm=gpdm, p_tdsdate=p_tdsdate, p_tdedate=p_tdedate)

