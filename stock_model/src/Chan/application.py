import numpy as np
import pandas as pd
import datetime as dt
from Chan.data_structure import *
from Chan.indicator import *
from quantax.data_query_advance import local_get_stock_day_adv, local_get_stock_min_adv

beiji = ['000004','000088','000403','000504','000520','000526','000546','000552','000553','000571','000589','000610','000636','000761','000767','000793','000796','000803','000813','000836','000888','000897','000900','000915','000925','000950','002005','002014','002029','002034','002043','002046','002069','002073','002084','002088','002105','002114','002125','002132','002159','002193','002206','002213','002221','002269','002286','002291','002298','002303','002313','002315','002324','002327','002343','002355','002381','002397','002398','002426','002438','002441','002445','002446','002452','002524','002567','002585','002609','002612','002623','002627','002668','002672','002676','002681','002695','002696','002707','002712','002713','002734','002743','002749','002768','002776','002780','002781','002810','002829','002833','002845','002847','002853','002858','002866','002869','002876','002884','002891','002897','002905','002906','002909','002930','002933','002935','002940','002942','002946','002947','002949','002956','002960','002961','002965','002967','002968','002970','002971','002972','002977','002979','002982','002985','300016','300019','300031','300032','300034','300035','300038','300041','300065','300078','300087','300098','300102','300108','300120','300121','300138','300151','300179','300192','300196','300198','300206','300208','300214','300215','300218','300224','300242','300248','300258','300259','300275','300302','300307','300331','300338','300352','300360','300363','300374','300378','300379','300404','300415','300416','300424','300427','300428','300437','300438','300439','300440','300441','300449','300452','300453','300455','300460','300470','300475','300486','300490','300494','300497','300499','300503','300505','300506','300509','300511','300512','300525','300530','300545','300548','300559','300562','300566','300570','300575','300580','300587','300590','300592','300593','300598','300600','300603','300604','300609','300622','300629','300635','300636','300639','300640','300651','300655','300657','300663','300671','300673','300680','300684','300686','300687','300692','300696','300703','300705','300715','300723','300729','300731','300733','300735','300738','300740','300753','300758','300763','300766','300767','300769','300787','300788','300791','300792','300801','300806','300815','300816','300823','300825','300827','300829','600037','600075','600103','600114','600129','600131','600162','600172','600195','600277','600312','600316','600320','600385','600390','600416','600428','600448','600461','600478','600480','600482','600491','600496','600515','600528','600546','600556','600562','600593','600624','600715','600732','600740','600751','600771','600785','600805','600812','600851','600862','600881','600882','600884','600966','600976','600987','600990','601015','601038','601137','601595','601677','601965','601968','601996','603001','603003','603005','603010','603011','603026','603076','603083','603086','603093','603096','603103','603129','603136','603158','603161','603167','603179','603181','603186','603187','603195','603197','603208','603212','603229','603232','603238','603239','603256','603283','603290','603297','603301','603308','603313','603315','603319','603332','603348','603351','603353','603359','603363','603365','603368','603377','603378','603385','603392','603396','603416','603456','603506','603507','603516','603536','603538','603558','603566','603600','603612','603657','603661','603680','603688','603690','603697','603699','603700','603711','603719','603721','603730','603755','603757','603788','603813','603825','603839','603855','603856','603867','603887','603890','603893','603906','603916','603920','603949','603955','603987','603995','688002','688005','688006','688008','688011','688012','688015','688016','688018','688019','688020','688021','688023','688025','688029','688030','688036','688037','688039','688051','688058','688066','688080','688085','688086','688088','688099','688100','688108','688111','688116','688122','688126','688139','688158','688159','688166','688169','688177','688181','688188','688189','688196','688198','688200','688202','688208','688222','688233','688258','688266','688268','688278','688288','688298','688299','688300','688318','688333','688357','688358','688363','688366','688368','688369','688388','688389','688396','000001','000002','000008','000009','000012','000021','000027','000028','000030','000031','000034','000039','000046','000049','000050','000059','000060','000061','000062','000063','000066','000069','000078','000089','000090','000096','000100','000156','000157','000158','000166','000301','000333','000338','000400','000401','000402','000413','000415','000423','000425','000488','000501','000513','000516','000517','000528','000537','000538','000540','000543','000547','000550','000555','000559','000563','000564','000568','000581','000582','000591','000596','000597','000598','000600','000601','000603','000623','000625','000627','000629','000630','000651','000656','000657','000661','000667','000671','000672','000681','000683','000685','000686','000688','000690','000703','000708','000709','000710','000712','000717','000718','000719','000723','000725','000726','000728','000729','000732','000733','000735','000738','000739','000750','000758','000768','000776','000778','000783','000786','000789','000799','000800','000807','000818','000825','000830','000848','000858','000860','000869','000876','000877','000878','000881','000883','000887','000895','000898','000902','000910','000921','000930','000932','000933','000936','000937','000938','000951','000958','000959','000960','000961','000963','000967','000970','000975','000976','000977','000983','000987','000988','000990','000997','000998','000999','001696','001914','001979','002001','002002','002004','002007','002008','002010','002013','002016','002019','002020','002022','002023','002024','002025','002027','002028','002030','002032','002035','002036','002038','002041','002042','002044','002048','002049','002050','002056','002063','002064','002065','002074','002075','002078','002080','002081','002085','002091','002092','002099','002100','002106','002110','002118','002120','002123','002124','002126','002127','002128','002129','002131','002138','002142','002145','002146','002151','002152','002153','002155','002156','002157','002171','002174','002179','002180','002182','002183','002185','002191','002195','002202','002203','002204','002212','002214','002216','002217','002223','002230','002233','002236','002237','002239','002241','002242','002244','002249','002250','002251','002252','002254','002258','002262','002266','002268','002271','002273','002276','002281','002285','002287','002292','002294','002299','002301','002302','002304','002310','002311','002317','002318','002326','002332','002340','002351','002352','002353','002358','002367','002368','002371','002372','002373','002375','002382','002384','002385','002387','002389','002390','002396','002399','002402','002405','002407','002408','002409','002410','002414','002415','002416','002419','002421','002422','002423','002424','002428','002429','002430','002434','002439','002440','002444','002449','002456','002458','002459','002460','002461','002463','002465','002466','002468','002475','002481','002484','002488','002489','002493','002497','002498','002500','002505','002506','002507','002508','002511','002518','002531','002541','002544','002555','002557','002558','002563','002568','002572','002583','002589','002594','002595','002597','002600','002601','002602','002603','002605','002607','002610','002614','002616','002624','002625','002626','002635','002640','002641','002643','002648','002653','002665','002670','002673','002675','002677','002683','002690','002697','002698','002699','002701','002705','002706','002709','002714','002726','002727','002736','002737','002739','002745','002747','002755','002773','002777','002791','002793','002797','002798','002803','002807','002812','002815','002818','002821','002831','002832','002837','002838','002839','002841','002850','002851','002859','002867','002901','002912','002913','002916','002918','002920','002925','002926','002928','002936','002938','002939','002941','002948','002950','002958','002959','003816','300001','300002','300003','300009','300010','300012','300014','300015','300017','300020','300024','300026','300033','300036','300037','300043','300053','300054','300058','300059','300068','300070','300072','300073','300079','300080','300083','300085','300088','300113','300114','300115','300118','300119','300122','300123','300124','300131','300132','300133','300136','300142','300144','300145','300146','300149','300166','300168','300170','300177','300180','300182','300188','300194','300197','300203','300207','300212','300223','300226','300232','300233','300236','300244','300251','300253','300257','300271','300274','300285','300287','300294','300296','300297','300298','300300','300308','300315','300316','300319','300324','300326','300327','300328','300339','300347','300348','300349','300357','300365','300369','300373','300376','300377','300383','300394','300395','300398','300406','300408','300413','300418','300429','300433','300450','300451','300454','300456','300457','300458','300459','300463','300474','300476','300482','300487','300496','300498','300502','300523','300526','300527','300529','300552','300558','300567','300568','300572','300573','300577','300595','300596','300601','300602','300607','300613','300616','300618','300623','300624','300628','300630','300633','300634','300638','300659','300661','300662','300664','300666','300674','300676','300677','300679','300682','300685','300699','300702','300709','300724','300725','300726','300737','300741','300747','300750','300751','300755','300759','300760','300768','300770','300773','300775','300776','300777','300782','300783','300785','600000','600004','600007','600008','600009','600010','600011','600012','600015','600016','600018','600019','600021','600025','600026','600027','600028','600029','600030','600031','600036','600038','600039','600048','600050','600053','600056','600057','600060','600061','600062','600066','600068','600073','600079','600085','600089','600094','600104','600105','600109','600111','600115','600118','600120','600125','600132','600138','600141','600143','600150','600153','600155','600160','600161','600167','600170','600176','600177','600180','600183','600185','600188','600196','600197','600201','600206','600208','600210','600216','600219','600221','600223','600233','600236','600252','600256','600258','600260','600261','600271','600273','600276','600282','600285','600295','600297','600298','600299','600305','600309','600315','600323','600325','600326','600329','600332','600335','600337','600340','600346','600348','600350','600352','600362','600363','600366','600369','600372','600373','600376','600377','600378','600380','600383','600388','600392','600395','600398','600406','600409','600420','600422','600426','600436','600438','600446','600459','600460','600466','600486','600487','600489','600498','600500','600507','600511','600516','600519','600521','600522','600529','600531','600535','600536','600545','600547','600549','600557','600559','600563','600565','600566','600567','600570','600572','600577','600580','600582','600583','600584','600585','600587','600588','600597','600598','600600','600606','600612','600621','600637','600639','600641','600642','600643','600648','600655','600657','600660','600663','600667','600673','600674','600675','600682','600685','600688','600690','600694','600699','600702','600703','600704','600705','600711','600720','600728','600729','600733','600737','600739','600741','600742','600745','600748','600750','600754','600755','600756','600760','600763','600764','600765','600779','600782','600787','600795','600801','600803','600804','600808','600809','600811','600820','600823','600835','600837','600845','600846','600848','600859','600863','600867','600872','600875','600879','600885','600886','600887','600893','600895','600900','600909','600919','600926','600933','600958','600967','600970','600971','600973','600977','600984','600985','600988','600989','600993','600996','600998','600999','601000','601001','601003','601005','601006','601009','601012','601016','601018','601019','601021','601058','601066','601077','601088','601098','601099','601100','601108','601111','601117','601128','601138','601139','601155','601162','601166','601169','601186','601198','601200','601211','601216','601225','601229','601231','601233','601236','601238','601288','601311','601318','601319','601328','601333','601336','601360','601375','601377','601390','601398','601519','601555','601567','601577','601588','601598','601600','601601','601607','601615','601618','601628','601633','601636','601658','601666','601668','601669','601678','601688','601689','601698','601699','601717','601727','601766','601788','601799','601800','601801','601808','601811','601816','601818','601828','601838','601857','601858','601865','601866','601869','601872','601877','601878','601880','601881','601888','601898','601899','601901','601916','601919','601933','601939','601949','601952','601958','601966','601985','601988','601989','601990','601992','601997','601998','603008','603018','603019','603025','603027','603039','603043','603056','603060','603077','603113','603127','603128','603160','603180','603185','603218','603228','603233','603236','603259','603260','603267','603279','603288','603298','603305','603317','603328','603337','603338','603345','603355','603367','603369','603387','603429','603444','603466','603486','603489','603501','603515','603517','603520','603533','603556','603568','603579','603583','603587','603588','603589','603590','603596','603599','603601','603605','603606','603609','603613','603626','603638','603650','603658','603659','603666','603678','603686','603707','603708','603712','603713','603728','603737','603766','603786','603799','603801','603806','603808','603816','603833','603858','603866','603868','603871','603877','603881','603882','603883','603885','603888','603899','603915','603927','603939','603956','603960','603983','603986','603989','603993']

def extract_data_1d(code_list, start_date, end_date):
    candles = local_get_stock_day_adv(code_list, start=start_date, end=end_date).to_qfq()
    charts = []
    try:
        candles.data['ds'] = [x.strftime('%Y-%m-%d') for x in candles.data.reset_index().date.tolist()]
    except:
        candles.data['ds'] = [x.strftime('%Y-%m-%d') for x in candles.data.date.tolist()]
    for idx, candle in enumerate(candles.splits()):
        try:
            code = candle.data.head(1).reset_index().code.values[0]
        except:
            code = code_list[idx]
        chart = ChartFrame(code, 'day')
        chart.init_candles(candle.data.loc[:, ['open', 'close', 'high', 'low']].values, candle.data.ds.tolist())
        charts.append(chart)
    return candles, charts


def extract_data_min(code_list, start_date, end_date, min_level):
    candles = local_get_stock_min_adv(code_list, start=start_date, end=end_date, frequence=min_level).to_qfq()
    charts = []
    try:
        candles.data['ds'] = [x.strftime('%Y-%m-%d %H:%M:%S') for x in candles.data.reset_index().datetime.tolist()]
    except:
        candles.data['ds'] = [x.strftime('%Y-%m-%d %H:%M:%S') for x in candles.data.datetime.tolist()]
    for idx, candle in enumerate(candles.splits()):
        try:
            code = candle.data.head(1).reset_index().code.values[0]
        except:
            code = code_list[idx]
        chart = ChartFrame(code, min_level)
        chart.init_candles(candle.data.loc[:, ['open', 'close', 'high', 'low']].values, candle.data.ds.tolist())
        charts.append(chart)
    return candles, charts


if __name__ == '__main__':
    end_date = dt.datetime.today()
    # end_date = dt.datetime(2020, 10, 23)
    start_date = end_date - dt.timedelta(days=365)

    stocks = beiji
    # stocks = ['000528', '002049', '300529', '300607', '600518', '600588', '603877']
    # stocks = ['300638', '600516']
    # stocks = ['002621']

    candles, charts = extract_data_1d(stocks, start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d'))

    indicator = ChanIndicator()
    near_lines = [(chart.code, indicator.near_pivot_lines(chart)) for chart in charts]
    chosen = [(code, line) for code, line in near_lines if line > 0]
    print(chosen)




