import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

sns.set()

all_accuracies = [0.03802083432674408, 0.17135417461395264, 0.1770833432674408, 0.30520835518836975, 0.3177083432674408, 0.3447916805744171, 0.3604166805744171, 0.3958333432674408, 0.3973958492279053, 0.4052083492279053, 0.4411458671092987, 0.45364585518836975, 0.4385416805744171, 0.468229204416275, 0.4593750238418579, 0.4958333671092987, 0.48593753576278687, 0.5104166865348816, 0.48229169845581055, 0.5140625238418579, 0.526562511920929, 0.5348958373069763, 0.5395833849906921, 0.5078125, 0.5536458492279053, 0.5609375238418579, 0.596875011920929, 0.5703125, 0.5760416984558105, 0.5770833492279053, 0.5791667103767395, 0.5526041984558105, 0.5583333373069763, 0.6005208492279053, 0.5984375476837158, 0.5916666984558105, 0.5838541984558105, 0.5942708849906921, 0.5947917103767395, 0.5817708373069763, 0.5828125476837158, 0.5848958492279053, 0.5770833492279053, 0.6057292222976685, 0.6151041984558105, 0.6041666865348816, 0.6151041984558105, 0.6010416746139526, 0.6015625596046448, 0.6229166984558105, 0.6255208849906921, 0.6140625476837158, 0.6020833849906921, 0.6041666865348816, 0.6020833849906921, 0.6411458849906921, 0.6239583492279053, 0.6338542103767395, 0.6338542103767395, 0.6182292103767395, 0.5953125357627869, 0.6255208849906921, 0.5989583730697632, 0.6052083373069763, 0.635937511920929, 0.6078125238418579, 0.6036458611488342, 0.6250000596046448, 0.6078125238418579, 0.6098958849906921, 0.6130208373069763, 0.6145833730697632, 0.6208333373069763, 0.6078125238418579, 0.635937511920929, 0.651562511920929, 0.6427083611488342, 0.6171875596046448, 0.6057292222976685, 0.6234375238418579, 0.6187500357627869, 0.6218750476837158, 0.6208333373069763, 0.6328125596046448, 0.6114583611488342, 0.6057292222976685, 0.6369792222976685, 0.6239583492279053, 0.6286458373069763, 0.6239583492279053, 0.6156250238418579, 0.6140625476837158, 0.6390625238418579, 0.6192708611488342, 0.6276041865348816, 0.6177083849906921, 0.620312511920929, 0.6234375238418579, 0.6265625357627869, 0.6380208730697632, 0.6276041865348816, 0.6401041746139526, 0.6479166746139526, 0.6307291984558105, 0.6598958969116211, 0.6296875476837158, 0.6416667103767395, 0.6401041746139526, 0.6494792103767395, 0.6265625357627869, 0.6307291984558105, 0.6348958611488342, 0.6145833730697632, 0.6260417103767395, 0.6322916746139526, 0.6161458492279053, 0.6427083611488342, 0.6208333373069763, 0.6604167222976685, 0.6369792222976685, 0.6390625238418579, 0.6531250476837158, 0.6260417103767395, 0.6390625238418579, 0.6609375476837158, 0.6598958969116211, 0.6447917222976685, 0.6583333611488342, 0.6489583849906921, 0.6291667222976685, 0.6479166746139526, 0.6276041865348816, 0.6135417222976685, 0.6729167103767395, 0.6500000357627869, 0.6432291865348816, 0.6171875596046448, 0.6421875357627869, 0.6421875357627869, 0.6598958969116211, 0.6354166865348816, 0.628125011920929, 0.6609375476837158, 0.6375000476837158, 0.6364583373069763, 0.6510416865348816, 0.6244791746139526, 0.6463541984558105, 0.6354166865348816, 0.6390625238418579, 0.6489583849906921, 0.6411458849906921, 0.6609375476837158, 0.6468750238418579, 0.6572917103767395, 0.643750011920929, 0.6546875238418579, 0.6401041746139526, 0.6390625238418579, 0.6421875357627869, 0.6479166746139526, 0.6427083611488342, 0.6296875476837158, 0.6567708849906921, 0.635937511920929, 0.6494792103767395, 0.6427083611488342, 0.6307291984558105, 0.6177083849906921, 0.6661458611488342, 0.6354166865348816, 0.6552083492279053, 0.6640625596046448, 0.6588541865348816, 0.6296875476837158, 0.6307291984558105, 0.6395833492279053, 0.6328125596046448, 0.6609375476837158, 0.6687500476837158, 0.6500000357627869, 0.6588541865348816, 0.6687500476837158, 0.6505208611488342, 0.6302083730697632, 0.659375011920929, 0.6562500596046448, 0.6317708492279053, 0.6677083969116211, 0.6375000476837158, 0.6286458373069763, 0.6739583611488342, 0.6468750238418579, 0.6588541865348816, 0.6453125476837158, 0.6291667222976685, 0.643750011920929, 0.6552083492279053, 0.6317708492279053, 0.6598958969116211, 0.6473958492279053, 0.6536458730697632, 0.6463541984558105, 0.6442708373069763, 0.6505208611488342, 0.6630208492279053, 0.6432291865348816, 0.6567708849906921, 0.6380208730697632, 0.6291667222976685, 0.6489583849906921, 0.6734375357627869, 0.6598958969116211, 0.6609375476837158, 0.6390625238418579, 0.6348958611488342, 0.6562500596046448, 0.6531250476837158, 0.6572917103767395, 0.6500000357627869, 0.6500000357627869, 0.6479166746139526, 0.6468750238418579, 0.6645833849906921, 0.6692708730697632, 0.6531250476837158, 0.6614583730697632, 0.6406250596046448, 0.651562511920929, 0.6786458492279053, 0.6416667103767395, 0.6692708730697632, 0.6484375596046448, 0.6661458611488342, 0.6640625596046448, 0.6645833849906921, 0.6536458730697632, 0.6369792222976685, 0.6348958611488342, 0.6546875238418579, 0.6494792103767395, 0.6651042103767395, 0.6598958969116211, 0.6510416865348816, 0.6567708849906921, 0.6526042222976685, 0.6380208730697632, 0.6739583611488342, 0.6531250476837158, 0.6645833849906921, 0.6552083492279053, 0.6473958492279053, 0.6562500596046448, 0.6572917103767395, 0.6609375476837158, 0.659375011920929, 0.6427083611488342, 0.6343750357627869, 0.6739583611488342, 0.6625000238418579, 0.6708333492279053, 0.6520833969116211, 0.6562500596046448, 0.6588541865348816, 0.6567708849906921, 0.6520833969116211, 0.6640625596046448, 0.6619791984558105, 0.6536458730697632, 0.6427083611488342, 0.6468750238418579, 0.6390625238418579, 0.6588541865348816, 0.6500000357627869, 0.6546875238418579, 0.6614583730697632, 0.6531250476837158, 0.6447917222976685, 0.6562500596046448, 0.6619791984558105, 0.6723958849906921, 0.6656250357627869, 0.6546875238418579, 0.6625000238418579, 0.6687500476837158, 0.6416667103767395, 0.6557291746139526, 0.6401041746139526, 0.6656250357627869, 0.6677083969116211, 0.6567708849906921, 0.6598958969116211, 0.6510416865348816, 0.6572917103767395, 0.6541666984558105, 0.6531250476837158, 0.6489583849906921, 0.6682292222976685, 0.6666666865348816, 0.659375011920929, 0.6687500476837158, 0.682812511920929, 0.6578125357627869, 0.6442708373069763, 0.6619791984558105, 0.6802083849906921, 0.651562511920929, 0.6703125238418579, 0.6682292222976685, 0.6942708492279053, 0.6760417222976685, 0.6656250357627869, 0.6583333611488342, 0.6661458611488342, 0.6578125357627869, 0.6567708849906921, 0.6645833849906921, 0.6567708849906921, 0.6598958969116211, 0.6677083969116211, 0.6526042222976685, 0.6500000357627869, 0.6526042222976685, 0.6557291746139526, 0.6645833849906921, 0.6447917222976685, 0.6520833969116211, 0.6500000357627869, 0.6739583611488342, 0.6718750596046448, 0.6697916984558105, 0.6765625476837158, 0.6645833849906921, 0.6687500476837158, 0.6500000357627869, 0.6500000357627869, 0.6468750238418579, 0.682812511920929, 0.6651042103767395, 0.6484375596046448, 0.6531250476837158, 0.6536458730697632, 0.6578125357627869, 0.6723958849906921, 0.6598958969116211, 0.6578125357627869, 0.6776041984558105, 0.6687500476837158, 0.6625000238418579, 0.6718750596046448, 0.6651042103767395, 0.6703125238418579, 0.6640625596046448, 0.6588541865348816, 0.6468750238418579, 0.6942708492279053, 0.6723958849906921, 0.6692708730697632, 0.6604167222976685]

all_losses = [3.660447359085083, 2.9014883041381836, 2.8076560497283936, 2.4284424781799316, 2.266242504119873, 2.2499961853027344, 2.126141309738159, 2.0501718521118164, 2.012974500656128, 2.006805181503296, 1.9473271369934082, 1.8348994255065918, 1.8810924291610718, 1.7680885791778564, 1.7934256792068481, 1.6795294284820557, 1.6939029693603516, 1.628973364830017, 1.7116574048995972, 1.5994430780410767, 1.6172242164611816, 1.5725438594818115, 1.5272185802459717, 1.6263715028762817, 1.4769338369369507, 1.4821679592132568, 1.3960298299789429, 1.44745934009552, 1.4007415771484375, 1.3829810619354248, 1.3389402627944946, 1.477675199508667, 1.4099323749542236, 1.2992699146270752, 1.2886193990707397, 1.3686387538909912, 1.4023298025131226, 1.3463363647460938, 1.293533205986023, 1.3782049417495728, 1.3434207439422607, 1.3157230615615845, 1.3822218179702759, 1.2858860492706299, 1.2545509338378906, 1.2905102968215942, 1.2794281244277954, 1.3001788854599, 1.28895103931427, 1.2656110525131226, 1.2492274045944214, 1.2720664739608765, 1.2625329494476318, 1.2885360717773438, 1.2794716358184814, 1.1868977546691895, 1.2249579429626465, 1.200332760810852, 1.1633312702178955, 1.25146484375, 1.2862881422042847, 1.2466678619384766, 1.3051611185073853, 1.2592031955718994, 1.207075595855713, 1.244838833808899, 1.3182940483093262, 1.2366682291030884, 1.222609281539917, 1.2635729312896729, 1.2199969291687012, 1.2252877950668335, 1.2104682922363281, 1.2426906824111938, 1.2193737030029297, 1.1371887922286987, 1.1522115468978882, 1.2090359926223755, 1.268452763557434, 1.1748368740081787, 1.1907668113708496, 1.2205754518508911, 1.1952900886535645, 1.1960647106170654, 1.2304166555404663, 1.253043293952942, 1.1639024019241333, 1.1981582641601562, 1.1740314960479736, 1.1845945119857788, 1.2172284126281738, 1.2358624935150146, 1.156226634979248, 1.2170385122299194, 1.19456946849823, 1.2272207736968994, 1.1910969018936157, 1.1838754415512085, 1.195850133895874, 1.180152177810669, 1.1959031820297241, 1.1691261529922485, 1.100374698638916, 1.1880524158477783, 1.1012526750564575, 1.1969025135040283, 1.1381670236587524, 1.1365597248077393, 1.1307671070098877, 1.1442831754684448, 1.1449886560440063, 1.1756939888000488, 1.197033166885376, 1.1990337371826172, 1.1535025835037231, 1.2072590589523315, 1.1489677429199219, 1.186607003211975, 1.11637282371521, 1.0947890281677246, 1.1436086893081665, 1.125074863433838, 1.1976016759872437, 1.1506518125534058, 1.0933431386947632, 1.081484317779541, 1.1091302633285522, 1.1180716753005981, 1.1013554334640503, 1.1718698740005493, 1.0952619314193726, 1.1900973320007324, 1.2142713069915771, 1.0449528694152832, 1.1140693426132202, 1.1560957431793213, 1.2171130180358887, 1.1058708429336548, 1.1553499698638916, 1.0932137966156006, 1.161903738975525, 1.1685682535171509, 1.064914345741272, 1.152369737625122, 1.135412573814392, 1.1084446907043457, 1.1999249458312988, 1.1607407331466675, 1.1383968591690063, 1.1681047677993774, 1.1283289194107056, 1.1444697380065918, 1.079633116722107, 1.1087126731872559, 1.0602136850357056, 1.1280349493026733, 1.076183795928955, 1.1296700239181519, 1.1277556419372559, 1.1627439260482788, 1.1259948015213013, 1.1153855323791504, 1.1606996059417725, 1.0711922645568848, 1.1027541160583496, 1.120632529258728, 1.1250026226043701, 1.1674574613571167, 1.1537524461746216, 1.0736628770828247, 1.1698532104492188, 1.1270582675933838, 1.076999545097351, 1.0758570432662964, 1.1820076704025269, 1.175117015838623, 1.1126760244369507, 1.1478391885757446, 1.085309386253357, 1.0521548986434937, 1.1188066005706787, 1.0762791633605957, 1.0607059001922607, 1.134040117263794, 1.1497899293899536, 1.0977661609649658, 1.07683527469635, 1.14151132106781, 1.0568517446517944, 1.1520764827728271, 1.1628354787826538, 1.0374693870544434, 1.161785364151001, 1.059587836265564, 1.1136722564697266, 1.1481765508651733, 1.1492763757705688, 1.1091548204421997, 1.1406099796295166, 1.0389269590377808, 1.088776707649231, 1.0636723041534424, 1.130308747291565, 1.140733242034912, 1.0958553552627563, 1.043022632598877, 1.1512020826339722, 1.0667471885681152, 1.0882761478424072, 1.1924757957458496, 1.1168972253799438, 1.0968825817108154, 1.0506678819656372, 1.0677282810211182, 1.1000885963439941, 1.1435574293136597, 1.0991666316986084, 1.0890132188796997, 1.0976853370666504, 1.0762618780136108, 1.1147083044052124, 1.1130197048187256, 1.0955079793930054, 1.0911787748336792, 1.057212471961975, 1.0908427238464355, 1.0556279420852661, 1.1427541971206665, 1.091607928276062, 1.0071911811828613, 1.1048941612243652, 1.0472369194030762, 1.150045394897461, 1.0860822200775146, 1.0718281269073486, 1.0736032724380493, 1.106990933418274, 1.1558154821395874, 1.1516278982162476, 1.0934014320373535, 1.0896782875061035, 1.0514949560165405, 1.0582002401351929, 1.1164060831069946, 1.0777379274368286, 1.0734525918960571, 1.1375994682312012, 1.0176717042922974, 1.0728230476379395, 1.0547246932983398, 1.0724965333938599, 1.0812760591506958, 1.0415819883346558, 1.048016905784607, 1.0806176662445068, 1.0784822702407837, 1.1491672992706299, 1.168644666671753, 1.036926507949829, 1.0811349153518677, 1.0598856210708618, 1.0972636938095093, 1.0323702096939087, 1.0573203563690186, 1.0656265020370483, 1.1246708631515503, 1.031866192817688, 1.0460388660430908, 1.0721598863601685, 1.11871337890625, 1.0921322107315063, 1.0857096910476685, 1.089826226234436, 1.1066550016403198, 1.0947378873825073, 1.0573574304580688, 1.053568720817566, 1.0934364795684814, 1.0554194450378418, 1.064148187637329, 1.0333001613616943, 1.0417873859405518, 1.0834426879882812, 1.0390318632125854, 1.0587067604064941, 1.106510043144226, 1.086755633354187, 1.098537564277649, 1.0500900745391846, 1.0587319135665894, 1.0660951137542725, 1.0508922338485718, 1.110082745552063, 1.0489753484725952, 1.0959300994873047, 1.0858032703399658, 1.0872161388397217, 1.0098326206207275, 1.0532660484313965, 1.093226671218872, 1.0438673496246338, 0.9764826893806458, 1.0318892002105713, 1.0907347202301025, 1.061163067817688, 1.0391701459884644, 1.0853205919265747, 1.0352109670639038, 1.032716155052185, 0.9701747894287109, 1.0219862461090088, 1.048020601272583, 1.0637292861938477, 1.0889930725097656, 1.0673710107803345, 1.0794881582260132, 1.0658904314041138, 1.0342540740966797, 1.0441814661026, 1.031177043914795, 1.0680574178695679, 1.1055599451065063, 1.070766568183899, 1.085700511932373, 1.0545027256011963, 1.1261101961135864, 1.0796470642089844, 1.1024614572525024, 1.0294723510742188, 1.0297489166259766, 1.0318852663040161, 1.0298737287521362, 1.034635066986084, 1.0360920429229736, 1.0857478380203247, 1.089893102645874, 1.114135980606079, 0.9883958697319031, 1.0384602546691895, 1.0962737798690796, 1.0915690660476685, 1.0723235607147217, 1.0690524578094482, 1.0700243711471558, 1.0619200468063354, 1.0671459436416626, 1.045832633972168, 1.0254806280136108, 1.065258264541626, 1.0174100399017334, 1.0522648096084595, 1.0200190544128418, 1.0686650276184082, 1.056827425956726, 1.0944479703903198, 0.9926839470863342, 1.002618670463562, 1.0579934120178223, 1.059679627418518]

all_train_steps = [0, 50, 100, 150, 200, 250, 300, 350, 400, 450, 500, 550, 600, 650, 700, 750, 800, 850, 900, 950, 1000, 1050, 1100, 1150, 1200, 1250, 1300, 1350, 1400, 1450, 1500, 1550, 1600, 1650, 1700, 1750, 1800, 1850, 1900, 1950, 2000, 2050, 2100, 2150, 2200, 2250, 2300, 2350, 2400, 2450, 2500, 2550, 2600, 2650, 2700, 2750, 2800, 2850, 2900, 2950, 3000, 3050, 3100, 3150, 3200, 3250, 3300, 3350, 3400, 3450, 3500, 3550, 3600, 3650, 3700, 3750, 3800, 3850, 3900, 3950, 4000, 4050, 4100, 4150, 4200, 4250, 4300, 4350, 4400, 4450, 4500, 4550, 4600, 4650, 4700, 4750, 4800, 4850, 4900, 4950, 5000, 5050, 5100, 5150, 5200, 5250, 5300, 5350, 5400, 5450, 5500, 5550, 5600, 5650, 5700, 5750, 5800, 5850, 5900, 5950, 6000, 6050, 6100, 6150, 6200, 6250, 6300, 6350, 6400, 6450, 6500, 6550, 6600, 6650, 6700, 6750, 6800, 6850, 6900, 6950, 7000, 7050, 7100, 7150, 7200, 7250, 7300, 7350, 7400, 7450, 7500, 7550, 7600, 7650, 7700, 7750, 7800, 7850, 7900, 7950, 8000, 8050, 8100, 8150, 8200, 8250, 8300, 8350, 8400, 8450, 8500, 8550, 8600, 8650, 8700, 8750, 8800, 8850, 8900, 8950, 9000, 9050, 9100, 9150, 9200, 9250, 9300, 9350, 9400, 9450, 9500, 9550, 9600, 9650, 9700, 9750, 9800, 9850, 9900, 9950, 10000, 10050, 10100, 10150, 10200, 10250, 10300, 10350, 10400, 10450, 10500, 10550, 10600, 10650, 10700, 10750, 10800, 10850, 10900, 10950, 11000, 11050, 11100, 11150, 11200, 11250, 11300, 11350, 11400, 11450, 11500, 11550, 11600, 11650, 11700, 11750, 11800, 11850, 11900, 11950, 12000, 12050, 12100, 12150, 12200, 12250, 12300, 12350, 12400, 12450, 12500, 12550, 12600, 12650, 12700, 12750, 12800, 12850, 12900, 12950, 13000, 13050, 13100, 13150, 13200, 13250, 13300, 13350, 13400, 13450, 13500, 13550, 13600, 13650, 13700, 13750, 13800, 13850, 13900, 13950, 14000, 14050, 14100, 14150, 14200, 14250, 14300, 14350, 14400, 14450, 14500, 14550, 14600, 14650, 14700, 14750, 14800, 14850, 14900, 14950, 15000, 15050, 15100, 15150, 15200, 15250, 15300, 15350, 15400, 15450, 15500, 15550, 15600, 15650, 15700, 15750, 15800, 15850, 15900, 15950, 16000, 16050, 16100, 16150, 16200, 16250, 16300, 16350, 16400, 16450, 16500, 16550, 16600, 16650, 16700, 16750, 16800, 16850, 16900, 16950, 17000, 17050, 17100, 17150, 17200, 17250, 17300, 17350, 17400, 17450, 17500, 17550, 17600, 17650, 17700, 17750, 17800, 17850, 17900]

plt.figure(figsize=(14, 6))

plt.subplot(1, 2, 1)
plt.plot(all_train_steps, all_accuracies, color="orange", label="accuracy")
plt.title("Accuracy over training time")
plt.xlabel("train steps")
plt.ylabel("accuracy")

plt.subplot(1, 2, 2)
plt.plot(all_train_steps, all_losses, label="loss")
plt.title("Loss over training time")
plt.xlabel("train steps")
plt.ylabel("loss")

plt.show()