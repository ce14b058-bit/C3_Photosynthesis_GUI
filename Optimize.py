
import numpy as np
from gekko import GEKKO

def photosynthesis_optimize(weather,observed_data):
    Photosynthesis = {}
    Photosynthesis['vcmax25'] = 100  
    Photosynthesis['jmax25']= 100  
    Photosynthesis['tpu25'] = 5
    Photosynthesis['rd25']= 3 
    Photosynthesis['gm25']= 1
    photosynthesis_array = np.array([x for x in Photosynthesis.values()])
    R = 8.314/1000 
    ci = np.array(observed_data.loc[:,'Ci'])
    anet = np.array(observed_data.loc[:,'Anet'])
    temperature = weather.loc[0,'temperature']
    temperature_kelvin = temperature+273.15
    pressure = weather.loc[0,'pressure']
    o2 = weather.loc[0,'O2']
    Ko = 1000/pressure*np.exp(12.3772 - 23720/(R * 1000.0 * temperature_kelvin))
    Kc = 1000/pressure*np.exp(35.9774 - 80990 /(R * 1000.0 * temperature_kelvin))

    gm = lambda p : p*np.exp(20.01 - (49600 / (R * 1000.0 * temperature_kelvin)))/(1+np.exp((1400 * temperature_kelvin - 437400) / (R * 1000.0 * temperature_kelvin))) #p4

    cm = lambda p,i:ci[i] - anet[i]/(gm(p)) #p4

    rd = lambda p : p*np.exp(18.7145 - 46.39 /(R * temperature_kelvin)) #p3
    vc_max = lambda p : p * np.exp(26.355 - 65.33 / (R * temperature_kelvin)) #p0
    gamma = 1000/pressure*np.exp(11.187 - 24460/(R * 1000.0 * temperature_kelvin)) 
    # gamma = 1000/pressure*np.exp(9.036 - 18685/(R * 1000.0 * temperature_kelvin))
    J = lambda p: p * np.exp(17.71 - 43.9 / (R * temperature_kelvin))  #p1
    tpu = lambda p :p*np.exp(21.46 - (53100 / (R * 1000.0 * temperature_kelvin))) \
        /(1+np.exp((650 * temperature_kelvin- 201800) / (R* 1000.0 * temperature_kelvin))) #p2
    
    aj = lambda p,i :(1.0 - gamma /cm(p[4],i)) * (J(p[1]) * cm(p[4],i)) /(4 * cm(p[4],i) + 8 * gamma) # Light limited photosynthesis (vonCaemmerer 2000)

    # CO2 limited photosynthesis
    ac = lambda p,i : (1.0 - gamma / cm(p[4],i)) * (vc_max(p[0]) * cm(p[4],i)) /(cm(p[4],i) + Kc * (1.0 + o2 / Ko)) # Rubisco limited photosynthesis (vonCaemmerer 2000)

    ap = lambda p: (3.0 * tpu(p[2])) # TPU limited photosynthesis (vonCaemmerer 2000)

    anet_ac = lambda p,i : ac(p,i) - rd(p[3])
    anet_ap = lambda p : ap(p) - rd(p[3])
    anet_aj = lambda p,i : aj(p,i) - rd(p[3])
    
    n1 = len(observed_data.loc[observed_data.flag < 2, 'flag'])
    n2 = len(observed_data.loc[observed_data.flag < 3, 'flag'])
    n3 = len(observed_data.loc[observed_data.flag < 4, 'flag'])

    #sum of squared errors
    sse_ac = lambda p : np.sum(np.square(np.array([anet_ac(p,i) - observed_data.loc[i,'Anet'] for i in range(n1)]) ))
    sse_ap = lambda p : np.sum(np.square(np.array([anet_ap(p) - observed_data.loc[i,'Anet'] for i in range(n2,n3)]) ))
    sse_aj = lambda p : np.sum(np.square(np.array([anet_aj(p,i) - observed_data.loc[i,'Anet'] for i in range(n1,n2)]) ))

    m1 = GEKKO(remote = False)
    x1 = m1.Var(value=photosynthesis_array[0],lb = 10,ub = 1000)
    x2 = m1.Var(value=photosynthesis_array[1],lb = 10,ub = 1000)
    x3 = m1.Var(value=photosynthesis_array[2],lb = 0,ub = 30)
    x4 = m1.Var(value=photosynthesis_array[3],lb = 0.1,ub = 10)
    x5 = m1.Var(value=photosynthesis_array[4],lb = 0,ub = 3 )

    m1.Minimize(sse_ac([x1,x2,x3,x4,x5]))
    m1.Minimize(sse_aj([x1,x2,x3,x4,x5]))
    m1.Minimize(sse_ap([x1,x2,x3,x4,x5]))
    m1.solve()

    photosynthesis_array = np.array(x1.value + x2.value + x3.value + x4.value + x5.value)

    return photosynthesis_array,n1,n2-n1,n3-n2